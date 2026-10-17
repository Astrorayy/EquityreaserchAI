"""Market data.

Two providers, in order of preference:

1. :class:`YFinanceProvider` - **optional**, unofficial, indicative only, US
   tickers only.  Requires the ``yfinance`` package (see
   ``requirements-optional.txt``).  Every quote it returns is flagged
   ``is_official = False``.
2. :class:`ManualPriceProvider` - a price the analyst typed in, stored in
   SQLite.  Always labelled as manual.

If neither is available the application says so instead of inventing a price.
The Casablanca Stock Exchange is not covered by yfinance; a Managem price
therefore has to be entered manually or imported from the official bulletin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Dict, List, Optional

from config import Settings, get_settings
from financial_engine.schemas import Currency


@dataclass
class Quote:
    company_id: str
    ticker: str
    price: Optional[float]
    currency: Currency
    as_of: Optional[datetime] = None
    source_name: str = ""
    source_url: Optional[str] = None
    is_official: bool = False
    is_delayed: bool = True
    note: Optional[str] = None
    error: Optional[str] = None
    history: List[Dict[str, object]] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return self.price is not None and self.error is None

    @property
    def label(self) -> str:
        if self.is_official:
            return "source officielle"
        if self.source_name.startswith("Saisie"):
            return "saisie manuelle"
        return "source non officielle / indicative"


class MarketDataProvider:
    name = "base"

    def quote(self, company_id: str, ticker: str, currency: Currency, **kwargs) -> Quote:
        raise NotImplementedError


# --------------------------------------------------------------------------
# yfinance (optional)
# --------------------------------------------------------------------------


class YFinanceProvider(MarketDataProvider):
    name = "yfinance"

    def __init__(self, settings: Optional[Settings] = None, period: str = "5y"):
        self.settings = settings or get_settings()
        self.period = period

    @staticmethod
    def is_installed() -> bool:
        try:
            import yfinance  # noqa: F401

            return True
        except ImportError:
            return False

    @property
    def available(self) -> bool:
        return self.settings.enable_yfinance and self.is_installed()

    def quote(self, company_id: str, ticker: str, currency: Currency, period: Optional[str] = None) -> Quote:
        base = Quote(
            company_id=company_id, ticker=ticker, price=None, currency=currency,
            source_name="yfinance", source_url=f"https://finance.yahoo.com/quote/{ticker}",
        )
        if not self.settings.enable_yfinance:
            base.error = (
                "yfinance est desactive (ERAI_ENABLE_YFINANCE=false). "
                "Saisissez le cours manuellement."
            )
            return base
        if not self.is_installed():
            base.error = (
                "Le paquet optionnel 'yfinance' n'est pas installe. Installez-le avec "
                "'pip install -r requirements-optional.txt' ou saisissez le cours manuellement."
            )
            return base

        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover
            base.error = str(exc)
            return base

        try:
            handle = yf.Ticker(ticker)
            history = handle.history(period=period or self.period, auto_adjust=False)
        except Exception as exc:  # pragma: no cover - network/library dependent
            base.error = f"yfinance a échoué : {type(exc).__name__}: {exc}"
            return base

        if history is None or history.empty:
            base.error = (
                f"Aucune donnée de marché yfinance pour '{ticker}'. Les valeurs non "
                "américaines, notamment celles de la Bourse de Casablanca, ne sont "
                "généralement pas couvertes. Saisissez le cours manuellement."
            )
            return base

        try:
            last = history.iloc[-1]
            base.price = float(last["Close"])
            index = history.index[-1]
            base.as_of = index.to_pydatetime() if hasattr(index, "to_pydatetime") else None
            if base.as_of is not None and base.as_of.tzinfo is None:
                base.as_of = base.as_of.replace(tzinfo=timezone.utc)
        except Exception as exc:  # pragma: no cover
            base.error = f"Lecture du dernier cours impossible : {exc}"
            return base

        base.is_official = False
        base.is_delayed = True
        base.note = (
            "Source non officielle (yfinance / Yahoo Finance), cours différé, usage "
            "indicatif uniquement. Ne l'utilisez pas comme référence de marché "
            "réglementaire. Vérifiez le cours auprès de la place de cotation."
        )
        base.history = self._to_history(history)
        return base

    @staticmethod
    def _to_history(history, limit: int = 1300) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        try:
            tail = history.tail(limit)
            for index, row in tail.iterrows():
                stamp = index.to_pydatetime() if hasattr(index, "to_pydatetime") else index
                out.append(
                    {
                        "date": stamp.date().isoformat() if hasattr(stamp, "date") else str(stamp),
                        "close": float(row["Close"]),
                        "volume": float(row["Volume"]) if "Volume" in row and row["Volume"] == row["Volume"] else None,
                    }
                )
        except Exception:  # pragma: no cover
            return []
        return out


# --------------------------------------------------------------------------
# Manual price
# --------------------------------------------------------------------------


class ManualPriceProvider(MarketDataProvider):
    """A price typed by the analyst and stored in the local database."""

    name = "manual"

    FIELD = "market_price"

    def __init__(self, db=None, settings: Optional[Settings] = None):
        self.db = db
        self.settings = settings or get_settings()

    @property
    def available(self) -> bool:
        return self.db is not None

    def stored_price(self, company_id: str) -> Optional[float]:
        if self.db is None:
            return None
        from storage.repository import get_manual_overrides

        return get_manual_overrides(self.db, company_id).get(self.FIELD)

    def set_price(self, company_id: str, price: float, currency: Currency, note: str = "") -> None:
        if self.db is None:
            raise RuntimeError("No database configured for manual price entry")
        from storage.repository import set_manual_override

        set_manual_override(
            self.db, company_id, self.FIELD, float(price),
            note or f"Cours saisi manuellement ({currency.value}).",
        )

    def quote(self, company_id: str, ticker: str, currency: Currency, **kwargs) -> Quote:
        price = self.stored_price(company_id)
        quote = Quote(
            company_id=company_id,
            ticker=ticker,
            price=price,
            currency=currency,
            as_of=datetime.now(timezone.utc) if price is not None else None,
            source_name="Saisie manuelle de l'analyste",
            is_official=False,
            is_delayed=True,
        )
        if price is None:
            quote.error = (
                "Aucun cours disponible : ni fournisseur automatique, ni saisie "
                "manuelle pour cette société. Le cours peut être saisi dans l'onglet "
                "Paramètres."
            )
        else:
            quote.note = (
                "Cours saisi manuellement par l'analyste. Il ne provient d'aucune source "
                "automatisée et doit être vérifié."
            )
        return quote


# --------------------------------------------------------------------------
# Service
# --------------------------------------------------------------------------


class MarketDataService:
    """Try the automatic provider, then fall back to the manual price."""

    def __init__(self, db=None, settings: Optional[Settings] = None, http=None):
        self.settings = settings or get_settings()
        self.db = db
        self.automatic = YFinanceProvider(self.settings)
        self.manual = ManualPriceProvider(db, self.settings)

    @property
    def automatic_available(self) -> bool:
        return self.automatic.available

    def quote(self, company_id: str, ticker: str, currency: Currency, prefer_manual: bool = False) -> Quote:
        if prefer_manual:
            manual = self.manual.quote(company_id, ticker, currency)
            if manual.available:
                return manual

        auto = self.automatic.quote(company_id, ticker, currency)
        if auto.available:
            manual_price = self.manual.stored_price(company_id)
            if manual_price is not None:
                auto.note = (auto.note or "") + (
                    f" Un cours saisi manuellement ({manual_price:,.2f} {currency.value}) "
                    "existe aussi : la saisie manuelle est prioritaire si vous l'activez."
                )
            return auto

        manual = self.manual.quote(company_id, ticker, currency)
        if manual.available:
            manual.note = (manual.note or "") + (
                f" Cours automatique indisponible : {auto.error}"
            )
            return manual

        # Nothing available: report both failures rather than inventing a price.
        auto.error = (
            f"{auto.error}\n"
            f"Repli manuel : {manual.error}"
        )
        return auto

    def capabilities(self) -> Dict[str, object]:
        return {
            "yfinance_installed": YFinanceProvider.is_installed(),
            "yfinance_enabled": self.settings.enable_yfinance,
            "automatic_available": self.automatic_available,
            "manual_available": self.manual.available,
            "note": (
                "Les cours américains peuvent être récupérés via yfinance (non officiel). "
                "Aucun fournisseur automatique ne couvre la Bourse de Casablanca : "
                "le cours de Managem doit être saisi manuellement ou repris du bulletin "
                "officiel de la Bourse."
            ),
        }


def compute_market_cap(price: Optional[float], shares_millions: Optional[float]) -> Optional[float]:
    """Market capitalisation in millions of currency, or ``None``."""
    if price is None or shares_millions is None:
        return None
    if price <= 0 or shares_millions <= 0:
        return None
    return price * shares_millions
