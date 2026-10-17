"""Comparable-company (multiples) valuation.

Guardrails implemented here
---------------------------
* Peers quoted in another currency are **excluded** unless the caller supplies
  an explicit FX rate to the target's currency.  Mixing MAD, USD and EUR
  multiples silently would produce a meaningless range.
* A multiple is not computed when its denominator is zero, negative or
  missing: a P/E on a loss-making peer is reported as "n/a", not as a number.
* Every statistic carries the list of peers it was actually built from, so the
  UI can show exactly who was included.
* Financial institutions are channelled to a book-value / residual-income
  framework instead of a mechanical EV/EBITDA.

Excluded peers are never deleted from the result: they are returned with a
reason, which is what the "Comparables" page displays.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from financial_engine.dcf import DCFInputError
from financial_engine.schemas import ComparableCompany, Currency

# --------------------------------------------------------------------------
# Multiple definitions
# --------------------------------------------------------------------------

#: metric key -> (numerator, denominator, requires_positive_denominator)
MULTIPLE_DEFS: Dict[str, tuple[str, str, bool]] = {
    "ev_ebitda": ("enterprise_value", "ebitda", True),
    "ev_revenue": ("enterprise_value", "revenue", True),
    "pe": ("market_cap", "net_income", True),
    "pb": ("market_cap", "book_value", True),
}


@dataclass(frozen=True)
class TargetValuationInputs:
    """What is needed to translate a peer multiple into a value per share."""

    name: str
    currency: Currency
    net_debt: Optional[float]
    diluted_shares: Optional[float]          # millions
    ebitda: Optional[float] = None
    revenue: Optional[float] = None
    net_income: Optional[float] = None
    book_value: Optional[float] = None
    current_price: Optional[float] = None
    minority_interest: float = 0.0

    @property
    def is_ready(self) -> bool:
        return self.net_debt is not None and self.diluted_shares not in (None, 0)


@dataclass
class MultipleStats:
    key: str
    label: str
    count: int
    median: Optional[float]
    mean: Optional[float]
    minimum: Optional[float]
    maximum: Optional[float]
    q1: Optional[float]
    q3: Optional[float]
    peers_used: List[str] = field(default_factory=list)
    excluded: List[tuple[str, str]] = field(default_factory=list)
    sufficient: bool = False
    note: Optional[str] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "metric": self.key,
            "label": self.label,
            "count": self.count,
            "median": self.median,
            "mean": self.mean,
            "min": self.minimum,
            "max": self.maximum,
            "sufficient": self.sufficient,
            "note": self.note,
        }


@dataclass
class ImpliedValue:
    metric: str
    statistic: str          # "median" | "mean"
    multiple: float
    implied_enterprise_value: Optional[float]
    implied_equity_value: Optional[float]
    implied_value_per_share: Optional[float]
    basis: str
    formula: str


@dataclass
class MultiplesResult:
    target: TargetValuationInputs
    peers_included: List[ComparableCompany]
    peers_excluded: List[tuple[ComparableCompany, str]]
    stats: Dict[str, MultipleStats]
    implied: List[ImpliedValue] = field(default_factory=list)
    valuation_range: Optional[tuple[float, float]] = None
    median_implied_value: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    method_note: str = ""

    def summary_records(self) -> List[Dict[str, object]]:
        return [s.as_dict() for s in self.stats.values()]

    def implied_records(self) -> List[Dict[str, object]]:
        return [
            {
                "metric": v.metric,
                "statistic": v.statistic,
                "multiple": v.multiple,
                "implied_value_per_share": v.implied_value_per_share,
                "basis": v.basis,
            }
            for v in self.implied
        ]


_LABELS = {
    "ev_ebitda": "VE/EBITDA",
    "ev_revenue": "VE/CA",
    "pe": "P/E",
    "pb": "P/B",
}


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------


class MultiplesEngine:
    def __init__(
        self,
        target: TargetValuationInputs,
        fxto_target: Optional[Dict[Currency, float]] = None,
    ):
        self.target = target
        #: 1 unit of the peer currency expressed in the target currency
        self.fx = fxto_target or {}
        self.warnings: List[str] = []

    # -- normalisation -----------------------------------------------------

    def _peer_currency_status(self, peer: ComparableCompany) -> tuple[bool, Optional[str]]:
        """Is this peer's currency compatible with the target's?

        A multiple is a dimensionless ratio, so a peer quoted in another
        currency is *mathematically* comparable.  What is not comparable is the
        market regime (country risk, liquidity, sector mix) that produced it.
        By default such peers are therefore set aside, and re-admitted only when
        the analyst declares an FX rate - the documented signal that the
        difference was considered rather than ignored.
        """
        if peer.currency == self.target.currency:
            return True, None
        if peer.currency in self.fx:
            return True, None
        return (
            False,
            f"devise {peer.currency.value} différente de {self.target.currency.value} "
            "et aucun taux de change déclaré",
        )

    def _computed_multiple(self, peer: ComparableCompany, key: str) -> tuple[Optional[float], Optional[str]]:
        """Peer multiple: supplied if available, computed from raw data otherwise."""
        supplied = getattr(peer, key, None)
        if supplied is not None:
            if not math.isfinite(supplied) or supplied <= 0:
                return None, f"multiple déclaré non significatif ({supplied:.1f}x)"
            return supplied, None

        num_field, den_field, require_positive = MULTIPLE_DEFS[key]
        num = getattr(peer, num_field, None)
        den = getattr(peer, den_field, None)
        if num is None:
            return None, f"{num_field} indisponible"
        if den is None:
            return None, f"{den_field} indisponible"
        if den == 0:
            return None, f"{den_field} nul : division impossible"
        if require_positive and den < 0:
            return None, f"{den_field} négatif : le multiple n'a pas de sens"
        multiple = num / den
        if not math.isfinite(multiple) or multiple <= 0:
            return None, f"multiple non significatif ({multiple:.2f}x)"
        return multiple, None

    # -- statistics --------------------------------------------------------

    def compute_stats(self, peers: Sequence[ComparableCompany], key: str) -> MultipleStats:
        """Median / mean / quartiles of one multiple across the usable peers."""
        values: List[float] = []
        used: List[str] = []
        excluded: List[tuple[str, str]] = []

        for peer in peers:
            if not peer.include:
                excluded.append((peer.name, "exclu manuellement par l'analyste"))
                continue
            ok, reason = self._peer_currency_status(peer)
            if not ok:
                excluded.append((peer.name, reason or "devise incompatible"))
                continue
            multiple, reason = self._computed_multiple(peer, key)
            if multiple is None:
                excluded.append((peer.name, reason or "multiple non calculable"))
                continue
            values.append(multiple)
            used.append(peer.name)

        sufficient = len(values) >= 3
        note = None
        if len(values) == 0:
            note = "aucun comparable exploitable"
        elif len(values) < 3:
            note = (
                f"seulement {len(values)} comparable(s) : la médiane est indicative, "
                "pas statistiquement robuste"
            )

        if not values:
            return MultipleStats(key, _LABELS[key], 0, None, None, None, None, None, None,
                                  peers_used=[], excluded=excluded, sufficient=False, note=note)

        q = statistics.quantiles(values, n=4) if len(values) >= 4 else None
        return MultipleStats(
            key=key,
            label=_LABELS[key],
            count=len(values),
            median=statistics.median(values),
            mean=statistics.fmean(values),
            minimum=min(values),
            maximum=max(values),
            q1=q[0] if q else None,
            q3=q[2] if q else None,
            peers_used=used,
            excluded=excluded,
            sufficient=sufficient,
            note=note,
        )

    # -- implied valuation -------------------------------------------------

    def _implied_from_multiple(self, key: str, multiple: float, statistic: str) -> Optional[ImpliedValue]:
        t = self.target
        if key == "ev_ebitda":
            base = t.ebitda
            if base is None:
                return None
            implied_ev = multiple * base
            equity = implied_ev - (t.net_debt or 0.0) - (t.minority_interest or 0.0)
            ps = equity / t.diluted_shares if t.diluted_shares else None
            return ImpliedValue(
                metric=key, statistic=statistic, multiple=multiple,
                implied_enterprise_value=implied_ev, implied_equity_value=equity,
                implied_value_per_share=ps,
                basis=f"EBITDA {t.name} = {base:,.1f} m {t.currency.value}",
                formula="(multiple x EBITDA) - dette nette - intérêts minoritaires",
            )
        if key == "ev_revenue":
            base = t.revenue
            if base is None:
                return None
            implied_ev = multiple * base
            equity = implied_ev - (t.net_debt or 0.0) - (t.minority_interest or 0.0)
            ps = equity / t.diluted_shares if t.diluted_shares else None
            return ImpliedValue(
                metric=key, statistic=statistic, multiple=multiple,
                implied_enterprise_value=implied_ev, implied_equity_value=equity,
                implied_value_per_share=ps,
                basis=f"CA {t.name} = {base:,.1f} m {t.currency.value}",
                formula="(multiple x CA) - dette nette",
            )
        if key == "pe":
            base = t.net_income
            if base is None:
                return None
            equity = multiple * base
            ps = equity / t.diluted_shares if t.diluted_shares else None
            return ImpliedValue(
                metric=key, statistic=statistic, multiple=multiple,
                implied_enterprise_value=None, implied_equity_value=equity,
                implied_value_per_share=ps,
                basis=f"Résultat net {t.name} = {base:,.1f} m {t.currency.value}",
                formula="multiple x résultat net / actions diluées",
            )
        if key == "pb":
            base = t.book_value
            if base is None:
                return None
            equity = multiple * base
            ps = equity / t.diluted_shares if t.diluted_shares else None
            return ImpliedValue(
                metric=key, statistic=statistic, multiple=multiple,
                implied_enterprise_value=None, implied_equity_value=equity,
                implied_value_per_share=ps,
                basis=f"Capitaux propres {t.name} = {base:,.1f} m {t.currency.value}",
                formula="multiple x capitaux propres / actions diluées",
            )
        return None

    def run(
        self,
        peers: Sequence[ComparableCompany],
        metrics: Optional[Sequence[str]] = None,
        min_peers: int = 3,
    ) -> MultiplesResult:
        t = self.target
        metrics = list(metrics or ("ev_ebitda", "pe"))
        same_sector = {p.sector for p in peers if p.include and p.sector}
        if len(same_sector) > 1:
            self.warnings.append(
                "Les comparables appartiennent a plusieurs secteurs "
                f"({', '.join(sorted(s for s in same_sector if s))}) : "
                "vérifiez que les multiples sont bien comparables."
            )
        if not t.is_ready:
            self.warnings.append(
                "Dette nette ou nombre d'actions diluées indisponible pour la société "
                "cible : les multiples d'entreprise ne peuvent pas être convertis en "
                "valeur par action."
            )

        stats = {k: self.compute_stats(peers, k) for k in metrics}

        implied: List[ImpliedValue] = []
        for key, st in stats.items():
            if not st.sufficient or st.median is None:
                continue
            for statistic, multiple in (("median", st.median), ("mean", st.mean)):
                if multiple is None:
                    continue
                iv = self._implied_from_multiple(key, multiple, statistic)
                if iv is not None and iv.implied_value_per_share is not None:
                    implied.append(iv)

        per_share_values = [iv.implied_value_per_share for iv in implied if iv.implied_value_per_share is not None]
        valuation_range = (min(per_share_values), max(per_share_values)) if per_share_values else None
        median_implied = statistics.median(per_share_values) if per_share_values else None

        if not per_share_values:
            self.warnings.append(
                "Aucune valeur implicite n'a pu être calculée : les données de la "
                "société cible ou des comparables sont insuffisantes."
            )

        included = [p for p in peers if p.include]
        # Rebuild the exclusion list with the full peer objects (the per-metric
        # stats only carry names). A peer is excluded as soon as one of the
        # requested metrics cannot be computed for it.
        reasons: Dict[str, str] = {}
        for st in stats.values():
            for name, reason in st.excluded:
                reasons.setdefault(name, reason)
        excluded = [(p, reasons[p.name]) for p in peers if p.name in reasons]

        return MultiplesResult(
            target=t,
            peers_included=included,
            peers_excluded=excluded,
            stats=stats,
            implied=implied,
            valuation_range=valuation_range,
            median_implied_value=median_implied,
            warnings=list(self.warnings),
            method_note=(
                "Multiples de comparables cotés. La médiane est privilégiée car moins "
                "sensible aux valeurs extrêmes que la moyenne. Toute valeur implicite "
                "suppose que la société se négocie comme la médiane de ses pairs."
            ),
        )


# --------------------------------------------------------------------------
# Financial institutions: residual income / P/B framework
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ResidualIncomeInputs:
    book_value_0: float           # millions, equity at t0
    roe: float                    # sustainable return on equity, decimal
    cost_of_equity: float         # decimal
    growth: float                 # perpetual growth of book value, decimal
    payout_ratio: float           # share of earnings paid out, decimal
    forecast_years: int = 5
    shares_outstanding: float = 0.0
    currency: str = "USD"
    simulated: bool = False

    def __post_init__(self) -> None:
        if self.book_value_0 <= 0:
            raise DCFInputError("book_value_0 doit être strictement positif")
        if self.cost_of_equity <= 0:
            raise DCFInputError("cost_of_equity doit être strictement positif")
        if not (0.0 <= self.payout_ratio <= 1.0):
            raise DCFInputError("payout_ratio doit être compris entre 0 et 1")
        if self.growth >= self.cost_of_equity:
            raise DCFInputError(
                "la croissance doit être strictement inférieure au coût des fonds propres "
                "pour que la valeur terminale du revenu résiduel converge"
            )
        if self.forecast_years < 1:
            raise DCFInputError("forecast_years doit valoir au moins 1")


@dataclass
class ResidualIncomeResult:
    equity_value: float
    value_per_share: Optional[float]
    pv_residual_income: float
    pv_terminal_value: float
    book_value_0: float
    implied_pb: float
    years: List[Dict[str, float]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def residual_income_value(inputs: ResidualIncomeInputs) -> ResidualIncomeResult:
    """Residual income (Edwards-Bell-Ohlson) valuation for financial firms.

    ``BV_t = BV_{t-1} + ROE*BV_{t-1}*(1 - payout)`` and
    ``RI_t = (ROE - Ke) * BV_{t-1}``; equity value is ``BV_0 + PV(RI) +
    PV(terminal RI)``.
    """
    kept = 1.0 - inputs.payout_ratio
    rows: List[Dict[str, float]] = []
    bv = inputs.book_value_0
    pv_ri = 0.0

    for t in range(1, inputs.forecast_years + 1):
        earnings = inputs.roe * bv
        ri = (inputs.roe - inputs.cost_of_equity) * bv
        pv = ri / ((1.0 + inputs.cost_of_equity) ** t)
        pv_ri += pv
        rows.append(
            {
                "year": t,
                "book_value_start": bv,
                "earnings": earnings,
                "residual_income": ri,
                "discount_factor": 1.0 / ((1.0 + inputs.cost_of_equity) ** t),
                "pv_residual_income": pv,
            }
        )
        bv = bv + earnings * kept

    # terminal residual income grows at g, discounted from year N
    ri_next = (inputs.roe - inputs.cost_of_equity) * bv * (1.0 + inputs.growth)
    tv = ri_next / (inputs.cost_of_equity - inputs.growth)
    pv_tv = tv / ((1.0 + inputs.cost_of_equity) ** inputs.forecast_years)

    equity_value = inputs.book_value_0 + pv_ri + pv_tv
    per_share = equity_value / inputs.shares_outstanding if inputs.shares_outstanding else None

    warnings: List[str] = []
    if inputs.roe <= inputs.cost_of_equity:
        warnings.append(
            "Le ROE est inférieur au coût des fonds propres : la valeur résiduelle "
            "est négative. Pour une banque, cela implique une valorisation sous "
            "ses capitaux propres comptables."
        )
    if pv_tv < 0:
        warnings.append("La valeur terminale résiduelle est négative.")
    if not inputs.shares_outstanding:
        warnings.append("Nombre d'actions indisponible : valeur par action non calculable.")

    return ResidualIncomeResult(
        equity_value=equity_value,
        value_per_share=per_share,
        pv_residual_income=pv_ri,
        pv_terminal_value=pv_tv,
        book_value_0=inputs.book_value_0,
        implied_pb=equity_value / inputs.book_value_0,
        years=rows,
        warnings=warnings,
    )


def pb_framework(peer_prices_to_book: Sequence[float], target_book_value: float, shares: float, target_roe: Optional[float] = None, peer_roe: Optional[float] = None) -> Dict[str, Optional[float]]:
    """Simple P/B range for a financial institution, with a ROE justification.

    The 'justified P/B' is ``(ROE - g) / (Ke - g)``; when the target ROE and a
    peer cost of equity are supplied the two views can be compared.
    """
    valid = [m for m in peer_prices_to_book if m and m > 0 and math.isfinite(m)]
    out: Dict[str, Optional[float]] = {
        "peer_count": float(len(valid)),
        "median_pb": statistics.median(valid) if valid else None,
        "mean_pb": statistics.fmean(valid) if valid else None,
    }
    if out["median_pb"] and target_book_value and shares:
        equity = out["median_pb"] * target_book_value
        out["implied_equity_value"] = equity
        out["implied_value_per_share"] = equity / shares
    else:
        out["implied_equity_value"] = None
        out["implied_value_per_share"] = None
    out["target_roe"] = target_roe
    out["peer_roe"] = peer_roe
    return out


def justified_pb(roe: float, cost_of_equity: float, growth: float) -> Optional[float]:
    """Justified price-to-book from the Gordon growth relationship."""
    if cost_of_equity <= growth:
        return None
    return (roe - growth) / (cost_of_equity - growth)
