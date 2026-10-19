"""Find and add any listed company, beyond the pre-verified registry.

Two keyless sources are combined:

* **SEC EDGAR** ``company_tickers.json`` - official, every SEC registrant
  (US listings and foreign filers).  A match carries its CIK, which unlocks the
  automatic import of published financial statements.
* **Yahoo Finance search** - unofficial, worldwide coverage (Paris, Frankfurt,
  London, Toronto...).  Used only to identify a company; its metadata is shown
  as "à vérifier" and never treated as a financial fact.

Companies that neither source covers (e.g. the Casablanca Stock Exchange) can be
created manually.  Nothing here invents a field: what a source does not provide
stays empty or is explicitly flagged for verification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlencode

from financial_engine.schemas import CompanyProfile, Currency, Scale

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
YAHOO_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"
)

SOURCE_SEC = "SEC EDGAR (officiel)"
SOURCE_YAHOO = "Yahoo Finance (non officiel)"
SOURCE_MANUAL = "Saisie manuelle"

#: Yahoo ticker suffix -> (country, listing currency).  The listing currency is
#: only a starting point: the reporting currency can differ (ADRs, dual
#: listings), which is why every non-SEC profile is flagged for verification.
_SUFFIX_MARKETS: Dict[str, Tuple[str, Currency]] = {
    "PA": ("France", Currency.EUR),
    "DE": ("Allemagne", Currency.EUR),
    "F": ("Allemagne", Currency.EUR),
    "AS": ("Pays-Bas", Currency.EUR),
    "BR": ("Belgique", Currency.EUR),
    "MI": ("Italie", Currency.EUR),
    "MC": ("Espagne", Currency.EUR),
    "LS": ("Portugal", Currency.EUR),
    "VI": ("Autriche", Currency.EUR),
    "HE": ("Finlande", Currency.EUR),
    "IR": ("Irlande", Currency.EUR),
    "L": ("Royaume-Uni", Currency.GBP),
    "SW": ("Suisse", Currency.CHF),
    "TO": ("Canada", Currency.CAD),
    "V": ("Canada", Currency.CAD),
    "AX": ("Australie", Currency.AUD),
    "T": ("Japon", Currency.JPY),
    "JO": ("Afrique du Sud", Currency.ZAR),
}
_US_EXCHANGES = ("NASDAQ", "NYSE", "NYSEARCA", "NYSE AMERICAN", "AMEX", "BATS", "OTC", "MARCHÉS OTC")


@dataclass
class CompanyCandidate:
    """A search hit, before the analyst confirms and adds it."""

    name: str
    ticker: str
    source: str
    exchange: str = ""
    country: str = ""
    currency: Currency = Currency.OTHER
    sector: Optional[str] = None
    industry: Optional[str] = None
    sec_cik: Optional[str] = None
    fiscal_year_end: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        venue = self.exchange or "place inconnue"
        return f"{self.ticker} — {self.name} ({venue} · {self.source.split(' ')[0]})"


@dataclass
class DiscoveryResult:
    query: str
    candidates: List[CompanyCandidate] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Identifiers
# --------------------------------------------------------------------------


def make_company_id(ticker: str) -> str:
    """Stable, storage-safe identifier derived from the ticker (``AIR.PA`` -> ``AIR_PA``)."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", (ticker or "").strip().upper()).strip("_")
    return cleaned[:40]


def _normalise_fiscal_year_end(raw: Optional[str]) -> Optional[str]:
    """SEC ``"0628"`` or ``"06-28"`` -> ``"06-28"``; anything else -> ``None``."""
    if not raw:
        return None
    text = str(raw).strip()
    if re.fullmatch(r"\d{4}", text):
        text = f"{text[:2]}-{text[2:]}"
    match = re.fullmatch(r"(\d{2})-(\d{2})", text)
    if not match:
        return None
    month, day = int(match.group(1)), int(match.group(2))
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    return text


def market_from_ticker(ticker: str, exchange: str = "") -> Tuple[str, Currency]:
    """Country and listing currency inferred from a Yahoo ticker / exchange."""
    if "." in ticker:
        suffix = ticker.rsplit(".", 1)[1].upper()
        if suffix in _SUFFIX_MARKETS:
            return _SUFFIX_MARKETS[suffix]
        return "", Currency.OTHER
    if exchange.strip().upper() in _US_EXCHANGES or not exchange:
        return "États-Unis", Currency.USD
    return "", Currency.OTHER


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------


def _get_client(http=None):
    if http is not None:
        return http
    from data_sources.http_client import get_client

    return get_client()


def _matches(query: str, *fields: Optional[str]) -> int:
    """Relevance score (0 = no match).  Exact ticker beats name prefix beats substring."""
    needle = query.strip().lower()
    best = 0
    for position, value in enumerate(fields):
        if not value:
            continue
        hay = value.lower()
        if hay == needle:
            best = max(best, 100 - position)
        elif hay.startswith(needle):
            best = max(best, 60 - position)
        elif needle in hay:
            best = max(best, 30 - position)
    return best


def search_sec(query: str, limit: int = 10, http=None, settings=None) -> Tuple[List[CompanyCandidate], Optional[str]]:
    """Search the official SEC ticker table by ticker or company name."""
    from config import get_settings

    settings = settings or get_settings()
    client = _get_client(http)
    data, response = client.get_json(
        SEC_TICKERS_URL, extra_headers={"User-Agent": settings.sec_user_agent}
    )
    if not isinstance(data, dict):
        error = response.error if response is not None else "réponse vide"
        return [], f"Table SEC indisponible : {error}"

    scored: List[Tuple[int, CompanyCandidate]] = []
    for entry in data.values():
        try:
            ticker = str(entry["ticker"]).upper()
            title = str(entry["title"])
            cik = str(entry["cik_str"]).zfill(10)
        except (KeyError, TypeError):
            continue
        score = _matches(query, ticker, title)
        if score:
            scored.append(
                (
                    score,
                    CompanyCandidate(
                        name=title,
                        ticker=ticker,
                        source=SOURCE_SEC,
                        exchange="SEC",
                        country="",
                        currency=Currency.USD,
                        sec_cik=cik,
                    ),
                )
            )
    scored.sort(key=lambda item: (-item[0], item[1].ticker))
    # One registrant can have several share classes: keep the best ticker per CIK.
    seen: set = set()
    out: List[CompanyCandidate] = []
    for _, candidate in scored:
        if candidate.sec_cik in seen:
            continue
        seen.add(candidate.sec_cik)
        out.append(candidate)
        if len(out) >= limit:
            break
    return out, None


def search_yahoo(query: str, limit: int = 10, http=None) -> Tuple[List[CompanyCandidate], Optional[str]]:
    """Worldwide equity search through Yahoo Finance's public endpoint."""
    client = _get_client(http)
    url = f"{YAHOO_SEARCH_URL}?{urlencode({'q': query, 'quotesCount': max(limit, 5), 'newsCount': 0, 'lang': 'fr-FR'})}"
    data, response = client.get_json(
        url,
        extra_headers={"User-Agent": _BROWSER_UA, "Accept": "application/json"},
        allow_robots_override=True,
    )
    if not isinstance(data, dict):
        error = response.error if response is not None else "réponse vide"
        return [], f"Recherche Yahoo indisponible : {error}"

    out: List[CompanyCandidate] = []
    for quote in data.get("quotes") or []:
        if str(quote.get("quoteType", "")).upper() != "EQUITY":
            continue
        ticker = str(quote.get("symbol") or "").upper()
        name = quote.get("longname") or quote.get("shortname")
        if not ticker or not name:
            continue
        exchange = str(quote.get("exchDisp") or quote.get("exchange") or "")
        country, currency = market_from_ticker(ticker, exchange)
        notes = [
            "Identité issue de Yahoo Finance (source non officielle) : vérifiez la "
            "devise de reporting, la place et la clôture d'exercice dans le rapport annuel."
        ]
        if ticker.endswith(".L"):
            notes.append("Cotation londonienne : les cours Yahoo sont souvent en pence (GBp).")
        out.append(
            CompanyCandidate(
                name=str(name),
                ticker=ticker,
                source=SOURCE_YAHOO,
                exchange=exchange,
                country=country,
                currency=currency,
                sector=quote.get("sectorDisp") or quote.get("sector"),
                industry=quote.get("industryDisp") or quote.get("industry"),
                notes=notes,
            )
        )
        if len(out) >= limit:
            break
    return out, None


def search_companies(
    query: str,
    limit: int = 10,
    include_international: bool = True,
    http=None,
    settings=None,
) -> DiscoveryResult:
    """Combined search: SEC first (official), then Yahoo for other markets."""
    result = DiscoveryResult(query=query.strip())
    if len(result.query) < 2:
        result.errors.append("Saisissez au moins 2 caractères (nom ou ticker).")
        return result

    sec_hits, sec_error = search_sec(result.query, limit=limit, http=http, settings=settings)
    if sec_error:
        result.errors.append(sec_error)
    result.candidates.extend(sec_hits)

    if include_international:
        yahoo_hits, yahoo_error = search_yahoo(result.query, limit=limit, http=http)
        if yahoo_error:
            result.errors.append(yahoo_error)
        sec_tickers = {c.ticker for c in sec_hits}
        for hit in yahoo_hits:
            if hit.ticker in sec_tickers:
                # Same US listing: keep the SEC (official) version, borrow the sector.
                for official in result.candidates:
                    if official.ticker == hit.ticker:
                        official.sector = official.sector or hit.sector
                        official.exchange = hit.exchange or official.exchange
                continue
            result.candidates.append(hit)
    return result


def enrich_from_sec(candidate: CompanyCandidate, http=None, settings=None) -> CompanyCandidate:
    """Fill identity fields from the SEC submissions API (ground truth)."""
    if not candidate.sec_cik:
        return candidate
    from data_sources.sec_edgar import SecEdgarClient

    client = SecEdgarClient(http=http, settings=settings)
    data, error = client.submissions(candidate.sec_cik)
    if data is None:
        return replace(
            candidate,
            notes=candidate.notes + [f"Métadonnées SEC indisponibles : {error}"],
        )
    exchanges = [e for e in (data.get("exchanges") or []) if e]
    business = ((data.get("addresses") or {}).get("business") or {})
    foreign = bool(business.get("isForeignLocation"))
    country = candidate.country
    notes = list(candidate.notes)
    if foreign:
        country = str(business.get("stateOrCountryDescription") or "") or country
        notes.append(
            "Émetteur étranger enregistré à la SEC : la devise de reporting peut ne pas "
            "être l'USD (formulaire 20-F) — vérifiez-la avant toute valorisation."
        )
    elif business:
        country = "États-Unis"
    return replace(
        candidate,
        name=str(data.get("name") or candidate.name),
        exchange=exchanges[0] if exchanges else candidate.exchange,
        country=country,
        industry=data.get("sicDescription") or candidate.industry,
        fiscal_year_end=_normalise_fiscal_year_end(data.get("fiscalYearEnd")) or candidate.fiscal_year_end,
        notes=notes,
    )


# --------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------


def candidate_to_profile(candidate: CompanyCandidate) -> CompanyProfile:
    """Turn a confirmed candidate into a stored profile, flagging what is unverified."""
    ticker = candidate.ticker.strip().upper()
    notes = list(candidate.notes)
    if candidate.source == SOURCE_SEC:
        notes.append(
            "Société ajoutée depuis la table officielle SEC : les états financiers "
            "publiés peuvent être importés automatiquement (page États financiers)."
        )
    elif candidate.source == SOURCE_MANUAL:
        notes.append("Profil saisi manuellement : à vérifier dans les documents officiels.")
    if candidate.currency == Currency.OTHER:
        notes.append("Devise non reconnue : renseignez-la avant toute valorisation.")

    data_sources = [candidate.source]
    if candidate.sec_cik:
        data_sources.append("SEC EDGAR (formulaires 10-K, 10-Q, 20-F, 8-K)")
    else:
        data_sources.append("Rapports annuels et communiqués publiés par la société")

    return CompanyProfile(
        company_id=make_company_id(ticker),
        name=candidate.name.strip(),
        short_name=candidate.name.strip()[:40],
        ticker=ticker,
        exchange=candidate.exchange.strip() or "Non renseignée",
        country=candidate.country.strip() or "Non renseigné",
        currency=candidate.currency,
        sector=(candidate.sector or None),
        industry=(candidate.industry or None),
        fiscal_year_end=_normalise_fiscal_year_end(candidate.fiscal_year_end),
        reporting_scale=Scale.MILLIONS,
        sec_cik=candidate.sec_cik,
        identity_notes=" ".join(notes) or None,
        data_sources=data_sources,
    )


def manual_candidate(
    name: str,
    ticker: str,
    exchange: str = "",
    country: str = "",
    currency: Currency = Currency.OTHER,
    sector: Optional[str] = None,
    industry: Optional[str] = None,
    fiscal_year_end: Optional[str] = None,
) -> CompanyCandidate:
    """Candidate for a company no search source covers (e.g. Casablanca)."""
    return CompanyCandidate(
        name=name.strip(),
        ticker=ticker.strip().upper(),
        source=SOURCE_MANUAL,
        exchange=exchange.strip(),
        country=country.strip(),
        currency=currency,
        sector=(sector or "").strip() or None,
        industry=(industry or "").strip() or None,
        fiscal_year_end=fiscal_year_end,
    )


def existing_ids(profiles: Iterable[CompanyProfile]) -> set:
    return {p.company_id for p in profiles}
