"""SEC EDGAR connector (US official disclosures).

Uses only the documented, public SEC APIs:

* ``https://www.sec.gov/files/company_tickers.json`` - ticker -> CIK map
* ``https://data.sec.gov/submissions/CIK##########.json`` - filing history
* ``https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`` - all tagged facts
* ``https://data.sec.gov/api/xbrl/companyconcept/CIK##########/us-gaap/<tag>.json``

A descriptive User-Agent is mandatory (the SEC blocks anonymous callers), and
the connector is rate limited by :mod:`data_sources.http_client`.

Normalisation decisions
-----------------------
* SEC reports in **US dollars, absolute units**.  Every monetary value is
  divided by 1e6 here, so the rest of the application only ever sees millions.
* Share counts are likewise converted to **millions**.
* The fiscal year label is inferred from the period **end date**, not from the
  filing's ``fy`` field, which is unreliable (the same period can appear under
  two different ``fy`` labels across filings).
* When several filings report the same period, the value from the **most
  recently filed** document wins, and the older one is kept as a separate
  observation so restatements stay visible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from data_sources.http_client import HttpClient, HttpResponse, get_client
from financial_engine.schemas import (
    Currency,
    ExtractionStatus,
    FinancialDataSet,
    FiscalPeriod,
    LineItem,
    Observation,
    PeriodType,
    Scale,
    SourceRecord,
    SourceTier,
)

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_COMPANYCONCEPT = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"
SEC_FILING_INDEX = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{document}"

#: Amounts are stored in millions; SEC figures are raw units.
_UNITS_TO_MILLIONS = 1e6
#: Share counts are stored in millions of shares.
_SHARES_TO_MILLIONS = 1e6


class FactKind:
    FLOW = "flow"          # income statement / cash flow: needs a duration
    INSTANT = "instant"    # balance sheet: a point in time
    SHARES_FLOW = "shares_flow"
    SHARES_INSTANT = "shares_instant"
    PER_SHARE = "per_share"


#: canonical line item -> (us-gaap tags in priority order, kind)
CONCEPT_MAP: Dict[LineItem, Tuple[Sequence[str], str]] = {
    LineItem.REVENUE: (
        (
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
        ),
        FactKind.FLOW,
    ),
    LineItem.COST_OF_REVENUE: (
        ("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold", "CostOfServices"),
        FactKind.FLOW,
    ),
    LineItem.GROSS_PROFIT: (("GrossProfit",), FactKind.FLOW),
    LineItem.OPERATING_EXPENSES: (("OperatingExpenses", "CostsAndExpenses"), FactKind.FLOW),
    LineItem.OPERATING_INCOME: (("OperatingIncomeLoss",), FactKind.FLOW),
    LineItem.EBIT: (("OperatingIncomeLoss",), FactKind.FLOW),
    LineItem.DEPRECIATION_AMORTIZATION: (
        (
            "DepreciationDepletionAndAmortization",
            "DepreciationAndAmortization",
            "DepreciationAmortizationAndAccretionNet",
            "Depreciation",
        ),
        FactKind.FLOW,
    ),
    LineItem.INTEREST_EXPENSE: (
        ("InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt", "InterestAndDebtExpense"),
        FactKind.FLOW,
    ),
    LineItem.PRETAX_INCOME: (
        (
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        ),
        FactKind.FLOW,
    ),
    LineItem.INCOME_TAX_EXPENSE: (("IncomeTaxExpenseBenefit",), FactKind.FLOW),
    LineItem.NET_INCOME: (("NetIncomeLoss", "ProfitLoss"), FactKind.FLOW),
    LineItem.EPS_DILUTED: (
        ("EarningsPerShareDiluted",),
        FactKind.PER_SHARE,
    ),
    LineItem.CASH_AND_EQUIVALENTS: (
        ("CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
        FactKind.INSTANT,
    ),
    LineItem.ACCOUNTS_RECEIVABLE: (
        ("AccountsReceivableNetCurrent", "ReceivablesNetCurrent", "AccountsReceivableNet"),
        FactKind.INSTANT,
    ),
    LineItem.INVENTORY: (("InventoryNet", "InventoryFinishedGoods", "InventoryGross"), FactKind.INSTANT),
    LineItem.CURRENT_ASSETS: (("AssetsCurrent",), FactKind.INSTANT),
    LineItem.TOTAL_ASSETS: (("Assets",), FactKind.INSTANT),
    LineItem.ACCOUNTS_PAYABLE: (
        ("AccountsPayableCurrent", "AccountsPayableAndAccruedLiabilitiesCurrent"),
        FactKind.INSTANT,
    ),
    LineItem.CURRENT_LIABILITIES: (("LiabilitiesCurrent",), FactKind.INSTANT),
    LineItem.SHORT_TERM_DEBT: (
        ("LongTermDebtCurrent", "ShortTermBorrowings", "DebtCurrent"),
        FactKind.INSTANT,
    ),
    LineItem.LONG_TERM_DEBT: (("LongTermDebtNoncurrent", "LongTermDebt"), FactKind.INSTANT),
    LineItem.TOTAL_LIABILITIES: (("Liabilities",), FactKind.INSTANT),
    LineItem.TOTAL_EQUITY: (
        ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
        FactKind.INSTANT,
    ),
    LineItem.MINORITY_INTEREST: (("MinorityInterest",), FactKind.INSTANT),
    LineItem.OPERATING_CASH_FLOW: (
        ("NetCashProvidedByUsedInOperatingActivities",
         "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
        FactKind.FLOW,
    ),
    LineItem.CAPITAL_EXPENDITURE: (
        ("PaymentsToAcquirePropertyPlantAndEquipment",
         "PaymentsToAcquireProductiveAssets",
         "PaymentsForCapitalImprovements"),
        FactKind.FLOW,
    ),
    LineItem.DIVIDENDS_PAID: (
        ("PaymentsOfDividendsCommonStock", "PaymentsOfDividends", "PaymentsOfDividendsMinorityInterest"),
        FactKind.FLOW,
    ),
    LineItem.DILUTED_SHARES: (
        ("WeightedAverageNumberOfDilutedSharesOutstanding",),
        FactKind.SHARES_FLOW,
    ),
    LineItem.SHARES_OUTSTANDING: (
        ("WeightedAverageNumberOfSharesOutstandingBasic",),
        FactKind.SHARES_FLOW,
    ),
}

#: dei namespace tags
DEI_SHARES = ("EntityCommonStockSharesOutstanding",)

ANNUAL_MIN_DAYS = 330
ANNUAL_MAX_DAYS = 400
QUARTER_MIN_DAYS = 70
QUARTER_MAX_DAYS = 110


@dataclass
class SecFact:
    """One tagged fact as returned by the XBRL API, with its provenance."""

    tag: str
    value: float
    start: Optional[date]
    end: date
    form: str
    filed: date
    fy: Optional[int]
    fp: Optional[str]
    frame: Optional[str]
    accn: Optional[str]

    @property
    def duration_days(self) -> Optional[int]:
        if self.start is None:
            return None
        return (self.end - self.start).days

    @property
    def is_annual(self) -> bool:
        d = self.duration_days
        return d is not None and ANNUAL_MIN_DAYS <= d <= ANNUAL_MAX_DAYS

    @property
    def is_quarterly(self) -> bool:
        d = self.duration_days
        return d is not None and QUARTER_MIN_DAYS <= d <= QUARTER_MAX_DAYS


@dataclass
class SecFetchResult:
    company_id: str
    dataset: Optional[FinancialDataSet] = None
    sources: List[SourceRecord] = field(default_factory=list)
    facts: Dict[str, List[SecFact]] = field(default_factory=dict)
    filings: List[Dict[str, object]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    cik: Optional[str] = None
    entity_name: Optional[str] = None
    sic_description: Optional[str] = None
    fiscal_year_end: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.dataset is not None and not self.errors


# --------------------------------------------------------------------------
# Ticker / CIK resolution
# --------------------------------------------------------------------------


def normalise_cik(raw: str | int) -> str:
    digits = re.sub(r"\D", "", str(raw))
    return digits.zfill(10)


class SecEdgarClient:
    """Thin, cached wrapper over the SEC endpoints."""

    def __init__(self, http: Optional[HttpClient] = None, settings=None):
        from config import get_settings

        self.settings = settings or get_settings()
        self.http = http or get_client()
        self._ticker_map: Optional[Dict[str, str]] = None

    # -- raw endpoints -----------------------------------------------------

    def ticker_map(self) -> Dict[str, str]:
        if self._ticker_map is not None:
            return self._ticker_map
        data, response = self.http.get_json(
            SEC_TICKERS_URL, extra_headers={"User-Agent": self.settings.sec_user_agent}
        )
        mapping: Dict[str, str] = {}
        if isinstance(data, dict):
            for entry in data.values():
                try:
                    mapping[str(entry["ticker"]).upper()] = normalise_cik(entry["cik_str"])
                except (KeyError, TypeError):
                    continue
        if not mapping:
            # Offline or blocked: the registry CIK is still usable.
            pass
        self._ticker_map = mapping
        return mapping

    def resolve_cik(self, ticker: str, fallback: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
        """Return (cik, error). Prefers the live SEC map, falls back to config."""
        self.ticker_map()
        if self._ticker_map and ticker.upper() in self._ticker_map:
            return self._ticker_map[ticker.upper()], None
        if fallback:
            return normalise_cik(fallback), None
        return None, (
            f"CIK introuvable pour le ticker {ticker}. La table officielle SEC n'a pas "
            "pu être téléchargée et aucun CIK de repli n'est configuré."
        )

    def submissions(self, cik: str) -> tuple[Optional[dict], Optional[str]]:
        url = SEC_SUBMISSIONS.format(cik=normalise_cik(cik))
        data, response = self.http.get_json(
            url, extra_headers={"User-Agent": self.settings.sec_user_agent}
        )
        if data is None:
            return None, (response.error if response else "reponse vide")
        return data, None

    def company_facts(self, cik: str) -> tuple[Optional[dict], Optional[str]]:
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{normalise_cik(cik)}.json"
        data, response = self.http.get_json(
            url, extra_headers={"User-Agent": self.settings.sec_user_agent}
        )
        if data is None:
            return None, (response.error if response else "reponse vide")
        return data, None

    def company_concept(self, cik: str, tag: str, taxonomy: str = "us-gaap") -> tuple[Optional[dict], Optional[str]]:
        url = SEC_COMPANYCONCEPT.format(cik=normalise_cik(cik), tag=tag)
        data, response = self.http.get_json(
            url, extra_headers={"User-Agent": self.settings.sec_user_agent}
        )
        if data is None:
            return None, (response.error if response else "reponse vide")
        return data, None

    def filing_url(self, cik: str, accession: Optional[str], document: Optional[str] = None) -> Optional[str]:
        if not accession:
            return None
        acc = accession.replace("-", "")
        base = f"https://www.sec.gov/Archives/edgar/data/{int(normalise_cik(cik))}/{acc}"
        return f"{base}/{document}" if document else f"{base}/{accession}-index.htm"


# --------------------------------------------------------------------------
# Fact parsing
# --------------------------------------------------------------------------


def _parse_date(raw) -> Optional[date]:
    if not raw:
        return None
    try:
        return date.fromisoformat(str(raw)[:10])
    except ValueError:
        return None


def parse_concept_facts(tag: str, payload: dict) -> List[SecFact]:
    """Flatten an XBRL concept payload into :class:`SecFact` objects."""
    out: List[SecFact] = []
    units = payload.get("units", {})
    for unit, entries in units.items():
        # USD for money, shares for counts, USD/shares for per-share figures.
        if unit not in ("USD", "shares", "USD/shares", "pure"):
            continue
        for entry in entries:
            end = _parse_date(entry.get("end"))
            if end is None:
                continue
            filed = _parse_date(entry.get("filed")) or end
            try:
                value = float(entry["val"])
            except (KeyError, TypeError, ValueError):
                continue
            out.append(
                SecFact(
                    tag=tag,
                    value=value,
                    start=_parse_date(entry.get("start")),
                    end=end,
                    form=str(entry.get("form") or ""),
                    filed=filed,
                    fy=entry.get("fy"),
                    fp=entry.get("fp"),
                    frame=entry.get("frame"),
                    accn=entry.get("accn"),
                )
            )
    out.sort(key=lambda f: (f.end, f.filed))
    return out


def extract_facts(companyfacts: dict, limit_concepts: Optional[Iterable[str]] = None) -> Dict[str, List[SecFact]]:
    """All us-gaap + dei facts of interest, keyed by tag."""
    wanted = set(limit_concepts) if limit_concepts else None
    facts = companyfacts.get("facts", {})
    out: Dict[str, List[SecFact]] = {}
    for taxonomy in ("us-gaap", "dei", "ifrs-full"):
        for tag, payload in (facts.get(taxonomy) or {}).items():
            if wanted is not None and tag not in wanted:
                continue
            parsed = parse_concept_facts(tag, payload)
            if parsed:
                out[tag] = parsed
    return out


def _dedupe_latest_filed(facts: Sequence[SecFact]) -> Optional[SecFact]:
    """Most recently filed fact, and among equals the most recently restated."""
    if not facts:
        return None
    return max(facts, key=lambda f: (f.filed, f.value is not None))


def _pick(
    all_facts: Dict[str, List[SecFact]],
    tags: Sequence[str],
    kind: str,
) -> List[SecFact]:
    """Best available fact series for a canonical line item."""
    for tag in tags:
        facts = all_facts.get(tag)
        if not facts:
            continue
        if kind in (FactKind.FLOW, FactKind.SHARES_FLOW):
            selected = [f for f in facts if f.start is not None and f.is_annual]
        elif kind in (FactKind.INSTANT, FactKind.SHARES_INSTANT):
            selected = [f for f in facts if f.start is None]
        else:
            selected = [f for f in facts if f.start is not None and f.is_annual]
        if selected:
            return selected
    return []


# --------------------------------------------------------------------------
# Dataset construction
# --------------------------------------------------------------------------


class SecEdgarExtractor:
    """Builds a :class:`FinancialDataSet` from the SEC XBRL company facts."""

    def __init__(self, client: Optional[SecEdgarClient] = None, company_id: str = ""):
        self.client = client or SecEdgarClient()
        self.company_id = company_id

    def build(
        self,
        company_id: str,
        cik: str,
        currency: Currency = Currency.USD,
        years: int = 6,
        include_quarterly: bool = True,
        companyfacts: Optional[dict] = None,
        filings: Optional[List[Dict[str, object]]] = None,
        fiscal_year_end: Optional[str] = None,
    ) -> SecFetchResult:
        result = SecFetchResult(company_id=company_id, cik=normalise_cik(cik))
        # Nominal fiscal year end, e.g. "0628". Without it the fiscal year is
        # inferred from the calendar year of the period end, which mislabels
        # every quarter of a non-December filer.
        self.fiscal_year_end = fiscal_year_end or None

        if companyfacts is None:
            companyfacts, error = self.client.company_facts(cik)
            if companyfacts is None:
                result.errors.append(f"SEC company facts indisponibles : {error}")
                return result

        result.entity_name = companyfacts.get("entityName")
        all_facts = extract_facts(companyfacts)
        result.facts = all_facts
        if not all_facts:
            result.errors.append(
                "Aucun fait XBRL exploitable dans l'API SEC pour ce CIK. "
                "La société ne balise peut-être pas ses etats, ou la reponse est vide."
            )
            return result

        # ---- determine the fiscal horizon from the annual revenue series ---
        annual_revenue = [
            f for f in _pick(all_facts, CONCEPT_MAP[LineItem.REVENUE][0], FactKind.FLOW)
        ]
        year_ends: Dict[int, date] = {}
        for fact in annual_revenue:
            fy = self.fiscal_year_for(fact.end)
            current = year_ends.get(fy)
            if current is None or fact.end > current:
                year_ends[fy] = fact.end
        if not year_ends:
            result.errors.append(
                "Aucune serie annuelle de chiffre d'affaires n'a pu être identifiée dans "
                "les balises XBRL. Le DCF ne peut pas être construit automatiquement."
            )
            return result

        keep_years = sorted(year_ends)[-years:]

        dataset = FinancialDataSet(company_id=company_id, currency=currency)
        source_ids: Dict[str, SourceRecord] = {}
        restatement_notes: List[str] = []

        for item, (tags, kind) in CONCEPT_MAP.items():
            facts = _pick(all_facts, tags, kind)
            if not facts:
                continue
            for fy in keep_years:
                period_end = year_ends.get(fy)
                if period_end is None:
                    continue
                # facts whose annual window ends on this fiscal year end
                candidates = [
                    f for f in facts
                    if abs((f.end - period_end).days) <= 6
                    or (f.start is not None and abs((f.end - period_end).days) <= 6)
                ]
                if kind in (FactKind.INSTANT, FactKind.SHARES_INSTANT):
                    candidates = [
                        f for f in facts
                        if f.end <= period_end and (period_end - f.end).days <= 14
                    ]
                if not candidates:
                    continue

                chosen = _dedupe_latest_filed(candidates)
                if chosen is None:
                    continue

                # Restatements: more than one distinct value for the same period
                distinct = {round(f.value, 2) for f in candidates}
                if len(distinct) > 1:
                    restatement_notes.append(
                        f"{item.value} {fy}: {len(distinct)} valeurs distinctes publiees "
                        f"({', '.join(f'{v:,.0f}' for v in sorted(distinct))}); "
                        f"la plus récente ({chosen.value:,.0f}) est retenue, déposée le "
                        f"{chosen.filed.isoformat()}."
                    )

                value = self._to_millions(chosen.value, kind)
                if value is None:
                    continue

                period = FiscalPeriod(
                    fiscal_year=fy,
                    period_type=PeriodType.INSTANT if kind in (FactKind.INSTANT, FactKind.SHARES_INSTANT) else PeriodType.ANNUAL,
                    start_date=chosen.start,
                    end_date=chosen.end,
                )

                source_id = self._source_for(chosen, cik, source_ids, result)
                dataset.add(
                    Observation(
                        company_id=company_id,
                        line_item=item,
                        period=period,
                        value=value,
                        currency=currency,
                        reported_value=chosen.value,
                        reported_scale=Scale.UNITS,
                        reported_currency=currency,
                        status=ExtractionStatus.REPORTED,
                        source_id=source_id,
                        extraction_method=f"SEC XBRL us-gaap:{chosen.tag} ({chosen.form})",
                        notes=None,
                    )
                )

                # Annual-only periods also own the balance-sheet instant values,
                # so re-tag instants to the same annual label for the ratio layer.
                if kind in (FactKind.INSTANT, FactKind.SHARES_INSTANT):
                    dataset.add(
                        Observation(
                            company_id=company_id,
                            line_item=item,
                            period=FiscalPeriod(fiscal_year=fy, period_type=PeriodType.ANNUAL,
                                                end_date=chosen.end),
                            value=value,
                            currency=currency,
                            reported_value=chosen.value,
                            reported_scale=Scale.UNITS,
                            reported_currency=currency,
                            status=ExtractionStatus.REPORTED,
                            source_id=source_id,
                            extraction_method=(
                                f"SEC XBRL us-gaap:{chosen.tag} (instant au "
                                f"{chosen.end.isoformat()}, rattache a FY{fy})"
                            ),
                            notes="Point de bilan rattache a l'exercice annuel correspondant.",
                        )
                    )

        # ---- dei share count, if us-gaap has none ---------------------------
        if not _pick(all_facts, CONCEPT_MAP[LineItem.SHARES_OUTSTANDING][0], FactKind.SHARES_FLOW):
            dei_facts = _pick(all_facts, DEI_SHARES, FactKind.SHARES_INSTANT)
            if dei_facts:
                chosen = _dedupe_latest_filed(dei_facts[-1:])
                if chosen:
                    fy = self.fiscal_year_for(chosen.end)
                    source_id = self._source_for(chosen, cik, source_ids, result)
                    dataset.add(
                        Observation(
                            company_id=company_id,
                            line_item=LineItem.SHARES_OUTSTANDING,
                            period=FiscalPeriod(fiscal_year=fy, period_type=PeriodType.ANNUAL,
                                                end_date=chosen.end),
                            value=chosen.value / _SHARES_TO_MILLIONS,
                            currency=currency,
                            reported_value=chosen.value,
                            reported_scale=Scale.UNITS,
                            reported_currency=currency,
                            status=ExtractionStatus.REPORTED,
                            source_id=source_id,
                            extraction_method=f"SEC XBRL dei:{chosen.tag}",
                            notes="Nombre d'actions déclaré par le déposant.",
                        )
                    )

        if include_quarterly:
            self._add_quarterly(dataset, all_facts, year_ends, keep_years, cik, currency, source_ids, result)

        result.dataset = dataset
        result.sources = list(source_ids.values())
        result.notes.extend(restatement_notes)
        if filings:
            result.filings = filings

        if not dataset.periods(PeriodType.ANNUAL):
            result.errors.append(
                "Le référentiel SEC a repondu mais aucune période annuelle n'a pu être "
                "reconstruite (balises annuelles absentes ou non reconnues)."
            )
        return result

    # -- helpers -----------------------------------------------------------

    def fiscal_year_for(self, period_end: date) -> int:
        """Fiscal year label, honouring a non-December fiscal year end."""
        from data_sources.company_registry import fiscal_year_for

        return fiscal_year_for(period_end, getattr(self, "fiscal_year_end", None))

    @staticmethod
    def _to_millions(value: float, kind: str) -> Optional[float]:
        if kind in (FactKind.SHARES_FLOW, FactKind.SHARES_INSTANT):
            return value / _SHARES_TO_MILLIONS
        if kind == FactKind.PER_SHARE:
            return value  # already per share, never scaled
        if kind == FactKind.FLOW or kind == FactKind.INSTANT:
            return value / _UNITS_TO_MILLIONS
        return None

    def _source_for(
        self,
        fact: SecFact,
        cik: str,
        cache: Dict[str, SourceRecord],
        result: SecFetchResult,
    ) -> str:
        key = fact.accn or f"{fact.form}-{fact.filed.isoformat()}"
        source_id = f"sec:{key}"
        if source_id not in cache:
            cache[source_id] = SourceRecord(
                source_id=source_id,
                name=f"SEC {fact.form} filed {fact.filed.isoformat()}",
                url=self.client.filing_url(cik, fact.accn),
                publisher="U.S. Securities and Exchange Commission",
                publication_date=fact.filed,
                document_type=fact.form,
                tier=SourceTier.OFFICIAL,
                currency=Currency.USD,
                scale=Scale.UNITS,
                retrieved_at=datetime.now(timezone.utc),
                notes=f"Tag XBRL {fact.tag}; période {fact.start or 'instant'} -> {fact.end}.",
            )
        else:
            existing = cache[source_id]
            if f"{fact.tag}" not in (existing.notes or ""):
                cache[source_id] = existing.model_copy(
                    update={"notes": (existing.notes or "") + f" {fact.tag}"}
                )
        return source_id

    def _add_quarterly(
        self,
        dataset: FinancialDataSet,
        all_facts: Dict[str, List[SecFact]],
        year_ends: Dict[int, date],
        keep_years: List[int],
        cik: str,
        currency: Currency,
        source_ids: Dict[str, SourceRecord],
        result: SecFetchResult,
    ) -> None:
        """Add quarterly revenue and net income when the tags allow it."""
        for item in (LineItem.REVENUE, LineItem.NET_INCOME, LineItem.OPERATING_INCOME):
            tags, kind = CONCEPT_MAP[item]
            for tag in tags:
                facts = all_facts.get(tag)
                if not facts:
                    continue
                quarterly = [
                    f for f in facts
                    if f.start is not None and f.is_quarterly and f.form in ("10-Q", "10-Q/A")
                ]
                if not quarterly:
                    continue
                index: Dict[int, date] = {}
                for fact in quarterly:
                    if self.fiscal_year_for(fact.end) not in keep_years:
                        continue
                    index[(fact.end, fact.start)] = fact.end
                for (end, start), _ in index.items():
                    bucket = [f for f in quarterly if f.end == end and f.start == start]
                    chosen = _dedupe_latest_filed(bucket)
                    if chosen is None:
                        continue
                    fy = self.fiscal_year_for(chosen.end)
                    period_number = self._quarter_number(chosen, year_ends.get(fy))
                    source_id = self._source_for(chosen, cik, source_ids, result)
                    dataset.add(
                        Observation(
                            company_id=dataset.company_id,
                            line_item=item,
                            period=FiscalPeriod(
                                fiscal_year=fy,
                                period_type=PeriodType.QUARTERLY,
                                period_number=period_number,
                                start_date=chosen.start,
                                end_date=chosen.end,
                            ),
                            value=chosen.value / _UNITS_TO_MILLIONS,
                            currency=currency,
                            reported_value=chosen.value,
                            reported_scale=Scale.UNITS,
                            reported_currency=currency,
                            status=ExtractionStatus.REPORTED,
                            source_id=source_id,
                            extraction_method=f"SEC XBRL us-gaap:{chosen.tag} (10-Q)",
                        )
                    )
                break  # first tag with quarterly data wins for this line item

    @staticmethod
    def _quarter_number(fact: SecFact, fiscal_year_end_date: Optional[date]) -> Optional[int]:
        """Map a quarter end date onto T1..T4 of its fiscal year.

        Q4 ends on the fiscal year end (0 days back), Q1 on roughly 273 days
        back, and so on in 91-day steps.
        """
        if fiscal_year_end_date is None:
            return None
        days_back = (fiscal_year_end_date - fact.end).days
        if days_back < 0 or days_back > 380:
            return None
        return max(1, min(4, 4 - int(days_back // 91)))


def fetch_recent_filings(client: SecEdgarClient, cik: str, forms: Sequence[str] = ("10-K", "10-Q", "8-K"), limit: int = 25) -> List[Dict[str, object]]:
    """Recent filings of the requested types, newest first."""
    data, error = client.submissions(cik)
    if data is None:
        return []
    recent = (data.get("filings") or {}).get("recent") or {}
    forms_col = recent.get("form") or []
    out: List[Dict[str, object]] = []
    for i, form in enumerate(forms_col):
        if form not in forms:
            continue
        try:
            accn = recent["accessionNumber"][i]
            document = recent["primaryDocument"][i]
        except (KeyError, IndexError):
            continue
        out.append(
            {
                "form": form,
                "filed": recent["filingDate"][i],
                "report_date": (recent.get("reportDate") or [None] * len(forms_col))[i],
                "accession": accn,
                "document": document,
                "url": client.filing_url(cik, accn, document),
                "description": (recent.get("primaryDocDescription") or [None] * len(forms_col))[i],
            }
        )
        if len(out) >= limit:
            break
    return out


def submissions_metadata(client: SecEdgarClient, cik: str) -> Dict[str, object]:
    """Identity metadata straight from the SEC (ground truth for the profile)."""
    data, error = client.submissions(cik)
    if data is None:
        return {"error": error}
    return {
        "name": data.get("name"),
        "cik": normalise_cik(data.get("cik", cik)),
        "tickers": data.get("tickers"),
        "exchanges": data.get("exchanges"),
        "sic": data.get("sic"),
        "sic_description": data.get("sicDescription"),
        "fiscal_year_end": data.get("fiscalYearEnd"),
        "former_names": [
            f.get("name") for f in (data.get("formerNames") or [])
        ],
        "state_of_incorporation": data.get("stateOfIncorporation"),
    }


def load_company_from_sec(
    company_id: str,
    ticker: str,
    cik_fallback: Optional[str] = None,
    years: int = 6,
    http: Optional[HttpClient] = None,
) -> SecFetchResult:
    """Convenience entry point used by the UI and the AI tools."""
    client = SecEdgarClient(http=http)
    cik, error = client.resolve_cik(ticker, fallback=cik_fallback)
    if cik is None:
        result = SecFetchResult(company_id=company_id)
        result.errors.append(error or "CIK introuvable")
        return result

    # Metadata first: the nominal fiscal year end drives every period label.
    meta = submissions_metadata(client, cik)

    facts, facts_error = client.company_facts(cik)
    if facts is None:
        result = SecFetchResult(company_id=company_id, cik=cik)
        result.errors.append(f"SEC company facts indisponibles : {facts_error}")
        return result

    extractor = SecEdgarExtractor(client, company_id=company_id)
    result = extractor.build(
        company_id=company_id,
        cik=cik,
        years=years,
        companyfacts=facts,
        filings=fetch_recent_filings(client, cik),
        fiscal_year_end=meta.get("fiscal_year_end"),
    )
    result.sic_description = meta.get("sic_description")
    result.fiscal_year_end = meta.get("fiscal_year_end")
    result.entity_name = meta.get("name") or result.entity_name
    return result
