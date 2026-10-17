"""Canonical financial data schema.

Design rules enforced by this module
------------------------------------
1. **One canonical unit.**  Every monetary value handled by the analysis layer
   is expressed in *millions* of the company's reporting currency.  The
   originally reported figure, its scale and its currency are preserved next to
   the normalised value so any number displayed in the UI can be traced back to
   its source document.
2. **Missing is not zero.**  A missing value is represented by ``None`` (or by
   an ``ExtractionStatus.MISSING`` observation), never by ``0.0``.
3. **Provenance is mandatory.**  Every observation carries a source id, an
   extraction method and a verification status.
4. **Facts and opinions are separated.**  ``ExtractionStatus`` distinguishes
   what a company reported from what an analyst assumed.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Dict, Iterable, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --------------------------------------------------------------------------
# Units and currencies
# --------------------------------------------------------------------------


class Scale(str, Enum):
    """Reporting scale used by the source document."""

    UNITS = "units"
    THOUSANDS = "thousands"
    MILLIONS = "millions"
    BILLIONS = "billions"


SCALE_FACTORS: Dict[Scale, float] = {
    Scale.UNITS: 1.0,
    Scale.THOUSANDS: 1e3,
    Scale.MILLIONS: 1e6,
    Scale.BILLIONS: 1e9,
}

MILLIONS = 1e6


class Currency(str, Enum):
    MAD = "MAD"  # Moroccan dirham
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    CHF = "CHF"
    CAD = "CAD"
    AUD = "AUD"
    JPY = "JPY"
    ZAR = "ZAR"
    OTHER = "OTHER"


def to_millions(value: float, scale: Scale) -> float:
    """Convert a figure reported in ``scale`` into millions of currency."""
    return float(value) * SCALE_FACTORS[scale] / MILLIONS


def from_millions(value_millions: float, scale: Scale) -> float:
    """Convert a value expressed in millions back to the reported scale."""
    return float(value_millions) * MILLIONS / SCALE_FACTORS[scale]


# --------------------------------------------------------------------------
# Periods
# --------------------------------------------------------------------------


class PeriodType(str, Enum):
    ANNUAL = "annual"
    SEMI_ANNUAL = "semi_annual"
    QUARTERLY = "quarterly"
    TTM = "ttm"
    INSTANT = "instant"  # balance-sheet point in time


class FiscalPeriod(BaseModel):
    """A reporting period, explicitly fiscal (not assumed to be calendar)."""

    model_config = ConfigDict(frozen=True)

    fiscal_year: int = Field(ge=1900, le=2200)
    period_type: PeriodType = PeriodType.ANNUAL
    period_number: Optional[int] = Field(default=None, ge=1, le=4)
    start_date: Optional[date] = None
    end_date: Optional[date] = None

    @property
    def label(self) -> str:
        if self.period_type == PeriodType.ANNUAL:
            return f"FY{self.fiscal_year}"
        if self.period_type == PeriodType.SEMI_ANNUAL:
            return f"S{self.period_number or 1} {self.fiscal_year}"
        if self.period_type == PeriodType.QUARTERLY:
            return f"T{self.period_number} {self.fiscal_year}"
        if self.period_type == PeriodType.TTM:
            return f"TTM {self.fiscal_year}"
        return f"INSTANT {self.fiscal_year}"

    @property
    def is_flow(self) -> bool:
        """Income-statement / cash-flow statement items are flows."""
        return self.period_type != PeriodType.INSTANT

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.label


# --------------------------------------------------------------------------
# Line items
# --------------------------------------------------------------------------


class Statement(str, Enum):
    INCOME = "income"
    BALANCE = "balance"
    CASHFLOW = "cashflow"
    MARKET = "market"


class LineItem(str, Enum):
    """Canonical (normalised) line items supported by the platform."""

    # Income statement
    REVENUE = "revenue"
    COST_OF_REVENUE = "cost_of_revenue"
    GROSS_PROFIT = "gross_profit"
    OPERATING_EXPENSES = "operating_expenses"
    EBIT = "ebit"
    OPERATING_INCOME = "operating_income"
    DEPRECIATION_AMORTIZATION = "depreciation_amortization"
    EBITDA = "ebitda"
    INTEREST_EXPENSE = "interest_expense"
    PRETAX_INCOME = "pretax_income"
    INCOME_TAX_EXPENSE = "income_tax_expense"
    NET_INCOME = "net_income"
    EPS_DILUTED = "eps_diluted"

    # Balance sheet
    CASH_AND_EQUIVALENTS = "cash_and_equivalents"
    SHORT_TERM_INVESTMENTS = "short_term_investments"
    ACCOUNTS_RECEIVABLE = "accounts_receivable"
    INVENTORY = "inventory"
    CURRENT_ASSETS = "current_assets"
    TOTAL_ASSETS = "total_assets"
    ACCOUNTS_PAYABLE = "accounts_payable"
    CURRENT_LIABILITIES = "current_liabilities"
    SHORT_TERM_DEBT = "short_term_debt"
    LONG_TERM_DEBT = "long_term_debt"
    TOTAL_DEBT = "total_debt"
    TOTAL_LIABILITIES = "total_liabilities"
    TOTAL_EQUITY = "total_equity"
    NET_WORKING_CAPITAL = "net_working_capital"
    MINORITY_INTEREST = "minority_interest"

    # Cash flow
    OPERATING_CASH_FLOW = "operating_cash_flow"
    CAPITAL_EXPENDITURE = "capital_expenditure"
    FREE_CASH_FLOW = "free_cash_flow"
    DIVIDENDS_PAID = "dividends_paid"

    # Shares / market
    SHARES_OUTSTANDING = "shares_outstanding"
    DILUTED_SHARES = "diluted_shares"
    MARKET_PRICE = "market_price"
    MARKET_CAP = "market_cap"

    @property
    def statement(self) -> Statement:
        if self in _BALANCE_ITEMS:
            return Statement.BALANCE
        if self in _CASHFLOW_ITEMS:
            return Statement.CASHFLOW
        if self in _MARKET_ITEMS:
            return Statement.MARKET
        return Statement.INCOME


_BALANCE_ITEMS = {
    LineItem.CASH_AND_EQUIVALENTS,
    LineItem.SHORT_TERM_INVESTMENTS,
    LineItem.ACCOUNTS_RECEIVABLE,
    LineItem.INVENTORY,
    LineItem.CURRENT_ASSETS,
    LineItem.TOTAL_ASSETS,
    LineItem.ACCOUNTS_PAYABLE,
    LineItem.CURRENT_LIABILITIES,
    LineItem.SHORT_TERM_DEBT,
    LineItem.LONG_TERM_DEBT,
    LineItem.TOTAL_DEBT,
    LineItem.TOTAL_LIABILITIES,
    LineItem.TOTAL_EQUITY,
    LineItem.NET_WORKING_CAPITAL,
    LineItem.MINORITY_INTEREST,
}

_CASHFLOW_ITEMS = {
    LineItem.OPERATING_CASH_FLOW,
    LineItem.CAPITAL_EXPENDITURE,
    LineItem.FREE_CASH_FLOW,
    LineItem.DIVIDENDS_PAID,
}

_MARKET_ITEMS = {
    LineItem.SHARES_OUTSTANDING,
    LineItem.DILUTED_SHARES,
    LineItem.MARKET_PRICE,
    LineItem.MARKET_CAP,
}

#: Items whose natural sign is an outflow in most filings.  Used by validation
#: to warn when a source reports a positive "capex" that is really a magnitude.
OUTFLOW_ITEMS = {LineItem.CAPITAL_EXPENDITURE, LineItem.DIVIDENDS_PAID}


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


class ExtractionStatus(str, Enum):
    """How a figure came to exist - facts and opinions are kept apart."""

    REPORTED = "reported"        # copied verbatim from an official statement
    DERIVED = "derived"          # computed from reported items (FCF = OCF - capex)
    MANUAL = "manual"            # typed in by the user
    ESTIMATED = "estimated"      # analyst / AI assumption
    UNVERIFIED = "unverified"    # secondary source, not cross-checked
    MISSING = "missing"          # explicitly known to be unavailable


class SourceTier(int, Enum):
    OFFICIAL = 1       # regulator filing, company annual report, exchange
    SUPPLEMENTARY = 2  # reputable press, industry data, broker summaries
    DERIVED = 3        # computed inside this application


class SourceRecord(BaseModel):
    """A retrievable document or dataset a figure was taken from."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    name: str
    url: Optional[str] = None
    publisher: Optional[str] = None
    publication_date: Optional[date] = None
    document_type: Optional[str] = None  # e.g. "20-F", "annual report", "earnings release"
    tier: SourceTier = SourceTier.OFFICIAL
    currency: Optional[Currency] = None
    scale: Optional[Scale] = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    notes: Optional[str] = None

    def citation(self) -> str:
        bits = [self.publisher or self.name]
        if self.document_type:
            bits.append(self.document_type)
        if self.publication_date:
            bits.append(self.publication_date.isoformat())
        if self.url:
            bits.append(self.url)
        return " | ".join(bits)


class Observation(BaseModel):
    """One financial figure for one company, one period, one line item."""

    model_config = ConfigDict(frozen=True)

    company_id: str
    line_item: LineItem
    period: FiscalPeriod

    #: normalised value, always in MILLIONS of ``currency``
    value: Optional[float] = None
    currency: Currency = Currency.USD

    # audit trail of the original figure
    reported_value: Optional[float] = None
    reported_scale: Optional[Scale] = None
    reported_currency: Optional[Currency] = None

    status: ExtractionStatus = ExtractionStatus.REPORTED
    source_id: Optional[str] = None
    extraction_method: Optional[str] = None
    notes: Optional[str] = None

    @model_validator(mode="after")
    def _check_consistency(self) -> "Observation":
        if self.value is None and self.status not in (
            ExtractionStatus.MISSING,
            ExtractionStatus.UNVERIFIED,
        ):
            raise ValueError(
                f"Observation for {self.line_item.value} {self.period.label} has no "
                f"value but status={self.status.value}; use status='missing' instead."
            )
        return self

    @property
    def is_missing(self) -> bool:
        return self.value is None

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.company_id, self.line_item.value, self.period.label)


# --------------------------------------------------------------------------
# Company identity
# --------------------------------------------------------------------------


class CompanyProfile(BaseModel):
    """Identity card of a listed company.

    This model deliberately allows optional fields: unclear metadata is left
    ``None`` rather than guessed.
    """

    model_config = ConfigDict(frozen=True)

    company_id: str
    name: str
    short_name: str
    ticker: str
    exchange: str
    country: str
    currency: Currency
    sector: Optional[str] = None
    industry: Optional[str] = None
    description: Optional[str] = None
    fiscal_year_end: Optional[str] = None  # "MM-DD"
    website: Optional[str] = None
    ir_url: Optional[str] = None
    reporting_scale: Optional[Scale] = None

    # Identifiers used by official connectors
    sec_cik: Optional[str] = None
    isin: Optional[str] = None
    bloomberg_ticker: Optional[str] = None

    # Data-quality metadata
    listing_date: Optional[date] = None
    identity_notes: Optional[str] = None
    #: Explained discontinuities (spin-off, merger, change of scope...).
    #: Any statement dated before the cut-off must not be merged silently.
    history_cutoff: Optional[date] = None
    history_cutoff_note: Optional[str] = None

    data_sources: List[str] = Field(default_factory=list)

    @property
    def display_name(self) -> str:
        return f"{self.name} ({self.ticker})"


# --------------------------------------------------------------------------
# Container
# --------------------------------------------------------------------------


class FinancialDataSet(BaseModel):
    """A collection of observations with lookup helpers.

    Duplicate observations for the same (item, period) are resolved with an
    explicit precedence: reported > derived > manual > estimated > unverified.
    Nothing is averaged or silently overwritten.
    """

    company_id: str
    currency: Currency
    observations: List[Observation] = Field(default_factory=list)

    _PRECEDENCE = {
        ExtractionStatus.REPORTED: 0,
        ExtractionStatus.DERIVED: 1,
        ExtractionStatus.MANUAL: 2,
        ExtractionStatus.ESTIMATED: 3,
        ExtractionStatus.UNVERIFIED: 4,
        ExtractionStatus.MISSING: 5,
    }

    def add(self, obs: Observation) -> None:
        self.observations.append(obs)

    def extend(self, obs: Iterable[Observation]) -> None:
        self.observations.extend(obs)

    # -- lookups -----------------------------------------------------------

    def get(
        self,
        line_item: LineItem,
        period_label: str,
        include_estimated: bool = True,
    ) -> Optional[Observation]:
        """Best observation for an item/period, or ``None``."""
        candidates = [
            o
            for o in self.observations
            if o.line_item == line_item
            and o.period.label == period_label
            and (include_estimated or o.status not in (ExtractionStatus.ESTIMATED,))
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda o: self._PRECEDENCE.get(o.status, 99))

    def value(
        self,
        line_item: LineItem,
        period_label: str,
        include_estimated: bool = True,
    ) -> Optional[float]:
        obs = self.get(line_item, period_label, include_estimated=include_estimated)
        return None if obs is None else obs.value

    def periods(self, period_type: Optional[PeriodType] = PeriodType.ANNUAL) -> List[str]:
        """Period labels, chronologically sorted.

        Pass ``period_type=None`` to get every period regardless of its type.
        """
        labels = {
            o.period.label
            for o in self.observations
            if period_type is None or o.period.period_type == period_type
        }
        return sorted(labels, key=_period_sort_key)

    def latest_period_label(self, period_type: Optional[PeriodType] = PeriodType.ANNUAL) -> Optional[str]:
        labels = self.periods(period_type)
        return labels[-1] if labels else None

    def series(self, line_item: LineItem, period_type: Optional[PeriodType] = PeriodType.ANNUAL) -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {}
        for label in self.periods(period_type):
            out[label] = self.value(line_item, label)
        return out

    def duplicates(self) -> List[tuple[str, str, int]]:
        """(line item, period label, count) for every over-specified cell."""
        counts: Dict[tuple[str, str], int] = {}
        for o in self.observations:
            counts[(o.line_item.value, o.period.label)] = (
                counts.get((o.line_item.value, o.period.label), 0) + 1
            )
        return [
            (item, label, n)
            for (item, label), n in sorted(counts.items())
            if n > 1
        ]

    def sources(self) -> List[str]:
        return sorted({o.source_id for o in self.observations if o.source_id})


_FY_RE = None


def _period_sort_key(label: str) -> tuple[int, int]:
    """Sort labels chronologically: FY2023 -> (2023, 9), T2 2023 -> (2023, 2)."""
    import re

    m = re.match(r"^FY(\d{4})$", label)
    if m:
        return (int(m.group(1)), 9)
    m = re.match(r"^TTM (\d{4})$", label)
    if m:
        return (int(m.group(1)), 8)
    m = re.match(r"^S(\d) (\d{4})$", label)
    if m:
        return (int(m.group(2)), 5 + int(m.group(1)))
    m = re.match(r"^T(\d) (\d{4})$", label)
    if m:
        return (int(m.group(2)), int(m.group(1)))
    m = re.match(r"^INSTANT (\d{4})$", label)
    if m:
        return (int(m.group(1)), 9)
    return (9999, 99)


# --------------------------------------------------------------------------
# Valuation inputs
# --------------------------------------------------------------------------


class ScenarioName(str, Enum):
    BEAR = "bear"
    BASE = "base"
    BULL = "bull"


class ScenarioFlags(BaseModel):
    """Which inputs were touched by hand, so the UI can label them."""

    model_config = ConfigDict(frozen=True)

    manually_entered: List[str] = Field(default_factory=list)
    simulated: bool = False
    notes: Optional[str] = None


class ComparableCompany(BaseModel):
    """A peer observation used by the multiples engine."""

    model_config = ConfigDict(frozen=True)

    name: str
    ticker: Optional[str] = None
    sector: Optional[str] = None
    currency: Currency
    as_of: Optional[date] = None

    market_cap: Optional[float] = None          # millions, local currency
    enterprise_value: Optional[float] = None    # millions, local currency
    revenue: Optional[float] = None
    ebitda: Optional[float] = None
    net_income: Optional[float] = None
    book_value: Optional[float] = None

    pe: Optional[float] = None
    ev_ebitda: Optional[float] = None
    pb: Optional[float] = None

    source_id: Optional[str] = None
    include: bool = True
    notes: Optional[str] = None

    @field_validator("market_cap", "enterprise_value", "revenue", "ebitda", "net_income", "book_value")
    @classmethod
    def _finite(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return v
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("non-finite value")
        return v
