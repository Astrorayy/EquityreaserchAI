"""Data validation, plausibility checks and transparent derivations.

Nothing in this module ever *silently* repairs data:

* :func:`validate_dataset` only *reports* problems.
* :func:`derive_missing_items` adds clearly labelled ``DERIVED`` observations
  and never overwrites a ``REPORTED`` figure.

Every issue carries a stable ``code`` so the UI and the automated tests can
assert on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence

from financial_engine.schemas import (
    Currency,
    ExtractionStatus,
    FinancialDataSet,
    FiscalPeriod,
    LineItem,
    Observation,
    PeriodType,
)


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


#: Relative tolerance for accounting identities (1 %).
REL_TOL = 0.01
#: Absolute tolerance floor, in millions of currency.
ABS_TOL = 0.5


@dataclass(frozen=True)
class Issue:
    code: str
    severity: Severity
    message: str
    context: Dict[str, object] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{self.severity.value.upper()}] {self.code}: {self.message}"


def _similar(a: Optional[float], b: Optional[float], rel_tol: float = REL_TOL, abs_tol: float = ABS_TOL) -> bool:
    """Tolerance-aware comparison that treats ``None`` as 'cannot compare'."""
    if a is None or b is None:
        return True
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) <= max(abs_tol, rel_tol * scale)


# --------------------------------------------------------------------------
# Per-item plausibility
# --------------------------------------------------------------------------

#: Hard bounds on annual revenue (in millions of reporting currency).  Outside
#: this band the value is almost certainly a units error.
_REVENUE_ABS_MIN = 0.01
_REVENUE_ABS_MAX = 5_000_000.0


def check_plausible_values(observations: Sequence[Observation]) -> List[Issue]:
    issues: List[Issue] = []
    for obs in observations:
        if obs.value is None:
            continue
        v = obs.value
        ctx = {"item": obs.line_item.value, "period": obs.period.label, "value": v}

        if obs.line_item == LineItem.REVENUE:
            if v < _REVENUE_ABS_MIN:
                issues.append(Issue("IMPLAUSIBLE_MAGNITUDE", Severity.WARNING,
                                    "Chiffre d'affaires nul ou quasi nul — vérifiez l'échelle du document.", ctx))
            elif v > _REVENUE_ABS_MAX:
                issues.append(Issue("IMPLAUSIBLE_MAGNITUDE", Severity.ERROR,
                                    "Chiffre d'affaires supérieur à 5 000 Md — erreur d'unité probable.", ctx))

        if obs.line_item in (LineItem.SHARES_OUTSTANDING, LineItem.DILUTED_SHARES):
            if v <= 0:
                issues.append(Issue("IMPLAUSIBLE_MAGNITUDE", Severity.ERROR,
                                    "Le nombre d'actions doit être strictement positif.", ctx))

        if obs.line_item == LineItem.MARKET_PRICE and v <= 0:
            issues.append(Issue("IMPLAUSIBLE_MAGNITUDE", Severity.ERROR,
                                "Le cours de bourse doit être strictement positif.", ctx))
    return issues


def check_signs(observations: Sequence[Observation]) -> List[Issue]:
    """Capital expenditure and dividends are outflows.

    Companies report them either as a negative number or as a positive
    magnitude; both are acceptable, but the two conventions must not be mixed
    inside one period, because that would corrupt the free-cash-flow build.
    """
    issues: List[Issue] = []
    by_period: Dict[str, List[Observation]] = {}
    for obs in observations:
        if obs.line_item in (LineItem.CAPITAL_EXPENDITURE, LineItem.DIVIDENDS_PAID) and obs.value is not None:
            by_period.setdefault(obs.period.label, []).append(obs)

    for label, obs_list in by_period.items():
        signs = {o.line_item: ("positif" if o.value >= 0 else "négatif") for o in obs_list}
        values = list(signs.values())
        if "positif" in values and "négatif" in values:
            issues.append(Issue(
                "SIGN_CONVENTION_MIXED", Severity.WARNING,
                "Les investissements et les dividendes utilisent des conventions de signe "
                "opposées sur la même période. Normalisez le signe avant de lancer un DCF.",
                {"period": label, "items": signs},
            ))
    return issues


# --------------------------------------------------------------------------
# Accounting identities
# --------------------------------------------------------------------------


def _get(ds: FinancialDataSet, item: LineItem, label: str) -> Optional[float]:
    return ds.value(item, label)


def check_accounting_identities(ds: FinancialDataSet, periods: Optional[Iterable[str]] = None) -> List[Issue]:
    """Cross-check the standard relationships between reported line items."""
    issues: List[Issue] = []
    labels = list(periods) if periods is not None else _all_labels(ds)

    for label in labels:
        rev = _get(ds, LineItem.REVENUE, label)
        cogs = _get(ds, LineItem.COST_OF_REVENUE, label)
        gp = _get(ds, LineItem.GROSS_PROFIT, label)
        opex = _get(ds, LineItem.OPERATING_EXPENSES, label)
        ebit = _coalesce(_get(ds, LineItem.EBIT, label), _get(ds, LineItem.OPERATING_INCOME, label))
        da = _get(ds, LineItem.DEPRECIATION_AMORTIZATION, label)
        ebitda = _get(ds, LineItem.EBITDA, label)
        pretax = _get(ds, LineItem.PRETAX_INCOME, label)
        tax = _get(ds, LineItem.INCOME_TAX_EXPENSE, label)
        ni = _get(ds, LineItem.NET_INCOME, label)

        st_debt = _get(ds, LineItem.SHORT_TERM_DEBT, label)
        lt_debt = _get(ds, LineItem.LONG_TERM_DEBT, label)
        tot_debt = _get(ds, LineItem.TOTAL_DEBT, label)

        ocf = _get(ds, LineItem.OPERATING_CASH_FLOW, label)
        capex = _get(ds, LineItem.CAPITAL_EXPENDITURE, label)
        fcf = _get(ds, LineItem.FREE_CASH_FLOW, label)

        ta = _get(ds, LineItem.TOTAL_ASSETS, label)
        tl = _get(ds, LineItem.TOTAL_LIABILITIES, label)
        te = _get(ds, LineItem.TOTAL_EQUITY, label)

        ca = _get(ds, LineItem.CURRENT_ASSETS, label)
        cl = _get(ds, LineItem.CURRENT_LIABILITIES, label)
        inv = _get(ds, LineItem.INVENTORY, label)
        ar = _get(ds, LineItem.ACCOUNTS_RECEIVABLE, label)

        # 1. Gross profit
        if rev is not None and cogs is not None and gp is not None:
            expected = rev - abs(cogs) if cogs < 0 else rev - cogs
            if not _similar(gp, expected):
                issues.append(Issue(
                    "IDENTITY_GROSS_PROFIT", Severity.WARNING,
                    f"Marge brute ({gp:,.1f}) différente du chiffre d'affaires moins le "
                    f"coût des ventes ({expected:,.1f}). Le document source utilise "
                    "peut-être une autre définition du coût.",
                    {"period": label, "reported": gp, "implied": expected},
                ))

        # 2. Operating income vs revenue - cogs - opex
        if rev is not None and ebit is not None and (cogs is not None or opex is not None):
            c = (abs(cogs) if (cogs is not None and cogs < 0) else (cogs or 0.0))
            o = (abs(opex) if (opex is not None and opex < 0) else (opex or 0.0))
            implied = rev - c - o
            if cogs is not None and opex is not None and not _similar(ebit, implied):
                issues.append(Issue(
                    "IDENTITY_OPERATING_INCOME", Severity.INFO,
                    f"Résultat d'exploitation ({ebit:,.1f}) différent du calcul chiffre "
                    f"d'affaires moins coûts ({implied:,.1f}) ; des éléments non récurrents "
                    "sont peut-être inclus.",
                    {"period": label, "reported": ebit, "implied": implied},
                ))

        # 3. EBITDA = EBIT + D&A
        if ebit is not None and da is not None and ebitda is not None:
            d = abs(da)
            if not _similar(ebitda, ebit + d):
                issues.append(Issue(
                    "IDENTITY_EBITDA", Severity.WARNING,
                    f"EBITDA ({ebitda:,.1f}) différent de EBIT + amortissements ({ebit + d:,.1f}).",
                    {"period": label, "reported": ebitda, "implied": ebit + d},
                ))

        # 4. Net income = pretax - tax
        if pretax is not None and tax is not None and ni is not None:
            t = abs(tax)
            if not _similar(ni, pretax - t):
                issues.append(Issue(
                    "IDENTITY_NET_INCOME", Severity.INFO,
                    f"Résultat net ({ni:,.1f}) différent du résultat avant impôt moins "
                    f"l'impôt ({pretax - t:,.1f}) ; intérêts minoritaires ou activités "
                    "arrêtées possibles.",
                    {"period": label, "reported": ni, "implied": pretax - t},
                ))

        # 5. Total debt = short-term + long-term
        if tot_debt is not None and (st_debt is not None or lt_debt is not None):
            implied = (abs(st_debt) if st_debt is not None else 0.0) + (
                abs(lt_debt) if lt_debt is not None else 0.0
            )
            if not _similar(tot_debt, implied):
                issues.append(Issue(
                    "IDENTITY_TOTAL_DEBT", Severity.WARNING,
                    f"Dette totale ({tot_debt:,.1f}) différente de dette court terme + "
                    f"long terme ({implied:,.1f}).",
                    {"period": label, "reported": tot_debt, "implied": implied},
                ))

        # 6. Free cash flow = operating cash flow - capex
        if ocf is not None and capex is not None and fcf is not None:
            implied = ocf - abs(capex)
            if not _similar(fcf, implied):
                issues.append(Issue(
                    "IDENTITY_FREE_CASH_FLOW", Severity.WARNING,
                    f"Flux de trésorerie disponible ({fcf:,.1f}) différent du flux "
                    f"d'exploitation moins les investissements ({implied:,.1f}).",
                    {"period": label, "reported": fcf, "implied": implied},
                ))

        # 7. Balance sheet: assets = liabilities + equity
        if ta is not None and tl is not None and te is not None and not _similar(ta, tl + te, rel_tol=0.02):
            issues.append(Issue(
                "IDENTITY_BALANCE_SHEET", Severity.WARNING,
                f"Total assets ({ta:,.1f}) différent du passif + capitaux propres ({tl + te:,.1f}).",
                {"period": label, "assets": ta, "liabilities_plus_equity": tl + te},
            ))

        # 8. Structural: inventory and receivables cannot exceed current assets
        if ca is not None:
            for item_name, value in (("inventory", inv), ("accounts receivable", ar)):
                if value is not None and value > ca * 1.02:
                    issues.append(Issue(
                        "STRUCTURAL_COMPONENT_TOO_LARGE", Severity.WARNING,
                        f"{item_name.capitalize()} ({value:,.1f}) dépasse l'actif courant ({ca:,.1f}).",
                        {"period": label},
                    ))

        # 9. Current liabilities present with current assets -> working capital usable
        if ca is not None and cl is not None and inv is not None and ca < 0:
            issues.append(Issue(
                "NEGATIVE_CURRENT_ASSETS", Severity.INFO,
                "Actif courant négatif — vérifiez la convention de signe.",
                {"period": label},
            ))

    return issues


def _coalesce(*values: Optional[float]) -> Optional[float]:
    for v in values:
        if v is not None:
            return v
    return None


def _all_labels(ds: FinancialDataSet) -> List[str]:
    """Every period label in the dataset, chronologically sorted."""
    return ds.periods(period_type=None)


# --------------------------------------------------------------------------
# Cross-period and cross-source checks
# --------------------------------------------------------------------------


def check_currency_consistency(ds: FinancialDataSet) -> List[Issue]:
    issues: List[Issue] = []
    for obs in ds.observations:
        if obs.reported_currency is not None and obs.reported_currency != obs.currency:
            issues.append(Issue(
                "CURRENCY_MISMATCH", Severity.WARNING,
                f"{obs.line_item.value} {obs.period.label} est publié en "
                f"{obs.reported_currency.value} mais normalisé en {obs.currency.value} ; "
                "un taux de change doit être documenté.",
                {"period": obs.period.label, "item": obs.line_item.value},
            ))
    return issues


def check_duplicates(ds: FinancialDataSet) -> List[Issue]:
    return [
        Issue(
            "DUPLICATE_OBSERVATION", Severity.INFO,
            f"{item} {label} est présent {n} fois ; la source de plus haute priorité est utilisée.",
            {"item": item, "period": label, "count": n},
        )
        for item, label, n in ds.duplicates()
    ]


def check_restatements(
    current: FinancialDataSet,
    previous: Optional[FinancialDataSet],
    rel_tol: float = 0.02,
) -> List[Issue]:
    """Flag items whose value changed between two vintages of the same period."""
    if previous is None:
        return []
    issues: List[Issue] = []
    for obs in current.observations:
        if obs.value is None:
            continue
        old = previous.get(obs.line_item, obs.period.label)
        if old is None or old.value is None:
            continue
        if not _similar(obs.value, old.value, rel_tol=rel_tol):
            delta = obs.value - old.value
            pct = 100.0 * delta / old.value if old.value else float("inf")
            issues.append(Issue(
                "RESTATEMENT_DETECTED", Severity.WARNING,
                f"{obs.line_item.value} {obs.period.label} a changé de {pct:+.1f}% "
                f"({old.value:,.1f} -> {obs.value:,.1f}) entre deux sources ; "
                "le retraitement doit être documenté avant utilisation.",
                {"period": obs.period.label, "item": obs.line_item.value, "delta": delta},
            ))
    return issues


def check_quarter_sum(
    ds: FinancialDataSet,
    line_item: LineItem = LineItem.REVENUE,
    year: Optional[int] = None,
    period_type: PeriodType = PeriodType.QUARTERLY,
) -> List[Issue]:
    """Compare the sum of quarterly figures against the annual figure."""
    issues: List[Issue] = []
    labels = ds.periods(period_type)
    by_year: Dict[int, List[str]] = {}
    for label in labels:
        m = __import__("re").match(r"^T(\d) (\d{4})$", label)
        if m:
            by_year.setdefault(int(m.group(2)), []).append(label)

    for yr, qlabels in by_year.items():
        if year is not None and yr != year:
            continue
        if len(qlabels) != 4:
            if qlabels:
                issues.append(Issue(
                    "INCOMPLETE_QUARTERS", Severity.INFO,
                    f"Seulement {len(qlabels)}/4 trimestres disponibles pour {yr} ; les "
                    "comptes annuels ne peuvent pas être reconstruits à partir des trimestres.",
                    {"year": yr},
                ))
            continue
        total = 0.0
        complete = True
        for q in qlabels:
            v = ds.value(line_item, q)
            if v is None:
                complete = False
                break
            total += v
        annual = ds.value(line_item, f"FY{yr}") if period_type != PeriodType.ANNUAL else None
        if complete and annual is not None and not _similar(total, annual, rel_tol=0.03):
            issues.append(Issue(
                "QUARTER_SUM_MISMATCH", Severity.WARNING,
                f"Somme de {line_item.value} sur les quatre trimestres de {yr} ({total:,.1f}) "
                f"différente du chiffre annuel ({annual:,.1f}).",
                {"year": yr, "sum_quarters": total, "annual": annual},
            ))
    return issues


def check_missing_critical(ds: FinancialDataSet, required: Optional[Sequence[LineItem]] = None) -> List[Issue]:
    """Report line items that are absent for the most recent annual period."""
    required = required or (
        LineItem.REVENUE,
        LineItem.EBIT,
        LineItem.NET_INCOME,
        LineItem.OPERATING_CASH_FLOW,
        LineItem.CAPITAL_EXPENDITURE,
    )
    labels = ds.periods(PeriodType.ANNUAL)
    issues: List[Issue] = []
    if not labels:
        issues.append(Issue(
            "NO_ANNUAL_DATA", Severity.ERROR,
            "Aucun exercice annuel disponible pour cette société. Un DCF ne peut pas être construit.",
            {},
        ))
        return issues
    latest = labels[-1]
    for item in required:
        obs = ds.get(item, latest)
        if obs is None or obs.value is None:
            issues.append(Issue(
                "MISSING_CRITICAL_ITEM", Severity.WARNING,
                f"{item.value} est absent pour {latest}. Il devra être saisi "
                "manuellement ou téléchargé depuis un document officiel.",
                {"period": latest, "item": item.value},
            ))
    return issues


# --------------------------------------------------------------------------
# Derivations (explicit, never overwriting reported figures)
# --------------------------------------------------------------------------


@dataclass
class DerivationReport:
    created: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    impossible: List[str] = field(default_factory=list)


def derive_missing_items(ds: FinancialDataSet) -> DerivationReport:
    """Add DERIVED observations where a strict identity allows it.

    Only created when *all* inputs are present and the target is absent.
    """
    report = DerivationReport()

    for ptype in (PeriodType.ANNUAL, PeriodType.SEMI_ANNUAL, PeriodType.QUARTERLY):
        for label in ds.periods(ptype):
            period = _period_for_label(ds, label)
            if period is None:
                continue

            rev = ds.value(LineItem.REVENUE, label)
            cogs = ds.value(LineItem.COST_OF_REVENUE, label)
            gp = ds.value(LineItem.GROSS_PROFIT, label)
            if gp is None and rev is not None and cogs is not None:
                value = rev - (abs(cogs) if cogs < 0 else cogs)
                ds.add(_derived(ds, LineItem.GROSS_PROFIT, period, value,
                                "revenue - cost_of_revenue"))
                report.created.append(f"gross_profit {label}")
            elif gp is not None:
                report.skipped.append(f"gross_profit {label}")

            st = ds.value(LineItem.SHORT_TERM_DEBT, label)
            lt = ds.value(LineItem.LONG_TERM_DEBT, label)
            if ds.value(LineItem.TOTAL_DEBT, label) is None and (st is not None or lt is not None):
                value = abs(st or 0.0) + abs(lt or 0.0)
                ds.add(_derived(ds, LineItem.TOTAL_DEBT, period, value,
                                "short_term_debt + long_term_debt"))
                report.created.append(f"total_debt {label}")

            ebit = _coalesce(ds.value(LineItem.EBIT, label), ds.value(LineItem.OPERATING_INCOME, label))
            da = ds.value(LineItem.DEPRECIATION_AMORTIZATION, label)
            if ds.value(LineItem.EBITDA, label) is None and ebit is not None and da is not None:
                ds.add(_derived(ds, LineItem.EBITDA, period, ebit + abs(da), "ebit + d_and_a"))
                report.created.append(f"ebitda {label}")

            ocf = ds.value(LineItem.OPERATING_CASH_FLOW, label)
            capex = ds.value(LineItem.CAPITAL_EXPENDITURE, label)
            if ds.value(LineItem.FREE_CASH_FLOW, label) is None and ocf is not None and capex is not None:
                ds.add(_derived(ds, LineItem.FREE_CASH_FLOW, period, ocf - abs(capex),
                                "operating_cash_flow - capital_expenditure"))
                report.created.append(f"free_cash_flow {label}")

            ca = ds.value(LineItem.CURRENT_ASSETS, label)
            cl = ds.value(LineItem.CURRENT_LIABILITIES, label)
            inv = ds.value(LineItem.INVENTORY, label)
            if ds.value(LineItem.NET_WORKING_CAPITAL, label) is None and ca is not None and cl is not None:
                value = (ca - (inv or 0.0)) - cl
                ds.add(_derived(ds, LineItem.NET_WORKING_CAPITAL, period, value,
                                "(current_assets - inventory) - current_liabilities"))
                report.created.append(f"net_working_capital {label}")

            price = ds.value(LineItem.MARKET_PRICE, label)
            shares = _coalesce(ds.value(LineItem.DILUTED_SHARES, label), ds.value(LineItem.SHARES_OUTSTANDING, label))
            if ds.value(LineItem.MARKET_CAP, label) is None and price is not None and shares is not None:
                ds.add(_derived(ds, LineItem.MARKET_CAP, period, price * shares,
                                "market_price * diluted_shares"))
                report.created.append(f"market_cap {label}")

    return report


def _derived(ds: FinancialDataSet, item: LineItem, period: FiscalPeriod, value: float, formula: str) -> Observation:
    return Observation(
        company_id=ds.company_id,
        line_item=item,
        period=period,
        value=value,
        currency=ds.currency,
        status=ExtractionStatus.DERIVED,
        source_id="derived:internal",
        extraction_method=formula,
        notes=f"Calculé par l'application : {formula}. Poste non publié tel quel.",
    )


def _period_for_label(ds: FinancialDataSet, label: str) -> Optional[FiscalPeriod]:
    for obs in ds.observations:
        if obs.period.label == label:
            return obs.period
    return None


# --------------------------------------------------------------------------
# Aggregate entry point
# --------------------------------------------------------------------------


@dataclass
class DataQualityReport:
    issues: List[Issue] = field(default_factory=list)

    @property
    def errors(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    @property
    def infos(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == Severity.INFO]

    def by_code(self, code: str) -> List[Issue]:
        return [i for i in self.issues if i.code == code]

    @property
    def score(self) -> int:
        """0-100 confidence indicator; purely a communication device."""
        score = 100 - 25 * len(self.errors) - 8 * len(self.warnings) - 2 * len(self.infos)
        return max(0, min(100, score))

    @property
    def is_usable(self) -> bool:
        return not self.errors


def validate_dataset(
    ds: FinancialDataSet,
    previous: Optional[FinancialDataSet] = None,
) -> DataQualityReport:
    """Run every validation rule and aggregate the issues."""
    issues: List[Issue] = []
    issues += check_plausible_values(ds.observations)
    issues += check_signs(ds.observations)
    issues += check_accounting_identities(ds)
    issues += check_currency_consistency(ds)
    issues += check_duplicates(ds)
    issues += check_restatements(ds, previous)
    issues += check_missing_critical(ds)
    return DataQualityReport(issues=issues)
