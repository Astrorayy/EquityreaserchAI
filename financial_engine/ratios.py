"""Financial ratio engine.

Every metric carries:

* its **value** (or ``None``),
* its **formula** as a readable string,
* an **interpretation** note for the analyst,
* the **inputs** actually used,
* a **status** distinguishing reported / calculated / estimated figures,
* a **reason** whenever it cannot be computed.

Rules enforced here
-------------------
* A ratio is never produced when its denominator is missing, zero, or
  economically meaningless (a P/E on a loss-making company is not a number,
  it is a warning).
* Nothing is defaulted to zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from financial_engine.schemas import (
    ExtractionStatus,
    FinancialDataSet,
    LineItem,
    Observation,
    PeriodType,
)


@dataclass(frozen=True)
class Metric:
    key: str
    label: str           # French UI label
    value: Optional[float]
    unit: str            # "percent" | "multiple" | "currency_millions" | "currency_per_share"
    formula: str
    interpretation: str
    inputs: Dict[str, Optional[float]] = field(default_factory=dict)
    status: ExtractionStatus = ExtractionStatus.DERIVED
    reason: Optional[str] = None
    period: Optional[str] = None

    @property
    def is_available(self) -> bool:
        return self.value is not None and self.reason is None

    def as_dict(self) -> Dict[str, object]:
        return {
            "metric": self.key,
            "label": self.label,
            "value": self.value,
            "unit": self.unit,
            "formula": self.formula,
            "period": self.period,
            "status": self.status.value,
            "reason": self.reason,
        }


# --------------------------------------------------------------------------
# Arithmetic guards
# --------------------------------------------------------------------------


def safe_div(
    numerator: Optional[float],
    denominator: Optional[float],
    *,
    require_positive_denominator: bool = False,
    label: str = "ratio",
) -> tuple[Optional[float], Optional[str]]:
    """Divide, or explain precisely why the division is not legitimate.

    Returns ``(value, reason)``; exactly one of the two is ``None``.
    """
    if numerator is None and denominator is None:
        return None, "le numérateur et le dénominateur sont tous deux absents"
    if numerator is None:
        return None, "le numérateur est absent"
    if denominator is None:
        return None, "le dénominateur est absent"
    if denominator == 0:
        return None, f"le dénominateur est nul : le {label} n'est pas défini"
    if require_positive_denominator and denominator < 0:
        return None, (
            f"le dénominateur est négatif : le {label} n'a pas de sens économique"
        )
    return numerator / denominator, None


def _pct(x: Optional[float]) -> Optional[float]:
    return None if x is None else x * 100.0


# --------------------------------------------------------------------------
# Definitions
# --------------------------------------------------------------------------

#: Definitional notes shared by the UI tooltips.  Kept in one place so that
#: the whole application uses the same convention.
DEFINITIONS = {
    "ebitda": "EBIT + amortissements (valeur publiée si disponible, calculée sinon).",
    "net_debt": "Dette totale moins trésorerie et équivalents. Une valeur négative signifie une trésorerie nette.",
    "fcf": "Flux de trésorerie d'exploitation moins investissements.",
    "ev": "Capitalisation boursière + dette nette + intérêts minoritaires.",
    "roic": "NOPAT / capital investi, avec capital investi = dette totale + capitaux propres - trésorerie.",
    "nopat": "EBIT x (1 - taux d'imposition effectif).",
}

_EBIT_ALIASES = (LineItem.EBIT, LineItem.OPERATING_INCOME)


class RatioEngine:
    """Computes the standard metric set for one company dataset."""

    def __init__(self, dataset: FinancialDataSet):
        self.ds = dataset

    # -- primitive accessors ---------------------------------------------

    def _v(self, item: LineItem, period: str) -> Optional[float]:
        return self.ds.value(item, period)

    def _observation(self, item: LineItem, period: str) -> Optional[Observation]:
        return self.ds.get(item, period)

    def _ebit(self, period: str) -> tuple[Optional[float], str]:
        for item in _EBIT_ALIASES:
            v = self._v(item, period)
            if v is not None:
                return v, item.value
        return None, "ebit"

    def _ebitda(self, period: str) -> tuple[Optional[float], str]:
        direct = self._v(LineItem.EBITDA, period)
        if direct is not None:
            return direct, "EBITDA (publié)"
        ebit, _ = self._ebit(period)
        da = self._v(LineItem.DEPRECIATION_AMORTIZATION, period)
        if ebit is not None and da is not None:
            return ebit + abs(da), "EBIT + amortissements (calculé)"
        return None, "EBITDA (indisponible)"

    def _tax_rate(self, period: str) -> tuple[Optional[float], str]:
        tax = self._v(LineItem.INCOME_TAX_EXPENSE, period)
        pretax = self._v(LineItem.PRETAX_INCOME, period)
        if tax is not None and pretax not in (None, 0):
            return abs(tax) / pretax, "effective rate (tax expense / pretax income)"
        return None, "taux d'impôt effectif indisponible"

    def net_debt(self, period: str) -> tuple[Optional[float], Optional[str]]:
        debt = self._v(LineItem.TOTAL_DEBT, period)
        cash = self._v(LineItem.CASH_AND_EQUIVALENTS, period)
        if debt is None and cash is None:
            return None, "ni la dette totale ni la trésorerie ne sont disponibles"
        if debt is None:
            return None, "dette totale absente : la dette nette ne peut pas être établie"
        return debt - (cash or 0.0), None

    def enterprise_value(self, period: str) -> tuple[Optional[float], Optional[str]]:
        mcap = self._v(LineItem.MARKET_CAP, period)
        if mcap is None:
            return None, "capitalisation boursière absente"
        nd, reason = self.net_debt(period)
        if nd is None:
            return None, reason
        minority = self._v(LineItem.MINORITY_INTEREST, period) or 0.0
        return mcap + nd + minority, None

    # -- individual metrics ----------------------------------------------

    def revenue_growth(self, period: str, previous: str) -> Metric:
        cur = self._v(LineItem.REVENUE, period)
        prev = self._v(LineItem.REVENUE, previous)
        value, reason = safe_div(
            None if (cur is None or prev is None) else cur - prev,
            prev,
            require_positive_denominator=True,
            label="revenue growth",
        )
        return Metric(
            key="revenue_growth",
            label="Croissance du chiffre d'affaires",
            value=None if value is None else value * 100.0,
            unit="percent",
            formula=f"(CA {period} - CA {previous}) / CA {previous}",
            interpretation=(
                "Croissance du chiffre d'affaires d'une période a l'autre. "
                "A interpréter en tenant compte de la cyclicité du secteur."
            ),
            inputs={"revenue_current": cur, "revenue_previous": prev},
            period=period,
            reason=reason,
        )

    def gross_margin(self, period: str) -> Metric:
        gp = self._v(LineItem.GROSS_PROFIT, period)
        rev = self._v(LineItem.REVENUE, period)
        value, reason = safe_div(gp, rev, label="gross margin")
        return Metric(
            key="gross_margin",
            label="Marge brute",
            value=_pct(value),
            unit="percent",
            formula="Marge brute / Chiffre d'affaires",
            interpretation=(
                "Part du chiffre d'affaires restant après le coût des ventes. "
                "Très sensible aux prix des matières premières pour une minière "
                "et aux prix NAND pour un fabricant de mémoire."
            ),
            inputs={"gross_profit": gp, "revenue": rev},
            period=period,
            reason=reason,
        )

    def operating_margin(self, period: str) -> Metric:
        ebit, src = self._ebit(period)
        rev = self._v(LineItem.REVENUE, period)
        value, reason = safe_div(ebit, rev, label="operating margin")
        return Metric(
            key="operating_margin",
            label="Marge opérationnelle",
            value=_pct(value),
            unit="percent",
            formula="Résultat opérationnel (EBIT) / Chiffre d'affaires",
            interpretation=(
                "Rentabilité de l'exploitation avant éléments financiers et fiscaux. "
                f"Source de l'EBIT : {src}."
            ),
            inputs={"ebit": ebit, "revenue": rev},
            period=period,
            reason=reason,
        )

    def ebitda_margin(self, period: str) -> Metric:
        ebitda, src = self._ebitda(period)
        rev = self._v(LineItem.REVENUE, period)
        value, reason = safe_div(ebitda, rev, label="ebitda margin")
        return Metric(
            key="ebitda_margin",
            label="Marge EBITDA",
            value=_pct(value),
            unit="percent",
            formula="EBITDA / Chiffre d'affaires",
            interpretation=(
                "Marge opérationnelle avant amortissements. " + DEFINITIONS["ebitda"] +
                f" Source : {src}."
            ),
            inputs={"ebitda": ebitda, "revenue": rev},
            period=period,
            reason=reason,
        )

    def net_margin(self, period: str) -> Metric:
        ni = self._v(LineItem.NET_INCOME, period)
        rev = self._v(LineItem.REVENUE, period)
        value, reason = safe_div(ni, rev, label="net margin")
        return Metric(
            key="net_margin",
            label="Marge nette",
            value=_pct(value),
            unit="percent",
            formula="Résultat net / Chiffre d'affaires",
            interpretation=(
                "Rentabilité finale après charges financières et impôts, "
                "attribuable a l'ensemble des actionnaires."
            ),
            inputs={"net_income": ni, "revenue": rev},
            period=period,
            reason=reason,
        )

    def roe(self, period: str, previous: Optional[str] = None) -> Metric:
        ni = self._v(LineItem.NET_INCOME, period)
        eq = self._v(LineItem.TOTAL_EQUITY, period)
        eq_prev = self._v(LineItem.TOTAL_EQUITY, previous) if previous else None
        avg_eq = eq if eq_prev is None or eq is None else (eq + eq_prev) / 2.0
        used = "capitaux propres moyens" if eq_prev is not None else "capitaux propres de cloture"
        value, reason = safe_div(ni, avg_eq, require_positive_denominator=True, label="ROE")
        return Metric(
            key="roe",
            label="Rentabilité des capitaux propres (ROE)",
            value=_pct(value),
            unit="percent",
            formula="Résultat net / Capitaux propres",
            interpretation=(
                f"Rendement pour les actionnaires, calcule sur les {used}. "
                "A comparer au coût des fonds propres."
            ),
            inputs={"net_income": ni, "equity": eq, "equity_previous": eq_prev},
            period=period,
            reason=reason,
        )

    def roic(self, period: str) -> Metric:
        ebit, _ = self._ebit(period)
        tax_rate, tax_src = self._tax_rate(period)
        debt = self._v(LineItem.TOTAL_DEBT, period)
        eq = self._v(LineItem.TOTAL_EQUITY, period)
        cash = self._v(LineItem.CASH_AND_EQUIVALENTS, period)

        if ebit is None or tax_rate is None or debt is None or eq is None:
            missing = []
            if ebit is None:
                missing.append("EBIT")
            if tax_rate is None:
                missing.append("taux d'imposition effectif")
            if debt is None:
                missing.append("dette totale")
            if eq is None:
                missing.append("capitaux propres")
            return Metric(
                key="roic", label="Rendement du capital investi (ROIC)", value=None,
                unit="percent", formula="NOPAT / Capital investi",
                interpretation=DEFINITIONS["roic"], period=period,
                reason="éléments manquants : " + ", ".join(missing),
            )

        nopat = ebit * (1.0 - tax_rate)
        invested = debt + eq - (cash or 0.0)
        value, reason = safe_div(nopat, invested, require_positive_denominator=True, label="ROIC")
        return Metric(
            key="roic",
            label="Rendement du capital investi (ROIC)",
            value=_pct(value),
            unit="percent",
            formula="NOPAT / (Dette totale + Capitaux propres - Trésorerie)",
            interpretation=(
                DEFINITIONS["roic"] + f" Taux d'imposition utilisé : {tax_src}."
            ),
            inputs={"ebit": ebit, "tax_rate": tax_rate, "nopat": nopat,
                    "debt": debt, "equity": eq, "cash": cash, "invested_capital": invested},
            period=period, reason=reason,
        )

    def net_debt_metric(self, period: str) -> Metric:
        nd, reason = self.net_debt(period)
        debt = self._v(LineItem.TOTAL_DEBT, period)
        cash = self._v(LineItem.CASH_AND_EQUIVALENTS, period)
        return Metric(
            key="net_debt", label="Dette nette",
            value=nd, unit="currency_millions",
            formula="Dette totale - Trésorerie et équivalents",
            interpretation=DEFINITIONS["net_debt"],
            inputs={"total_debt": debt, "cash": cash},
            period=period, reason=reason,
        )

    def net_debt_to_ebitda(self, period: str) -> Metric:
        nd, nd_reason = self.net_debt(period)
        ebitda, _ = self._ebitda(period)
        value, reason = safe_div(
            nd, ebitda, require_positive_denominator=True, label="net debt / EBITDA"
        )
        if reason is None:
            reason = nd_reason
        return Metric(
            key="net_debt_to_ebitda", label="Dette nette / EBITDA",
            value=value, unit="multiple",
            formula="Dette nette / EBITDA",
            interpretation=(
                "Nombre d'années d'EBITDA nécessaires pour rembourser la dette nette. "
                "Au-delà de 3x, le levier devient un risque majeur ; non calculable "
                "lorsque l'EBITDA est négatif."
            ),
            inputs={"net_debt": nd, "ebitda": ebitda},
            period=period, reason=reason,
        )

    def operating_cash_flow(self, period: str) -> Metric:
        ocf = self._v(LineItem.OPERATING_CASH_FLOW, period)
        return Metric(
            key="operating_cash_flow", label="Flux de trésorerie d'exploitation",
            value=ocf, unit="currency_millions",
            formula="Tel que publié dans le tableau des flux",
            interpretation="Capacité de l'exploitation a générer de la trésorerie.",
            inputs={"operating_cash_flow": ocf}, period=period,
            reason=None if ocf is not None else "non publié pour cette période",
        )

    def free_cash_flow(self, period: str) -> Metric:
        ocf = self._v(LineItem.OPERATING_CASH_FLOW, period)
        capex = self._v(LineItem.CAPITAL_EXPENDITURE, period)
        reported = self._v(LineItem.FREE_CASH_FLOW, period)
        if reported is not None:
            return Metric(
                key="free_cash_flow", label="Flux de trésorerie disponible (FCF)",
                value=reported, unit="currency_millions",
                formula="Tel que publié, ou OCF - Investissements",
                interpretation=DEFINITIONS["fcf"],
                inputs={"operating_cash_flow": ocf, "capex": capex, "reported_fcf": reported},
                period=period,
            )
        if ocf is None or capex is None:
            missing = "flux d'exploitation" if ocf is None else "investissements"
            return Metric(
                key="free_cash_flow", label="Flux de trésorerie disponible (FCF)",
                value=None, unit="currency_millions",
                formula="OCF - Investissements", interpretation=DEFINITIONS["fcf"],
                period=period, reason=f"element manquant : {missing}",
            )
        return Metric(
            key="free_cash_flow", label="Flux de trésorerie disponible (FCF)",
            value=ocf - abs(capex), unit="currency_millions",
            formula="Flux d'exploitation - Investissements",
            interpretation=DEFINITIONS["fcf"] + " Calcule par l'application (donnée derivee).",
            inputs={"operating_cash_flow": ocf, "capex": capex},
            status=ExtractionStatus.DERIVED, period=period,
        )

    def capex_intensity(self, period: str) -> Metric:
        capex = self._v(LineItem.CAPITAL_EXPENDITURE, period)
        rev = self._v(LineItem.REVENUE, period)
        value, reason = safe_div(
            None if capex is None else abs(capex), rev, label="capex intensity"
        )
        return Metric(
            key="capex_intensity", label="Intensité capitalistique",
            value=_pct(value), unit="percent",
            formula="Investissements / Chiffre d'affaires",
            interpretation=(
                "Part du chiffre d'affaires réinvestie. Structurellement élevée "
                "dans les mines et les semi-conducteurs; un pic de capex pese sur "
                "le FCF a court terme."
            ),
            inputs={"capex": capex, "revenue": rev}, period=period, reason=reason,
        )

    def pe_ratio(self, period: str) -> Metric:
        mcap = self._v(LineItem.MARKET_CAP, period)
        ni = self._v(LineItem.NET_INCOME, period)
        price = self._v(LineItem.MARKET_PRICE, period)
        shares = self._v(LineItem.DILUTED_SHARES, period) or self._v(LineItem.SHARES_OUTSTANDING, period)
        eps = None if (ni is None or not shares) else ni / shares
        if mcap is None and price is not None and shares:
            mcap = price * shares
        value, reason = safe_div(mcap, ni, require_positive_denominator=True, label="P/E")
        return Metric(
            key="pe_ratio", label="Cours / Bénéfice (P/E)",
            value=value, unit="multiple",
            formula="Capitalisation boursière / Résultat net",
            interpretation=(
                "Nombre d'années de bénéfices actuels capitalisées dans le cours. "
                "Non significatif lorsque la société est en perte ou lorsque le "
                "bénéfice est cycliquement au plus haut."
            ),
            inputs={"market_cap": mcap, "net_income": ni, "eps_diluted": eps},
            period=period, reason=reason,
        )

    def ev_to_ebitda(self, period: str) -> Metric:
        ev, ev_reason = self.enterprise_value(period)
        ebitda, src = self._ebitda(period)
        value, reason = safe_div(
            ev, ebitda, require_positive_denominator=True, label="EV/EBITDA"
        )
        if reason is None:
            reason = ev_reason
        return Metric(
            key="ev_to_ebitda", label="VE / EBITDA",
            value=value, unit="multiple",
            formula="Valeur d'entreprise / EBITDA",
            interpretation=(
                "Multiple de référence pour les comparables. " + DEFINITIONS["ev"] +
                f" EBITDA retenu : {src}."
            ),
            inputs={"enterprise_value": ev, "ebitda": ebitda},
            period=period, reason=reason,
        )

    def price_to_book(self, period: str) -> Metric:
        mcap = self._v(LineItem.MARKET_CAP, period)
        eq = self._v(LineItem.TOTAL_EQUITY, period)
        value, reason = safe_div(mcap, eq, require_positive_denominator=True, label="P/B")
        return Metric(
            key="price_to_book", label="Cours / Capitaux propres (P/B)",
            value=value, unit="multiple",
            formula="Capitalisation boursière / Capitaux propres",
            interpretation=(
                "Pertinent pour les sociétés financières et les groupes a forte "
                "intensité d'actifs. A croiser avec le ROE : un P/B élevé se "
                "justifie par un ROE durablement supérieur au coût des fonds propres."
            ),
            inputs={"market_cap": mcap, "total_equity": eq}, period=period, reason=reason,
        )

    def interest_coverage(self, period: str) -> Metric:
        ebit, _ = self._ebit(period)
        interest = self._v(LineItem.INTEREST_EXPENSE, period)
        value, reason = safe_div(
            ebit, None if interest is None else abs(interest),
            require_positive_denominator=True, label="interest coverage",
        )
        return Metric(
            key="interest_coverage", label="Couverture des frais financiers",
            value=value, unit="multiple",
            formula="EBIT / Charges d'intérêts",
            interpretation=(
                "Nombre de fois que l'exploitation couvre les intérêts. "
                "En dessous de 2x, la structure financière est fragile."
            ),
            inputs={"ebit": ebit, "interest_expense": interest}, period=period, reason=reason,
        )

    def dividend_payout(self, period: str) -> Metric:
        div = self._v(LineItem.DIVIDENDS_PAID, period)
        ni = self._v(LineItem.NET_INCOME, period)
        value, reason = safe_div(
            None if div is None else abs(div), ni,
            require_positive_denominator=True, label="payout ratio",
        )
        return Metric(
            key="dividend_payout", label="Taux de distribution",
            value=_pct(value), unit="percent",
            formula="Dividendes verses / Résultat net",
            interpretation="Part du bénéfice redistribuee aux actionnaires.",
            inputs={"dividends_paid": div, "net_income": ni}, period=period, reason=reason,
        )

    def market_cap(self, period: str) -> Metric:
        mcap = self._v(LineItem.MARKET_CAP, period)
        price = self._v(LineItem.MARKET_PRICE, period)
        shares = self._v(LineItem.DILUTED_SHARES, period) or self._v(LineItem.SHARES_OUTSTANDING, period)
        if mcap is None and price is not None and shares:
            mcap = price * shares
        return Metric(
            key="market_cap", label="Capitalisation boursière",
            value=mcap, unit="currency_millions",
            formula="Cours x Actions en circulation",
            interpretation="Valeur de marché des fonds propres (donnée de marché).",
            inputs={"price": price, "shares": shares}, period=period,
            reason=None if mcap is not None else "cours ou nombre d'actions indisponible",
        )

    def enterprise_value_metric(self, period: str) -> Metric:
        ev, reason = self.enterprise_value(period)
        return Metric(
            key="enterprise_value", label="Valeur d'entreprise",
            value=ev, unit="currency_millions",
            formula="Capitalisation + Dette nette + Intérêts minoritaires",
            interpretation=DEFINITIONS["ev"],
            inputs={}, period=period, reason=reason,
        )

    # -- aggregate ---------------------------------------------------------

    def compute_all(self, period: str, previous: Optional[str] = None) -> Dict[str, Metric]:
        """Every metric that can be computed for ``period``."""
        metrics: List[Metric] = [
            self.market_cap(period),
            self.enterprise_value_metric(period),
            self.net_debt_metric(period),
            self.operating_cash_flow(period),
            self.free_cash_flow(period),
            self.gross_margin(period),
            self.operating_margin(period),
            self.ebitda_margin(period),
            self.net_margin(period),
            self.capex_intensity(period),
            self.net_debt_to_ebitda(period),
            self.roic(period),
            self.interest_coverage(period),
            self.pe_ratio(period),
            self.ev_to_ebitda(period),
            self.price_to_book(period),
            self.dividend_payout(period),
        ]
        if previous:
            metrics.insert(0, self.revenue_growth(period, previous))
        metrics.append(self.roe(period, previous))
        return {m.key: m for m in metrics}

    def history(self, line_items: Sequence[LineItem], period_type: PeriodType = PeriodType.ANNUAL) -> Dict[str, Dict[str, Optional[float]]]:
        """{item: {period_label: value}} for chart building."""
        return {
            item.value: {
                label: self.ds.value(item, label)
                for label in self.ds.periods(period_type)
            }
            for item in line_items
        }

    def margin_series(self, period_type: PeriodType = PeriodType.ANNUAL) -> Dict[str, Dict[str, Optional[float]]]:
        """Gross / operating / EBITDA / net margin for every period available."""
        out: Dict[str, Dict[str, Optional[float]]] = {
            "gross_margin": {}, "operating_margin": {}, "ebitda_margin": {}, "net_margin": {},
        }
        for label in self.ds.periods(period_type):
            for key, fn in (
                ("gross_margin", self.gross_margin),
                ("operating_margin", self.operating_margin),
                ("ebitda_margin", self.ebitda_margin),
                ("net_margin", self.net_margin),
            ):
                metric = fn(label)
                out[key][label] = metric.value
        return out


def missing_items_for_dcf(dataset: FinancialDataSet, period: str) -> List[str]:
    """Human-readable list of the inputs a DCF still needs for ``period``."""
    needed = {
        LineItem.REVENUE: "chiffre d'affaires",
        LineItem.TOTAL_DEBT: "dette totale",
        LineItem.CASH_AND_EQUIVALENTS: "trésorerie",
        LineItem.DILUTED_SHARES: "actions diluées",
    }
    missing = [label for item, label in needed.items() if dataset.value(item, period) is None]
    return missing
