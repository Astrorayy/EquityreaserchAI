"""Discounted cash flow engine (FCFF / WACC), fully transparent.

Conventions enforced here
-------------------------
* Every monetary input is in **millions** of ``currency``; shares are in
  **millions**; per-share outputs are in currency units.
* ``revenue_growth``, ``ebit_margin``, ``da_pct_revenue``, ``capex_pct_revenue``
  and ``nwc_pct_revenue`` are decimals (``0.08`` == 8 %).
* ``capex_pct_revenue`` and ``da_pct_revenue`` are *positive magnitudes*.
* FCFF is discounted at the WACC; the WACC must exceed the perpetual growth
  rate, otherwise the terminal value is meaningless and the run is rejected.

The model is deliberately a pure function of its inputs: no I/O, no globals,
which is what makes it unit-testable to the cent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Union

Number = Union[int, float]
Series = Union[Number, Sequence[Number]]


class DCFInputError(ValueError):
    """Raised when the assumptions are internally inconsistent."""


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DCFInputs:
    """Complete assumption set for one valuation run."""

    #: last actual full-year revenue, in millions
    base_revenue: float
    #: number of explicit forecast years (>= 1)
    forecast_years: int
    #: revenue growth per forecast year, decimal
    revenue_growth: Series
    #: EBIT margin per forecast year, decimal
    ebit_margin: Series
    #: effective tax rate applied to EBIT, decimal
    tax_rate: float
    #: depreciation & amortisation as a % of revenue, positive magnitude
    da_pct_revenue: Series
    #: capital expenditure as a % of revenue, positive magnitude
    capex_pct_revenue: Series
    #: net working capital as a % of revenue (level, not change)
    nwc_pct_revenue: Series
    #: weighted average cost of capital, decimal
    wacc: float
    #: perpetual growth rate, decimal
    terminal_growth: float

    # --- bridge from enterprise value to equity value -------------------
    #: total debt minus cash, in millions (positive == net debt)
    net_debt: float = 0.0
    #: e.g. pension deficits, preferred stock, minority interests
    other_senior_claims: float = 0.0
    #: e.g. surplus real estate, equity stakes, tax assets
    non_operating_assets: float = 0.0

    #: diluted shares, in millions
    shares_outstanding: float = 0.0

    # --- optional refinements -------------------------------------------
    #: net working capital at t0; defaults to nwc_pct_revenue * base_revenue
    base_nwc: Optional[float] = None
    #: discount cash flows at mid-year instead of year-end
    mid_year_discounting: bool = False

    # --- labelling -------------------------------------------------------
    currency: str = "USD"
    units: str = "millions"
    scenario: str = "base"
    label: str = ""

    # --- original inputs (kept separate from normalised ones) ------------
    #: names of the fields entered by hand by the analyst
    manually_entered: tuple[str, ...] = ()
    #: True when the figures are synthetic test values, not company data
    simulated: bool = False

    def __post_init__(self) -> None:
        if self.forecast_years < 1:
            raise DCFInputError("forecast_years doit valoir au moins 1")
        if self.base_revenue <= 0:
            raise DCFInputError(
                "base_revenue doit être strictement positif : un DCF sur un chiffre "
                "d'affaires nul ou négatif n'a pas de sens. Utilisez un modèle de scénarios."
            )
        if not (0.0 <= self.tax_rate <= 0.95):
            raise DCFInputError("tax_rate doit être compris entre 0 et 0,95")
        if self.wacc <= 0:
            raise DCFInputError("wacc doit être strictement positif")
        if self.shares_outstanding <= 0:
            raise DCFInputError(
                "shares_outstanding doit être strictement positif ; sans nombre d'actions "
                "dilué, aucune valeur intrinsèque par action ne peut être calculée."
            )
        if self.terminal_growth <= -1.0:
            raise DCFInputError("terminal_growth doit être supérieur à -100 %")

        # Series lengths must match the horizon.  The WACC > terminal growth
        # relationship is deliberately NOT checked here: it is enforced by
        # run_dcf(), which can be told to inspect the boundary of the valid
        # region instead of failing (the sensitivity grid relies on that).
        for field_name in (
            "revenue_growth", "ebit_margin", "da_pct_revenue",
            "capex_pct_revenue", "nwc_pct_revenue",
        ):
            broadcast(getattr(self, field_name), self.forecast_years, field_name)

    # -- helpers ----------------------------------------------------------

    def series(self, field_name: str) -> List[float]:
        """Broadcast a scalar to the forecast horizon, or validate a list."""
        raw = getattr(self, field_name)
        return broadcast(raw, self.forecast_years, field_name)

    def with_overrides(self, **kwargs) -> "DCFInputs":
        """Return a copy with fields replaced (used by the sensitivity engine)."""
        return replace(self, **kwargs)


def broadcast(value: Series, length: int, name: str = "series") -> List[float]:
    """Turn ``0.05`` into ``[0.05] * length`` and validate explicit lists."""
    if isinstance(value, (int, float)):
        return [float(value)] * length
    seq = [float(v) for v in value]
    if len(seq) == 0:
        raise DCFInputError(f"{name} est vide")
    if len(seq) == 1:
        return seq * length
    if len(seq) != length:
        raise DCFInputError(
            f"{name} contient {len(seq)} valeurs alors que l'horizon de prévision est de {length} ans"
        )
    return seq


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------


@dataclass
class DCFYear:
    index: int
    fiscal_year: int
    revenue: float
    ebit: float
    nopat: float
    da: float
    capex: float
    nwc: float
    change_nwc: float
    fcff: float
    discount_factor: float
    pv_fcff: float
    ebitda: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "year": self.fiscal_year,
            "revenue": self.revenue,
            "ebit": self.ebit,
            "nopat": self.nopat,
            "da": self.da,
            "capex": self.capex,
            "nwc": self.nwc,
            "change_nwc": self.change_nwc,
            "fcff": self.fcff,
            "discount_factor": self.discount_factor,
            "pv_fcff": self.pv_fcff,
            "ebitda": self.ebitda,
        }


@dataclass
class DCFResult:
    currency: str
    units: str
    scenario: str
    label: str
    years: List[DCFYear]
    wacc: float
    terminal_growth: float
    tax_rate: float

    pv_explicit: float
    terminal_value: float
    pv_terminal_value: float
    enterprise_value: float

    net_debt: float
    other_senior_claims: float
    non_operating_assets: float
    equity_value: float
    shares_outstanding: float
    value_per_share: float

    current_price: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    inputs: Optional[DCFInputs] = None

    # -- diagnostics ------------------------------------------------------

    @property
    def tv_share_of_ev(self) -> float:
        """How much of the enterprise value comes from the terminal value."""
        if self.enterprise_value == 0:
            return float("nan")
        return self.pv_terminal_value / self.enterprise_value

    @property
    def final_year_ebitda(self) -> float:
        return self.years[-1].ebitda if self.years else 0.0

    @property
    def implied_exit_multiple(self) -> Optional[float]:
        """EV/EBITDA implied by the terminal value - a sanity check."""
        ebitda = self.final_year_ebitda
        if not ebitda:
            return None
        return self.terminal_value / ebitda

    @property
    def upside(self) -> Optional[float]:
        if self.current_price is None or self.current_price <= 0:
            return None
        return self.value_per_share / self.current_price - 1.0

    @property
    def reliability(self) -> str:
        """A blunt, honest label: 'high' | 'medium' | 'low'."""
        if not self.inputs:
            return "unknown"
        hard = [w for w in self.warnings if w.startswith("CRITIQUE")]
        if hard:
            return "low"
        if self.tv_share_of_ev > 0.85 or self.warnings:
            return "medium"
        return "high"

    def to_table(self) -> List[Dict[str, float]]:
        return [y.as_dict() for y in self.years]

    def summary(self) -> Dict[str, float]:
        return {
            "enterprise_value": self.enterprise_value,
            "equity_value": self.equity_value,
            "value_per_share": self.value_per_share,
            "pv_explicit": self.pv_explicit,
            "pv_terminal_value": self.pv_terminal_value,
            "tv_share_of_ev": self.tv_share_of_ev,
            "net_debt": self.net_debt,
            "shares_outstanding": self.shares_outstanding,
            "wacc": self.wacc,
            "terminal_growth": self.terminal_growth,
            "implied_exit_multiple": self.implied_exit_multiple or float("nan"),
        }


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------


def run_dcf(
    inputs: DCFInputs,
    current_price: Optional[float] = None,
    base_year: Optional[int] = None,
    inspect_only: bool = False,
) -> DCFResult:
    """Run a FCFF DCF valuation.

    Parameters
    ----------
    inputs:
        The assumption set. Invalid combinations raise :class:`DCFInputError`.
    current_price:
        Optional market price, used only to compute the implied upside.
    base_year:
        Fiscal year of ``base_revenue``. Defaults to the current calendar year - 1.
    inspect_only:
        When True the terminal growth / WACC sanity checks are downgraded to
        warnings instead of hard failures (used by the sensitivity grid, which
        legitimately explores the *boundary* of the valid region).
    """
    import datetime as _dt

    n = inputs.forecast_years
    growth = inputs.series("revenue_growth")
    margin = inputs.series("ebit_margin")
    da_pct = inputs.series("da_pct_revenue")
    capex_pct = inputs.series("capex_pct_revenue")
    nwc_pct = inputs.series("nwc_pct_revenue")

    if base_year is None:
        base_year = _dt.date.today().year - 1

    warnings: List[str] = []

    if not inspect_only and inputs.terminal_growth >= inputs.wacc:
        raise DCFInputError("terminal_growth doit être strictement inférieur au WACC")
    if inputs.terminal_growth >= inputs.wacc:
        warnings.append(
            "CRITIQUE : le taux de croissance perpétuelle n'est pas inférieur au WACC — "
            "la valeur terminale de cette cellule n'est pas exploitable."
        )

    prior_revenue = inputs.base_revenue
    prior_nwc = (
        inputs.base_nwc if inputs.base_nwc is not None else nwc_pct[0] * inputs.base_revenue
    )

    years: List[DCFYear] = []
    pv_explicit = 0.0

    for i in range(n):
        revenue = prior_revenue * (1.0 + growth[i])
        ebit = revenue * margin[i]
        nopat = ebit * (1.0 - inputs.tax_rate)
        da = revenue * da_pct[i]
        capex = revenue * capex_pct[i]
        nwc = revenue * nwc_pct[i]
        change_nwc = nwc - prior_nwc

        fcff = nopat + da - capex - change_nwc

        # discount factor: mid-year convention shifts the exponent by -0.5
        exponent = (i + 1) - (0.5 if inputs.mid_year_discounting else 0.0)
        df = 1.0 / ((1.0 + inputs.wacc) ** exponent)
        pv = fcff * df

        years.append(
            DCFYear(
                index=i + 1,
                fiscal_year=base_year + 1 + i,
                revenue=revenue,
                ebit=ebit,
                nopat=nopat,
                da=da,
                capex=capex,
                nwc=nwc,
                change_nwc=change_nwc,
                fcff=fcff,
                discount_factor=df,
                pv_fcff=pv,
                ebitda=ebit + da,
            )
        )
        pv_explicit += pv
        prior_revenue = revenue
        prior_nwc = nwc

    # --- terminal value: Gordon growth on the final explicit FCFF --------
    final_fcff = years[-1].fcff
    spread = inputs.wacc - inputs.terminal_growth
    if spread <= 0:
        terminal_value = float("nan")
        pv_terminal = float("nan")
    else:
        terminal_value = final_fcff * (1.0 + inputs.terminal_growth) / spread
        df_final = years[-1].discount_factor
        pv_terminal = terminal_value * df_final

    enterprise_value = pv_explicit + pv_terminal

    equity_value = (
        enterprise_value
        - inputs.net_debt
        - inputs.other_senior_claims
        + inputs.non_operating_assets
    )
    value_per_share = equity_value / inputs.shares_outstanding

    # --- diagnostics ------------------------------------------------------
    if final_fcff <= 0:
        warnings.append(
            "Le flux de trésorerie disponible de la dernière année est négatif : la "
            "valeur terminale de Gordon serait négative et la valorisation fragile."
        )
    if any(y.fcff < 0 for y in years):
        negatives = sum(1 for y in years if y.fcff < 0)
        warnings.append(f"{negatives} exercice(s) sur {n} présentent un flux de trésorerie disponible négatif.")
    if pv_explicit + pv_terminal != 0 and not math.isnan(pv_terminal):
        share = pv_terminal / enterprise_value if enterprise_value else float("nan")
        if share > 0.85:
            warnings.append(
                f"{share:.0%} de la valeur d'entreprise provient de la valeur terminale ; "
                "la valorisation est très sensible à l'hypothèse de croissance perpétuelle."
            )
    if inputs.terminal_growth > 0.05:
        warnings.append(
            f"Le taux de croissance perpétuelle ({inputs.terminal_growth:.1%}) dépasse 5 %, "
            "au-dessus de la croissance nominale mondiale de long terme de la plupart des secteurs."
        )
    if inputs.net_debt < 0:
        warnings.append(
            "La dette nette est négative (trésorerie nette) : elle est ajoutée à la "
            "valeur d'entreprise dans le passage aux fonds propres."
        )
    if equity_value < 0:
        warnings.append(
            "CRITIQUE : la valeur des fonds propres est négative — la dette nette dépasse "
            "la valeur d'entreprise. La valeur par action n'a pas de sens."
        )

    return DCFResult(
        currency=inputs.currency,
        units=inputs.units,
        scenario=inputs.scenario,
        label=inputs.label,
        years=years,
        wacc=inputs.wacc,
        terminal_growth=inputs.terminal_growth,
        tax_rate=inputs.tax_rate,
        pv_explicit=pv_explicit,
        terminal_value=terminal_value,
        pv_terminal_value=pv_terminal,
        enterprise_value=enterprise_value,
        net_debt=inputs.net_debt,
        other_senior_claims=inputs.other_senior_claims,
        non_operating_assets=inputs.non_operating_assets,
        equity_value=equity_value,
        shares_outstanding=inputs.shares_outstanding,
        value_per_share=value_per_share,
        current_price=current_price,
        warnings=warnings,
        inputs=inputs,
    )


# --------------------------------------------------------------------------
# Cross-validation of the FMV bridge
# --------------------------------------------------------------------------


def reconcile(result: DCFResult, tol: float = 1e-6) -> List[str]:
    """Re-check the enterprise-to-equity bridge of a finished run.

    Returns a list of problems; an empty list means the bridge is consistent.
    Used by the unit tests and by the "show your work" panel.
    """
    problems: List[str] = []
    if abs(result.pv_explicit + result.pv_terminal_value - result.enterprise_value) > tol * max(
        1.0, abs(result.enterprise_value)
    ):
        problems.append("enterprise value != PV(explicit FCFF) + PV(terminal value)")

    expected_equity = (
        result.enterprise_value
        - result.net_debt
        - result.other_senior_claims
        + result.non_operating_assets
    )
    if abs(expected_equity - result.equity_value) > tol * max(1.0, abs(expected_equity)):
        problems.append("la valeur des fonds propres ne se réconcilie pas avec le passage valeur d'entreprise -> fonds propres")

    if result.shares_outstanding:
        expected_ps = result.equity_value / result.shares_outstanding
        if abs(expected_ps - result.value_per_share) > tol * max(1.0, abs(expected_ps)):
            problems.append("valeur intrinsèque par action != fonds propres / actions diluées")
    else:
        problems.append("shares_outstanding n'est pas exploitable")

    return problems


# --------------------------------------------------------------------------
# WACC
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WACCInputs:
    """Inputs for the weighted average cost of capital."""

    risk_free_rate: float
    equity_risk_premium: float
    beta: float
    cost_of_debt: float
    tax_rate: float
    market_value_equity: float
    market_value_debt: float

    def __post_init__(self) -> None:
        if self.market_value_equity < 0 or self.market_value_debt < 0:
            raise DCFInputError("les valeurs de marché ne peuvent pas être négatives")
        if self.market_value_equity + self.market_value_debt <= 0:
            raise DCFInputError("capitaux propres + dette doivent être strictement positifs pour calculer un WACC")


@dataclass
class WACCResult:
    cost_of_equity: float
    after_tax_cost_of_debt: float
    weight_equity: float
    weight_debt: float
    wacc: float
    components: Dict[str, float] = field(default_factory=dict)


def compute_wacc(inputs: WACCInputs) -> WACCResult:
    """CAPM cost of equity, after-tax cost of debt, weights at market value.

    ``WACC = E/(D+E) * Ke + D/(D+E) * Kd * (1 - tax)``
    with ``Ke = Rf + beta * ERP``.
    """
    cost_of_equity = inputs.risk_free_rate + inputs.beta * inputs.equity_risk_premium
    after_tax_kd = inputs.cost_of_debt * (1.0 - inputs.tax_rate)
    total = inputs.market_value_equity + inputs.market_value_debt
    we = inputs.market_value_equity / total
    wd = inputs.market_value_debt / total
    wacc = we * cost_of_equity + wd * after_tax_kd
    return WACCResult(
        cost_of_equity=cost_of_equity,
        after_tax_cost_of_debt=after_tax_kd,
        weight_equity=we,
        weight_debt=wd,
        wacc=wacc,
        components={
            "risk_free_rate": inputs.risk_free_rate,
            "equity_risk_premium": inputs.equity_risk_premium,
            "beta": inputs.beta,
            "cost_of_debt": inputs.cost_of_debt,
            "tax_rate": inputs.tax_rate,
        },
    )
