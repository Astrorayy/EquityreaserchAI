"""Sensitivity and scenario analysis on top of the DCF engine.

Three tools are provided:

* :func:`wacc_growth_grid` - the classic WACC x perpetual-growth table.
* :func:`one_way_sensitivity` - vary a single field over a range.
* :func:`tornado_analysis` - rank the assumptions by their impact.

All of them re-run the deterministic DCF engine; no approximation formula is
used, so the numbers in the tables reconcile exactly with a full re-run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

from financial_engine.dcf import DCFInputError, DCFInputs, DCFResult, broadcast, run_dcf


@dataclass
class SensitivityGrid:
    row_label: str
    col_label: str
    row_values: List[float]
    col_values: List[float]
    #: matrix[row][col] of intrinsic value per share; NaN where not meaningful
    values: List[List[float]]
    #: same shape, enterprise value
    enterprise_values: List[List[float]] = field(default_factory=list)
    current_price: Optional[float] = None

    def cell(self, row_value: float, col_value: float) -> Optional[float]:
        r = _nearest_index(self.row_values, row_value)
        c = _nearest_index(self.col_values, col_value)
        if r is None or c is None:
            return None
        v = self.values[r][c]
        return None if math.isnan(v) else v

    def upside_matrix(self) -> List[List[float]]:
        if not self.current_price:
            return []
        return [
            [float("nan") if math.isnan(v) else v / self.current_price - 1.0 for v in row]
            for row in self.values
        ]

    def to_records(self) -> List[Dict[str, float]]:
        out: List[Dict[str, float]] = []
        for i, rv in enumerate(self.row_values):
            for j, cv in enumerate(self.col_values):
                out.append(
                    {
                        self.row_label: rv,
                        self.col_label: cv,
                        "value_per_share": self.values[i][j],
                        "enterprise_value": (
                            self.enterprise_values[i][j]
                            if self.enterprise_values
                            else float("nan")
                        ),
                    }
                )
        return out


def _nearest_index(values: Sequence[float], target: float) -> Optional[int]:
    if not values:
        return None
    best = min(range(len(values)), key=lambda i: abs(values[i] - target))
    if abs(values[best] - target) > 1e-9 + 1e-6 * max(1.0, abs(target)):
        return None
    return best


def _frange(start: float, stop: float, step: float) -> List[float]:
    """Inclusive float range, robust to binary rounding."""
    if step == 0:
        raise ValueError("step doit être non nul")
    n = int(round((stop - start) / step))
    return [round(start + i * step, 10) for i in range(n + 1)]


# --------------------------------------------------------------------------
# WACC x terminal growth
# --------------------------------------------------------------------------


def wacc_growth_grid(
    inputs: DCFInputs,
    current_price: Optional[float] = None,
    base_year: Optional[int] = None,
    wacc_values: Optional[Sequence[float]] = None,
    growth_values: Optional[Sequence[float]] = None,
    wacc_span: float = 0.02,
    growth_span: float = 0.01,
    steps: int = 5,
) -> SensitivityGrid:
    """Intrinsic value per share for each (WACC, perpetual growth) pair.

    The diagonal where ``growth >= wacc`` is left as NaN rather than being
    filled with a meaningless number.
    """
    if wacc_values is None:
        wacc_values = _frange(inputs.wacc - wacc_span, inputs.wacc + wacc_span, 2 * wacc_span / (steps - 1)) if steps > 1 else [inputs.wacc]
    if growth_values is None:
        growth_values = _frange(inputs.terminal_growth - growth_span, inputs.terminal_growth + growth_span, 2 * growth_span / (steps - 1)) if steps > 1 else [inputs.terminal_growth]

    values: List[List[float]] = []
    evs: List[List[float]] = []

    for w in wacc_values:
        row_v: List[float] = []
        row_ev: List[float] = []
        for g in growth_values:
            if g >= w or w <= 0:
                row_v.append(float("nan"))
                row_ev.append(float("nan"))
                continue
            try:
                res = run_dcf(
                    inputs.with_overrides(wacc=w, terminal_growth=g),
                    current_price=current_price,
                    base_year=base_year,
                    inspect_only=True,
                )
                row_v.append(res.value_per_share)
                row_ev.append(res.enterprise_value)
            except DCFInputError:
                row_v.append(float("nan"))
                row_ev.append(float("nan"))
        values.append(row_v)
        evs.append(row_ev)

    return SensitivityGrid(
        row_label="wacc",
        col_label="terminal_growth",
        row_values=list(wacc_values),
        col_values=list(growth_values),
        values=values,
        enterprise_values=evs,
        current_price=current_price,
    )


# --------------------------------------------------------------------------
# One-way sensitivity
# --------------------------------------------------------------------------

#: fields that can be shocked, and whether they hold a per-year series
SCALAR_FIELDS = {
    "wacc",
    "terminal_growth",
    "tax_rate",
    "base_revenue",
    "net_debt",
    "shares_outstanding",
    "other_senior_claims",
    "non_operating_assets",
}
SERIES_FIELDS = {
    "revenue_growth",
    "ebit_margin",
    "da_pct_revenue",
    "capex_pct_revenue",
    "nwc_pct_revenue",
}


def apply_field(inputs: DCFInputs, field_name: str, value) -> DCFInputs:
    """Replace a scalar field, or set every year of a series field."""
    if field_name in SCALAR_FIELDS:
        return replace(inputs, **{field_name: float(value)})
    if field_name in SERIES_FIELDS:
        n = inputs.forecast_years
        if isinstance(value, (int, float)):
            series = [float(value)] * n
        else:
            series = broadcast(value, n, field_name)
        return replace(inputs, **{field_name: series})
    raise KeyError(f"unknown DCF field '{field_name}'")


def one_way_sensitivity(
    inputs: DCFInputs,
    field_name: str,
    values: Sequence[float],
    current_price: Optional[float] = None,
    base_year: Optional[int] = None,
) -> List[Dict[str, float]]:
    """Value per share as a single assumption is varied, everything else fixed."""
    records: List[Dict[str, float]] = []
    for v in values:
        try:
            res = run_dcf(
                apply_field(inputs, field_name, v),
                current_price=current_price,
                base_year=base_year,
                inspect_only=True,
            )
            records.append(
                {
                    field_name: float(v),
                    "value_per_share": res.value_per_share,
                    "enterprise_value": res.enterprise_value,
                    "equity_value": res.equity_value,
                    "valid": 1.0,
                }
            )
        except DCFInputError:
            records.append(
                {
                    field_name: float(v),
                    "value_per_share": float("nan"),
                    "enterprise_value": float("nan"),
                    "equity_value": float("nan"),
                    "valid": 0.0,
                }
            )
            continue
        # run_dcf(inspect_only=True) returns NaN rather than raising when the
        # combination is out of bounds; mark those cells as invalid too so the
        # chart can show a gap instead of a misleading line.
        last = records[-1]
        if not math.isfinite(last["value_per_share"]):
            last["valid"] = 0.0
    return records


# --------------------------------------------------------------------------
# Tornado
# --------------------------------------------------------------------------


@dataclass
class TornadoBar:
    field_name: str
    label: str
    low_input: float
    high_input: float
    low_value: float
    high_value: float
    base_value: float

    @property
    def swing(self) -> float:
        return abs(self.high_value - self.low_value)

    @property
    def relative_swing(self) -> float:
        if self.base_value == 0:
            return float("nan")
        return self.swing / abs(self.base_value)


@dataclass
class TornadoResult:
    base: DCFResult
    bars: List[TornadoBar]

    def sorted_bars(self) -> List[TornadoBar]:
        return sorted(self.bars, key=lambda b: b.swing, reverse=True)

    def to_records(self) -> List[Dict[str, float]]:
        return [
            {
                "field": b.field_name,
                "label": b.label,
                "low_input": b.low_input,
                "high_input": b.high_input,
                "low_value": b.low_value,
                "high_value": b.high_value,
                "base_value": b.base_value,
                "swing": b.swing,
            }
            for b in self.sorted_bars()
        ]


#: default levers and their plausible shock widths, used by the UI and the AI
DEFAULT_TORNADO_SPECS: List[Tuple[str, str, float, float]] = [
    # field, label (FR), low delta, high delta  (deltas are absolute)
    ("wacc", "CMPC (WACC)", -0.01, 0.01),
    ("terminal_growth", "Croissance perpetuelle", -0.005, 0.005),
    ("ebit_margin", "Marge opérationnelle", -0.02, 0.02),
    ("revenue_growth", "Croissance du chiffre d'affaires", -0.02, 0.02),
    ("capex_pct_revenue", "Intensité des investissements", -0.01, 0.02),
    ("tax_rate", "Taux d'imposition", -0.03, 0.03),
]


def _shift_series(inputs: DCFInputs, field_name: str, delta: float) -> DCFInputs:
    current = inputs.series(field_name)
    shifted = [v + delta for v in current]
    return replace(inputs, **{field_name: shifted})


def tornado_analysis(
    inputs: DCFInputs,
    specs: Optional[Sequence[Tuple[str, str, float, float]]] = None,
    current_price: Optional[float] = None,
    base_year: Optional[int] = None,
) -> TornadoResult:
    """Rank assumptions by the size of the swing they create in value per share."""
    base = run_dcf(inputs, current_price=current_price, base_year=base_year)
    specs = specs or DEFAULT_TORNADO_SPECS
    bars: List[TornadoBar] = []

    for field_name, label, low_delta, high_delta in specs:
        try:
            if field_name in SCALAR_FIELDS:
                cur = getattr(inputs, field_name)
                low_input = cur + low_delta
                high_input = cur + high_delta
                low_inputs = replace(inputs, **{field_name: low_input})
                high_inputs = replace(inputs, **{field_name: high_input})
            elif field_name in SERIES_FIELDS:
                cur_series = inputs.series(field_name)
                cur = cur_series[0]
                low_input = cur + low_delta
                high_input = cur + high_delta
                low_inputs = _shift_series(inputs, field_name, low_delta)
                high_inputs = _shift_series(inputs, field_name, high_delta)
            else:
                continue

            low_res = run_dcf(low_inputs, inspect_only=True, base_year=base_year)
            high_res = run_dcf(high_inputs, inspect_only=True, base_year=base_year)
        except DCFInputError:
            continue

        bars.append(
            TornadoBar(
                field_name=field_name,
                label=label,
                low_input=low_input,
                high_input=high_input,
                low_value=low_res.value_per_share,
                high_value=high_res.value_per_share,
                base_value=base.value_per_share,
            )
        )

    return TornadoResult(base=base, bars=bars)


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------


@dataclass
class ScenarioDefinition:
    name: str
    label: str
    overrides: Dict[str, object] = field(default_factory=dict)
    notes: str = ""


def build_scenarios(
    base_inputs: DCFInputs,
    bear: Optional[ScenarioDefinition] = None,
    bull: Optional[ScenarioDefinition] = None,
) -> List[DCFResult]:
    """Run the base case plus explicit bear and bull overrides.

    Overrides are applied on top of the base assumption set; anything not
    overridden stays identical, which keeps the comparison honest.
    """
    results = [run_dcf(base_inputs)]

    for scen in (bear, bull):
        if scen is None:
            continue
        overridden = base_inputs
        for field_name, value in scen.overrides.items():
            overridden = apply_field(overridden, field_name, value)
        overridden = replace(overridden, scenario=scen.name, label=scen.label)
        results.append(run_dcf(overridden))

    return results


def default_bear_bull(
    inputs: DCFInputs,
    growth_delta: float = -0.03,
    margin_delta: float = -0.03,
    wacc_delta: float = 0.015,
) -> Tuple[ScenarioDefinition, ScenarioDefinition]:
    """Symmetric bear / bull around the base case.

    The deltas are *relative shifts applied by the analyst*, not forecasts.
    """
    bear = ScenarioDefinition(
        name="bear",
        label="Pessimiste",
        overrides={
            "revenue_growth": [g + growth_delta for g in inputs.series("revenue_growth")],
            "ebit_margin": [m + margin_delta for m in inputs.series("ebit_margin")],
            "wacc": inputs.wacc + wacc_delta,
            "terminal_growth": min(inputs.terminal_growth, inputs.wacc + wacc_delta - 0.005),
        },
        notes="Croissance et marge réduites, taux d'actualisation relevé.",
    )
    bull = ScenarioDefinition(
        name="bull",
        label="Optimiste",
        overrides={
            "revenue_growth": [g - growth_delta for g in inputs.series("revenue_growth")],
            "ebit_margin": [m - margin_delta for m in inputs.series("ebit_margin")],
            "wacc": max(0.005, inputs.wacc - wacc_delta),
            "terminal_growth": min(
                inputs.terminal_growth + 0.005, max(0.005, inputs.wacc - wacc_delta) - 0.005
            ),
        },
        notes="Croissance et marge ameliorees, taux d'actualisation reduit.",
    )
    return bear, bull


def scenario_table(results: Sequence[DCFResult]) -> List[Dict[str, object]]:
    """Compact comparison table for the UI and the report."""
    rows: List[Dict[str, object]] = []
    for r in results:
        rows.append(
            {
                "scenario": r.scenario,
                "label": r.label or r.scenario,
                "currency": r.currency,
                "enterprise_value": r.enterprise_value,
                "equity_value": r.equity_value,
                "value_per_share": r.value_per_share,
                "wacc": r.wacc,
                "terminal_growth": r.terminal_growth,
                "tv_share_of_ev": r.tv_share_of_ev,
                "upside": r.upside,
                "reliability": r.reliability,
                "warnings": len(r.warnings),
            }
        )
    return rows
