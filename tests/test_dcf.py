"""Unit tests for the DCF engine.

The reference case is small enough to be verified by hand; every expected
figure below is derived from the closed-form formulas in the project brief.

Reference case
--------------
base revenue 1 000, growth 10 %, EBIT margin 20 %, tax 25 %, D&A 5 % of
revenue, capex 6 % of revenue, NWC 10 % of revenue, WACC 10 %, g 2 %,
net debt 200, 100 m shares.

    revenue 1 = 1 100      revenue 2 = 1 210
    FCFF   1 = 165 + 55 - 66 - 10      = 144.0
    FCFF   2 = 181.5 + 60.5 - 72.6 - 11 = 158.4
    PV(FCFF) = 144/1.1 + 158.4/1.21    = 261.818181...
    TV  = 158.4 * 1.02 / 0.08          = 2 019.6
    PV(TV) = 2 019.6 / 1.21            = 1 669.090909...
    EV  = 1 930.909090...
    Equity = 1 930.909090 - 200        = 1 730.909090...
    Value per share = 17.309090...
"""

from __future__ import annotations

import math

import pytest

from financial_engine.dcf import (
    DCFInputError,
    DCFInputs,
    WACCInputs,
    broadcast,
    compute_wacc,
    reconcile,
    run_dcf,
)
from financial_engine.schemas import Currency, Scale, to_millions, from_millions


REFERENCE = dict(
    base_revenue=1000.0,
    forecast_years=2,
    revenue_growth=0.10,
    ebit_margin=0.20,
    tax_rate=0.25,
    da_pct_revenue=0.05,
    capex_pct_revenue=0.06,
    nwc_pct_revenue=0.10,
    wacc=0.10,
    terminal_growth=0.02,
    net_debt=200.0,
    shares_outstanding=100.0,
    currency="USD",
)


def make_inputs(**overrides) -> DCFInputs:
    params = dict(REFERENCE)
    params.update(overrides)
    return DCFInputs(**params)


# --------------------------------------------------------------------------
# Reference case
# --------------------------------------------------------------------------


def test_reference_case_line_by_line():
    result = run_dcf(make_inputs(), current_price=15.0, base_year=2024)

    assert [y.fiscal_year for y in result.years] == [2025, 2026]

    y1, y2 = result.years
    assert y1.revenue == pytest.approx(1100.0)
    assert y1.ebit == pytest.approx(220.0)
    assert y1.nopat == pytest.approx(165.0)
    assert y1.da == pytest.approx(55.0)
    assert y1.capex == pytest.approx(66.0)
    assert y1.nwc == pytest.approx(110.0)
    assert y1.change_nwc == pytest.approx(10.0)
    assert y1.fcff == pytest.approx(144.0)

    assert y2.revenue == pytest.approx(1210.0)
    assert y2.change_nwc == pytest.approx(11.0)
    assert y2.fcff == pytest.approx(158.4)

    assert y1.discount_factor == pytest.approx(1 / 1.1)
    assert y2.discount_factor == pytest.approx(1 / 1.21)
    assert y1.pv_fcff == pytest.approx(144.0 / 1.1)
    assert y2.pv_fcff == pytest.approx(158.4 / 1.21)


def test_reference_case_terminal_value():
    result = run_dcf(make_inputs(), base_year=2024)
    assert result.terminal_value == pytest.approx(2019.6)
    assert result.pv_terminal_value == pytest.approx(2019.6 / 1.21)


def test_reference_case_enterprise_to_equity_bridge():
    result = run_dcf(make_inputs(), current_price=15.0, base_year=2024)
    assert result.pv_explicit == pytest.approx(144.0 / 1.1 + 158.4 / 1.21)
    assert result.enterprise_value == pytest.approx(1930.90909090909)
    assert result.equity_value == pytest.approx(1730.90909090909)
    assert result.value_per_share == pytest.approx(17.30909090909)
    assert result.current_price == 15.0
    assert result.upside == pytest.approx(17.30909090909 / 15.0 - 1.0)


def test_reconcile_is_clean_for_reference_case():
    result = run_dcf(make_inputs())
    assert reconcile(result) == []


def test_manual_bridge_matches_formula():
    """EV - net debt - senior claims + non-operating assets == equity value."""
    result = run_dcf(
        make_inputs(net_debt=250.0, other_senior_claims=30.0, non_operating_assets=80.0)
    )
    manual = result.enterprise_value - 250.0 - 30.0 + 80.0
    assert result.equity_value == pytest.approx(manual)
    assert reconcile(result) == []


def test_net_debt_is_subtracted_once_only():
    """Sanity guard: changing net debt moves the equity value by exactly that amount."""
    a = run_dcf(make_inputs(net_debt=100.0))
    b = run_dcf(make_inputs(net_debt=300.0))
    assert b.enterprise_value == pytest.approx(a.enterprise_value)
    assert a.equity_value - b.equity_value == pytest.approx(200.0)


# --------------------------------------------------------------------------
# Invalid combinations
# --------------------------------------------------------------------------


def test_terminal_growth_equal_to_wacc_is_rejected():
    with pytest.raises(DCFInputError, match="strictement inférieur"):
        run_dcf(make_inputs(terminal_growth=0.10))


def test_terminal_growth_above_wacc_is_rejected():
    with pytest.raises(DCFInputError, match="strictement inférieur"):
        run_dcf(make_inputs(wacc=0.08, terminal_growth=0.09))


def test_zero_wacc_is_rejected():
    with pytest.raises(DCFInputError, match="wacc"):
        make_inputs(wacc=0.0)


def test_negative_wacc_is_rejected():
    with pytest.raises(DCFInputError, match="wacc"):
        make_inputs(wacc=-0.05, terminal_growth=0.0)


def test_zero_forecast_horizon_is_rejected():
    with pytest.raises(DCFInputError, match="forecast_years"):
        make_inputs(forecast_years=0)


def test_zero_shares_is_rejected():
    with pytest.raises(DCFInputError, match="shares_outstanding"):
        make_inputs(shares_outstanding=0.0)


def test_zero_or_negative_revenue_is_rejected():
    with pytest.raises(DCFInputError, match="base_revenue"):
        make_inputs(base_revenue=0.0)
    with pytest.raises(DCFInputError, match="base_revenue"):
        make_inputs(base_revenue=-50.0)


def test_tax_rate_out_of_range_is_rejected():
    with pytest.raises(DCFInputError, match="tax_rate"):
        make_inputs(tax_rate=1.5)
    with pytest.raises(DCFInputError, match="tax_rate"):
        make_inputs(tax_rate=-0.1)


def test_inspect_only_returns_nan_instead_of_raising():
    """The sensitivity grid explores the boundary of the valid region."""
    inputs = DCFInputs(**{**REFERENCE, "wacc": 0.02, "terminal_growth": 0.03})
    result = run_dcf(inputs, inspect_only=True)
    assert math.isnan(result.terminal_value)
    assert math.isnan(result.value_per_share)
    assert any("CRITIQUE" in w for w in result.warnings)


def test_invalid_growth_is_rejected_by_run_dcf_not_by_construction():
    """DCFInputs is a container; the legality of the combination is a run-time
    property, which is what makes the sensitivity grid possible."""
    inputs = DCFInputs(**{**REFERENCE, "wacc": 0.02, "terminal_growth": 0.03})
    with pytest.raises(DCFInputError):
        run_dcf(inputs)


# --------------------------------------------------------------------------
# Series broadcasting
# --------------------------------------------------------------------------


def test_scalar_is_broadcast_over_the_horizon():
    inputs = make_inputs(forecast_years=5, revenue_growth=0.05)
    assert inputs.series("revenue_growth") == [0.05] * 5


def test_explicit_series_is_preserved():
    inputs = make_inputs(
        forecast_years=3, revenue_growth=[0.10, 0.05, 0.02], ebit_margin=[0.20, 0.18, 0.16]
    )
    assert inputs.series("revenue_growth") == [0.10, 0.05, 0.02]
    assert inputs.series("ebit_margin") == [0.20, 0.18, 0.16]


def test_series_of_wrong_length_is_rejected():
    with pytest.raises(DCFInputError, match="horizon de prévision"):
        make_inputs(forecast_years=3, revenue_growth=[0.10, 0.05])
    with pytest.raises(DCFInputError, match="horizon de prévision"):
        broadcast([0.1, 0.2, 0.3], 5, "revenue_growth")


def test_single_element_series_is_broadcast():
    assert broadcast([0.07], 3, "x") == [0.07, 0.07, 0.07]


def test_empty_series_is_rejected():
    with pytest.raises(DCFInputError, match="est vide"):
        broadcast([], 3, "x")


# --------------------------------------------------------------------------
# Negative cash flows and warnings
# --------------------------------------------------------------------------


def test_negative_fcff_produces_warnings_and_negative_value():
    result = run_dcf(make_inputs(ebit_margin=-0.05))
    assert all(y.fcff < 0 for y in result.years)
    assert result.value_per_share < 0
    assert any("flux de trésorerie disponible négatif" in w for w in result.warnings)
    assert result.reliability == "low" or result.reliability == "medium"


def test_heavy_capex_can_make_fcff_negative():
    result = run_dcf(make_inputs(capex_pct_revenue=0.40))
    assert result.years[0].fcff < 0
    assert result.reliability in {"low", "medium"}


def test_high_terminal_share_is_flagged():
    result = run_dcf(make_inputs(wacc=0.07, terminal_growth=0.03, forecast_years=3))
    assert result.tv_share_of_ev > 0.75
    assert any("valeur terminale" in w for w in result.warnings)


def test_high_perpetual_growth_is_flagged():
    result = run_dcf(make_inputs(wacc=0.14, terminal_growth=0.06))
    assert any("dépasse 5 %" in w for w in result.warnings)


def test_negative_net_debt_is_flagged_and_adds_value():
    result = run_dcf(make_inputs(net_debt=-150.0))
    assert any("trésorerie nette" in w for w in result.warnings)
    assert result.equity_value > result.enterprise_value


def test_negative_equity_value_is_critical():
    result = run_dcf(make_inputs(net_debt=100_000.0))
    assert result.equity_value < 0
    assert any(w.startswith("CRITIQUE") for w in result.warnings)
    assert result.reliability == "low"


# --------------------------------------------------------------------------
# Discounting conventions
# --------------------------------------------------------------------------


def test_mid_year_discounting_increases_value():
    end_year = run_dcf(make_inputs())
    mid_year = run_dcf(make_inputs(mid_year_discounting=True))
    assert mid_year.enterprise_value > end_year.enterprise_value
    assert mid_year.years[0].discount_factor == pytest.approx(1 / 1.1**0.5)
    assert mid_year.years[1].discount_factor == pytest.approx(1 / 1.1**1.5)


def test_discount_factors_are_positive_and_decreasing():
    result = run_dcf(make_inputs(forecast_years=6))
    dfs = [y.discount_factor for y in result.years]
    assert all(d > 0 for d in dfs)
    assert all(b < a for a, b in zip(dfs, dfs[1:]))


def test_terminal_value_uses_final_year_fcff_not_first():
    r_final = run_dcf(make_inputs(forecast_years=3))
    assert r_final.terminal_value == pytest.approx(
        r_final.years[-1].fcff * 1.02 / (0.10 - 0.02)
    )
    assert r_final.terminal_value != pytest.approx(
        r_final.years[0].fcff * 1.02 / (0.10 - 0.02)
    )


def test_base_nwc_override_changes_the_working_capital_draw():
    low = run_dcf(make_inputs(base_nwc=0.0))
    high = run_dcf(make_inputs(base_nwc=500.0))
    assert low.years[0].change_nwc == pytest.approx(110.0)
    assert high.years[0].change_nwc == pytest.approx(-390.0)
    assert high.enterprise_value > low.enterprise_value


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


def test_implied_exit_multiple_is_reported():
    result = run_dcf(make_inputs())
    assert result.implied_exit_multiple is not None
    assert result.implied_exit_multiple == pytest.approx(2019.6 / (242.0 + 60.5))


def test_tv_share_of_enterprise_value_sums_to_one():
    result = run_dcf(make_inputs())
    assert result.pv_explicit / result.enterprise_value + result.tv_share_of_ev == pytest.approx(1.0)


def test_summary_and_table_shapes():
    result = run_dcf(make_inputs())
    table = result.to_table()
    assert len(table) == 2
    assert set(table[0]) >= {"year", "revenue", "ebit", "fcff", "pv_fcff"}
    summary = result.summary()
    assert summary["value_per_share"] == pytest.approx(result.value_per_share)


def test_with_overrides_does_not_mutate_original():
    base = make_inputs()
    shocked = base.with_overrides(wacc=0.12)
    assert base.wacc == 0.10
    assert shocked.wacc == 0.12


# --------------------------------------------------------------------------
# WACC
# --------------------------------------------------------------------------


def test_compute_wacc_capm():
    res = compute_wacc(
        WACCInputs(
            risk_free_rate=0.04,
            equity_risk_premium=0.06,
            beta=1.2,
            cost_of_debt=0.06,
            tax_rate=0.25,
            market_value_equity=1000.0,
            market_value_debt=500.0,
        )
    )
    assert res.cost_of_equity == pytest.approx(0.112)
    assert res.after_tax_cost_of_debt == pytest.approx(0.045)
    assert res.weight_equity == pytest.approx(2 / 3)
    assert res.weight_debt == pytest.approx(1 / 3)
    assert res.wacc == pytest.approx(2 / 3 * 0.112 + 1 / 3 * 0.045)
    assert res.weight_equity + res.weight_debt == pytest.approx(1.0)


def test_compute_wacc_all_equity():
    res = compute_wacc(
        WACCInputs(
            risk_free_rate=0.03,
            equity_risk_premium=0.05,
            beta=1.0,
            cost_of_debt=0.05,
            tax_rate=0.20,
            market_value_equity=100.0,
            market_value_debt=0.0,
        )
    )
    assert res.wacc == pytest.approx(0.08)


def test_compute_wacc_rejects_empty_capital_structure():
    with pytest.raises(DCFInputError):
        compute_wacc(
            WACCInputs(0.04, 0.06, 1.0, 0.05, 0.25, market_value_equity=0.0, market_value_debt=0.0)
        )


# --------------------------------------------------------------------------
# Currency and unit consistency
# --------------------------------------------------------------------------


def test_scale_conversion_round_trip():
    assert to_millions(1500.0, Scale.THOUSANDS) == pytest.approx(1.5)
    assert to_millions(1.5, Scale.BILLIONS) == pytest.approx(1500.0)
    assert to_millions(2_000_000.0, Scale.UNITS) == pytest.approx(2.0)
    assert from_millions(1.5, Scale.THOUSANDS) == pytest.approx(1500.0)


def test_per_share_output_is_in_currency_units_not_millions():
    """Equity of 1 730.9 m over 100 m shares must be ~17.3 currency units."""
    result = run_dcf(make_inputs())
    assert 1.0 < result.value_per_share < 1000.0
    assert result.value_per_share == pytest.approx(
        result.equity_value / result.shares_outstanding
    )


def test_result_carries_the_input_currency():
    result = run_dcf(make_inputs(currency=Currency.MAD.value))
    assert result.currency == "MAD"
    assert result.units == "millions"


def test_scenario_label_is_propagated():
    result = run_dcf(make_inputs(scenario="bear", label="Pessimiste"))
    assert result.scenario == "bear"
    assert result.label == "Pessimiste"


def test_linearity_in_shares():
    """Doubling the share count halves the intrinsic value per share."""
    a = run_dcf(make_inputs(shares_outstanding=100.0))
    b = run_dcf(make_inputs(shares_outstanding=200.0))
    assert b.value_per_share == pytest.approx(a.value_per_share / 2)
    assert b.equity_value == pytest.approx(a.equity_value)


def test_growth_of_zero_produces_flat_revenue():
    result = run_dcf(make_inputs(revenue_growth=0.0, forecast_years=4))
    revenues = [y.revenue for y in result.years]
    assert revenues == pytest.approx([1000.0] * 4)
    assert all(abs(y.change_nwc) < 1e-9 for y in result.years)
