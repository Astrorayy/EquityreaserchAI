"""Unit tests for the sensitivity, scenario and multiples engines."""

from __future__ import annotations

import math

import pytest

from data_sources.sample import SyntheticParams, synthetic_dataset
from financial_engine.dcf import DCFInputs
from financial_engine.multiples import (
    MultiplesEngine,
    ResidualIncomeInputs,
    TargetValuationInputs,
    justified_pb,
    pb_framework,
    residual_income_value,
)
from financial_engine.schemas import ComparableCompany, Currency
from financial_engine.sensitivity import (
    DEFAULT_TORNADO_SPECS,
    apply_field,
    build_scenarios,
    default_bear_bull,
    one_way_sensitivity,
    scenario_table,
    tornado_analysis,
    wacc_growth_grid,
)
from financial_engine.dcf import run_dcf

BASE = dict(
    base_revenue=1000.0,
    forecast_years=5,
    revenue_growth=0.05,
    ebit_margin=0.18,
    tax_rate=0.25,
    da_pct_revenue=0.07,
    capex_pct_revenue=0.09,
    nwc_pct_revenue=0.12,
    wacc=0.10,
    terminal_growth=0.02,
    net_debt=300.0,
    shares_outstanding=80.0,
    currency="MAD",
)


@pytest.fixture()
def inputs() -> DCFInputs:
    return DCFInputs(**BASE)


# --------------------------------------------------------------------------
# WACC x growth grid
# --------------------------------------------------------------------------


def test_grid_shape(inputs: DCFInputs):
    grid = wacc_growth_grid(inputs, steps=5, wacc_span=0.02, growth_span=0.01)
    assert len(grid.row_values) == 5
    assert len(grid.col_values) == 5
    assert len(grid.values) == 5
    assert all(len(row) == 5 for row in grid.values)
    assert grid.row_label == "wacc"
    assert grid.col_label == "terminal_growth"


def test_grid_centre_matches_the_base_case(inputs: DCFInputs):
    grid = wacc_growth_grid(inputs, steps=5)
    base = run_dcf(inputs)
    centre = grid.cell(inputs.wacc, inputs.terminal_growth)
    assert centre is not None
    assert centre == pytest.approx(base.value_per_share, rel=1e-9)


def test_grid_marks_the_invalid_region_as_nan(inputs: DCFInputs):
    grid = wacc_growth_grid(
        inputs,
        wacc_values=[0.06, 0.10],
        growth_values=[0.02, 0.05, 0.08],
    )
    # growth 0.08 >= wacc 0.06 -> invalid cell
    assert math.isnan(grid.values[0][2])
    # growth 0.02 and 0.05 are both below wacc 0.06 -> valid cells
    assert not math.isnan(grid.values[0][0])
    assert not math.isnan(grid.values[0][1])
    # wacc 0.10 > all growth rates -> all valid
    assert all(not math.isnan(v) for v in grid.values[1])


def test_grid_value_decreases_as_wacc_increases(inputs: DCFInputs):
    grid = wacc_growth_grid(inputs, wacc_values=[0.08, 0.10, 0.12],
                            growth_values=[inputs.terminal_growth])
    values = [row[0] for row in grid.values]
    assert values[0] > values[1] > values[2]


def test_grid_value_increases_with_terminal_growth(inputs: DCFInputs):
    grid = wacc_growth_grid(inputs, wacc_values=[inputs.wacc],
                            growth_values=[0.005, 0.015, 0.025])
    values = grid.values[0]
    assert values[0] < values[1] < values[2]


def test_grid_upside_matrix_requires_a_price(inputs: DCFInputs):
    grid = wacc_growth_grid(inputs, steps=3)
    assert grid.upside_matrix() == []
    priced = wacc_growth_grid(inputs, steps=3, current_price=10.0)
    upside = priced.upside_matrix()
    assert len(upside) == 3


def test_grid_records_are_flat_and_complete(inputs: DCFInputs):
    grid = wacc_growth_grid(inputs, steps=3)
    records = grid.to_records()
    assert len(records) == 9
    assert set(records[0]) == {"wacc", "terminal_growth", "value_per_share", "enterprise_value"}


def test_grid_cell_returns_none_for_an_absent_value(inputs: DCFInputs):
    grid = wacc_growth_grid(inputs, steps=3)
    assert grid.cell(0.999, 0.02) is None


# --------------------------------------------------------------------------
# One-way sensitivity
# --------------------------------------------------------------------------


def test_one_way_sensitivity_is_monotonic_in_wacc(inputs: DCFInputs):
    records = one_way_sensitivity(inputs, "wacc", [0.08, 0.10, 0.12, 0.14])
    values = [r["value_per_share"] for r in records]
    assert values == sorted(values, reverse=True)


def test_one_way_sensitivity_handles_invalid_combinations(inputs: DCFInputs):
    records = one_way_sensitivity(inputs, "wacc", [0.01])  # below terminal growth
    assert records[0]["valid"] == 0.0
    assert math.isnan(records[0]["value_per_share"])


def test_apply_field_scalar_and_series(inputs: DCFInputs):
    shocked = apply_field(inputs, "wacc", 0.13)
    assert shocked.wacc == pytest.approx(0.13)
    assert shocked.revenue_growth == inputs.revenue_growth

    margined = apply_field(inputs, "ebit_margin", 0.22)
    assert margined.series("ebit_margin") == [0.22] * 5

    listed = apply_field(inputs, "ebit_margin", [0.1, 0.12, 0.14, 0.16, 0.18])
    assert listed.series("ebit_margin") == [0.1, 0.12, 0.14, 0.16, 0.18]


def test_apply_field_rejects_unknown_name(inputs: DCFInputs):
    with pytest.raises(KeyError):
        apply_field(inputs, "not_a_field", 1.0)


# --------------------------------------------------------------------------
# Tornado
# --------------------------------------------------------------------------


def test_tornado_ranks_by_swing(inputs: DCFInputs):
    result = tornado_analysis(inputs)
    bars = result.sorted_bars()
    assert len(bars) >= 4
    swings = [b.swing for b in bars]
    assert swings == sorted(swings, reverse=True)
    assert all(s >= 0 for s in swings)


def test_tornado_covers_the_default_levers(inputs: DCFInputs):
    result = tornado_analysis(inputs)
    names = {b.field_name for b in result.bars}
    expected = {spec[0] for spec in DEFAULT_TORNADO_SPECS}
    assert names == expected


def test_tornado_base_value_matches_a_plain_run(inputs: DCFInputs):
    result = tornado_analysis(inputs)
    assert result.base.value_per_share == pytest.approx(run_dcf(inputs).value_per_share)


def test_tornado_records_are_serialisable(inputs: DCFInputs):
    records = tornado_analysis(inputs).to_records()
    assert records
    assert set(records[0]) >= {"field", "label", "low_value", "high_value", "swing"}


def test_wacc_is_among_the_most_influential_levers(inputs: DCFInputs):
    """A genuine property of any DCF: the discount rate always matters a lot."""
    result = tornado_analysis(inputs)
    top_three = {b.field_name for b in result.sorted_bars()[:3]}
    assert "wacc" in top_three
    assert all(b.swing > 0 for b in result.sorted_bars())


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------


def test_bear_base_bull_ordering(inputs: DCFInputs):
    bear, bull = default_bear_bull(inputs)
    results = build_scenarios(inputs, bear, bull)
    by_name = {r.scenario: r for r in results}
    assert set(by_name) == {"base", "bear", "bull"}
    assert by_name["bear"].value_per_share < by_name["base"].value_per_share
    assert by_name["bull"].value_per_share > by_name["base"].value_per_share
    assert by_name["bear"].wacc > by_name["bull"].wacc


def test_scenarios_keep_the_untouched_assumptions_identical(inputs: DCFInputs):
    """Only growth, margin, WACC and g move; the balance-sheet inputs do not.

    Capital expenditure is expressed as a % of revenue, so its absolute value
    legitimately follows the revenue path - what must stay constant is the
    *ratio* and the financing inputs.
    """
    bear, bull = default_bear_bull(inputs)
    results = {r.scenario: r for r in build_scenarios(inputs, bear, bull)}
    for scen in ("bear", "bull"):
        assert results[scen].net_debt == pytest.approx(results["base"].net_debt)
        assert results[scen].shares_outstanding == pytest.approx(results["base"].shares_outstanding)
        assert results[scen].tax_rate == pytest.approx(results["base"].tax_rate)
    # capex intensity is unchanged in every scenario
    base_ratio = results["base"].years[0].capex / results["base"].years[0].revenue
    bear_ratio = results["bear"].years[0].capex / results["bear"].years[0].revenue
    assert bear_ratio == pytest.approx(base_ratio)


def test_scenario_table_shape(inputs: DCFInputs):
    bear, bull = default_bear_bull(inputs)
    rows = scenario_table(build_scenarios(inputs, bear, bull))
    assert len(rows) == 3
    assert set(rows[0]) >= {
        "scenario", "value_per_share", "enterprise_value", "equity_value",
        "wacc", "terminal_growth", "upside",
    }


def test_build_scenarios_without_bear_bull_returns_only_base(inputs: DCFInputs):
    results = build_scenarios(inputs)
    assert len(results) == 1
    assert results[0].scenario == "base"


# --------------------------------------------------------------------------
# Multiples
# --------------------------------------------------------------------------


def target() -> TargetValuationInputs:
    return TargetValuationInputs(
        name="Target Co", currency=Currency.MAD, net_debt=1000.0,
        diluted_shares=100.0, ebitda=800.0, revenue=5000.0,
        net_income=400.0, book_value=3000.0, current_price=25.0,
    )


def peer(name: str, ev_ebitda: float, currency=Currency.MAD, include: bool = True,
         ebitda: float = 500.0, net_income: float = 100.0) -> ComparableCompany:
    return ComparableCompany(
        name=name, currency=currency, include=include,
        ev_ebitda=ev_ebitda, ebitda=ebitda, net_income=net_income,
        market_cap=50.0 * net_income, enterprise_value=ev_ebitda * ebitda,
        revenue=2000.0, book_value=2000.0,
    )


def test_peer_statistics_median_and_mean():
    peers = [peer("A", 6.0), peer("B", 8.0), peer("C", 10.0)]
    engine = MultiplesEngine(target())
    stats = engine.compute_stats(peers, "ev_ebitda")
    assert stats.count == 3
    assert stats.median == pytest.approx(8.0)
    assert stats.mean == pytest.approx(8.0)
    assert stats.minimum == pytest.approx(6.0)
    assert stats.maximum == pytest.approx(10.0)
    assert stats.sufficient


def test_peer_excluded_manually_is_not_in_the_statistic():
    peers = [peer("A", 6.0), peer("B", 8.0), peer("C", 10.0, include=False)]
    stats = MultiplesEngine(target()).compute_stats(peers, "ev_ebitda")
    assert stats.count == 2
    assert stats.median == pytest.approx(7.0)
    assert not stats.sufficient
    assert any(n == "C" for n, _ in stats.excluded)


def test_peer_in_another_currency_is_excluded_without_an_fx_rate():
    peers = [peer("A", 6.0), peer("B", 8.0), peer("USD", 4.0, currency=Currency.USD)]
    stats = MultiplesEngine(target()).compute_stats(peers, "ev_ebitda")
    assert stats.count == 2
    assert any(n == "USD" and "devise" in r for n, r in stats.excluded)


def test_peer_in_another_currency_is_admitted_with_a_declared_fx_rate():
    peers = [peer("A", 6.0), peer("B", 8.0), peer("USD", 4.0, currency=Currency.USD)]
    engine = MultiplesEngine(target(), fxto_target={Currency.USD: 10.0})
    stats = engine.compute_stats(peers, "ev_ebitda")
    assert stats.count == 3


def test_peer_with_negative_ebitda_is_excluded():
    peers = [peer("A", 6.0), peer("B", 8.0), peer("LOSS", 0.0, ebitda=-100.0)]
    peers[2] = ComparableCompany(
        name="LOSS", currency=Currency.MAD, ebitda=-100.0, enterprise_value=1000.0,
        market_cap=500.0, net_income=-50.0, revenue=900.0,
    )
    stats = MultiplesEngine(target()).compute_stats(peers, "ev_ebitda")
    assert stats.count == 2
    assert any(n == "LOSS" and "négatif" in r for n, r in stats.excluded)


def test_multiple_is_computed_from_raw_peer_data_when_absent():
    p = ComparableCompany(
        name="Raw", currency=Currency.MAD, enterprise_value=900.0, ebitda=100.0,
        market_cap=800.0, net_income=80.0,
    )
    stats = MultiplesEngine(target()).compute_stats([p], "ev_ebitda")
    assert stats.median == pytest.approx(9.0)


def test_insufficient_peers_are_flagged_not_hidden():
    stats = MultiplesEngine(target()).compute_stats([peer("A", 6.0)], "ev_ebitda")
    assert stats.count == 1
    assert not stats.sufficient
    assert "indicative" in stats.note


def test_implied_value_from_median_ev_ebitda():
    peers = [peer("A", 6.0), peer("B", 8.0), peer("C", 10.0)]
    result = MultiplesEngine(target()).run(peers, metrics=["ev_ebitda"])
    median_implied = [iv for iv in result.implied if iv.statistic == "median"][0]
    # 8.0 x EBITDA 800 = 6400 EV, less net debt 1000 = 5400 equity, / 100 shares = 54
    assert median_implied.implied_enterprise_value == pytest.approx(6400.0)
    assert median_implied.implied_equity_value == pytest.approx(5400.0)
    assert median_implied.implied_value_per_share == pytest.approx(54.0)
    assert result.valuation_range == (pytest.approx(54.0), pytest.approx(54.0))


def test_median_implied_value_is_reported():
    peers = [peer("A", 6.0), peer("B", 8.0), peer("C", 10.0)]
    result = MultiplesEngine(target()).run(peers, metrics=["ev_ebitda"])
    assert result.median_implied_value == pytest.approx(54.0)


def test_run_reports_insufficient_data_rather_than_inventing_a_value():
    result = MultiplesEngine(target()).run([peer("A", 6.0)], metrics=["ev_ebitda"])
    assert result.implied == []
    assert result.valuation_range is None
    assert result.warnings


def test_run_flags_a_target_without_net_debt_inputs():
    t = TargetValuationInputs(name="No data", currency=Currency.MAD, net_debt=None, diluted_shares=None)
    result = MultiplesEngine(t).run([peer("A", 6.0), peer("B", 8.0), peer("C", 10.0)])
    assert any("actions" in w or "Dette nette" in w for w in result.warnings)


def test_peer_without_denominator_is_excluded_from_pe():
    p = ComparableCompany(name="NoEarnings", currency=Currency.MAD, market_cap=1000.0)
    stats = MultiplesEngine(target()).compute_stats([p, peer("A", 6.0), peer("B", 8.0)], "pe")
    assert p.name in {n for n, _ in stats.excluded}


def test_multi_sector_peer_set_raises_a_warning():
    p1 = peer("Miner1", 6.0)
    p2 = peer("Bank1", 9.0)
    p2 = p2.model_copy(update={"sector": "Banques"})
    p1 = p1.model_copy(update={"sector": "Mines"})
    result = MultiplesEngine(target()).run([p1, p2], metrics=["ev_ebitda"])
    assert any("secteurs" in w for w in result.warnings)


# --------------------------------------------------------------------------
# Residual income / P-B framework
# --------------------------------------------------------------------------


def test_residual_income_value_when_roe_exceeds_cost_of_equity():
    result = residual_income_value(
        ResidualIncomeInputs(
            book_value_0=1000.0, roe=0.15, cost_of_equity=0.10, growth=0.03,
            payout_ratio=0.5, forecast_years=5, shares_outstanding=100.0,
        )
    )
    assert result.equity_value > 1000.0
    assert result.implied_pb > 1.0
    assert result.value_per_share == pytest.approx(result.equity_value / 100.0)
    assert len(result.years) == 5


def test_residual_income_warns_when_roe_is_below_cost_of_equity():
    result = residual_income_value(
        ResidualIncomeInputs(
            book_value_0=1000.0, roe=0.06, cost_of_equity=0.10, growth=0.02,
            payout_ratio=0.5, forecast_years=5, shares_outstanding=100.0,
        )
    )
    assert result.equity_value < 1000.0
    assert result.implied_pb < 1.0
    assert any("ROE" in w for w in result.warnings)


def test_residual_income_rejects_growth_above_cost_of_equity():
    with pytest.raises(Exception):
        ResidualIncomeInputs(
            book_value_0=1000.0, roe=0.15, cost_of_equity=0.08, growth=0.09,
            payout_ratio=0.5, forecast_years=5, shares_outstanding=100.0,
        )


def test_pb_framework_uses_the_peer_median():
    out = pb_framework([1.2, 1.5, 1.8], target_book_value=2000.0, shares=100.0)
    assert out["peer_count"] == 3.0
    assert out["median_pb"] == pytest.approx(1.5)
    assert out["implied_value_per_share"] == pytest.approx(1.5 * 2000.0 / 100.0)


def test_pb_framework_without_peers_returns_none():
    out = pb_framework([], target_book_value=2000.0, shares=100.0)
    assert out["peer_count"] == 0.0
    assert out["median_pb"] is None
    assert out["implied_value_per_share"] is None


def test_justified_pb():
    # (ROE - g) / (Ke - g) = (0.12 - 0.02) / (0.10 - 0.02) = 1.25
    assert justified_pb(roe=0.12, cost_of_equity=0.10, growth=0.02) == pytest.approx(1.25)
    # no finite justified P/B when the cost of equity does not exceed g
    assert justified_pb(roe=0.12, cost_of_equity=0.02, growth=0.03) is None


# --------------------------------------------------------------------------
# Sector models
# --------------------------------------------------------------------------


def test_baseline_assumptions_are_traceable_to_history():
    from financial_engine.sector_models import derive_baseline_assumptions

    ds = synthetic_dataset(SyntheticParams(company_id="BASE"))
    baseline = derive_baseline_assumptions(ds)
    assert baseline.suggested["revenue_growth"] == pytest.approx(0.06, abs=1e-6)
    assert baseline.suggested["ebit_margin"] == pytest.approx(0.17, abs=1e-6)
    assert baseline.suggested["capex_pct_revenue"] == pytest.approx(0.10, abs=1e-6)
    assert baseline.suggested["shares_outstanding"] == pytest.approx(100.0)
    assert "TCAM" in baseline.provenance["revenue_growth"]
    assert baseline.missing == []
    assert any("HISTORIQUES" in n for n in baseline.notes)


def test_baseline_assumptions_report_gaps_instead_of_guessing():
    from financial_engine.sector_models import derive_baseline_assumptions
    from financial_engine.schemas import FinancialDataSet

    empty = FinancialDataSet(company_id="EMPTY", currency=Currency.USD)
    baseline = derive_baseline_assumptions(empty)
    assert baseline.suggested == {}
    assert "revenue_growth" in baseline.missing
    assert "shares_outstanding" in baseline.missing


def test_cycle_normalisation_finds_the_mid_cycle_margin():
    from financial_engine.sector_models import normalise_memory_cycle

    ds = synthetic_dataset(SyntheticParams(company_id="CYCL"))
    cycle = normalise_memory_cycle(ds)
    assert cycle.mid_cycle_margin == pytest.approx(0.17, abs=1e-6)
    assert cycle.peak_margin == pytest.approx(0.17, abs=1e-6)
    assert cycle.normalised_ebit is not None
    assert cycle.cycle_spread_pp == pytest.approx(0.0, abs=1e-9)


def test_cycle_normalisation_warns_when_history_is_too_short():
    from financial_engine.sector_models import normalise_memory_cycle

    ds = synthetic_dataset(SyntheticParams(company_id="SHORT", years=2))
    cycle = normalise_memory_cycle(ds)
    assert any("exercice" in w for w in cycle.warnings)


def test_cycle_normalisation_handles_no_data():
    from financial_engine.sector_models import normalise_memory_cycle
    from financial_engine.schemas import FinancialDataSet

    cycle = normalise_memory_cycle(FinancialDataSet(company_id="NONE", currency=Currency.USD))
    assert cycle.mid_cycle_margin is None
    assert cycle.warnings


def test_mine_project_npv_is_linear_in_price():
    from financial_engine.sector_models import MineProjectInputs, mine_project_npv

    base = mine_project_npv(
        MineProjectInputs(
            name="Simulated Pit", annual_production=100_000.0, unit="oz",
            commodity_price=1800.0, cash_cost=1100.0, annual_sustaining_capex=20.0,
            tax_rate=0.30, discount_rate=0.09, mine_life_years=10,
        )
    )
    higher = mine_project_npv(
        MineProjectInputs(
            name="Simulated Pit", annual_production=100_000.0, unit="oz",
            commodity_price=2000.0, cash_cost=1100.0, annual_sustaining_capex=20.0,
            tax_rate=0.30, discount_rate=0.09, mine_life_years=10,
        )
    )
    assert base.unit_cash_margin == pytest.approx(700.0)
    assert higher.npv > base.npv
    assert base.annual_fcf == pytest.approx((180.0 - 110.0 - 20.0) * 0.70)
    assert any("vérifiés" in w for w in base.warnings)


def test_mine_project_flags_a_negative_cash_margin():
    from financial_engine.sector_models import MineProjectInputs, mine_project_npv

    result = mine_project_npv(
        MineProjectInputs(
            name="Unprofitable", annual_production=1000.0, unit="t",
            commodity_price=100.0, cash_cost=150.0, annual_sustaining_capex=1.0,
            tax_rate=0.25, discount_rate=0.10, mine_life_years=5,
        )
    )
    assert any("perte" in w for w in result.warnings)


def test_commodity_margin_impact_is_first_order():
    from financial_engine.sector_models import commodity_margin_impact

    out = commodity_margin_impact(
        revenue=1000.0, ebit=200.0, commodity_revenue_share=0.6,
        price_change=0.10, variable_cost_share=0.3,
    )
    # exposed revenue 600, +10 % -> +60 revenue, +18 cost, +42 EBIT
    assert out["revenue_change"] == pytest.approx(60.0)
    assert out["cost_change"] == pytest.approx(18.0)
    assert out["ebit_change"] == pytest.approx(42.0)
    assert out["new_ebit"] == pytest.approx(242.0)


def test_sector_risk_lists_exist_and_are_generic():
    from financial_engine.sector_models import sector_risks

    assert len(sector_risks("mining")) >= 5
    assert len(sector_risks("semiconductors")) >= 5
    assert len(sector_risks("financials")) >= 3
    assert sector_risks("unknown-sector") == []
