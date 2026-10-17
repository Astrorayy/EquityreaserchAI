"""Unit tests for the ratio engine.

Uses the synthetic, internally consistent dataset from
:mod:`data_sources.sample` - never a real company's figures.
"""

from __future__ import annotations

import pytest

from data_sources.sample import SyntheticParams, synthetic_dataset
from financial_engine.ratios import RatioEngine, safe_div
from financial_engine.schemas import (
    Currency,
    ExtractionStatus,
    FinancialDataSet,
    FiscalPeriod,
    LineItem,
    Observation,
    PeriodType,
)


@pytest.fixture()
def dataset() -> FinancialDataSet:
    return synthetic_dataset(SyntheticParams(company_id="TESTCO", currency=Currency.MAD))


@pytest.fixture()
def engine(dataset: FinancialDataSet) -> RatioEngine:
    return RatioEngine(dataset)


# --------------------------------------------------------------------------
# safe_div
# --------------------------------------------------------------------------


def test_safe_div_returns_value_when_legitimate():
    value, reason = safe_div(10.0, 2.0)
    assert value == pytest.approx(5.0)
    assert reason is None


def test_safe_div_rejects_zero_denominator():
    value, reason = safe_div(10.0, 0.0)
    assert value is None
    assert "nul" in reason


def test_safe_div_rejects_missing_numerator():
    value, reason = safe_div(None, 2.0)
    assert value is None
    assert "numérateur" in reason


def test_safe_div_rejects_missing_denominator():
    value, reason = safe_div(10.0, None)
    assert value is None
    assert "dénominateur" in reason


def test_safe_div_rejects_both_missing():
    value, reason = safe_div(None, None)
    assert value is None
    assert "tous deux" in reason


def test_safe_div_rejects_negative_denominator_when_required():
    value, reason = safe_div(10.0, -2.0, require_positive_denominator=True)
    assert value is None
    assert "négatif" in reason


def test_safe_div_allows_negative_numerator():
    value, reason = safe_div(-10.0, 2.0)
    assert value == pytest.approx(-5.0)
    assert reason is None


# --------------------------------------------------------------------------
# Aggregate metrics on the synthetic model
# --------------------------------------------------------------------------


def test_revenue_growth_matches_the_parameters(engine: RatioEngine):
    metric = engine.revenue_growth("FY2024", "FY2023")
    assert metric.is_available
    assert metric.value == pytest.approx(6.0, abs=0.01)  # the synthetic CAGR is 6 %
    assert metric.unit == "percent"
    assert "CA" in metric.formula


def test_all_metrics_have_a_formula_and_interpretation(engine: RatioEngine):
    metrics = engine.compute_all("FY2024", previous="FY2023")
    assert len(metrics) >= 15
    for key, metric in metrics.items():
        assert metric.formula, f"{key} has no formula"
        assert metric.interpretation, f"{key} has no interpretation"


def test_margins_are_built_from_the_parameters(engine: RatioEngine):
    assert engine.gross_margin("FY2024").value == pytest.approx(35.0, abs=0.01)
    # opex is 18 % of revenue, so the EBIT margin is 35 - 18 = 17 %
    assert engine.operating_margin("FY2024").value == pytest.approx(17.0, abs=0.01)
    assert engine.ebitda_margin("FY2024").value == pytest.approx(25.0, abs=0.01)
    # net margin = (EBIT - interest) x (1 - tax) / revenue, with interest fixed
    # at 5 % of the constant debt stock, hence the margin drifts up as revenue grows
    ebit = engine.ds.value(LineItem.EBIT, "FY2024")
    interest = engine.ds.value(LineItem.INTEREST_EXPENSE, "FY2024")
    revenue = engine.ds.value(LineItem.REVENUE, "FY2024")
    expected_nm = 100.0 * (ebit - interest) * (1 - 0.25) / revenue
    assert engine.net_margin("FY2024").value == pytest.approx(expected_nm, abs=1e-6)


def test_net_debt_is_debt_minus_cash(engine: RatioEngine):
    metric = engine.net_debt_metric("FY2024")
    assert metric.is_available
    assert metric.value == pytest.approx(4000.0 - 1800.0)


def test_net_debt_to_ebitda(engine: RatioEngine):
    metric = engine.net_debt_to_ebitda("FY2024")
    ebitda = engine.ds.value(LineItem.EBITDA, "FY2024")
    assert metric.value == pytest.approx(2200.0 / ebitda)


def test_free_cash_flow_is_ocf_minus_capex(engine: RatioEngine):
    metric = engine.free_cash_flow("FY2024")
    ocf = engine.ds.value(LineItem.OPERATING_CASH_FLOW, "FY2024")
    capex = engine.ds.value(LineItem.CAPITAL_EXPENDITURE, "FY2024")
    assert metric.value == pytest.approx(ocf - abs(capex))


def test_market_metrics_use_price_times_shares(engine: RatioEngine):
    mcap = engine.market_cap("FY2024")
    assert mcap.value == pytest.approx(100.0 * 120.0)


def test_enterprise_value_adds_net_debt_to_market_cap(engine: RatioEngine):
    ev = engine.enterprise_value_metric("FY2024")
    assert ev.value == pytest.approx(100.0 * 120.0 + 4000.0 - 1800.0)


def test_pe_and_ev_ebitda_are_consistent(engine: RatioEngine):
    pe = engine.pe_ratio("FY2024")
    ev_ebitda = engine.ev_to_ebitda("FY2024")
    ni = engine.ds.value(LineItem.NET_INCOME, "FY2024")
    ebitda = engine.ds.value(LineItem.EBITDA, "FY2024")
    assert pe.value == pytest.approx(12_000.0 / ni)
    assert ev_ebitda.value == pytest.approx((12_000.0 + 2_200.0) / ebitda)


def test_roic_uses_nopat_over_invested_capital(engine: RatioEngine):
    metric = engine.roic("FY2024")
    ebit = engine.ds.value(LineItem.EBIT, "FY2024")
    debt = engine.ds.value(LineItem.TOTAL_DEBT, "FY2024")
    equity = engine.ds.value(LineItem.TOTAL_EQUITY, "FY2024")
    cash = engine.ds.value(LineItem.CASH_AND_EQUIVALENTS, "FY2024")
    expected = ebit * 0.75 / (debt + equity - cash)
    assert metric.value == pytest.approx(expected * 100.0, rel=1e-6)


def test_capex_intensity(engine: RatioEngine):
    assert engine.capex_intensity("FY2024").value == pytest.approx(10.0, abs=0.01)


# --------------------------------------------------------------------------
# Failure modes: a ratio is never fabricated
# --------------------------------------------------------------------------


def _sparse() -> FinancialDataSet:
    """A dataset missing most denominators."""
    ds = FinancialDataSet(company_id="SPARSE", currency=Currency.USD)
    period = FiscalPeriod(fiscal_year=2024, period_type=PeriodType.ANNUAL)
    ds.add(
        Observation(
            company_id="SPARSE", line_item=LineItem.REVENUE, period=period,
            value=1000.0, currency=Currency.USD, status=ExtractionStatus.REPORTED,
            source_id="test",
        )
    )
    return ds


def test_missing_denominator_produces_a_reason_not_a_number():
    engine = RatioEngine(_sparse())
    metric = engine.net_margin("FY2024")
    assert metric.value is None
    assert metric.reason
    assert not metric.is_available


def test_zero_denominator_is_flagged():
    ds = FinancialDataSet(company_id="ZERO", currency=Currency.USD)
    period = FiscalPeriod(fiscal_year=2024, period_type=PeriodType.ANNUAL)
    ds.add(Observation(company_id="ZERO", line_item=LineItem.REVENUE, period=period,
                       value=0.0, currency=Currency.USD, status=ExtractionStatus.REPORTED,
                       source_id="test"))
    ds.add(Observation(company_id="ZERO", line_item=LineItem.GROSS_PROFIT, period=period,
                       value=100.0, currency=Currency.USD, status=ExtractionStatus.REPORTED,
                       source_id="test"))
    metric = RatioEngine(ds).gross_margin("FY2024")
    assert metric.value is None
    assert "nul" in metric.reason


def test_pe_is_not_computed_on_a_loss():
    ds = FinancialDataSet(company_id="LOSS", currency=Currency.USD)
    period = FiscalPeriod(fiscal_year=2024, period_type=PeriodType.ANNUAL)
    ds.add(Observation(company_id="LOSS", line_item=LineItem.NET_INCOME, period=period,
                       value=-500.0, currency=Currency.USD, status=ExtractionStatus.REPORTED,
                       source_id="t"))
    ds.add(Observation(company_id="LOSS", line_item=LineItem.MARKET_CAP, period=period,
                       value=4000.0, currency=Currency.USD, status=ExtractionStatus.REPORTED,
                       source_id="t"))
    metric = RatioEngine(ds).pe_ratio("FY2024")
    assert metric.value is None
    assert "négatif" in metric.reason


def test_ev_ebitda_is_not_computed_on_negative_ebitda():
    ds = FinancialDataSet(company_id="NEG", currency=Currency.USD)
    period = FiscalPeriod(fiscal_year=2024, period_type=PeriodType.ANNUAL)
    for item, value in (
        (LineItem.EBITDA, -100.0),
        (LineItem.MARKET_CAP, 2000.0),
        (LineItem.TOTAL_DEBT, 500.0),
        (LineItem.CASH_AND_EQUIVALENTS, 100.0),
    ):
        ds.add(Observation(company_id="NEG", line_item=item, period=period,
                           value=value, currency=Currency.USD,
                           status=ExtractionStatus.REPORTED, source_id="t"))
    metric = RatioEngine(ds).ev_to_ebitda("FY2024")
    assert metric.value is None
    assert metric.reason


def test_net_debt_requires_total_debt():
    ds = FinancialDataSet(company_id="ND", currency=Currency.USD)
    period = FiscalPeriod(fiscal_year=2024, period_type=PeriodType.ANNUAL)
    ds.add(Observation(company_id="ND", line_item=LineItem.CASH_AND_EQUIVALENTS, period=period,
                       value=500.0, currency=Currency.USD, status=ExtractionStatus.REPORTED,
                       source_id="t"))
    metric = RatioEngine(ds).net_debt_metric("FY2024")
    assert metric.value is None
    assert "dette totale" in metric.reason


def test_ebitda_is_derived_from_ebit_plus_da_when_absent():
    ds = FinancialDataSet(company_id="DER", currency=Currency.USD)
    period = FiscalPeriod(fiscal_year=2024, period_type=PeriodType.ANNUAL)
    for item, value in (
        (LineItem.REVENUE, 1000.0),
        (LineItem.EBIT, 200.0),
        (LineItem.DEPRECIATION_AMORTIZATION, 50.0),
    ):
        ds.add(Observation(company_id="DER", line_item=item, period=period,
                           value=value, currency=Currency.USD,
                           status=ExtractionStatus.REPORTED, source_id="t"))
    metric = RatioEngine(ds).ebitda_margin("FY2024")
    assert metric.value == pytest.approx(25.0)
    assert "calculé" in metric.interpretation


def test_revenue_growth_needs_a_previous_period(engine: RatioEngine):
    metric = engine.revenue_growth("FY2024", "FY1999")
    assert metric.value is None
    assert "dénominateur" in metric.reason


def test_ratio_history_shapes(engine: RatioEngine):
    history = engine.history([LineItem.REVENUE, LineItem.NET_INCOME])
    assert set(history) == {"revenue", "net_income"}
    assert list(history["revenue"]) == ["FY2021", "FY2022", "FY2023", "FY2024"]
    assert all(v is not None for v in history["revenue"].values())


def test_margin_series_has_one_entry_per_period(engine: RatioEngine):
    series = engine.margin_series()
    for key in ("gross_margin", "operating_margin", "ebitda_margin", "net_margin"):
        assert len(series[key]) == 4
