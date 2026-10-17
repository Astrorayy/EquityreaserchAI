"""Unit tests for the validation rules and the transparent derivations."""

from __future__ import annotations

import pytest

from data_sources.sample import SyntheticParams, synthetic_dataset
from financial_engine.schemas import (
    Currency,
    ExtractionStatus,
    FinancialDataSet,
    FiscalPeriod,
    LineItem,
    Observation,
    PeriodType,
    Scale,
    from_millions,
    to_millions,
)
from financial_engine.validation import (
    Severity,
    check_accounting_identities,
    check_currency_consistency,
    check_duplicates,
    check_missing_critical,
    check_plausible_values,
    check_quarter_sum,
    check_restatements,
    check_signs,
    derive_missing_items,
    validate_dataset,
)

ANNUAL_2024 = FiscalPeriod(fiscal_year=2024, period_type=PeriodType.ANNUAL)
ANNUAL_2023 = FiscalPeriod(fiscal_year=2023, period_type=PeriodType.ANNUAL)


def obs(item: LineItem, value, period=ANNUAL_2024, company="TEST", status=ExtractionStatus.REPORTED,
        currency=Currency.USD, source="test:source", reported_currency=None, scale=None) -> Observation:
    return Observation(
        company_id=company, line_item=item, period=period, value=value, currency=currency,
        status=status, source_id=source,
        reported_value=value if scale else None,
        reported_scale=scale,
        reported_currency=reported_currency,
    )


def ds_with(*observations: Observation, company: str = "TEST", currency=Currency.USD) -> FinancialDataSet:
    ds = FinancialDataSet(company_id=company, currency=currency)
    ds.extend(observations)
    return ds


# --------------------------------------------------------------------------
# Units and scales
# --------------------------------------------------------------------------


def test_scale_conversions():
    assert to_millions(1_500_000.0, Scale.UNITS) == pytest.approx(1.5)
    assert to_millions(1500.0, Scale.THOUSANDS) == pytest.approx(1.5)
    assert to_millions(1500.0, Scale.MILLIONS) == pytest.approx(1500.0)
    assert to_millions(1.5, Scale.BILLIONS) == pytest.approx(1500.0)
    assert from_millions(1500.0, Scale.BILLIONS) == pytest.approx(1.5)


def test_thousands_vs_millions_are_not_confused():
    """A figure reported in thousands must be 1000x smaller once normalised."""
    thousands = to_millions(2_500_000.0, Scale.THOUSANDS)
    millions = to_millions(2_500.0, Scale.MILLIONS)
    assert thousands == pytest.approx(millions) == pytest.approx(2500.0)


# --------------------------------------------------------------------------
# Missing data is never zero
# --------------------------------------------------------------------------


def test_missing_value_cannot_be_smuggled_in_as_reported():
    with pytest.raises(ValueError):
        Observation(
            company_id="T", line_item=LineItem.REVENUE, period=ANNUAL_2024,
            value=None, currency=Currency.USD, status=ExtractionStatus.REPORTED,
        )


def test_missing_status_allows_a_null_value():
    o = Observation(
        company_id="T", line_item=LineItem.REVENUE, period=ANNUAL_2024,
        value=None, currency=Currency.USD, status=ExtractionStatus.MISSING,
    )
    assert o.is_missing
    assert o.value is None


def test_missing_critical_items_are_reported():
    ds = ds_with(obs(LineItem.REVENUE, 1000.0), obs(LineItem.NET_INCOME, 100.0))
    issues = check_missing_critical(ds)
    codes = {i.code for i in issues}
    assert "MISSING_CRITICAL_ITEM" in codes
    missing_items = {i.context["item"] for i in issues}
    assert "ebit" in missing_items
    assert "operating_cash_flow" in missing_items


def test_no_annual_data_is_an_error():
    ds = ds_with(obs(LineItem.REVENUE, 100.0, period=FiscalPeriod(fiscal_year=2024, period_type=PeriodType.QUARTERLY, period_number=1)))
    issues = check_missing_critical(ds)
    assert any(i.code == "NO_ANNUAL_DATA" and i.severity == Severity.ERROR for i in issues)


# --------------------------------------------------------------------------
# Accounting identities
# --------------------------------------------------------------------------


def test_clean_synthetic_dataset_has_no_errors():
    ds = synthetic_dataset(SyntheticParams(company_id="CLEAN"))
    report = validate_dataset(ds)
    assert report.errors == [], [str(e) for e in report.errors]
    assert report.is_usable
    assert report.score == 100


def test_gross_profit_identity_is_checked():
    ds = ds_with(
        obs(LineItem.REVENUE, 1000.0),
        obs(LineItem.COST_OF_REVENUE, 600.0),
        obs(LineItem.GROSS_PROFIT, 500.0),  # should be 400
    )
    issues = check_accounting_identities(ds)
    assert any(i.code == "IDENTITY_GROSS_PROFIT" for i in issues)


def test_gross_profit_identity_accepts_negative_cogs_convention():
    ds = ds_with(
        obs(LineItem.REVENUE, 1000.0),
        obs(LineItem.COST_OF_REVENUE, -600.0),
        obs(LineItem.GROSS_PROFIT, 400.0),
    )
    issues = check_accounting_identities(ds)
    assert not any(i.code == "IDENTITY_GROSS_PROFIT" for i in issues)


def test_ebitda_identity_is_checked():
    ds = ds_with(
        obs(LineItem.EBIT, 200.0),
        obs(LineItem.DEPRECIATION_AMORTIZATION, 50.0),
        obs(LineItem.EBITDA, 300.0),  # should be 250
    )
    issues = check_accounting_identities(ds)
    assert any(i.code == "IDENTITY_EBITDA" for i in issues)


def test_total_debt_identity_is_checked():
    ds = ds_with(
        obs(LineItem.SHORT_TERM_DEBT, 100.0),
        obs(LineItem.LONG_TERM_DEBT, 400.0),
        obs(LineItem.TOTAL_DEBT, 900.0),  # should be 500
    )
    issues = check_accounting_identities(ds)
    assert any(i.code == "IDENTITY_TOTAL_DEBT" for i in issues)


def test_free_cash_flow_identity_is_checked():
    ds = ds_with(
        obs(LineItem.OPERATING_CASH_FLOW, 500.0),
        obs(LineItem.CAPITAL_EXPENDITURE, 200.0),
        obs(LineItem.FREE_CASH_FLOW, 100.0),  # should be 300
    )
    issues = check_accounting_identities(ds)
    assert any(i.code == "IDENTITY_FREE_CASH_FLOW" for i in issues)


def test_balance_sheet_identity_is_checked():
    ds = ds_with(
        obs(LineItem.TOTAL_ASSETS, 1000.0),
        obs(LineItem.TOTAL_LIABILITIES, 400.0),
        obs(LineItem.TOTAL_EQUITY, 300.0),  # 400 + 300 = 700 != 1000
    )
    issues = check_accounting_identities(ds)
    assert any(i.code == "IDENTITY_BALANCE_SHEET" for i in issues)


def test_net_income_identity_is_informational_only():
    ds = ds_with(
        obs(LineItem.PRETAX_INCOME, 300.0),
        obs(LineItem.INCOME_TAX_EXPENSE, 75.0),
        obs(LineItem.NET_INCOME, 400.0),
    )
    issues = [i for i in check_accounting_identities(ds) if i.code == "IDENTITY_NET_INCOME"]
    assert issues and all(i.severity == Severity.INFO for i in issues)


def test_inventory_cannot_exceed_current_assets():
    ds = ds_with(
        obs(LineItem.CURRENT_ASSETS, 100.0),
        obs(LineItem.INVENTORY, 200.0),
    )
    issues = check_accounting_identities(ds)
    assert any(i.code == "STRUCTURAL_COMPONENT_TOO_LARGE" for i in issues)


def test_identity_tolerance_allows_small_rounding():
    ds = ds_with(
        obs(LineItem.REVENUE, 1000.0),
        obs(LineItem.COST_OF_REVENUE, 600.0),
        obs(LineItem.GROSS_PROFIT, 399.5),  # 0.05 % off, within tolerance
    )
    issues = check_accounting_identities(ds)
    assert not any(i.code == "IDENTITY_GROSS_PROFIT" for i in issues)


# --------------------------------------------------------------------------
# Plausibility and signs
# --------------------------------------------------------------------------


def test_zero_revenue_is_flagged_as_implausible():
    ds = ds_with(obs(LineItem.REVENUE, 0.005))
    issues = check_plausible_values(ds.observations)
    assert any(i.code == "IMPLAUSIBLE_MAGNITUDE" for i in issues)


def test_absurd_revenue_is_an_error():
    ds = ds_with(obs(LineItem.REVENUE, 9_000_000.0))
    issues = check_plausible_values(ds.observations)
    assert any(i.code == "IMPLAUSIBLE_MAGNITUDE" and i.severity == Severity.ERROR for i in issues)


def test_negative_share_count_is_an_error():
    ds = ds_with(obs(LineItem.SHARES_OUTSTANDING, -5.0))
    issues = check_plausible_values(ds.observations)
    assert any(i.severity == Severity.ERROR for i in issues)


def test_mixed_sign_conventions_are_flagged():
    ds = ds_with(
        obs(LineItem.CAPITAL_EXPENDITURE, 200.0),
        obs(LineItem.DIVIDENDS_PAID, -50.0),
    )
    issues = check_signs(ds.observations)
    assert any(i.code == "SIGN_CONVENTION_MIXED" for i in issues)


def test_consistent_sign_convention_is_accepted():
    ds = ds_with(
        obs(LineItem.CAPITAL_EXPENDITURE, 200.0),
        obs(LineItem.DIVIDENDS_PAID, 50.0),
    )
    assert check_signs(ds.observations) == []


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def test_currency_mismatch_is_reported():
    ds = ds_with(
        obs(LineItem.REVENUE, 1000.0, currency=Currency.MAD, reported_currency=Currency.USD)
    )
    issues = check_currency_consistency(ds)
    assert any(i.code == "CURRENCY_MISMATCH" for i in issues)


def test_matching_currency_is_not_reported():
    ds = ds_with(
        obs(LineItem.REVENUE, 1000.0, currency=Currency.MAD, reported_currency=Currency.MAD)
    )
    assert check_currency_consistency(ds) == []


def test_duplicate_observations_are_reported_not_silently_merged():
    ds = ds_with(
        obs(LineItem.REVENUE, 1000.0, source="a"),
        obs(LineItem.REVENUE, 1010.0, source="b"),
    )
    issues = check_duplicates(ds)
    assert len(issues) == 1
    assert issues[0].context["count"] == 2
    # the reported source wins on lookup
    assert ds.value(LineItem.REVENUE, "FY2024") in (1000.0, 1010.0)


def test_restatement_between_two_vintages_is_detected():
    old = ds_with(obs(LineItem.REVENUE, 1000.0))
    new = ds_with(obs(LineItem.REVENUE, 1100.0, source="restated"))
    issues = check_restatements(new, old)
    assert any(i.code == "RESTATEMENT_DETECTED" for i in issues)


def test_identical_vintages_produce_no_restatement_warning():
    old = ds_with(obs(LineItem.REVENUE, 1000.0))
    new = ds_with(obs(LineItem.REVENUE, 1000.0, source="same"))
    assert check_restatements(new, old) == []


# --------------------------------------------------------------------------
# Quarterly vs annual
# --------------------------------------------------------------------------


def test_quarter_sum_mismatch_is_detected():
    ds = FinancialDataSet(company_id="Q", currency=Currency.USD)
    for q, value in enumerate([100.0, 110.0, 120.0, 130.0], start=1):
        ds.add(obs(LineItem.REVENUE, value,
                   period=FiscalPeriod(fiscal_year=2024, period_type=PeriodType.QUARTERLY, period_number=q)))
    ds.add(obs(LineItem.REVENUE, 800.0, period=ANNUAL_2024))  # 460 actual sum
    issues = check_quarter_sum(ds)
    assert any(i.code == "QUARTER_SUM_MISMATCH" for i in issues)


def test_incomplete_quarters_are_flagged_as_info():
    ds = FinancialDataSet(company_id="Q", currency=Currency.USD)
    for q, value in enumerate([100.0, 110.0], start=1):
        ds.add(obs(LineItem.REVENUE, value,
                   period=FiscalPeriod(fiscal_year=2024, period_type=PeriodType.QUARTERLY, period_number=q)))
    issues = check_quarter_sum(ds)
    assert any(i.code == "INCOMPLETE_QUARTERS" for i in issues)


# --------------------------------------------------------------------------
# Derivations
# --------------------------------------------------------------------------


def test_derivations_fill_gaps_with_derived_status():
    ds = ds_with(
        obs(LineItem.REVENUE, 1000.0),
        obs(LineItem.COST_OF_REVENUE, 600.0),
        obs(LineItem.EBIT, 200.0),
        obs(LineItem.DEPRECIATION_AMORTIZATION, 50.0),
        obs(LineItem.OPERATING_CASH_FLOW, 300.0),
        obs(LineItem.CAPITAL_EXPENDITURE, 120.0),
        obs(LineItem.CURRENT_ASSETS, 400.0),
        obs(LineItem.CURRENT_LIABILITIES, 150.0),
        obs(LineItem.INVENTORY, 100.0),
    )
    report = derive_missing_items(ds)

    assert ds.value(LineItem.GROSS_PROFIT, "FY2024") == pytest.approx(400.0)
    assert ds.value(LineItem.EBITDA, "FY2024") == pytest.approx(250.0)
    assert ds.value(LineItem.FREE_CASH_FLOW, "FY2024") == pytest.approx(180.0)
    # NWC = (current assets - inventory) - current liabilities = (400-100)-150
    assert ds.value(LineItem.NET_WORKING_CAPITAL, "FY2024") == pytest.approx(150.0)

    by_item = {o.line_item: o for o in ds.observations if o.status == ExtractionStatus.DERIVED}
    assert by_item[LineItem.GROSS_PROFIT].extraction_method == "revenue - cost_of_revenue"
    for derived in by_item.values():
        assert "Poste non publié tel quel." in derived.notes
    assert len(report.created) == 4


def test_derivation_never_overwrites_a_reported_figure():
    ds = ds_with(
        obs(LineItem.REVENUE, 1000.0),
        obs(LineItem.COST_OF_REVENUE, 600.0),
        obs(LineItem.GROSS_PROFIT, 999.0),  # suspicious but reported
    )
    derive_missing_items(ds)
    assert ds.value(LineItem.GROSS_PROFIT, "FY2024") == pytest.approx(999.0)
    assert not any(o.status == ExtractionStatus.DERIVED for o in ds.observations)


def test_derivation_is_impossible_without_inputs():
    ds = ds_with(obs(LineItem.REVENUE, 1000.0))
    derive_missing_items(ds)
    assert ds.value(LineItem.GROSS_PROFIT, "FY2024") is None
    assert ds.value(LineItem.EBITDA, "FY2024") is None
    assert ds.value(LineItem.FREE_CASH_FLOW, "FY2024") is None


def test_total_debt_derivation_takes_absolute_values():
    ds = ds_with(
        obs(LineItem.SHORT_TERM_DEBT, 100.0),
        obs(LineItem.LONG_TERM_DEBT, 400.0),
    )
    derive_missing_items(ds)
    assert ds.value(LineItem.TOTAL_DEBT, "FY2024") == pytest.approx(500.0)


def test_market_cap_derivation():
    ds = ds_with(obs(LineItem.MARKET_PRICE, 25.0), obs(LineItem.DILUTED_SHARES, 40.0))
    derive_missing_items(ds)
    assert ds.value(LineItem.MARKET_CAP, "FY2024") == pytest.approx(1000.0)


# --------------------------------------------------------------------------
# Aggregate report
# --------------------------------------------------------------------------


def test_report_score_decreases_with_issues():
    clean = validate_dataset(synthetic_dataset(SyntheticParams(company_id="CLEAN2")))
    dirty = validate_dataset(
        ds_with(
            obs(LineItem.REVENUE, 0.001),
            obs(LineItem.SHARES_OUTSTANDING, -1.0),
        )
    )
    assert clean.score == 100
    assert dirty.score < clean.score
    assert dirty.errors


def test_report_helpers_classify_severities():
    report = validate_dataset(
        ds_with(
            obs(LineItem.REVENUE, 1000.0),
            obs(LineItem.COST_OF_REVENUE, 600.0),
            obs(LineItem.GROSS_PROFIT, 500.0),
        )
    )
    assert report.warnings
    assert all(i.severity == Severity.WARNING for i in report.warnings)
    assert report.by_code("IDENTITY_GROSS_PROFIT")


def test_validation_never_mutates_the_dataset():
    ds = synthetic_dataset(SyntheticParams(company_id="IMMUT"))
    before = len(ds.observations)
    validate_dataset(ds)
    assert len(ds.observations) == before
