"""Clearly labelled SIMULATED data.

Purpose
-------
The application must be demonstrable and testable without downloading anything,
and without ever presenting invented figures as real company data.  This module
therefore builds a synthetic, internally consistent three-statement model that
is *always* flagged:

* every observation carries ``status=ExtractionStatus.MANUAL`` and
  ``source_id="simulated:demo"``,
* every note begins with ``SIMULE``,
* :func:`synthetic_dataset` docstring and the returned dataset expose
  ``is_simulated`` through the conventions below.

Nothing produced here may be presented as Managem's or SanDisk's accounts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional

from financial_engine.schemas import (
    Currency,
    ExtractionStatus,
    FinancialDataSet,
    FiscalPeriod,
    LineItem,
    Observation,
    PeriodType,
    Scale,
)

SIMULATION_SOURCE_ID = "simulated:demo"
SAMPLE_NOTE = "SIMULÉ — donnée synthétique de démonstration, ne correspond à aucune société réelle."


@dataclass
class SyntheticParams:
    """Parameters of the synthetic model. Defaults are deliberately bland."""

    company_id: str = "SIMULATED"
    currency: Currency = Currency.MAD
    base_revenue: float = 10_000.0        # millions
    revenue_growth: float = 0.06
    gross_margin: float = 0.35
    opex_pct_revenue: float = 0.18
    da_pct_revenue: float = 0.08
    capex_pct_revenue: float = 0.10
    tax_rate: float = 0.25
    interest_rate_on_debt: float = 0.05
    ocf_conversion: float = 0.85          # OCF / EBITDA
    days_receivable: float = 55.0
    days_inventory: float = 80.0
    days_payable: float = 45.0
    cash_and_equivalents: float = 1_800.0
    short_term_debt: float = 900.0
    long_term_debt: float = 3_100.0
    shares_outstanding: float = 100.0     # millions
    price: float = 120.0
    first_year: int = 2021
    years: int = 4


def synthetic_dataset(params: Optional[SyntheticParams] = None) -> FinancialDataSet:
    """Build a synthetic annual dataset (3 balance-sheet + P&L + cash-flow).

    The accounting identities used by :mod:`financial_engine.validation` hold
    **exactly** by construction, so a clean validation report on this dataset is
    meaningful (it proves the validator works, not that a real company is
    consistent).
    """
    p = params or SyntheticParams()
    ds = FinancialDataSet(company_id=p.company_id, currency=p.currency)

    for i in range(p.years):
        year = p.first_year + i
        period = FiscalPeriod(
            fiscal_year=year,
            period_type=PeriodType.ANNUAL,
            start_date=date(year, 1, 1),
            end_date=date(year, 12, 31),
        )
        revenue = p.base_revenue * ((1.0 + p.revenue_growth) ** i)
        cogs = revenue * (1.0 - p.gross_margin)
        gross_profit = revenue - cogs
        opex = revenue * p.opex_pct_revenue
        ebit = gross_profit - opex
        da = revenue * p.da_pct_revenue
        ebitda = ebit + da

        total_debt = p.short_term_debt + p.long_term_debt
        interest = total_debt * p.interest_rate_on_debt
        pretax = ebit - interest
        tax = pretax * p.tax_rate
        net_income = pretax - tax

        ocf = ebitda * p.ocf_conversion
        capex = revenue * p.capex_pct_revenue
        fcf = ocf - capex

        # working capital, consistent with the balance sheet below
        receivables = revenue * p.days_receivable / 365.0
        inventory = cogs * p.days_inventory / 365.0
        payables = cogs * p.days_payable / 365.0
        current_assets = p.cash_and_equivalents + receivables + inventory
        current_liabilities = payables + p.short_term_debt
        nwc = (current_assets - inventory) - current_liabilities

        # non-current assets adjusted so that assets == liabilities + equity
        non_current = revenue * 1.10
        total_assets = current_assets + non_current
        total_liabilities = current_liabilities + p.long_term_debt
        total_equity = total_assets - total_liabilities

        dividends = net_income * 0.30
        market_cap = p.shares_outstanding * p.price

        values: Dict[LineItem, float] = {
            LineItem.REVENUE: revenue,
            LineItem.COST_OF_REVENUE: cogs,
            LineItem.GROSS_PROFIT: gross_profit,
            LineItem.OPERATING_EXPENSES: opex,
            LineItem.EBIT: ebit,
            LineItem.OPERATING_INCOME: ebit,
            LineItem.DEPRECIATION_AMORTIZATION: da,
            LineItem.EBITDA: ebitda,
            LineItem.INTEREST_EXPENSE: interest,
            LineItem.PRETAX_INCOME: pretax,
            LineItem.INCOME_TAX_EXPENSE: tax,
            LineItem.NET_INCOME: net_income,
            LineItem.CASH_AND_EQUIVALENTS: p.cash_and_equivalents,
            LineItem.ACCOUNTS_RECEIVABLE: receivables,
            LineItem.INVENTORY: inventory,
            LineItem.CURRENT_ASSETS: current_assets,
            LineItem.TOTAL_ASSETS: total_assets,
            LineItem.ACCOUNTS_PAYABLE: payables,
            LineItem.CURRENT_LIABILITIES: current_liabilities,
            LineItem.SHORT_TERM_DEBT: p.short_term_debt,
            LineItem.LONG_TERM_DEBT: p.long_term_debt,
            LineItem.TOTAL_DEBT: total_debt,
            LineItem.TOTAL_LIABILITIES: total_liabilities,
            LineItem.TOTAL_EQUITY: total_equity,
            LineItem.NET_WORKING_CAPITAL: nwc,
            LineItem.OPERATING_CASH_FLOW: ocf,
            LineItem.CAPITAL_EXPENDITURE: capex,
            LineItem.FREE_CASH_FLOW: fcf,
            LineItem.DIVIDENDS_PAID: dividends,
            LineItem.SHARES_OUTSTANDING: p.shares_outstanding,
            LineItem.DILUTED_SHARES: p.shares_outstanding,
            LineItem.MARKET_PRICE: p.price,
            LineItem.MARKET_CAP: market_cap,
        }

        for item, value in values.items():
            ds.add(
                Observation(
                    company_id=p.company_id,
                    line_item=item,
                    period=period,
                    value=value,
                    currency=p.currency,
                    reported_value=value,
                    reported_scale=Scale.MILLIONS,
                    reported_currency=p.currency,
                    status=ExtractionStatus.MANUAL,
                    source_id=SIMULATION_SOURCE_ID,
                    extraction_method="synthetic model (tests / demonstration)",
                    notes=SAMPLE_NOTE,
                )
            )

    return ds


#: The subset of line items shown in the manual-entry form of the UI.
MANUAL_ENTRY_ITEMS: List[LineItem] = [
    LineItem.REVENUE,
    LineItem.COST_OF_REVENUE,
    LineItem.OPERATING_EXPENSES,
    LineItem.DEPRECIATION_AMORTIZATION,
    LineItem.INTEREST_EXPENSE,
    LineItem.INCOME_TAX_EXPENSE,
    LineItem.NET_INCOME,
    LineItem.CASH_AND_EQUIVALENTS,
    LineItem.SHORT_TERM_DEBT,
    LineItem.LONG_TERM_DEBT,
    LineItem.CURRENT_ASSETS,
    LineItem.CURRENT_LIABILITIES,
    LineItem.INVENTORY,
    LineItem.TOTAL_ASSETS,
    LineItem.TOTAL_EQUITY,
    LineItem.OPERATING_CASH_FLOW,
    LineItem.CAPITAL_EXPENDITURE,
    LineItem.DIVIDENDS_PAID,
    LineItem.SHARES_OUTSTANDING,
    LineItem.MARKET_PRICE,
]


def manual_dataset_from_values(
    company_id: str,
    currency: Currency,
    fiscal_year: int,
    values: Dict[LineItem, Optional[float]],
    scale: Scale = Scale.MILLIONS,
    note: str = "Saisie manuelle par l'analyste.",
) -> FinancialDataSet:
    """Turn a dictionary typed by the analyst into a dataset.

    ``None`` and empty entries are skipped - never coerced to zero - so a
    half-filled form produces a dataset with *missing* values, which the
    validator reports explicitly.
    """
    from financial_engine.schemas import to_millions

    ds = FinancialDataSet(company_id=company_id, currency=currency)
    period = FiscalPeriod(fiscal_year=fiscal_year, period_type=PeriodType.ANNUAL)
    for item, raw in values.items():
        if raw is None:
            continue
        if isinstance(raw, str) and not raw.strip():
            continue
        numeric = float(raw)
        ds.add(
            Observation(
                company_id=company_id,
                line_item=item,
                period=period,
                value=to_millions(numeric, scale),
                currency=currency,
                reported_value=numeric,
                reported_scale=scale,
                reported_currency=currency,
                status=ExtractionStatus.MANUAL,
                source_id="manual:analyst",
                extraction_method="manual entry",
                notes=note,
            )
        )
    return ds
