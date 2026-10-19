"""Chart layer: Plotly figures for the research terminal.

Every function returns a :class:`plotly.graph_objects.Figure` and takes plain
data (a ``FinancialDataSet``, ``DCFResult``, sensitivity grid, …) so the figures
can be unit-rendered in tests without a Streamlit runtime.

Conventions
-----------
* Values are in **millions** of the reporting currency unless the axis title
  says otherwise; per-share values are in currency units.
* Missing data is never coerced to zero: series gaps stay gaps.
* Scenario colours are fixed (bear = red, base = blue, bull = green) so the
  same shade means the same thing on every page.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import pandas as pd
import plotly.graph_objects as go

from financial_engine.dcf import DCFResult
from financial_engine.multiples import MultiplesResult
from financial_engine.ratios import Metric
from financial_engine.schemas import (
    ComparableCompany,
    ExtractionStatus,
    FinancialDataSet,
    LineItem,
    PeriodType,
)
from financial_engine.sector_models import CycleNormalisation
from financial_engine.sensitivity import SensitivityGrid, TornadoResult

# --------------------------------------------------------------------------
# Palette / layout
# --------------------------------------------------------------------------

INK = "#ffffff"          # card background (charts sit on white cards)
PANEL = "#ffffff"        # panel background
GRID = "#e9ecf3"         # gridlines
AXIS = "#d5d9e4"
TEXT = "#0e1318"
MUTED = "#6b7280"
BLUE = "#7d2ae8"         # primary series (Canva purple)
GOLD = "#ffb020"
GOLD_TEXT = "#b26a00"    # readable amber for annotation text on white
GREEN = "#22c55e"
GREEN_TEXT = "#0f9d58"
RED = "#f43f5e"
AMBER = "#ff7a59"
VIOLET = "#a855f7"
CYAN = "#00c4cc"
FONT_FAMILY = "'Plus Jakarta Sans', Inter, 'Segoe UI', sans-serif"

SCENARIO_COLORS = {
    "bear": RED,
    "pessimiste": RED,
    "base": BLUE,
    "central": BLUE,
    "bull": GREEN,
    "optimiste": GREEN,
}

SEQUENCE = [BLUE, GOLD, GREEN, VIOLET, CYAN, RED]

ASSET_CLASS_NOTE = "Valeurs en millions de la devise de reporting, sauf indication contraire."


def _currency_text(currency: object) -> str:
    """Currency enums stringify as ``Currency.USD``; always show the code."""
    return getattr(currency, "value", None) or str(currency)


def _layout(
    title: str,
    *,
    y_title: str = "",
    x_title: str = "",
    height: int = 380,
    legend: bool = True,
    note: Optional[str] = None,
) -> Dict[str, object]:
    if note:
        title = (
            f"{title}<br><span style='font-size:11.5px;color:{MUTED};font-weight:400'>"
            f"{note}</span>"
        )
    return {
        "title": {
            "text": title,
            "font": {"size": 15, "color": TEXT, "family": FONT_FAMILY, "weight": 700},
            "x": 0.01,
            "xanchor": "left",
        },
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": TEXT, "family": FONT_FAMILY, "size": 12},
        # French number formatting: decimal comma, thin-space thousands.
        "separators": ", ",
        "height": height,
        "margin": {"l": 56, "r": 24, "t": 64, "b": 44},
        "hovermode": "x unified",
        "hoverlabel": {
            "bgcolor": "#ffffff",
            "bordercolor": "#e3e6ef",
            "font": {"family": FONT_FAMILY, "color": TEXT, "size": 12},
        },
        "barcornerradius": 6,
        "bargap": 0.28,
        "transition": {"duration": 450, "easing": "cubic-in-out"},
        "dragmode": "zoom",
        "xaxis": {
            "title": {"text": x_title, "font": {"size": 12, "color": MUTED}},
            "gridcolor": "rgba(0,0,0,0)",
            "zerolinecolor": AXIS,
            "linecolor": AXIS,
            "showline": True,
            "ticks": "",
            "tickfont": {"color": MUTED},
            "showspikes": True,
            "spikecolor": "rgba(125, 42, 232, 0.45)",
            "spikethickness": 1,
            "spikedash": "dot",
            "spikemode": "across",
        },
        "yaxis": {
            "title": {"text": y_title, "font": {"size": 12, "color": MUTED}},
            "gridcolor": GRID,
            "griddash": "dot",
            "zerolinecolor": AXIS,
            "linecolor": "rgba(0,0,0,0)",
            "tickfont": {"color": MUTED},
        },
        "showlegend": legend,
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
            "font": {"color": MUTED, "size": 11},
        },
        "colorway": SEQUENCE,
    }


def polish(fig: go.Figure) -> go.Figure:
    """Uniform finishing touches; never changes the plotted values."""
    fig.update_traces(
        selector={"type": "scatter"},
        line={"width": 2.6},
        marker={"size": 7, "line": {"width": 1.5, "color": "#ffffff"}},
    )
    fig.update_traces(
        selector={"type": "bar"},
        marker_line_width=0,
        opacity=0.92,
    )
    return fig


def _period_frame(dataset: FinancialDataSet, items: Sequence[LineItem]) -> pd.DataFrame:
    """Periods as index (oldest first) and one column per line item."""
    periods = dataset.periods(PeriodType.ANNUAL)
    series = {item: dataset.series(item, PeriodType.ANNUAL) for item in items}
    frame = pd.DataFrame(series, index=periods)
    if not frame.empty:
        # Chronological order: periods come back sorted, but be explicit.
        frame = frame.reindex(sorted(frame.index))
    return frame


def _empty_figure(message: str, height: int = 320) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 13, "color": MUTED},
    )
    fig.update_layout(**_layout("", height=height, legend=False))
    return fig


# --------------------------------------------------------------------------
# Financial statements
# --------------------------------------------------------------------------

def revenue_and_earnings(dataset: FinancialDataSet) -> go.Figure:
    frame = _period_frame(dataset, [LineItem.REVENUE, LineItem.NET_INCOME])
    if frame.empty or frame[LineItem.REVENUE].dropna().empty:
        return _empty_figure(
            "Aucun chiffre d'affaires annuel disponible. Chargez les comptes ou "
            "utilisez la saisie manuelle pour alimenter ce graphique."
        )
    fig = go.Figure()
    fig.add_bar(
        x=frame.index.tolist(),
        y=frame[LineItem.REVENUE].tolist(),
        name="Chiffre d'affaires",
        marker_color=BLUE,
        hovertemplate="%{x}<br>CA : %{y:,.0f}<extra></extra>",
    )
    fig.add_scatter(
        x=frame.index.tolist(),
        y=frame[LineItem.NET_INCOME].tolist(),
        name="Résultat net",
        mode="lines+markers",
        line={"color": GOLD, "width": 2.5},
        marker={"size": 7},
        connectgaps=False,
        hovertemplate="%{x}<br>Résultat net : %{y:,.0f}<extra></extra>",
    )
    layout = _layout(
        "Chiffre d'affaires et résultat net (annuel)",
        y_title="Millions",
        note=ASSET_CLASS_NOTE,
    )
    fig.update_layout(**layout)
    return fig


def margin_trends(dataset: FinancialDataSet, engine_metrics: Optional[Sequence[Metric]] = None) -> go.Figure:
    """Gross / operating / net margins in percent, computed from reported items."""
    frame = _period_frame(
        dataset,
        [
            LineItem.REVENUE,
            LineItem.GROSS_PROFIT,
            LineItem.EBIT,
            LineItem.OPERATING_INCOME,
            LineItem.NET_INCOME,
        ],
    )
    if frame.empty or frame[LineItem.REVENUE].dropna().empty:
        return _empty_figure("Marges non calculables : le chiffre d'affaires est absent.")

    revenue = frame[LineItem.REVENUE]
    operating = frame[LineItem.EBIT].fillna(frame[LineItem.OPERATING_INCOME])
    margins = {
        "Marge brute": (frame[LineItem.GROSS_PROFIT] / revenue) * 100,
        "Marge opérationnelle": (operating / revenue) * 100,
        "Marge nette": (frame[LineItem.NET_INCOME] / revenue) * 100,
    }
    fig = go.Figure()
    for name, series in margins.items():
        if series.dropna().empty:
            continue
        fig.add_scatter(
            x=frame.index.tolist(),
            y=series.round(2).tolist(),
            name=name,
            mode="lines+markers",
            connectgaps=False,
            hovertemplate="%{x}<br>" + name + " : %{y:.1f} %<extra></extra>",
        )
    if not fig.data:
        return _empty_figure(
            "Aucune marge calculable : il manque le résultat brut, opérationnel ou net."
        )
    layout = _layout(
        "Marges (% du chiffre d'affaires)",
        y_title="%",
        note="Les trous correspondent à des données absentes, jamais à des zéros.",
    )
    fig.update_layout(**layout)
    return fig


def cash_flow_bars(dataset: FinancialDataSet) -> go.Figure:
    frame = _period_frame(
        dataset,
        [LineItem.OPERATING_CASH_FLOW, LineItem.CAPITAL_EXPENDITURE, LineItem.FREE_CASH_FLOW],
    )
    if frame.dropna(how="all").empty:
        return _empty_figure(
            "Flux de trésorerie non disponibles. Le tableau des flux doit être "
            "saisi ou extrait du rapport annuel."
        )
    fig = go.Figure()
    if frame[LineItem.OPERATING_CASH_FLOW].notna().any():
        fig.add_bar(
            x=frame.index.tolist(),
            y=frame[LineItem.OPERATING_CASH_FLOW].tolist(),
            name="Flux d'exploitation",
            marker_color=CYAN,
        )
    if frame[LineItem.CAPITAL_EXPENDITURE].notna().any():
        fig.add_bar(
            x=frame.index.tolist(),
            y=(-frame[LineItem.CAPITAL_EXPENDITURE]).tolist(),
            name="Investissements (signe inversé)",
            marker_color=RED,
        )
    if frame[LineItem.FREE_CASH_FLOW].notna().any():
        fig.add_scatter(
            x=frame.index.tolist(),
            y=frame[LineItem.FREE_CASH_FLOW].tolist(),
            name="Free cash flow",
            mode="lines+markers",
            line={"color": GOLD, "width": 2.5},
            connectgaps=False,
        )
    fig.update_layout(barmode="relative")
    layout = _layout("Flux de trésorerie (annuel)", y_title="Millions")
    fig.update_layout(**layout)
    return fig


def balance_sheet_trend(dataset: FinancialDataSet) -> go.Figure:
    frame = _period_frame(
        dataset,
        [
            LineItem.TOTAL_ASSETS,
            LineItem.TOTAL_EQUITY,
            LineItem.TOTAL_DEBT,
            LineItem.CASH_AND_EQUIVALENTS,
        ],
    )
    if frame.dropna(how="all").empty:
        return _empty_figure("Bilan non disponible : actif, capitaux propres et dette absents.")
    fig = go.Figure()
    for name, item, color in (
        ("Total actif", LineItem.TOTAL_ASSETS, BLUE),
        ("Capitaux propres", LineItem.TOTAL_EQUITY, GREEN),
        ("Dette totale", LineItem.TOTAL_DEBT, RED),
        ("Trésorerie", LineItem.CASH_AND_EQUIVALENTS, GOLD),
    ):
        series = frame[item]
        if series.dropna().empty:
            continue
        fig.add_scatter(
            x=frame.index.tolist(),
            y=series.tolist(),
            name=name,
            mode="lines+markers",
            line={"color": color, "width": 2.2},
            connectgaps=False,
        )
    layout = _layout("Structure financière (annuel)", y_title="Millions")
    fig.update_layout(**layout)
    return fig


# --------------------------------------------------------------------------
# Valuation
# --------------------------------------------------------------------------

def dcf_bridge(result: DCFResult) -> go.Figure:
    """Waterfall from discounted cash flows to equity value."""
    steps = [
        ("VA des FCFF explicites", result.pv_explicit, BLUE),
        ("VA valeur terminale", result.pv_terminal_value, VIOLET),
        ("Valeur d'entreprise", result.enterprise_value, GOLD),
        ("Dette nette", -result.net_debt, RED),
    ]
    if result.other_senior_claims:
        steps.append(("Autres créances senior", -result.other_senior_claims, RED))
    if result.non_operating_assets:
        steps.append(("Actifs hors exploitation", result.non_operating_assets, GREEN))
    steps.append(("Valeur des capitaux propres", result.equity_value, GREEN))

    labels = [s[0] for s in steps]
    values = [s[1] for s in steps]
    colors = [s[2] for s in steps]
    measures = ["relative"] * (len(steps) - 1) + ["total"]

    fig = go.Figure(
        go.Waterfall(
            x=labels,
            y=values,
            measure=measures,
            connector={"line": {"color": GRID}},
            increasing={"marker": {"color": GREEN}},
            decreasing={"marker": {"color": RED}},
            totals={"marker": {"color": GOLD}},
            text=[f"{v:,.0f}" for v in values],
            textposition="outside",
            textfont={"color": TEXT},
            hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>",
        )
    )
    layout = _layout(
        "Passage de la valeur d'entreprise à la valeur des capitaux propres",
        y_title=f"Millions {result.currency}",
        note=(
            f"Scénario {result.scenario} — WACC {result.wacc:.2%}, "
            f"croissance terminale {result.terminal_growth:.2%}. "
            "La dette nette n'est retirée qu'une seule fois."
        ),
        height=430,
        legend=False,
    )
    fig.update_layout(**layout)
    return fig


def fcff_projection(result: DCFResult) -> go.Figure:
    if not result.years:
        return _empty_figure("Aucune projection FCFF disponible.")
    years = [y.fiscal_year for y in result.years]
    fig = go.Figure()
    fig.add_bar(
        x=years,
        y=[y.fcff for y in result.years],
        name="FCFF",
        marker_color=BLUE,
        hovertemplate="%{x}<br>FCFF : %{y:,.0f}<extra></extra>",
    )
    fig.add_scatter(
        x=years,
        y=[y.discount_factor for y in result.years],
        name="Facteur d'actualisation",
        mode="lines+markers",
        yaxis="y2",
        line={"color": GOLD, "width": 2, "dash": "dot"},
        hovertemplate="%{x}<br>Facteur : %{y:.3f}<extra></extra>",
    )
    fig.add_scatter(
        x=years,
        y=[y.pv_fcff for y in result.years],
        name="FCFF actualisé",
        mode="lines+markers",
        line={"color": CYAN, "width": 2.2},
        hovertemplate="%{x}<br>VA : %{y:,.0f}<extra></extra>",
    )
    layout = _layout(
        "Projection du free cash flow de l'entreprise (FCFF)",
        y_title=f"Millions {result.currency}",
        note=(
            "La valeur terminale représente "
            f"{result.pv_terminal_value / result.enterprise_value:.0%} de la valeur "
            "d'entreprise — surveillez ce ratio."
            if result.enterprise_value
            else None
        ),
    )
    layout["yaxis2"] = {
        "title": {"text": "Facteur", "font": {"size": 12, "color": MUTED}},
        "overlaying": "y",
        "side": "right",
        "showgrid": False,
        "tickfont": {"color": MUTED},
        "range": [0, 1.1],
    }
    fig.update_layout(**layout)
    return fig


def scenario_comparison(results: Sequence[DCFResult], current_price: Optional[float] = None) -> go.Figure:
    if not results:
        return _empty_figure("Aucun scénario calculé.")
    labels = [f"{r.scenario.capitalize()}" for r in results]
    values = [r.value_per_share for r in results]
    colors = [
        SCENARIO_COLORS.get(r.scenario.lower(), BLUE) for r in results
    ]
    fig = go.Figure()
    fig.add_bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"{v:,.2f}" for v in values],
        textposition="outside",
        textfont={"color": TEXT},
        hovertemplate="%{x}<br>Valeur par action : %{y:,.2f}<extra></extra>",
    )
    if current_price:
        fig.add_hline(
            y=current_price,
            line={"color": GOLD, "width": 2, "dash": "dash"},
            annotation_text=f"Cours actuel {current_price:,.2f}",
            annotation_position="top left",
            annotation_font={"color": GOLD_TEXT},
        )
    layout = _layout(
        "Valeur par action par scénario",
        y_title=f"Par action ({results[0].currency})",
        note="Chaque barre est un jeu d'hypothèses explicité ; ce ne sont pas des prévisions.",
        legend=False,
        height=400,
    )
    fig.update_layout(**layout)
    return fig


def sensitivity_heatmap(grid: SensitivityGrid) -> go.Figure:
    values = grid.values
    if not values or not values[0]:
        return _empty_figure("Grille de sensibilité vide.")

    # Hide the invalid (WACC <= g) cells: they are NaN and must not look like 0.
    z = [[None if (v is None or (isinstance(v, float) and pd.isna(v))) else v for v in row] for row in values]
    text = [
        ["" if cell is None else f"{cell:,.1f}" for cell in row] for row in z
    ]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[f"{c:.2%}" for c in grid.col_values],
            y=[f"{r:.2%}" for r in grid.row_values],
            colorscale=[
                [0.0, "#f8b4bf"],
                [0.25, "#ffe2b8"],
                [0.5, "#ffffff"],
                [0.75, "#bfdcfd"],
                [1.0, "#b6ecd0"],
            ],
            text=text,
            texttemplate="%{text}",
            textfont={"size": 11, "color": TEXT},
            colorbar={"title": {"text": "Ach/action", "side": "right"}, "tickfont": {"color": MUTED}},
            hoverongaps=False,
            hovertemplate=(
                f"{grid.row_label}=%{{y}}<br>{grid.col_label}=%{{x}}<br>"
                "Valeur : %{z:,.2f}<extra></extra>"
            ),
        )
    )
    note = (
        "Les cellules vides sont des combinaisons invalides "
        f"({grid.row_label} ≤ {grid.col_label}) ou non calculables."
    )
    layout = _layout(
        f"Sensibilité : {grid.row_label} × {grid.col_label}",
        x_title=grid.col_label,
        y_title=grid.row_label,
        note=note,
        legend=False,
        height=440,
    )
    fig.update_layout(**layout)
    return fig


def tornado_chart(tornado: TornadoResult) -> go.Figure:
    bars = [b for b in tornado.bars if b is not None]
    if not bars:
        return _empty_figure("Analyse de sensibilité unitaire non disponible.")
    # Most impactful first, so the chart reads top-down.
    bars = sorted(
        bars, key=lambda b: abs((b.high_value or 0) - (b.low_value or 0)), reverse=True
    )
    labels = [b.label for b in bars]
    base = tornado.base.value_per_share
    lows = [b.low_value for b in bars]
    highs = [b.high_value for b in bars]

    fig = go.Figure()
    fig.add_bar(
        y=labels,
        x=[base - lo for lo in lows],
        base=[lo for lo in lows],
        orientation="h",
        name="Hypothèse basse",
        marker_color=RED,
        hovertemplate="%{y}<br>Valeur basse : %{base:,.2f}<extra></extra>",
    )
    fig.add_bar(
        y=labels,
        x=[hi - base for hi in highs],
        base=[base for _ in highs],
        orientation="h",
        name="Hypothèse haute",
        marker_color=GREEN,
        hovertemplate="%{y}<br>Valeur haute : %{x:,.2f}<extra></extra>",
    )
    fig.add_vline(
        x=base,
        line={"color": GOLD, "width": 2, "dash": "dash"},
        annotation_text=f"Base {base:,.2f}",
        annotation_font={"color": GOLD_TEXT},
    )
    layout = _layout(
        "Sensibilité de la valeur par action aux hypothèses clés",
        x_title=f"Valeur par action ({tornado.base.currency})",
        note="Les barres vont de l'hypothèse défavorable (rouge) à favorable (verte).",
        height=max(360, 60 + 40 * len(bars)),
        legend=False,
    )
    layout["barmode"] = "overlay"
    fig.update_layout(**layout)
    return fig


# --------------------------------------------------------------------------
# Comparables
# --------------------------------------------------------------------------

def peer_multiples(peers: Sequence[ComparableCompany], metric: str = "ev_ebitda") -> go.Figure:
    """Horizontal bars of a multiple, target peers vs. the company under review."""
    rows = []
    for peer in peers:
        value = getattr(peer, metric, None)
        if value is None:
            continue
        rows.append((peer.name, value))
    if not rows:
        return _empty_figure(
            "Aucun multiple de comparables disponible pour cette métrique."
        )
    rows.sort(key=lambda item: item[1])
    names = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = [GOLD if i == len(rows) - 1 else BLUE for i in range(len(rows))]

    label = {
        "ev_ebitda": "VE/EBITDA",
        "pe": "Cours/Bénéfice (P/E)",
        "pb": "Cours/Valeur comptable (P/B)",
    }.get(metric, metric)

    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{v:,.1f}×" for v in values],
            textposition="outside",
            textfont={"color": TEXT},
            hovertemplate="%{y}<br>" + label + " : %{x:,.1f}×<extra></extra>",
        )
    )
    layout = _layout(
        f"Comparables — {label}",
        x_title=f"{label} (×)",
        note=(
            "Les multiples ne sont comparables qu'à périmètre comptable et devise "
            "identiques ; les pairs d'une autre devise sont écartés par défaut."
        ),
        height=max(320, 90 + 34 * len(names)),
        legend=False,
    )
    fig.update_layout(**layout)
    return fig


def valuation_range(result: MultiplesResult) -> go.Figure:
    implied = list(result.implied or [])
    points = [
        (f"{iv.metric} · {iv.statistic} {iv.multiple:,.1f}×", iv.implied_value_per_share)
        for iv in implied
        if iv.implied_value_per_share is not None
    ]
    if not points:
        return _empty_figure(
            "Aucune valorisation par comparables calculée : les pairs ou les métriques "
            "de la société sont insuffisants."
        )
    points.sort(key=lambda item: item[1])
    labels = [p[0] for p in points]
    values = [p[1] for p in points]
    fig = go.Figure()
    fig.add_scatter(
        x=values,
        y=labels,
        mode="markers",
        marker={"size": 13, "color": BLUE},
        name="Multiple appliqué",
        hovertemplate="%{y}<br>%{x:,.2f} par action<extra></extra>",
    )
    if result.median_implied_value:
        fig.add_vline(
            x=result.median_implied_value,
            line={"color": GOLD, "width": 2, "dash": "dash"},
            annotation_text=f"Médiane {result.median_implied_value:,.2f}",
            annotation_font={"color": GOLD_TEXT},
        )
    if result.target.current_price:
        fig.add_vline(
            x=result.target.current_price,
            line={"color": GREEN, "width": 2, "dash": "dot"},
            annotation_text=f"Cours {result.target.current_price:,.2f}",
            annotation_font={"color": GREEN_TEXT},
        )
    layout = _layout(
        "Valeur par action implicite par multiple",
        x_title=f"Par action ({_currency_text(result.target.currency)})",
        note="Chaque point applique la médiane ou la moyenne des pairs à la métrique de la société.",
        legend=False,
        height=max(320, 130 + 46 * len(labels)),
    )
    fig.update_layout(**layout)
    return fig


# --------------------------------------------------------------------------
# Sector
# --------------------------------------------------------------------------

def cycle_normalisation_chart(cycle: CycleNormalisation) -> go.Figure:
    if not cycle.periods:
        return _empty_figure("Normalisation cyclique non calculable.")
    fig = go.Figure()
    fig.add_scatter(
        x=cycle.periods,
        y=[m * 100 if m is not None else None for m in cycle.margins],
        name="Marge opérationnelle",
        mode="lines+markers",
        connectgaps=False,
        line={"color": BLUE, "width": 2.4},
    )
    if cycle.mid_cycle_margin is not None:
        fig.add_hline(
            y=cycle.mid_cycle_margin * 100,
            line={"color": GOLD, "width": 2, "dash": "dash"},
            annotation_text=f"Marge médiane de cycle {cycle.mid_cycle_margin:.1%}",
            annotation_font={"color": GOLD_TEXT},
        )
    if cycle.latest_margin is not None:
        fig.add_scatter(
            x=[cycle.periods[-1]],
            y=[cycle.latest_margin * 100],
            mode="markers",
            name="Dernière période observée",
            marker={"size": 12, "color": GREEN, "symbol": "diamond"},
        )
    layout = _layout(
        "Marge opérationnelle sur le cycle",
        y_title="%",
        note=(
            "Point haut "
            f"{cycle.peak_margin:.0%} / point bas {cycle.trough_margin:.0%} / "
            f"amplitude {cycle.cycle_spread_pp:.0f} pts"
            if cycle.peak_margin is not None and cycle.trough_margin is not None
            else None
        ),
    )
    fig.update_layout(**layout)
    return fig


def metric_bars(metrics: Sequence[Metric], title: str = "Ratios financiers") -> go.Figure:
    usable = [m for m in metrics if m.value is not None]
    if not usable:
        return _empty_figure("Aucun ratio calculable avec les données disponibles.")

    def _text(metric: Metric) -> str:
        if metric.unit == "percent":
            return f"{metric.value:,.1f} %"
        if metric.unit == "multiple":
            return f"{metric.value:,.1f}×"
        return f"{metric.value:,.2f}"

    fig = go.Figure(
        go.Bar(
            x=[m.label for m in usable],
            y=[m.value for m in usable],
            marker_color=[
                GOLD if m.status == ExtractionStatus.DERIVED else BLUE for m in usable
            ],
            text=[_text(m) for m in usable],
            textposition="outside",
            textfont={"color": TEXT},
            hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>",
        )
    )
    layout = _layout(
        title,
        note=(
            "Ratios calculés par le moteur interne à partir des données chargées ; "
            "un ratio non calculable est absent, jamais nul."
        ),
        legend=False,
        height=max(360, 60 + 26 * len(usable)),
    )
    fig.update_layout(**layout)
    return fig
