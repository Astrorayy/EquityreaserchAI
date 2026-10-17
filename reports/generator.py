"""Report generation (self-contained HTML + lightweight PDF).

The report is the deliverable a human hands to someone else, so the rules are
stricter than in the UI:

* Every figure carries its status (publié / calculé / saisi / estimation) and its
  source identifier.  Nothing is dropped for looking untidy.
* Facts and projections live in **separate** sections and the projection
  sections start with an explicit "hypothèses" block.
* Sections with no data are rendered as "non disponible" with the reason, never
  omitted silently and never filled with a placeholder number.
* The AI narrative, when present, is quoted in a clearly labelled block and
  never mixed into the financial tables.
"""

from __future__ import annotations

import datetime as _dt
import html
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from dashboard.components import (
    DISCLAIMER,
    fmt_date,
    fmt_multiple,
    fmt_number,
    fmt_pct,
    fmt_timestamp,
    line_item_label,
    status_label,
    tier_label,
)
from financial_engine.dcf import DCFResult
from financial_engine.multiples import MultiplesResult
from financial_engine.ratios import Metric
from financial_engine.schemas import (
    CompanyProfile,
    FinancialDataSet,
    LineItem,
    SourceRecord,
)
from financial_engine.sensitivity import SensitivityGrid, TornadoResult
from financial_engine.validation import DataQualityReport, Severity

logger = logging.getLogger(__name__)

#: The ten sections required by the brief, in order.
REPORT_SECTIONS = (
    "resume",
    "identite",
    "donnees_historiques",
    "ratios",
    "hypotheses",
    "dcf",
    "sensibilite",
    "comparables",
    "risques",
    "sources",
)

SECTION_TITLES = {
    "resume": "1. Résumé et périmètre",
    "identite": "2. Identité de la société",
    "donnees_historiques": "3. Données financières historiques",
    "ratios": "4. Ratios et analyse financière",
    "hypotheses": "5. Hypothèses de valorisation",
    "dcf": "6. Valorisation par les flux (DCF)",
    "sensibilite": "7. Analyse de sensibilité",
    "comparables": "8. Sociétés comparables",
    "risques": "9. Risques et limites",
    "sources": "10. Sources, qualité des données et avertissements",
}

KEY_ITEMS = (
    LineItem.REVENUE,
    LineItem.GROSS_PROFIT,
    LineItem.EBIT,
    LineItem.EBITDA,
    LineItem.NET_INCOME,
    LineItem.OPERATING_CASH_FLOW,
    LineItem.CAPITAL_EXPENDITURE,
    LineItem.FREE_CASH_FLOW,
    LineItem.TOTAL_ASSETS,
    LineItem.TOTAL_EQUITY,
    LineItem.TOTAL_DEBT,
    LineItem.CASH_AND_EQUIVALENTS,
    LineItem.DILUTED_SHARES,
)

UNAVAILABLE = "Non disponible — aucune donnée sourcée ne permet de renseigner cette section."


@dataclass
class ReportContext:
    """Everything the report may show. Nothing is inferred inside the template."""

    profile: CompanyProfile
    dataset: FinancialDataSet
    quality: Optional[DataQualityReport] = None
    sources: List[SourceRecord] = field(default_factory=list)
    ratio_metrics: List[Metric] = field(default_factory=list)
    assumptions: Dict[str, object] = field(default_factory=dict)
    assumption_provenance: Dict[str, str] = field(default_factory=dict)
    dcf_results: List[DCFResult] = field(default_factory=list)
    sensitivity: Optional[SensitivityGrid] = None
    tornado: Optional[TornadoResult] = None
    multiples: Optional[MultiplesResult] = None
    current_price: Optional[float] = None
    ai_narrative: Optional[str] = None
    ai_narrative_sources: List[str] = field(default_factory=list)
    extra_notes: List[str] = field(default_factory=list)
    generated_at: _dt.datetime = field(default_factory=lambda: _dt.datetime.now())
    simulated: bool = False
    sections: Sequence[str] = REPORT_SECTIONS

    def source_by_id(self, source_id: Optional[str]) -> Optional[SourceRecord]:
        if not source_id:
            return None
        for source in self.sources:
            if source.source_id == source_id:
                return source
        return None


# --------------------------------------------------------------------------
# HTML helpers
# --------------------------------------------------------------------------

def _esc(text: object) -> str:
    return html.escape("" if text is None else str(text))


def _css() -> str:
    return """
    :root { --ink:#0f172a; --muted:#64748b; --line:#d7dee8; --panel:#f7f9fc;
            --gold:#a16207; --blue:#1d4ed8; --red:#b91c1c; --green:#15803d; }
    * { box-sizing: border-box; }
    body { font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
           color: var(--ink); margin: 0 auto; max-width: 900px; padding: 32px 28px 80px;
           line-height: 1.5; background: #fff; }
    header.cover { border-bottom: 3px solid var(--ink); padding-bottom: 16px; margin-bottom: 26px; }
    header.cover h1 { margin: 0 0 4px; font-size: 27px; }
    header.cover .sub { color: var(--muted); font-size: 13px; }
    h2 { font-size: 18px; margin-top: 34px; padding-bottom: 6px; border-bottom: 1px solid var(--line); }
    h3 { font-size: 15px; margin-top: 22px; color: #1e293b; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0 18px; font-size: 12.5px; }
    th, td { border: 1px solid var(--line); padding: 6px 8px; text-align: right; vertical-align: top; }
    th:first-child, td:first-child { text-align: left; }
    thead th { background: var(--panel); font-weight: 600; }
    tbody tr:nth-child(even) { background: #fbfcfe; }
    .num { font-variant-numeric: tabular-nums; }
    .badge { display:inline-block; padding:1px 7px; border-radius:9px; font-size:10.5px;
             border:1px solid var(--line); color:var(--muted); background:#fff; }
    .badge.reported { color: var(--green); border-color: #bbf7d0; background:#f0fdf4; }
    .badge.derived  { color: var(--blue);  border-color: #bfdbfe; background:#eff6ff; }
    .badge.manual   { color: var(--gold);  border-color: #fde68a; background:#fffbeb; }
    .badge.estimated{ color: #7c3aed;      border-color: #ddd6fe; background:#f5f3ff; }
    .badge.missing  { color: var(--red);   border-color: #fecaca; background:#fef2f2; }
    .note { background: var(--panel); border-left: 4px solid var(--blue);
            padding: 10px 14px; margin: 14px 0; font-size: 12.5px; }
    .warn { background: #fffbeb; border-left: 4px solid var(--gold); }
    .danger { background: #fef2f2; border-left: 4px solid var(--red); }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 14px 0; }
    .kpi { border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; }
    .kpi .label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .03em; }
    .kpi .value { font-size: 19px; font-weight: 600; margin-top: 4px; }
    .kpi .sub { font-size: 11px; color: var(--muted); margin-top: 3px; }
    .ai { background: #f5f3ff; border-left: 4px solid #7c3aed; padding: 12px 15px; font-size: 12.5px; }
    footer { margin-top: 44px; border-top: 1px solid var(--line); padding-top: 12px;
             font-size: 11px; color: var(--muted); }
    @media print { body { max-width: none; padding: 0 6mm; } h2 { page-break-after: avoid; } table { page-break-inside: auto; } }
    """


def _badge(status: object) -> str:
    key = getattr(status, "value", str(status))
    css = {"reported": "reported", "derived": "derived", "manual": "manual",
           "estimated": "estimated", "missing": "missing"}.get(key, "")
    return f'<span class="badge {css}">{_esc(status_label(status) if hasattr(status, "value") else status)}</span>'


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]], numeric_from: int = 1) -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body_rows = []
    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            cls = ' class="num"' if i >= numeric_from else ""
            cells.append(f"<td{cls}>{cell if i >= numeric_from else _esc(cell)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _kpi(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="sub">{_esc(sub)}</div>' if sub else ""
    return (
        f'<div class="kpi"><div class="label">{_esc(label)}</div>'
        f'<div class="value">{value}</div>{sub_html}</div>'
    )


def _currency_text(currency: object) -> str:
    """Currency enums stringify as ``Currency.USD``; always show the code."""
    return getattr(currency, "value", None) or str(currency)


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------

def _section_resume(ctx: ReportContext) -> str:
    profile = ctx.profile
    latest = ctx.dataset.latest_period_label()
    parts = [f"<h2>{_esc(SECTION_TITLES['resume'])}</h2>"]
    if ctx.simulated:
        parts.append(
            '<div class="note danger"><strong>Jeu de données SIMULÉ.</strong> Ce document '
            "est généré à partir de données synthétiques destinées aux tests. Il ne décrit "
            "aucune société réelle et ne doit jamais être diffusé comme une analyse.</div>"
        )
    base = next((r for r in ctx.dcf_results if r.scenario.lower() == "base"), None)
    if base is None and ctx.dcf_results:
        base = ctx.dcf_results[0]

    kpis = [
        _kpi("Société", _esc(profile.name), f"{profile.ticker} · {profile.exchange}"),
        _kpi("Devise de reporting", profile.currency.value, "Valeurs en millions"),
        _kpi("Dernière période", _esc(latest or "—"), "États financiers annuels"),
    ]
    if base is not None:
        kpis.append(_kpi("Valeur DCF (base)", f"{fmt_number(base.value_per_share, 2)} {base.currency}",
                         f"WACC {fmt_pct(base.wacc)} · g {fmt_pct(base.terminal_growth)}"))
    if ctx.current_price:
        kpis.append(_kpi("Cours de référence", f"{fmt_number(ctx.current_price, 2)} {_currency_text(profile.currency)}",
                         "Saisi ou récupéré — vérifier la date"))
    parts.append(f'<div class="grid">{"".join(kpis)}</div>')

    coverage = _coverage_summary(ctx)
    parts.append(f'<div class="note">{coverage}</div>')
    if ctx.quality is not None and ctx.quality.errors:
        parts.append(
            '<div class="note danger">Contrôles de cohérence : '
            + "; ".join(_esc(i.message) for i in ctx.quality.errors[:6])
            + "</div>"
        )
    parts.append(f'<div class="note warn">{_esc(DISCLAIMER.replace("**", ""))}</div>')
    return "".join(parts)


def _coverage_summary(ctx: ReportContext) -> str:
    periods = ctx.dataset.periods()
    if not periods:
        return "Aucune donnée financière n'est chargée pour cette société : les sections 3, 4 et 6 resteront vides."
    total = len(KEY_ITEMS) * len(periods)
    present = 0
    for item in KEY_ITEMS:
        for period in periods:
            observation = ctx.dataset.get(item, period)
            if observation is not None and not observation.is_missing:
                present += 1
    pct = present / total if total else 0
    return (
        f"Couverture des postes clés : {present}/{total} cellules renseignées "
        f"({pct:.0%}) sur {len(periods)} exercice(s). Les cellules absentes restent "
        "vides : elles ne sont jamais remplacées par zéro."
    )


def _section_identite(ctx: ReportContext) -> str:
    p = ctx.profile
    rows = [
        ("Nom", _esc(p.name)),
        ("Nom court", _esc(p.short_name)),
        ("Ticker", _esc(p.ticker)),
        ("Place", _esc(p.exchange)),
        ("Pays", _esc(p.country)),
        ("Secteur", _esc(p.sector or "—")),
        ("Industrie", _esc(p.industry or "—")),
        ("Devise de reporting", p.currency.value),
        ("Clôture d'exercice", _esc(p.fiscal_year_end or "—")),
        ("ISIN", _esc(p.isin or "—")),
        ("CIK (SEC)", _esc(p.sec_cik or "—")),
        ("Site", _esc(p.website or "—")),
        ("Relations investisseurs", _esc(p.ir_url or "—")),
        ("Périmètre historique", _esc(p.identity_notes or "—")),
    ]
    if p.history_cutoff:
        rows.append(("Historique limité à partir du", fmt_date(p.history_cutoff)))
        if p.history_cutoff_note:
            rows.append(("Motif", _esc(p.history_cutoff_note)))
    html_rows = [(label, value) for label, value in rows]
    table = _table(["Caractéristique", "Valeur"], html_rows, numeric_from=99)
    body = f"<h2>{_esc(SECTION_TITLES['identite'])}</h2>{table}"
    if p.description:
        body += f'<div class="note">{_esc(p.description)}</div>'
    return body


def _section_historique(ctx: ReportContext) -> str:
    periods = ctx.dataset.periods()
    body = [f"<h2>{_esc(SECTION_TITLES['donnees_historiques'])}</h2>"]
    if not periods:
        return "".join(body) + f'<div class="note warn">{_esc(UNAVAILABLE)}</div>'

    body.append(
        '<div class="note">Valeurs en millions de la devise de reporting. '
        "Le statut indique l'origine de chaque chiffre : "
        + " · ".join(
            [
                _badge(s)
                for s in (
                    "reported",
                    "derived",
                    "manual",
                    "estimated",
                )
            ]
        )
        + " — un « absent » n'est jamais un zéro.</div>"
    )
    headers = ["Poste", *periods, "Statut (dernière période)", "Source"]
    rows: List[List[str]] = []
    for item in KEY_ITEMS:
        cells: List[str] = [line_item_label(item)]
        last_status = "missing"
        last_source = "—"
        for period in periods:
            observation = ctx.dataset.get(item, period)
            if observation is None or observation.is_missing:
                cells.append("—")
                continue
            cells.append(fmt_number(observation.value))
            last_status = observation.status.value if hasattr(observation.status, "value") else str(observation.status)
            if observation.source_id:
                source = ctx.source_by_id(observation.source_id)
                last_source = source.name if source else observation.source_id
        cells.append(_badge(last_status))
        cells.append(_esc(last_source))
        rows.append(cells)
    body.append(_table(headers, rows, numeric_from=1))
    return "".join(body)


def _metric_value_text(metric: Metric) -> str:
    """Render a ratio value with the right unit (values arrive already scaled)."""
    if metric.value is None:
        return "—"
    if metric.unit == "percent":
        return f"{metric.value:,.1f} %".replace(",", " ")
    if metric.unit == "multiple":
        return f"{metric.value:,.2f}×".replace(",", " ")
    if metric.unit == "currency_per_share":
        return f"{metric.value:,.2f}".replace(",", " ")
    return f"{metric.value:,.0f}".replace(",", " ")


def _metric_inputs_text(metric: Metric, limit: int = 4) -> str:
    if not metric.inputs:
        return "—"
    parts = []
    for key, value in list(metric.inputs.items())[:limit]:
        shown = "—" if value is None else f"{value:,.0f}".replace(",", " ")
        parts.append(f"{key}={shown}")
    return ", ".join(parts)


def _section_ratios(ctx: ReportContext) -> str:
    body = [f"<h2>{_esc(SECTION_TITLES['ratios'])}</h2>"]
    if not ctx.ratio_metrics:
        body.append(f'<div class="note warn">{_esc(UNAVAILABLE)}</div>')
        return "".join(body)
    rows = []
    for metric in ctx.ratio_metrics:
        rows.append(
            [
                metric.label,
                _metric_value_text(metric),
                _esc(metric.formula or "—"),
                _esc(_metric_inputs_text(metric)),
                _badge(metric.status),
            ]
        )
    body.append(_table(["Ratio", "Valeur", "Formule", "Données d'entrée", "Statut"], rows, numeric_from=1))
    failed = [m for m in ctx.ratio_metrics if m.value is None]
    if failed:
        body.append("<h3>Ratios non calculables</h3><ul>")
        for metric in failed:
            body.append(f"<li>{_esc(metric.label)} — {_esc(metric.reason or 'donnée absente')}</li>")
        body.append("</ul>")
    return "".join(body)


_PERCENT_HINTS = ("wacc", "growth", "margin", "rate", "pct", "share_of", "ratio_of")


def _assumption_text(key: str, value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return _esc(value)
    lowered = key.lower()
    if any(hint in lowered for hint in ("wacc", "growth", "margin", "tax", "rate")):
        return f"{value:.2%}"
    return f"{value:,.2f}".replace(",", " ")


def _section_hypotheses(ctx: ReportContext) -> str:
    body = [f"<h2>{_esc(SECTION_TITLES['hypotheses'])}</h2>"]
    if not ctx.assumptions:
        body.append(f'<div class="note warn">{_esc(UNAVAILABLE)}</div>')
        return "".join(body)
    body.append(
        '<div class="note warn">Ces valeurs sont des <strong>choix d\'analyste</strong>, '
        "pas des données publiées. Elles sont modifiables et doivent être justifiées "
        "avant toute utilisation externe.</div>"
    )
    rows = [
        [key, _assumption_text(key, value), _esc(ctx.assumption_provenance.get(key, "hypothèse analyste"))]
        for key, value in ctx.assumptions.items()
    ]
    body.append(_table(["Hypothèse", "Valeur", "Origine"], rows, numeric_from=1))
    return "".join(body)


def _dcf_years_table(result: DCFResult) -> str:
    headers = ["Exercice", "CA", "EBIT", "NOPAT", "Amort.", "Capex", "Δ BFR", "FCFF", "Facteur", "VA"]
    rows = []
    for year in result.years:
        rows.append(
            [
                str(year.fiscal_year),
                fmt_number(year.revenue),
                fmt_number(year.ebit),
                fmt_number(year.nopat),
                fmt_number(year.da),
                fmt_number(year.capex),
                fmt_number(year.change_nwc),
                fmt_number(year.fcff),
                f"{year.discount_factor:,.3f}",
                fmt_number(year.pv_fcff),
            ]
        )
    return _table(headers, rows, numeric_from=1)


def _section_dcf(ctx: ReportContext) -> str:
    body = [f"<h2>{_esc(SECTION_TITLES['dcf'])}</h2>"]
    if not ctx.dcf_results:
        body.append(
            '<div class="note warn">Aucune valorisation DCF n\'a encore été exécutée pour '
            "cette société. Lancez le calcul dans la page Valorisation : les hypothèses y "
            "sont saisies explicitement.</div>"
        )
        return "".join(body)
    body.append(
        '<div class="note">Méthode : FCFF (free cash flow de l\'entreprise) actualisé au WACC, '
        "valeur terminale par croissance perpétuelle avec contrainte WACC &gt; g. "
        "La dette nette est retirée <em>une seule fois</em>, au niveau de la valeur d'entreprise.</div>"
    )
    headers = ["Scénario", "WACC", "g", "VE", "Dette nette", "Valeur des fonds propres", "Par action"]
    rows = []
    for result in ctx.dcf_results:
        rows.append(
            [
                _esc(result.scenario.capitalize()),
                fmt_pct(result.wacc),
                fmt_pct(result.terminal_growth),
                fmt_number(result.enterprise_value),
                fmt_number(result.net_debt),
                fmt_number(result.equity_value),
                f"<strong>{fmt_number(result.value_per_share, 2)}</strong>",
            ]
        )
    body.append(_table(headers, rows, numeric_from=1))

    base = next((r for r in ctx.dcf_results if r.scenario.lower() == "base"), ctx.dcf_results[0])
    body.append(f"<h3>Détail du scénario { _esc(base.scenario) }</h3>")
    body.append(_dcf_years_table(base))
    body.append(
        '<div class="note">'
        f"VA des FCFF explicites : {fmt_number(base.pv_explicit)} · "
        f"VA de la valeur terminale : {fmt_number(base.pv_terminal_value)} "
        f"({(base.pv_terminal_value / base.enterprise_value):.0%} de la VE) · "
        f"Valeur d'entreprise : {fmt_number(base.enterprise_value)} · "
        f"Fonds propres : {fmt_number(base.equity_value)} · "
        f"Actions diluées : {fmt_number(base.shares_outstanding, 1)} millions."
        "</div>"
    )
    warnings = [w for r in ctx.dcf_results for w in r.warnings]
    if warnings:
        body.append(
            '<div class="note warn"><strong>Avertissements du calcul :</strong><ul>'
            + "".join(f"<li>{_esc(w)}</li>" for w in warnings)
            + "</ul></div>"
        )
    return "".join(body)


def _section_sensibilite(ctx: ReportContext) -> str:
    body = [f"<h2>{_esc(SECTION_TITLES['sensibilite'])}</h2>"]
    if ctx.sensitivity is None:
        body.append(
            '<div class="note warn">Sensibilité non calculée. Plutôt qu\'un graphique, le '
            "rapport HTML imprimable reproduit ci-dessous la grille de valeurs si elle est "
            "disponible.</div>"
        )
        return "".join(body)
    grid = ctx.sensitivity
    headers = [f"{grid.row_label} \\ {grid.col_label}"] + [f"{c:.2%}" for c in grid.col_values]
    rows = []
    for row_value, line in zip(grid.row_values, grid.values):
        cells = [f"{row_value:.2%}"]
        for value in line:
            if value is None or (isinstance(value, float) and value != value):
                cells.append("n.d.")
            else:
                cells.append(fmt_number(value, 2))
        rows.append(cells)
    body.append(_table(headers, rows, numeric_from=1))
    body.append(
        '<div class="note">« n.d. » signale une combinaison invalide (WACC ≤ g) ou non '
        "calculable. Les cellules ne sont jamais extrapolées.</div>"
    )
    if ctx.tornado is not None and ctx.tornado.bars:
        rows = [
            [
                bar.label,
                fmt_number(bar.low_value, 2),
                fmt_number(bar.base_value, 2),
                fmt_number(bar.high_value, 2),
                fmt_number(abs((bar.high_value or 0) - (bar.low_value or 0)), 2),
            ]
            for bar in sorted(
                ctx.tornado.bars,
                key=lambda b: abs((b.high_value or 0) - (b.low_value or 0)),
                reverse=True,
            )
        ]
        body.append("<h3>Hypothèses les plus influentes</h3>")
        body.append(_table(["Hypothèse", "Valeur défavorable", "Base", "Valeur favorable", "Amplitude"], rows, numeric_from=1))
    return "".join(body)


def _section_comparables(ctx: ReportContext) -> str:
    body = [f"<h2>{_esc(SECTION_TITLES['comparables'])}</h2>"]
    result = ctx.multiples
    if result is None:
        body.append(
            '<div class="note warn">Analyse de comparables non disponible : ajoutez des pairs '
            "et lancez le calcul depuis la page Comparables.</div>"
        )
        return "".join(body)
    if result.peers_included:
        headers = ["Pair", "Devise", "P/E", "VE/EBITDA", "P/B", "Capitalisation", "Note"]
        rows = [
            [
                peer.name,
                peer.currency.value,
                fmt_multiple(peer.pe, 1),
                fmt_multiple(peer.ev_ebitda, 1),
                fmt_multiple(peer.pb, 1),
                fmt_number(peer.market_cap),
                _esc(peer.notes or "—"),
            ]
            for peer in result.peers_included
        ]
        body.append(_table(headers, rows, numeric_from=1))
    if result.stats:
        rows = []
        for stats in result.stats.values():
            rows.append(
                [
                    stats.label,
                    fmt_multiple(stats.minimum, 1),
                    fmt_multiple(stats.median, 1),
                    fmt_multiple(stats.mean, 1),
                    fmt_multiple(stats.maximum, 1),
                    str(stats.count),
                    "oui" if stats.sufficient else "non — trop peu de pairs",
                ]
            )
        if rows:
            body.append("<h3>Statistiques des multiples retenus</h3>")
            body.append(
                _table(
                    ["Multiple", "Min", "Médiane", "Moyenne", "Max", "N pairs", "Robustesse"],
                    rows,
                    numeric_from=1,
                )
            )
    if result.implied:
        rows = []
        for implied in result.implied:
            rows.append(
                [
                    f"{implied.metric} ({implied.statistic} = {fmt_multiple(implied.multiple, 1)})",
                    fmt_number(implied.implied_enterprise_value),
                    fmt_number(implied.implied_equity_value),
                    fmt_number(implied.implied_value_per_share, 2),
                    _esc(implied.basis or "—"),
                ]
            )
        body.append("<h3>Valeur implicite par action</h3>")
        body.append(
            _table(
                ["Base de multiple", "VE implicite", "Fonds propres implicites", "Par action", "Assiette"],
                rows,
                numeric_from=1,
            )
        )
        if result.valuation_range:
            body.append(
                '<div class="note">Fourchette implicite : '
                f"{fmt_number(result.valuation_range[0], 2)} – {fmt_number(result.valuation_range[1], 2)} "
                f"{_currency_text(result.target.currency)} par action · médiane "
                f"{fmt_number(result.median_implied_value, 2)}.</div>"
            )
    if result.peers_excluded:
        body.append(
            '<div class="note warn">Pairs écartés : '
            + "; ".join(
                f"{_esc(peer.name)} — {_esc(reason)}" for peer, reason in result.peers_excluded
            )
            + ". Les multiples ne sont pas comparables entre devises ou périmètres "
            "comptables différents.</div>"
        )
    if result.method_note:
        body.append(f'<div class="note">{_esc(result.method_note)}</div>')
    if result.warnings:
        body.append(
            '<div class="note warn"><ul>'
            + "".join(f"<li>{_esc(w)}</li>" for w in result.warnings)
            + "</ul></div>"
        )
    return "".join(body)


def _section_risques(ctx: ReportContext, sector_risks: Sequence[str] = (), method_notes: Sequence[str] = ()) -> str:
    body = [f"<h2>{_esc(SECTION_TITLES['risques'])}</h2>"]
    items: List[str] = list(sector_risks)
    items.extend(method_notes)
    items.extend(ctx.extra_notes)
    items.extend(
        [
            "Les projections reposent sur des hypothèses d'analyste : une variation du WACC "
            "ou de la croissance de 100 points de base modifie sensiblement la valeur par action "
            "(voir la grille de sensibilité).",
            "Les données des états financiers peuvent être retraitées par l'émetteur ; "
            "comparez toujours au document d'origine (voir section 10).",
        ]
    )
    if ctx.dataset.currency.value == "MAD" and any(
        s.currency and s.currency.value != "MAD" for s in ctx.sources
    ):
        items.append(
            "Des documents en devise étrangère coexistent avec ceux en dirham : les montants "
            "restent dans leur devise d'origine et ne sont pas convertis automatiquement."
        )
    body.append("<ul>" + "".join(f"<li>{_esc(i)}</li>" for i in items) + "</ul>")
    return "".join(body)


def _section_sources(ctx: ReportContext) -> str:
    body = [f"<h2>{_esc(SECTION_TITLES['sources'])}</h2>"]
    if not ctx.sources:
        body.append(
            '<div class="note warn">Aucun document source n\'est enregistré : les chiffres '
            "affichés ne sont pas traçables. Chargez les états financiers avant de diffuser "
            "ce rapport.</div>"
        )
    else:
        rows = []
        for source in ctx.sources:
            url = (
                f'<a href="{_esc(source.url)}">{_esc(source.url)}</a>' if source.url else "—"
            )
            rows.append(
                [
                    source.source_id,
                    source.name,
                    source.publisher or "—",
                    tier_label(source.tier) if source.tier else "—",
                    fmt_date(source.publication_date),
                    fmt_timestamp(source.retrieved_at),
                    url,
                ]
            )
        body.append(
            _table(
                ["Id", "Document", "Éditeur", "Niveau", "Publié le", "Récupéré le", "Lien"],
                rows,
                numeric_from=99,
            )
        )
    if ctx.quality is not None:
        body.append("<h3>Contrôles automatiques</h3>")
        if not ctx.quality.issues:
            body.append('<div class="note">Aucune anomalie détectée par les contrôles automatiques.</div>')
        else:
            rows = [
                [issue.severity.value, issue.code, issue.message]
                for issue in sorted(ctx.quality.issues, key=lambda i: (i.severity != Severity.ERROR, i.severity != Severity.WARNING))
            ]
            body.append(_table(["Sévérité", "Code", "Message"], rows, numeric_from=99))
    body.append(f'<div class="note warn">{_esc(DISCLAIMER.replace("**", ""))}</div>')
    if ctx.ai_narrative:
        body.append("<h3>Note générée par l'assistant IA (à revérifier)</h3>")
        body.append(f'<div class="ai">{_esc(ctx.ai_narrative)}</div>')
        body.append(
            '<div class="note warn">Cette note est un texte généré automatiquement à partir '
            "des données ci-dessus. Elle ne constitue pas une donnée financière et doit être "
            "revérifiée avant diffusion.</div>"
        )
    return "".join(body)


# --------------------------------------------------------------------------
# Assemblies
# --------------------------------------------------------------------------

def build_html(ctx: ReportContext, sector_risks: Sequence[str] = (), method_notes: Sequence[str] = ()) -> str:
    """Render the complete report as a single self-contained HTML document."""
    profile = ctx.profile
    header = f"""
    <header class="cover">
      <h1>{_esc(profile.name)} — Note de recherche</h1>
      <div class="sub">
        {_esc(profile.ticker)} · {_esc(profile.exchange)} · {_esc(profile.country)} ·
        devise {profile.currency.value} · valeurs en millions ·
        généré le {fmt_timestamp(ctx.generated_at)}
        {' · <strong>DONNÉES SIMULÉES</strong>' if ctx.simulated else ''}
      </div>
    </header>
    """
    renderers = {
        "resume": lambda: _section_resume(ctx),
        "identite": lambda: _section_identite(ctx),
        "donnees_historiques": lambda: _section_historique(ctx),
        "ratios": lambda: _section_ratios(ctx),
        "hypotheses": lambda: _section_hypotheses(ctx),
        "dcf": lambda: _section_dcf(ctx),
        "sensibilite": lambda: _section_sensibilite(ctx),
        "comparables": lambda: _section_comparables(ctx),
        "risques": lambda: _section_risques(ctx, sector_risks, method_notes),
        "sources": lambda: _section_sources(ctx),
    }
    body = "".join(renderers[name]() for name in ctx.sections if name in renderers)
    footer = (
        '<footer>Document produit par Equity Research AI — plateforme locale d\'analyse. '
        "Les chiffres publiés proviennent des documents listés en section 10 ; les projections "
        "sont des hypothèses. Ce document n'est pas un conseil en investissement.</footer>"
    )
    title = f"Note de recherche — {profile.name}"
    return (
        "<!DOCTYPE html>\n<html lang=\"fr\">\n<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>{_esc(title)}</title>\n"
        f"<style>{_css()}</style>\n</head>\n<body>\n"
        f"{header}{body}{footer}\n</body>\n</html>\n"
    )


def _plain_text(ctx: ReportContext) -> List[str]:
    """Line-by-line plain text for the PDF (and for console inspection)."""
    profile = ctx.profile
    lines: List[str] = [
        f"{profile.name} — Note de recherche",
        f"{profile.ticker} | {profile.exchange} | {profile.country} | devise {profile.currency.value}",
        f"Valeurs en millions de la devise de reporting. Généré le {fmt_timestamp(ctx.generated_at)}",
        "",
    ]
    if ctx.simulated:
        lines += ["*** DONNÉES SIMULÉES — document de test, aucune société réelle ***", ""]

    lines.append(SECTION_TITLES["resume"])
    latest = ctx.dataset.latest_period_label()
    lines.append(f"- Dernière période chargée : {latest or 'aucune'}")
    lines.append(f"- {_coverage_summary(ctx)}")
    if ctx.current_price:
        lines.append(f"- Cours de référence : {fmt_number(ctx.current_price, 2)} {_currency_text(profile.currency)}")
    lines.append(f"- {DISCLAIMER.replace('**', '')}")
    lines.append("")

    lines.append(SECTION_TITLES["identite"])
    for label, value in (
        ("Secteur", profile.sector or "—"),
        ("Industrie", profile.industry or "—"),
        ("Clôture d'exercice", profile.fiscal_year_end or "—"),
        ("ISIN", profile.isin or "—"),
        ("CIK", profile.sec_cik or "—"),
        ("Site", profile.website or "—"),
        ("Note de périmètre", profile.identity_notes or "—"),
    ):
        lines.append(f"- {label} : {value}")
    if profile.history_cutoff:
        lines.append(
            f"- Historique limité à partir du {fmt_date(profile.history_cutoff)} : "
            f"{profile.history_cutoff_note or 'périmètre antérieur non comparables'}"
        )
    lines.append("")

    periods = ctx.dataset.periods()
    lines.append(SECTION_TITLES["donnees_historiques"])
    if not periods:
        lines.append("- " + UNAVAILABLE)
    else:
        for item in KEY_ITEMS:
            cells = []
            for period in periods:
                observation = ctx.dataset.get(item, period)
                if observation is None or observation.is_missing:
                    cells.append(f"{period}: —")
                else:
                    cells.append(f"{period}: {fmt_number(observation.value)} ({status_label(observation.status)})")
            lines.append("- " + line_item_label(item) + " → " + " | ".join(cells))
    lines.append("")

    lines.append(SECTION_TITLES["ratios"])
    if not ctx.ratio_metrics:
        lines.append("- " + UNAVAILABLE)
    for metric in ctx.ratio_metrics:
        if metric.value is None:
            lines.append(f"- {metric.label} : non calculable ({metric.reason or 'donnée absente'})")
        else:
            detail = " / ".join(filter(None, [metric.formula, _metric_inputs_text(metric)]))
            lines.append(f"- {metric.label} : {_metric_value_text(metric)}  [{detail}]")
    lines.append("")

    lines.append(SECTION_TITLES["hypotheses"])
    if not ctx.assumptions:
        lines.append("- " + UNAVAILABLE)
    for key, value in ctx.assumptions.items():
        shown = f"{value:,.4f}" if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)
        lines.append(f"- {key} = {shown}  ({ctx.assumption_provenance.get(key, 'hypothèse analyste')})")
    lines.append("")

    lines.append(SECTION_TITLES["dcf"])
    if not ctx.dcf_results:
        lines.append("- Aucune valorisation DCF exécutée.")
    for result in ctx.dcf_results:
        lines.append(
            f"- {result.scenario}: WACC {result.wacc:.2%}, g {result.terminal_growth:.2%}, "
            f"VE {fmt_number(result.enterprise_value)}, dette nette {fmt_number(result.net_debt)}, "
            f"fonds propres {fmt_number(result.equity_value)}, "
            f"par action {fmt_number(result.value_per_share, 2)} {result.currency}"
        )
        lines.append(
            f"    VA FCFF explicites {fmt_number(result.pv_explicit)} | "
            f"VA valeur terminale {fmt_number(result.pv_terminal_value)}"
        )
        for warning in result.warnings:
            lines.append(f"    ! {warning}")
    lines.append("")

    lines.append(SECTION_TITLES["sensibilite"])
    if ctx.sensitivity is None:
        lines.append("- Grille de sensibilité non calculée.")
    else:
        grid = ctx.sensitivity
        lines.append(f"- {grid.row_label} (lignes) x {grid.col_label} (colonnes), valeurs par action")
        lines.append("    " + " ".join(f"{c:.2%}".rjust(10) for c in grid.col_values))
        for row_value, line in zip(grid.row_values, grid.values):
            cells = []
            for value in line:
                if value is None or (isinstance(value, float) and value != value):
                    cells.append("n.d.".rjust(10))
                else:
                    cells.append(f"{value:,.2f}".rjust(10))
            lines.append(f"    {row_value:.2%} " + "".join(cells))
    if ctx.tornado is not None and ctx.tornado.bars:
        for bar in sorted(ctx.tornado.bars, key=lambda b: abs((b.high_value or 0) - (b.low_value or 0)), reverse=True):
            lines.append(
                f"- {bar.label}: {fmt_number(bar.low_value, 2)} → {fmt_number(bar.high_value, 2)} "
                f"(base {fmt_number(bar.base_value, 2)})"
            )
    lines.append("")

    lines.append(SECTION_TITLES["comparables"])
    if ctx.multiples is None:
        lines.append("- Analyse de comparables non disponible.")
    else:
        for peer in ctx.multiples.peers_included:
            lines.append(
                f"- {peer.name} ({_currency_text(peer.currency)}) : P/E {fmt_multiple(peer.pe)}, "
                f"VE/EBITDA {fmt_multiple(peer.ev_ebitda)}, P/B {fmt_multiple(peer.pb)}"
            )
        for stats in ctx.multiples.stats.values():
            lines.append(
                f"- {stats.label} : min {fmt_multiple(stats.minimum)}, "
                f"médiane {fmt_multiple(stats.median)}, moyenne {fmt_multiple(stats.mean)}, "
                f"max {fmt_multiple(stats.maximum)} sur {stats.count} pair(s)"
            )
        if ctx.multiples.implied:
            lines.append("- Valeur implicite par action :")
            for implied in ctx.multiples.implied:
                lines.append(
                    f"    {implied.metric} ({implied.statistic} = {fmt_multiple(implied.multiple)}) : "
                    f"{fmt_number(implied.implied_value_per_share, 2)}"
                )
        if ctx.multiples.peers_excluded:
            lines.append(
                "- Pairs écartés : "
                + "; ".join(f"{peer.name} ({reason})" for peer, reason in ctx.multiples.peers_excluded)
            )
    lines.append("")

    lines.append(SECTION_TITLES["risques"])
    for note in ctx.extra_notes:
        lines.append(f"- {note}")
    lines.append("- Sensibilité forte aux hypothèses de WACC et de croissance terminale.")
    lines.append("- Données historiques susceptibles d'être retraitées par l'émetteur.")
    lines.append("")

    lines.append(SECTION_TITLES["sources"])
    if not ctx.sources:
        lines.append("- Aucun document source enregistré : chiffres non traçables.")
    for source in ctx.sources:
        lines.append(
            f"- [{source.source_id}] {source.name} — {source.publisher or 'éditeur inconnu'} — "
            f"publié le {fmt_date(source.publication_date)} — récupéré le {fmt_timestamp(source.retrieved_at)} — "
            f"{source.url or 'URL absente'}"
        )
    if ctx.quality is not None and ctx.quality.issues:
        lines.append("- Contrôles automatiques :")
        for issue in ctx.quality.issues:
            lines.append(f"    [{issue.severity.value}] {issue.code}: {issue.message}")
    lines.append("")
    lines.append(DISCLAIMER.replace("**", ""))
    return lines


# Windows ships these TTF files; the first one found is used so accented French
# text and the × / — glyphs render correctly in the PDF.
_FONT_CANDIDATES = (
    (Path(r"C:\Windows\Fonts\segoeui.ttf"), Path(r"C:\Windows\Fonts\segoeuib.ttf")),
    (Path(r"C:\Windows\Fonts\arial.ttf"), Path(r"C:\Windows\Fonts\arialbd.ttf")),
    (Path(r"C:\Windows\Fonts\calibri.ttf"), Path(r"C:\Windows\Fonts\calibrib.ttf")),
    (Path(r"C:\Windows\Fonts\verdana.ttf"), Path(r"C:\Windows\Fonts\verdanab.ttf")),
)

_LATIN1_SUBSTITUTIONS = {
    "×": "x", "—": "-", "–": "-", "’": "'", "‘": "'", "“": '"', "”": '"',
    "…": "...", "≤": "<=", "≥": ">=", "≈": "~", "€": "EUR", "→": "->",
    "Δ": "D", "™": "", "•": "-", "\u202f": " ", "\u00a0": " ",
}


def _latin1(text: str) -> str:
    for source, target in _LATIN1_SUBSTITUTIONS.items():
        text = text.replace(source, target)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def build_pdf(ctx: ReportContext, path: Path) -> Path:
    """Write a compact PDF version of the report. Returns the path written."""
    from fpdf import FPDF

    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()

    font_family = "Helvetica"
    bold_family = "Helvetica"
    for regular, bold in _FONT_CANDIDATES:
        if regular.exists():
            try:
                pdf.add_font("Report", "", str(regular))
                if bold.exists():
                    pdf.add_font("Report", "B", str(bold))
                font_family = "Report"
                bold_family = "Report" if bold.exists() else "Report"
                break
            except Exception as exc:  # pragma: no cover - font dependent
                logger.warning("Police %s illisible (%s) : repli sur Helvetica.", regular, exc)

    def write(text: str, size: float = 9.0, bold: bool = False) -> None:
        style = "B" if bold else ""
        family = bold_family if bold else font_family
        pdf.set_font(family, style, size)
        safe = text if font_family == "Report" else _latin1(text)
        # ``multi_cell`` leaves the cursor at the right margin by default, which
        # makes the next call fail with "not enough horizontal space"; force the
        # cursor back to the left margin and let the cell consume the full width.
        pdf.set_x(pdf.l_margin)
        try:
            pdf.multi_cell(
                w=pdf.epw,
                h=4.6,
                text=safe,
                new_x="LMARGIN",
                new_y="NEXT",
            )
        except Exception:  # pragma: no cover - defensive
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(
                w=pdf.epw,
                h=4.6,
                text=_latin1(text),
                new_x="LMARGIN",
                new_y="NEXT",
            )

    for index, line in enumerate(_plain_text(ctx)):
        if index == 0:
            write(line, size=15, bold=True)
        elif line.startswith(("1. ", "2. ", "3. ", "4. ", "5. ", "6. ", "7. ", "8. ", "9. ", "10.")):
            pdf.ln(2.5)
            write(line, size=11.5, bold=True)
        else:
            write(line)
    pdf.output(str(path))
    return path


def report_filename(profile: CompanyProfile, generated_at: _dt.datetime, extension: str) -> str:
    stamp = generated_at.strftime("%Y%m%d-%H%M")
    safe_ticker = "".join(ch for ch in profile.ticker if ch.isalnum()) or "company"
    return f"note-recherche-{safe_ticker}-{stamp}.{extension}"


def generate_report(
    ctx: ReportContext,
    reports_dir: Path,
    formats: Sequence[str] = ("html", "pdf"),
    sector_risks: Sequence[str] = (),
    method_notes: Sequence[str] = (),
) -> Dict[str, Path]:
    """Write the requested formats and return the paths actually written."""
    written: Dict[str, Path] = {}
    for fmt in formats:
        fmt = fmt.lower()
        if fmt == "html":
            target = reports_dir / report_filename(ctx.profile, ctx.generated_at, "html")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(build_html(ctx, sector_risks, method_notes), encoding="utf-8")
            written["html"] = target
        elif fmt == "pdf":
            target = reports_dir / report_filename(ctx.profile, ctx.generated_at, "pdf")
            try:
                written["pdf"] = build_pdf(ctx, target)
            except Exception as exc:
                logger.error("Génération PDF impossible : %s", exc)
                raise
        else:
            raise ValueError(f"Format de rapport non pris en charge : {fmt!r} (attendu : html, pdf)")
    return written
