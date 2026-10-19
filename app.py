"""Equity Research AI — Streamlit terminal.

Entry point of the application. Eight pages, French UI by default, every figure
carrying its source and status. The application is designed to be fully usable
**without any API key**: the AI assistant is the only feature that degrades, and
it says so explicitly.

Run with::

    .\\.venv\\Scripts\\python.exe -m streamlit run app.py
"""

from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

from config import get_settings
from dashboard import charts, components as C
from data_sources.company_registry import (
    get_profile,
    list_profiles,
    search as search_profiles,
)
from financial_engine.schemas import (
    ComparableCompany,
    CompanyProfile,
    Currency,
    ExtractionStatus,
    FinancialDataSet,
    LineItem,
    PeriodType,
    Scale,
)
from financial_engine.validation import Severity
from i18n import SUPPORTED as SUPPORTED_LANGS
from i18n import get_lang, set_lang, t
from storage.database import Database
from storage import repository as repo
import workflows

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("app")

# Public search backends log one INFO line per HTTP response; keep the terminal
# readable while preserving warnings and errors.
for _noisy in ("primp", "ddgs", "httpx"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

#: Numeric multipliers for the reporting scales a document can use.  Kept here
#: (and not inferred) because reading "1 234" as millions instead of thousands is
#: a factor-1 000 error nobody notices in a finished table.
SCALE_FACTORS = {
    Scale.UNITS.value: 1.0,
    Scale.THOUSANDS.value: 1_000.0,
    Scale.MILLIONS.value: 1_000_000.0,
    Scale.BILLIONS.value: 1_000_000_000.0,
}

PAGES = (
    "dashboard",
    "research",
    "statements",
    "valuation",
    "comparables",
    "assistant",
    "sources",
    "settings",
)
PAGE_ICONS = {
    "dashboard": "▦",
    "research": "◱",
    "statements": "▤",
    "valuation": "∑",
    "comparables": "⚖",
    "assistant": "✦",
    "sources": "⛁",
    "settings": "⚙",
}


# ==========================================================================
# Bootstrap
# ==========================================================================

st.set_page_config(
    page_title="Equity Research AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px; }
    h1, h2, h3 { letter-spacing: .01em; }
    /* KPI strip */
    .erai-kpi {
        border: 1px solid #233149; border-radius: 10px; padding: 10px 14px;
        background: linear-gradient(180deg, #131d33 0%, #0e1729 100%);
        height: 100%;
    }
    .erai-kpi .label { font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: #94a3b8; }
    .erai-kpi .value { font-size: 21px; font-weight: 600; margin-top: 2px; }
    .erai-kpi .sub { font-size: 11px; color: #94a3b8; margin-top: 2px; }
    /* Status pills */
    .erai-pill {
        display: inline-block; padding: 1px 9px; border-radius: 10px; font-size: 11px;
        border: 1px solid #2c3a55; color: #cbd5e1; background: #131d33; margin-right: 6px;
    }
    .erai-pill.ok { color: #6ee7b7; border-color: #14532d; background: #052e1b; }
    .erai-pill.warn { color: #fcd34d; border-color: #713f12; background: #2a1d05; }
    .erai-pill.bad { color: #fca5a5; border-color: #7f1d1d; background: #2a0a0a; }
    .erai-pill.info { color: #93c5fd; border-color: #1e3a8a; background: #0b1930; }
    /* Provenance line under a figure */
    .erai-prov { font-size: 11.5px; color: #8ba0bd; margin-top: -6px; }
    .erai-section {
        border-left: 3px solid #4c8dff; padding-left: 10px; margin: 6px 0 14px;
    }
    .erai-quote { background:#111a2e; border-left:3px solid #a78bfa; padding:10px 14px;
                  border-radius:6px; font-size:13px; }
    .erai-warn { background:#2a1d05; border-left:3px solid #fbbf24; padding:10px 14px;
                 border-radius:6px; font-size:12.5px; }
    .erai-danger { background:#2a0a0a; border-left:3px solid #f87171; padding:10px 14px;
                   border-radius:6px; font-size:12.5px; }
    .erai-tool { font-size:12px; color:#8ba0bd; font-family: ui-monospace, Consolas, monospace; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_db(path_str: str) -> Database:
    database = Database(Path(path_str))
    database.initialise()
    return database


def settings():
    return get_settings()


def db() -> Database:
    return get_db(str(settings().db_path))


def seed_registry_companies() -> None:
    """Make the verified registry profiles available in the database."""
    if st.session_state.get("_seeded") == settings().db_path.stat().st_mtime if settings().db_path.exists() else False:
        return
    database = db()
    for profile in list_profiles():
        repo.upsert_company(database, profile)
    st.session_state["_seeded"] = True


def data_version() -> int:
    return int(st.session_state.get("data_version", 0))


def bump_data_version() -> None:
    st.session_state["data_version"] = data_version() + 1


def load_dataset_cached(company_id: str, version: int) -> FinancialDataSet:
    """Load a company's dataset; ``version`` invalidates the cache on writes."""
    return repo.load_dataset(db(), company_id)


def get_dataset(company_id: str) -> FinancialDataSet:
    return load_dataset_cached(company_id, data_version())


def current_company_id() -> Optional[str]:
    return st.session_state.get("company_id")


def current_profile() -> Optional[CompanyProfile]:
    company_id = current_company_id()
    if not company_id:
        return None
    profile = get_profile(company_id) or repo.get_company(db(), company_id)
    return profile


def sales_pitch_note() -> None:
    st.caption(C.DISCLAIMER.replace("**", ""))


def pill(text: str, kind: str = "") -> str:
    return f'<span class="erai-pill {kind}">{text}</span>'


def kpi(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return (
        f'<div class="erai-kpi"><div class="label">{label}</div>'
        f'<div class="value">{value}</div>{sub_html}</div>'
    )


def kpi_row(items: Sequence[Tuple[str, str, str]]) -> None:
    if not items:
        return
    for column, (label, value, sub) in zip(st.columns(len(items)), items):
        with column:
            st.markdown(kpi(label, value, sub), unsafe_allow_html=True)


def show_frame(frame: pd.DataFrame, empty_message: str = "Aucune donnée à afficher.", **kwargs) -> None:
    if frame is None or frame.empty:
        st.info(empty_message)
        return
    st.dataframe(frame, width="stretch", hide_index=True, **kwargs)


def chart(fig, caption: Optional[str] = None) -> None:
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
    if caption:
        st.markdown(f'<div class="erai-prov">{caption}</div>', unsafe_allow_html=True)


def missing_banner(items: Sequence[str], intro: str = "Données absentes") -> None:
    if not items:
        return
    joined = ", ".join(items)
    st.markdown(
        f'<div class="erai-warn"><strong>{intro} :</strong> {joined}. '
        "Ces valeurs restent vides : elles ne sont remplacées par aucun chiffre implicite.</div>",
        unsafe_allow_html=True,
    )


def simulated_banner(dataset: FinancialDataSet) -> None:
    if workflows.is_simulated(dataset):
        st.markdown(
            '<div class="erai-danger"><strong>Jeu de données SIMULÉ</strong> — données '
            "synthétiques de démonstration, sans lien avec une société réelle. À utiliser "
            "uniquement pour tester les calculs et l'interface.</div>",
            unsafe_allow_html=True,
        )
    if workflows.has_manual_entries(dataset):
        st.markdown(
            '<div class="erai-warn"><strong>Valeurs saisies à la main</strong> — elles ne '
            "proviennent pas d'un document publié et doivent être revérifiées avant toute "
            "utilisation.</div>",
            unsafe_allow_html=True,
        )


# ==========================================================================
# Sidebar
# ==========================================================================

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("### ◮ Equity Research AI")
        st.caption("Analyse actions — toute société cotée (Managem et SanDisk préconfigurées)")

        # A company added below is selected on the next run: a widget's state
        # cannot be changed once the widget has been drawn in the current run.
        if st.session_state.pop("_reset_company", False):
            st.session_state.pop("company_id", None)
            st.session_state.pop("company_selector", None)
        pending = st.session_state.pop("_pending_company", None)
        if pending:
            st.session_state["company_id"] = pending
            st.session_state.pop("company_selector", None)
            for key in ("valuation_result", "multiples_result", "sec_filings"):
                st.session_state.pop(key, None)

        # --- company selector ------------------------------------------
        database = db()
        companies = repo.list_companies(database)
        if not companies:
            companies = list_profiles()
        if not companies:
            st.error("Aucune société en base.")
            return PAGES[0]

        options = {_company_label(p): p.company_id for p in companies}
        labels = list(options)
        current = current_company_id()
        index = 0
        if current:
            for i, label in enumerate(labels):
                if options[label] == current:
                    index = i
                    break
        selected_label = st.selectbox(
            t("nav.company"),
            labels,
            index=index,
            key="company_selector",
            help="Sociétés vérifiées dans le registre, plus toute société que vous avez ajoutée.",
        )
        selected_id = options[selected_label]
        if selected_id != current:
            st.session_state["company_id"] = selected_id
            for key in ("valuation_result", "multiples_result", "sec_filings"):
                st.session_state.pop(key, None)
            st.rerun()

        with st.expander("🔎 Rechercher une autre société"):
            _company_finder()
        with st.expander("✍️ Ajouter une société manuellement"):
            _manual_company_form()

        profile = current_profile()
        dataset = get_dataset(selected_id) if selected_id else None
        if profile is not None:
            flags = [
                pill(profile.currency.value, "info"),
                pill(profile.exchange, ""),
            ]
            if dataset is not None:
                if workflows.is_simulated(dataset):
                    flags.append(pill("SIMULÉ", "bad"))
                elif workflows.has_manual_entries(dataset):
                    flags.append(pill("saisie manuelle", "warn"))
                elif dataset.observations:
                    flags.append(pill("données chargées", "ok"))
                else:
                    flags.append(pill("aucune donnée", "warn"))
            st.markdown(" ".join(flags), unsafe_allow_html=True)

        st.divider()

        # --- navigation -------------------------------------------------
        available = {key: f"{PAGE_ICONS[key]}  {t(f'nav.{key}')}" for key in PAGES}
        if profile is not None and dataset is not None and not dataset.observations:
            available["valuation"] = f"{PAGE_ICONS['valuation']}  {t('nav.valuation')} ⚠"
        page = st.radio(
            "Navigation",
            options=list(available),
            format_func=lambda key: available[key],
            label_visibility="collapsed",
            key="nav_page",
        )

        st.divider()
        st.markdown(f"**{t('nav.language')}**")
        lang = st.radio(
            "Langue",
            options=list(SUPPORTED_LANGS),
            index=list(SUPPORTED_LANGS).index(get_lang()),
            format_func=lambda code: {"fr": "Français", "en": "English"}.get(code, code),
            horizontal=True,
            label_visibility="collapsed",
            key="lang_selector",
        )
        if lang != get_lang():
            set_lang(lang)
            st.rerun()

        st.divider()
        _sidebar_status()
    return page


def _company_label(profile: CompanyProfile) -> str:
    return f"{profile.name} ({profile.ticker})" if profile.ticker else profile.name


def _company_finder() -> None:
    """Search any listed company (SEC official table + Yahoo worldwide)."""
    from data_sources import company_discovery as cd

    with st.form("company_search_form"):
        query = st.text_input("Nom ou ticker", placeholder="ex. Airbus, NVDA, LVMH, Micron")
        international = st.checkbox(
            "Inclure les places hors US (Yahoo, non officiel)", value=True
        )
        submitted = st.form_submit_button("Rechercher")
    if submitted:
        with st.spinner("Recherche des sociétés…"):
            try:
                st.session_state["company_search"] = cd.search_companies(
                    query, limit=8, include_international=international
                )
            except Exception as exc:  # network-dependent: surfaced, never hidden
                logger.warning("Recherche de société échouée : %s", exc)
                st.error(f"Recherche impossible : {exc}")
                st.session_state.pop("company_search", None)

    result = st.session_state.get("company_search")
    if result is None:
        st.caption(
            "Sources : table officielle SEC (sociétés cotées aux États-Unis, import "
            "automatique des comptes) et Yahoo Finance (autres places, identité à vérifier)."
        )
        return
    for error in result.errors:
        st.caption(f"⚠ {error}")
    if not result.candidates:
        st.info(
            "Aucune société trouvée. Pour une place non couverte (ex. Bourse de "
            "Casablanca), utilisez l'ajout manuel."
        )
        return
    labels = [candidate.label for candidate in result.candidates]
    choice = st.selectbox(
        "Résultats",
        list(range(len(labels))),
        format_func=lambda i: labels[i],
        key="company_search_choice",
    )
    candidate = result.candidates[int(choice)]
    st.caption(
        f"{candidate.source} · {candidate.country or 'pays à vérifier'} · "
        f"{candidate.currency.value} · {candidate.sector or 'secteur non renseigné'}"
    )
    if st.button("➕ Ajouter et ouvrir", key="company_add", type="primary"):
        if candidate.sec_cik:
            with st.spinner("Lecture de l'identité officielle (SEC)…"):
                candidate = cd.enrich_from_sec(candidate)
        _add_company(cd.candidate_to_profile(candidate))


def _manual_company_form() -> None:
    """Create a company no search source covers (e.g. Casablanca listings)."""
    from data_sources import company_discovery as cd

    with st.form("manual_company_form"):
        name = st.text_input("Nom de la société *")
        ticker = st.text_input("Ticker / code *", placeholder="ex. ATW")
        exchange = st.text_input("Place de cotation", placeholder="ex. Bourse de Casablanca")
        country = st.text_input("Pays", placeholder="ex. Maroc")
        currency = st.selectbox("Devise de reporting", [c.value for c in Currency])
        sector = st.text_input("Secteur", placeholder="ex. Banques")
        fiscal_year_end = st.text_input("Clôture d'exercice (MM-JJ)", value="12-31")
        submitted = st.form_submit_button("Ajouter et ouvrir")
    if not submitted:
        return
    if not name.strip() or not ticker.strip():
        st.error("Le nom et le ticker sont obligatoires.")
        return
    candidate = cd.manual_candidate(
        name=name,
        ticker=ticker,
        exchange=exchange,
        country=country,
        currency=Currency(currency),
        sector=sector,
        fiscal_year_end=fiscal_year_end,
    )
    _add_company(cd.candidate_to_profile(candidate))


def _add_company(profile: CompanyProfile) -> None:
    """Store a new company (or reopen it if already known) and select it."""
    if not profile.company_id:
        st.error("Ticker invalide : impossible de créer un identifiant.")
        return
    database = db()
    known = list_profiles() + repo.list_companies(database)
    for existing in known:
        same_cik = bool(profile.sec_cik) and existing.sec_cik == profile.sec_cik
        same_ticker = existing.ticker.upper() == profile.ticker.upper() and existing.exchange == profile.exchange
        if existing.company_id == profile.company_id or same_cik or same_ticker:
            st.session_state["_pending_company"] = existing.company_id
            st.session_state.pop("company_search", None)
            st.rerun()
    repo.upsert_company(database, profile)
    st.session_state["_pending_company"] = profile.company_id
    st.session_state.pop("company_search", None)
    st.rerun()


def _sidebar_status() -> None:
    app_settings = settings()
    ai_ok = app_settings.ai_enabled
    search_ok = app_settings.search_enabled
    rows = [
        ("Assistant IA", "actif" if ai_ok else "désactivé", "ok" if ai_ok else ""),
        (
            "Recherche web",
            app_settings.search_provider if search_ok else "désactivée",
            "ok" if search_ok else "",
        ),
    ]
    for label, value, kind in rows:
        st.markdown(f'<span class="erai-pill {kind}">{label} : {value}</span>', unsafe_allow_html=True)
    st.caption(f"Base : {app_settings.db_path.name}")


# ==========================================================================
# Page 1 — Dashboard
# ==========================================================================

def page_dashboard() -> None:
    profile = current_profile()
    if profile is None:
        st.info("Sélectionnez une société dans la barre latérale.")
        return
    dataset = get_dataset(profile.company_id)

    st.markdown(f"## {profile.name}")
    st.markdown(
        f'<div class="erai-section">{profile.sector or "secteur non renseigné"} · '
        f"{profile.exchange} · {profile.country} · reporting en "
        f"{profile.currency.value} (millions)</div>",
        unsafe_allow_html=True,
    )
    simulated_banner(dataset)

    if not dataset.observations:
        st.markdown(
            '<div class="erai-warn"><strong>Aucune donnée financière chargée.</strong> '
            "Rendez-vous dans « États financiers » pour importer les comptes (SEC EDGAR "
            "pour les valeurs américaines, PDF du rapport annuel, ou saisie manuelle "
            "explicite).</div>",
            unsafe_allow_html=True,
        )
        _dashboard_empty_steps(profile)
        return

    latest = dataset.latest_period_label()
    previous_periods = [p for p in dataset.periods() if p != latest] if latest else []
    previous = previous_periods[-1] if previous_periods else ""

    def value(item: LineItem, period: Optional[str]) -> Optional[float]:
        return dataset.value(item, period) if period else None

    revenue = value(LineItem.REVENUE, latest)
    revenue_prev = value(LineItem.REVENUE, previous)
    net_income = value(LineItem.NET_INCOME, latest)
    margin = (
        revenue and value(LineItem.EBIT, latest) and value(LineItem.EBIT, latest) / revenue
    )
    net_debt, _nd_reason = workflows.RatioEngine(dataset).net_debt(latest) if latest else (None, None)
    price = value(LineItem.MARKET_PRICE, latest)
    shares = value(LineItem.DILUTED_SHARES, latest) or value(LineItem.SHARES_OUTSTANDING, latest)
    market_cap = price * shares if price and shares else None

    quality = workflows.assess(dataset)

    valuation = st.session_state.get("valuation_result")
    base_value = (
        valuation["base"]["value_per_share"]
        if isinstance(valuation, dict) and valuation.get("base")
        else None
    )

    kpis = [
        ("Chiffre d'affaires", C.fmt_number(revenue), f"{latest or '—'} · millions {profile.currency.value}"),
        ("Résultat net", C.fmt_number(net_income), f"{latest or '—'}"),
        ("Marge opérationnelle", C.fmt_pct(margin), "EBIT / chiffre d'affaires"),
        ("Dette nette", C.fmt_number(net_debt), "dette totale − trésorerie"),
        (
            "Capitalisation",
            C.fmt_number(market_cap),
            f"cours {C.fmt_number(price, 2)} × {C.fmt_number(shares, 1)} M actions"
            if (price and shares)
            else "cours ou actions non renseignés",
        ),
    ]
    if base_value is not None:
        kpis.append(
            (
                "Valeur DCF (session)",
                C.fmt_number(base_value, 2),
                f"{valuation['base'].get('currency', '')} par action — hypothèses à revoir",
            )
        )
    kpi_row(kpis)

    st.divider()
    left, right = st.columns(2)
    with left:
        chart(charts.revenue_and_earnings(dataset), C.REPORTED_FIGURES_NOTE)
    with right:
        chart(
            charts.margin_trends(dataset),
            "Marge brute / opérationnelle / nette, en % du chiffre d'affaires.",
        )

    left, right = st.columns(2)
    with left:
        chart(charts.cash_flow_bars(dataset))
    with right:
        chart(charts.balance_sheet_trend(dataset))

    st.divider()
    st.markdown("### Qualité des données")
    status_col, coverage_col = st.columns([1, 2])
    with status_col:
        st.markdown(f"**{C.quality_badge(quality)}**")
        st.caption(
            f"{len(quality.errors)} erreur(s), {len(quality.warnings)} avertissement(s), "
            f"{len(quality.infos)} information(s)."
        )
        if quality.errors:
            st.markdown('<div class="erai-danger">Contrôles en échec : corrigez-les avant '
                        "d'utiliser les chiffres.</div>", unsafe_allow_html=True)
    with coverage_col:
        key_items = [LineItem.REVENUE, LineItem.EBIT, LineItem.NET_INCOME,
                     LineItem.TOTAL_ASSETS, LineItem.TOTAL_EQUITY, LineItem.TOTAL_DEBT,
                     LineItem.CASH_AND_EQUIVALENTS, LineItem.OPERATING_CASH_FLOW,
                     LineItem.CAPITAL_EXPENDITURE, LineItem.DILUTED_SHARES]
        absent = C.missing_items(dataset, key_items)
        if absent:
            missing_banner(absent, "Postes clés absents à la dernière période")
        else:
            st.success("Tous les postes clés nécessaires à la valorisation sont renseignés.")

    top_issues = sorted(quality.issues, key=lambda i: (i.severity != Severity.ERROR, i.severity != Severity.WARNING))[:5]
    if top_issues:
        show_frame(
            pd.DataFrame(
                [
                    {
                        "Sévérité": C.severity_icon(issue.severity),
                        "Code": issue.code,
                        "Message": issue.message,
                    }
                    for issue in top_issues
                ]
            )
        )
    sales_pitch_note()


def _dashboard_empty_steps(profile: CompanyProfile) -> None:
    st.markdown("#### Comment charger les comptes")
    if profile.sec_cik:
        st.markdown(
            f"1. **Import automatique SEC EDGAR** (recommandé) — CIK `{profile.sec_cik}`. "
            "Page *États financiers* → *Importer depuis la SEC*."
        )
    else:
        st.markdown(
            "1. **Import du rapport annuel** — la page *États financiers* accepte un PDF : "
            "les tableaux sont extraits, vous confirmez les lignes avant enregistrement."
        )
    st.markdown("2. **Saisie manuelle** — pour les chiffres publiés dans un document non lisible par machine.")
    st.markdown(
        "3. **Jeu de démonstration simulé** — pour tester les calculs, explicitement étiqueté "
        "comme non réel."
    )
    if profile.ir_url:
        st.link_button("Site relations investisseurs", profile.ir_url)


# ==========================================================================
# Page 2 — Company research
# ==========================================================================

def page_research() -> None:
    profile = current_profile()
    if profile is None:
        st.info("Sélectionnez une société dans la barre latérale.")
        return
    dataset = get_dataset(profile.company_id)

    st.markdown(f"## Recherche société — {profile.name}")
    if profile.description:
        st.markdown(f'<div class="erai-quote">{profile.description}</div>', unsafe_allow_html=True)

    user_added = get_profile(profile.company_id) is None
    if user_added:
        next_step = (
            "importez les comptes publiés via **États financiers → SEC EDGAR**"
            if profile.sec_cik
            else "importez un rapport annuel PDF ou saisissez les comptes dans **États financiers**"
        )
        col_info, col_remove = st.columns([4, 1])
        with col_info:
            st.info(
                f"Société ajoutée par vous. Prochaine étape : {next_step}. "
                "Vérifiez l'identité ci-dessous dans les documents officiels."
            )
        with col_remove:
            if st.button("Retirer de la liste", key=f"remove_{profile.company_id}"):
                repo.delete_company(db(), profile.company_id)
                st.session_state["_reset_company"] = True
                st.rerun()

    left, right = st.columns([3, 2])
    with left:
        st.markdown("### Identité et périmètre")
        identity = pd.DataFrame(
            [
                {"Caractéristique": "Nom", "Valeur": profile.name},
                {"Caractéristique": "Ticker", "Valeur": profile.ticker},
                {"Caractéristique": "Place", "Valeur": profile.exchange},
                {"Caractéristique": "Pays", "Valeur": profile.country},
                {"Caractéristique": "Secteur", "Valeur": profile.sector or C.MISSING_GLYPH},
                {"Caractéristique": "Industrie", "Valeur": profile.industry or C.MISSING_GLYPH},
                {"Caractéristique": "Devise de reporting", "Valeur": profile.currency.value},
                {"Caractéristique": "Clôture d'exercice", "Valeur": profile.fiscal_year_end or C.MISSING_GLYPH},
                {"Caractéristique": "ISIN", "Valeur": profile.isin or C.MISSING_GLYPH},
                {"Caractéristique": "CIK (SEC)", "Valeur": profile.sec_cik or C.MISSING_GLYPH},
                {"Caractéristique": "Site", "Valeur": profile.website or C.MISSING_GLYPH},
            ]
        )
        show_frame(identity)
        if profile.identity_notes:
            st.markdown(f'<div class="erai-warn">{profile.identity_notes}</div>', unsafe_allow_html=True)

    with right:
        st.markdown("### Risques sectoriels documentés")
        risks, methods = workflows.sector_context(profile.company_id, profile)
        if risks:
            for risk in risks:
                st.markdown(f"- {risk}")
        else:
            st.caption("Aucun risque sectoriel documenté pour ce secteur dans le registre.")
        st.markdown("### Méthodes de valorisation recommandées")
        if methods:
            for method in methods:
                st.markdown(f"- {method}")
        else:
            st.caption("Aucune méthode particulière recommandée : DCF + comparables.")

    if profile.history_cutoff:
        st.markdown(
            '<div class="erai-warn"><strong>Périmètre historique restreint.</strong> '
            f"Les exercices antérieurs au {C.fmt_date(profile.history_cutoff)} proviennent d'un "
            f"périmètre comptable différent : {profile.history_cutoff_note or 'non comparables.'}"
            "</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("### Documents officiels")
    documents = _official_documents(profile)
    if documents:
        show_frame(pd.DataFrame(documents))
    else:
        st.info(
            "Aucun document catalogué pour cette société. Utilisez la page *Sources et "
            "qualité* pour enregistrer un document."
        )

    left, right = st.columns(2)
    with left:
        if profile.ir_url:
            st.link_button("↗ Relations investisseurs", profile.ir_url)
    with right:
        if profile.sec_cik:
            st.link_button(
                "↗ Dépôts SEC EDGAR",
                f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={profile.sec_cik}&type=10-K&dateb=&owner=include&count=20",
            )

    if profile.sec_cik:
        st.markdown("### Derniers dépôts réglementaires (SEC)")
        st.caption(
            "Récupération en direct depuis l'API publique de la SEC ; le contenu n'est "
            "enregistré qu'après validation dans la page *États financiers*."
        )
        if st.button("Récupérer les derniers dépôts", key="fetch_sec_filings"):
            with st.spinner("Interrogation de la SEC…"):
                from data_sources.sec_edgar import SecEdgarClient, fetch_recent_filings

                try:
                    client = SecEdgarClient(settings=settings())
                    filings = fetch_recent_filings(client, profile.sec_cik, limit=20)
                    rows = [
                        {
                            "Formulaire": f.get("form"),
                            "Déposé le": str(f.get("filing_date") or ""),
                            "Période": str(f.get("report_date") or ""),
                            "URL": f.get("url"),
                        }
                        for f in filings
                    ]
                    st.session_state["sec_filings"] = rows
                except Exception as exc:  # pragma: no cover - network dependent
                    st.error(f"Récupération impossible : {exc}")
        filings = st.session_state.get("sec_filings")
        if filings:
            show_frame(pd.DataFrame(filings))

    st.divider()
    _report_section(profile, dataset, extra_title="Note de recherche")


def _official_documents(profile: CompanyProfile) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        from data_sources.official_sources import document_rows, discover_official_sources

        result = discover_official_sources(profile.company_id, live=False)
        for document in document_rows(result.official_documents):
            rows.append(
                {
                    "Document": document["name"],
                    "Type": document["document_type"],
                    "Période": document["period"] or "",
                    "Exercice": document["fiscal_year"] or "",
                    "URL": document["url"],
                    "Vérifié le": C.fmt_date(document["verified_on"]),
                }
            )
        for note in result.errors:
            st.warning(note)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Catalogue de documents indisponible : %s", exc)
    return rows


# ==========================================================================
# Page 3 — Financial statements
# ==========================================================================

def page_statements() -> None:
    profile = current_profile()
    if profile is None:
        st.info("Sélectionnez une société dans la barre latérale.")
        return
    dataset = get_dataset(profile.company_id)

    st.markdown(f"## États financiers — {profile.name}")
    simulated_banner(dataset)

    with st.expander("📥 Charger / compléter les données", expanded=not dataset.observations):
        _data_loading_panel(profile)

    if not dataset.observations:
        st.info(
            "Aucune donnée en base pour cette société. Utilisez le panneau ci-dessus : "
            "import SEC, import PDF, saisie manuelle ou jeu de démonstration simulé."
        )
        return

    dataset, derived = workflows.ensure_derived(dataset)
    latest = dataset.latest_period_label()
    st.caption(
        f"{len(dataset.observations)} observations · {len(dataset.periods())} exercice(s) · "
        f"dernière période {latest or '—'} · valeurs en millions de {dataset.currency.value}."
    )

    tab_income, tab_balance, tab_cash, tab_ratios, tab_prov = st.tabs(
        ["Compte de résultat", "Bilan", "Flux de trésorerie", "Ratios", "Provenance"]
    )

    with tab_income:
        _statements_table(
            dataset,
            [
                LineItem.REVENUE, LineItem.COST_OF_REVENUE, LineItem.GROSS_PROFIT,
                LineItem.OPERATING_EXPENSES, LineItem.EBIT, LineItem.EBITDA,
                LineItem.DEPRECIATION_AMORTIZATION, LineItem.INTEREST_EXPENSE,
                LineItem.PRETAX_INCOME, LineItem.INCOME_TAX_EXPENSE, LineItem.NET_INCOME,
                LineItem.EPS_DILUTED,
            ],
        )
        chart(charts.revenue_and_earnings(dataset))
        chart(charts.margin_trends(dataset))

    with tab_balance:
        _statements_table(
            dataset,
            [
                LineItem.CASH_AND_EQUIVALENTS, LineItem.SHORT_TERM_INVESTMENTS,
                LineItem.ACCOUNTS_RECEIVABLE, LineItem.INVENTORY, LineItem.CURRENT_ASSETS,
                LineItem.TOTAL_ASSETS, LineItem.ACCOUNTS_PAYABLE, LineItem.CURRENT_LIABILITIES,
                LineItem.SHORT_TERM_DEBT, LineItem.LONG_TERM_DEBT, LineItem.TOTAL_DEBT,
                LineItem.TOTAL_LIABILITIES, LineItem.TOTAL_EQUITY,
                LineItem.NET_WORKING_CAPITAL, LineItem.MINORITY_INTEREST,
            ],
        )
        chart(charts.balance_sheet_trend(dataset))

    with tab_cash:
        _statements_table(
            dataset,
            [
                LineItem.OPERATING_CASH_FLOW, LineItem.CAPITAL_EXPENDITURE,
                LineItem.FREE_CASH_FLOW, LineItem.DIVIDENDS_PAID,
            ],
        )
        chart(charts.cash_flow_bars(dataset))

    with tab_ratios:
        period = st.selectbox(
            "Période", list(reversed(dataset.periods())), key="ratio_period"
        )
        metrics = workflows.ratio_metrics(dataset, period)
        if metrics:
            rows = []
            for metric in metrics.values():
                rows.append(
                    {
                        "Ratio": metric.label,
                        "Valeur": (
                            C.MISSING_GLYPH
                            if metric.value is None
                            else (
                                f"{metric.value:,.1f} %"
                                if metric.unit == "percent"
                                else f"{metric.value:,.2f}"
                            )
                        ),
                        "Unité": {
                            "percent": "%",
                            "multiple": "×",
                            "currency_millions": "millions",
                            "currency_per_share": "par action",
                        }.get(metric.unit, metric.unit),
                        "Formule": metric.formula,
                        "Données d'entrée": ", ".join(
                            f"{k}={C.MISSING_GLYPH if v is None else f'{v:,.0f}'}"
                            for k, v in (metric.inputs or {}).items()
                        ),
                        "Statut": C.status_label(metric.status),
                        "Lecture": metric.interpretation,
                        "Limite": metric.reason or "",
                    }
                )
            show_frame(pd.DataFrame(rows))
            chart(charts.metric_bars(list(metrics.values())))
        else:
            st.info("Aucun ratio calculable pour cette période.")

    with tab_prov:
        st.markdown("### Traçabilité de chaque valeur")
        st.caption(
            "Chaque observation conserve son statut, sa source, sa méthode d'extraction et "
            "sa période. Les lignes « Calculé » sont dérivées de valeurs publiées."
        )
        show_frame(
            C.observations_frame(dataset)[
                ["Poste", "Période", "Valeur", "Statut", "Source", "Méthode d'extraction", "Note"]
            ]
        )
        st.download_button(
            "Télécharger les observations (CSV)",
            data=C.observations_frame(dataset).to_csv(index=False).encode("utf-8"),
            file_name=f"observations-{profile.company_id}.csv",
            mime="text/csv",
        )
        if derived:
            st.caption("Postes dérivés automatiquement (statut « Calculé ») : " + ", ".join(derived[:12]))

    st.divider()
    _report_section(profile, dataset)


def _statements_table(dataset: FinancialDataSet, items: Sequence[LineItem]) -> None:
    periods = dataset.periods()
    if not periods:
        st.info("Aucune période annuelle.")
        return
    rows: List[Dict[str, Any]] = []
    for item in items:
        row: Dict[str, Any] = {"Poste": C.line_item_label(item)}
        statuses = set()
        for period in periods:
            observation = dataset.get(item, period)
            if observation is None or observation.is_missing:
                row[period] = C.MISSING_GLYPH
            else:
                row[period] = C.fmt_number(observation.value)
                statuses.add(observation.status)
        if not statuses:
            row["Statut"] = C.MISSING_GLYPH
        elif len(statuses) == 1:
            row["Statut"] = C.status_label(next(iter(statuses)))
        else:
            row["Statut"] = " / ".join(sorted(C.status_label(s) for s in statuses))
        rows.append(row)
    st.caption(
        "Montants en millions de la devise de reporting. « " + C.MISSING_GLYPH + " » "
        "signifie « donnée absente », jamais zéro."
    )
    show_frame(pd.DataFrame(rows))


def _data_loading_panel(profile: CompanyProfile) -> None:
    st.markdown(
        "Les sources sont traitées par ordre de fiabilité : **document officiel** > calcul > "
        "**saisie manuelle** > estimation. Aucune donnée n'est écrite sans que vous ayez vu "
        "ce qui a été extrait."
    )
    if profile.sec_cik:
        tab_sec, tab_pdf, tab_manual, tab_demo = st.tabs(
            ["SEC EDGAR", "PDF", "Saisie manuelle", "Démonstration"]
        )
    else:
        tab_pdf, tab_manual, tab_demo = st.tabs(["PDF", "Saisie manuelle", "Démonstration"])
        tab_sec = None

    if tab_sec is not None:
        with tab_sec:
            st.markdown(
                f"Import automatique des faits annuels publiés à la SEC pour **{profile.name}** "
                f"(CIK `{profile.sec_cik}`). Les valeurs sont marquées *Publié* et rattachées au "
                "dépôt d'origine."
            )
            years = st.number_input("Nombre d'exercices à récupérer", 2, 10, 6, key="sec_years")
            if st.button("Importer depuis la SEC", type="primary", key="sec_import"):
                with st.spinner("Téléchargement des états financiers publiés…"):
                    from data_sources.sec_edgar import load_company_from_sec

                    try:
                        result = load_company_from_sec(
                            profile.company_id,
                            profile.ticker,
                            cik_fallback=profile.sec_cik,
                            years=int(years),
                        )
                    except Exception as exc:  # pragma: no cover - network dependent
                        st.error(f"Échec de l'import SEC : {exc}")
                    else:
                        _store_sec_result(profile, result)

    with tab_pdf:
        st.markdown(
            "Déposez le PDF d'un rapport annuel ou d'un communiqué. Le texte et les tableaux "
            "sont extraits **localement**, avec le numéro de page : vous choisissez ensuite "
            "les lignes à enregistrer. Les PDF scannés (images) ne sont pas lisibles sans OCR "
            "et sont signalés comme tels."
        )
        uploaded = st.file_uploader("Fichier PDF", type=["pdf"], key="pdf_upload")
        if uploaded is not None:
            _pdf_import_flow(profile, uploaded)

    with tab_manual:
        st.markdown(
            "Saisissez les valeurs telles que publiées dans le document (montants en "
            "**millions**). Ces lignes seront enregistrées avec le statut *Saisie manuelle* "
            "et rappelées comme telles partout dans l'application."
        )
        _manual_entry_form(profile)

    with tab_demo:
        st.markdown(
            "Charge un jeu de données **synthétique** (simulé) pour tester les calculs, "
            "les graphiques et les rapports sans dépendre d'un document externe. "
            "Il est étiqueté « SIMULÉ » dans toute l'interface et dans les rapports."
        )
        if st.button("Charger le jeu de démonstration simulé", key="load_demo"):
            from data_sources.sample import synthetic_dataset

            dataset = workflows.relabel_dataset(
                synthetic_dataset(), profile.company_id, currency=profile.currency
            )
            repo.upsert_observations(db(), dataset.observations)
            repo.upsert_source(
                db(),
                repo.SourceRecord(
                    source_id="simulated:demo",
                    name="Jeu de démonstration SIMULÉ",
                    publisher="Equity Research AI",
                    document_type="données synthétiques",
                    currency=profile.currency,
                    scale=Scale.MILLIONS,
                    notes="Données générées pour les tests. Aucune société réelle.",
                ),
                profile.company_id,
            )
            bump_data_version()
            st.success("Jeu simulé chargé.")
            st.rerun()
        st.caption(
            "Le jeu simulé est compatible avec un profil réel mais ne décrit pas la société : "
            "il ne doit jamais servir de base à une décision."
        )


def _store_sec_result(profile: CompanyProfile, result) -> None:
    if getattr(result, "errors", None):
        for error in result.errors[:5]:
            st.warning(error)
    dataset = getattr(result, "dataset", None)
    if dataset is None or not dataset.observations:
        st.error(
            "L'import n'a produit aucune observation exploitable. Vérifiez l'identifiant "
            "CIK ou utilisez la saisie manuelle."
        )
        return
    database = db()
    stored = repo.upsert_observations(database, dataset.observations)
    for source in getattr(result, "sources", []) or []:
        repo.upsert_source(database, source, profile.company_id)
    bump_data_version()
    simulated_note = ""
    if workflows.is_simulated(dataset):
        simulated_note = " (jeu simulé)"
    st.success(
        f"{stored} observations enregistrées pour {profile.name}{simulated_note} · "
        f"{len(dataset.periods())} exercice(s) : {', '.join(dataset.periods())}."
    )
    for note in (getattr(result, "notes", []) or [])[:4]:
        st.caption(note)
    st.rerun()


def _manual_entry_form(profile: CompanyProfile) -> None:
    with st.form("manual_entry", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            fiscal_year = st.number_input(
                "Exercice (année de clôture)", min_value=1990, max_value=2100,
                value=_dt.date.today().year - 1, step=1,
            )
        with col2:
            currency = st.selectbox(
                "Devise", [c.value for c in Currency],
                index=[c.value for c in Currency].index(profile.currency.value),
            )
        with col3:
            scale = st.selectbox("Échelle du document", [s.value for s in Scale], index=2,
                                 help="Indiquez l'unité dans laquelle le document présente les montants : "
                                      "l'application convertit en millions.")

        st.markdown("**Montants (dans l'échelle choisie)**")
        fields = [
            (LineItem.REVENUE, "Toujours nécessaire pour valoriser"),
            (LineItem.EBIT, "Résultat d'exploitation"),
            (LineItem.EBITDA, "Optionnel si EBIT et amortissements sont saisis"),
            (LineItem.NET_INCOME, "Résultat net"),
            (LineItem.DEPRECIATION_AMORTIZATION, "Amortissements"),
            (LineItem.CAPITAL_EXPENDITURE, "Investissements de l'exercice"),
            (LineItem.OPERATING_CASH_FLOW, "Flux de trésorerie d'exploitation"),
            (LineItem.TOTAL_ASSETS, "Total du bilan"),
            (LineItem.TOTAL_EQUITY, "Capitaux propres"),
            (LineItem.TOTAL_DEBT, "Dette financière (court + long terme)"),
            (LineItem.CASH_AND_EQUIVALENTS, "Trésorerie et équivalents"),
            (LineItem.CURRENT_ASSETS, "Pour le besoin en fonds de roulement"),
            (LineItem.CURRENT_LIABILITIES, "Pour le besoin en fonds de roulement"),
            (LineItem.DILUTED_SHARES, "Nombre d'actions diluées (millions)"),
            (LineItem.MARKET_PRICE, "Cours de clôture (optionnel)"),
        ]
        values: Dict[LineItem, Optional[float]] = {}
        columns = st.columns(3)
        for index, (item, help_text) in enumerate(fields):
            with columns[index % 3]:
                raw = st.text_input(
                    C.line_item_label(item),
                    key=f"manual_{item.value}",
                    help=help_text,
                    placeholder="vide = non renseigné",
                )
                values[item] = _parse_manual_input(raw)

        note = st.text_input(
            "Note de saisie (document, page, remarque)",
            value="Saisie manuelle — source à préciser.",
            key="manual_note",
        )
        submitted = st.form_submit_button("Enregistrer la saisie", type="primary")

    if not submitted:
        return
    provided = {item: value for item, value in values.items() if value is not None}
    if not provided:
        st.warning("Aucune valeur saisie : rien n'a été enregistré.")
        return
    if LineItem.REVENUE not in provided:
        st.warning(
            "Sans chiffre d'affaires, la plupart des analyses restent indisponibles. "
            "La saisie est tout de même enregistrée."
        )
    from data_sources.sample import manual_dataset_from_values

    dataset = manual_dataset_from_values(
        profile.company_id,
        Currency(currency),
        int(fiscal_year),
        provided,
        scale=Scale(scale),
        note=note,
    )
    stored = repo.upsert_observations(db(), dataset.observations)
    repo.upsert_source(
        db(),
        repo.SourceRecord(
            source_id="manual:analyst",
            name="Saisie manuelle de l'analyste",
            publisher="Analyse interne",
            document_type="saisie manuelle",
            currency=Currency(currency),
            scale=Scale(scale),
            notes=note,
        ),
        profile.company_id,
    )
    bump_data_version()
    st.success(f"{stored} valeur(s) enregistrée(s) pour FY{int(fiscal_year)}.")
    st.rerun()


def _parse_manual_input(raw: str) -> Optional[float]:
    from data_sources.parsers import parse_number

    if raw is None or not str(raw).strip():
        return None
    value = parse_number(raw)
    if value is None:
        st.caption(f"« {raw} » n'a pas pu être interprété comme un nombre : valeur ignorée.")
    return value


def _pdf_import_flow(profile: CompanyProfile, uploaded) -> None:
    from data_sources.pdf_extractor import candidate_tables, extract_pdf

    tmp_dir = settings().data_dir / "uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    target = tmp_dir / Path(uploaded.name).name
    target.write_bytes(uploaded.getbuffer())
    with st.spinner("Extraction du texte et des tableaux…"):
        document = extract_pdf(target)

    if not document.ok:
        st.error(document.error or "Extraction impossible.")
        return
    summary = document.summary()
    st.caption(
        f"{summary['pages']} pages · {summary['tables']} tableau(x) détecté(s) · "
        f"{summary['scanned_pages']} page(s) image · extracteur : {summary['extractor']}"
    )
    for warning in document.warnings:
        st.warning(warning)
    if not document.tables:
        st.info(
            "Aucun tableau structuré détecté. Vous pouvez malgré tout chercher une ligne par "
            "son libellé ci-dessous, ou utiliser la saisie manuelle."
        )

    tables = candidate_tables(document)
    if tables:
        options = {
            f"p.{table.page} · tableau {table.index + 1} · {table.n_rows}×{table.n_cols}"
            + (f" · {table.title_hint[:60]}" if table.title_hint else ""): table
            for table in tables[:25]
        }
        choice = st.selectbox("Tableaux candidats", list(options), key="pdf_table_choice")
        table = options[choice]
        if table.scale:
            st.caption(
                f"Échelle détectée dans la page : ×{table.scale:,.0f} "
                f"({table.scale_evidence or 'contexte du document'})."
            )
        else:
            st.caption(
                "Échelle non détectée : vérifiez le document. Sans certitude, la table sera "
                "enregistrée dans l'unité indiquée ci-dessous."
            )
        st.dataframe(pd.DataFrame(table.rows, columns=[f"col_{i}" for i in range(table.n_cols)]),
                     width="stretch", height=280)

        row_labels = [row[0] for row in table.rows if row and row[0]]
        columns_labels = st.columns(3)
        with columns_labels[0]:
            item_name = st.selectbox(
                "Associer la ligne à un poste",
                options=[item.value for item in LineItem],
                format_func=lambda value: C.line_item_label(LineItem(value)),
                key="pdf_item",
            )
        with columns_labels[1]:
            row_choice = st.selectbox("Ligne du tableau", row_labels, key="pdf_row")
        with columns_labels[2]:
            scale_choice = st.selectbox(
                "Échelle des montants lus", [s.value for s in Scale],
                index=([s.value for s in Scale].index(profile.reporting_scale.value)
                       if profile.reporting_scale else 2),
                key="pdf_scale",
            )
        row = next((r for r in table.rows if r and r[0] == row_choice), None)
        if row:
            numbers = [
                value
                for value in (_parse_manual_input(cell) for cell in row[1:])
                if value is not None
            ]
            if numbers:
                period_labels = [f"Colonne {i + 1}: {C.fmt_number(n)}" for i, n in enumerate(numbers)]
                picked = st.selectbox("Colonne retenue (période)", period_labels, key="pdf_col")
                period_index = period_labels.index(picked)
                year = st.number_input(
                    "Exercice de cette colonne", min_value=1990, max_value=2100,
                    value=_dt.date.today().year - 1, step=1, key="pdf_year",
                )
                if st.button("Enregistrer cette ligne", type="primary", key="pdf_save"):
                    from data_sources.parsers import to_millions

                    raw_value = numbers[period_index]
                    value = to_millions(raw_value, SCALE_FACTORS[scale_choice])
                    if value is None:
                        st.error("Conversion d'unité impossible.")
                    else:
                        observation = repo.Observation(
                            company_id=profile.company_id,
                            line_item=LineItem(item_name),
                            period=repo.FiscalPeriod(fiscal_year=int(year), period_type=PeriodType.ANNUAL),
                            value=value,
                            currency=Currency(profile.currency.value),
                            reported_value=numbers[period_index],
                            reported_scale=Scale(scale_choice),
                            reported_currency=Currency(profile.currency.value),
                            status=ExtractionStatus.UNVERIFIED,
                            source_id=f"pdf:{Path(uploaded.name).name}",
                            extraction_method=(
                                f"{table.extractor}, page {table.page}, tableau {table.index + 1}, "
                                f"ligne {row_choice!r}, colonne {period_index + 1}"
                            ),
                            notes="Extraction PDF à confirmer par l'analyste avant usage.",
                        )
                        repo.upsert_observations(db(), [observation])
                        repo.upsert_source(
                            db(),
                            repo.SourceRecord(
                                source_id=f"pdf:{Path(uploaded.name).name}",
                                name=uploaded.name,
                                document_type="PDF importé",
                                currency=Currency(profile.currency.value),
                                scale=Scale(scale_choice),
                                notes="Import local, extraction assistée non vérifiée.",
                            ),
                            profile.company_id,
                        )
                        bump_data_version()
                        st.success(
                            f"{C.line_item_label(LineItem(item_name))} FY{int(year)} = "
                            f"{C.fmt_number(value)} millions enregistré (statut « Non vérifié »)."
                        )
            else:
                st.info("Aucun nombre exploitable sur cette ligne.")

    st.markdown("### Rechercher un libellé dans le document")
    needle = st.text_input("Libellé recherché (ex. chiffre d'affaires)", key="pdf_search")
    if needle:
        from data_sources.pdf_extractor import extract_labeled_values

        hits = extract_labeled_values(document, needle)
        if hits:
            show_frame(
                pd.DataFrame(
                    [
                        {
                            "Page": hit["page"],
                            "Libellé trouvé": hit["label_found"],
                            "Valeurs lues": ", ".join(C.fmt_number(n, 2) for n in hit["numbers"]),
                            "Échelle": f"×{hit['scale']:,.0f}" if hit["scale"] else "non détectée",
                            "Confiance": hit["confidence"],
                        }
                        for hit in hits
                    ]
                )
            )
        else:
            st.info("Libellé introuvable dans les tableaux extraits.")


# ==========================================================================
# Page 4 — Valuation
# ==========================================================================

def page_valuation() -> None:
    profile = current_profile()
    if profile is None:
        st.info("Sélectionnez une société dans la barre latérale.")
        return
    dataset = get_dataset(profile.company_id)
    simulated_banner(dataset)

    st.markdown(f"## Valorisation — {profile.name}")
    if not dataset.observations:
        st.warning(
            "Aucune donnée financière chargée : le DCF ne peut pas être calculé. "
            "Chargez les états financiers ou saisissez manuellement les agrégats."
        )
        return

    setup = workflows.build_dcf_inputs(dataset, forecast_years=st.session_state.get("dcf_years", 5))
    if setup.errors and setup.inputs is None:
        for error in setup.errors:
            st.error(error)
        missing_banner(setup.missing, "Hypothèses manquantes")
        return
    if setup.errors:
        for error in setup.errors:
            st.warning(error)

    st.markdown("### Hypothèses du modèle")
    st.caption(
        "Les valeurs proposées sont des **statistiques historiques** issues des comptes "
        "chargés (médianes, TCAM) ou des valeurs génériques explicitement signalées. "
        "Chaque ligne indique son origine ; modifiez librement, l'origine devient alors "
        "« hypothèse de l'analyste »."
    )

    with st.form("dcf_form"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            forecast_years = st.number_input(
                "Horizon (années)", 1, workflows.MAX_FORECAST_YEARS,
                int(st.session_state.get("dcf_years", 5)), key="dcf_years_input",
            )
            wacc_input = st.number_input(
                "WACC (%)", 0.5, 40.0,
                float(round(setup.assumptions.get("wacc", workflows.DEFAULT_WACC) * 100, 2)),
                step=0.25, key="dcf_wacc",
            )
        with col2:
            growth_input = st.number_input(
                "Croissance perpétuelle g (%)", -5.0, 15.0,
                float(round(setup.assumptions.get("terminal_growth", workflows.DEFAULT_TERMINAL_GROWTH) * 100, 2)),
                step=0.25, key="dcf_g",
            )
            revenue_growth_input = st.number_input(
                "Croissance du CA (%)", -50.0, 100.0,
                float(round(float(setup.assumptions.get("revenue_growth", 0.0)) * 100, 2)),
                step=0.5, key="dcf_growth",
            )
        with col3:
            ebit_margin_input = st.number_input(
                "Marge opérationnelle EBIT (%)", -50.0, 70.0,
                float(round(float(setup.assumptions.get("ebit_margin", workflows.DEFAULT_EBIT_MARGIN)) * 100, 2)),
                step=0.5, key="dcf_margin",
            )
            tax_input = st.number_input(
                "Taux d'impôt (%)", 0.0, 60.0,
                float(round(float(setup.assumptions.get("tax_rate", workflows.DEFAULT_TAX_RATE)) * 100, 2)),
                step=1.0, key="dcf_tax",
            )
        with col4:
            net_debt_input = st.number_input(
                "Dette nette (millions)", value=float(round(float(setup.assumptions.get("net_debt", 0.0)), 1)),
                step=10.0, key="dcf_net_debt",
                help="Dette financière moins trésorerie. Positive si endettée.",
            )
            shares_input = st.number_input(
                "Actions diluées (millions)", min_value=0.01,
                value=float(round(float(setup.assumptions.get("shares_outstanding", 1.0)), 2)),
                step=1.0, key="dcf_shares",
            )
            price_input = st.number_input(
                "Cours de référence (optionnel)", min_value=0.0,
                value=float(st.session_state.get("reference_price") or 0.0), step=0.5,
                key="dcf_price",
            )

        with st.expander("Hypothèses avancées (intensités)", expanded=False):
            adv1, adv2, adv3 = st.columns(3)
            with adv1:
                capex_input = st.number_input(
                    "Investissements / CA (%)", 0.0, 60.0,
                    float(round(float(setup.assumptions.get("capex_pct_revenue", 0.0)) * 100, 2)),
                    step=0.5, key="dcf_capex",
                )
            with adv2:
                da_input = st.number_input(
                    "Amortissements / CA (%)", 0.0, 40.0,
                    float(round(float(setup.assumptions.get("da_pct_revenue", 0.0)) * 100, 2)),
                    step=0.5, key="dcf_da",
                )
            with adv3:
                nwc_input = st.number_input(
                    "BFR / CA (%)", -20.0, 60.0,
                    float(round(float(setup.assumptions.get("nwc_pct_revenue", 0.0)) * 100, 2)),
                    step=0.5, key="dcf_nwc",
                )

        submitted = st.form_submit_button("Lancer la valorisation", type="primary")

    if submitted:
        overrides = {
            "revenue_growth": revenue_growth_input / 100.0,
            "ebit_margin": ebit_margin_input / 100.0,
            "capex_pct_revenue": capex_input / 100.0,
            "da_pct_revenue": da_input / 100.0,
            "nwc_pct_revenue": nwc_input / 100.0,
            "tax_rate": tax_input / 100.0,
            "net_debt": net_debt_input,
            "shares_outstanding": shares_input,
        }
        st.session_state["dcf_years"] = int(forecast_years)
        st.session_state["reference_price"] = float(price_input) if price_input else None
        with st.spinner("Calcul du DCF, des scénarios et de la sensibilité…"):
            bundle = workflows.run_valuation(
                dataset,
                wacc=wacc_input / 100.0,
                terminal_growth=growth_input / 100.0,
                forecast_years=int(forecast_years),
                overrides=overrides,
                current_price=float(price_input) if price_input else None,
                base_label="Page Valorisation",
            )
        if not bundle.ok:
            st.error("Valorisation impossible : " + "; ".join(bundle.setup.errors or bundle.warnings))
        else:
            st.session_state["valuation_result"] = _serialise_bundle(bundle)
            _save_valuation(bundle, profile)
            st.rerun()

    _render_assumption_provenance(setup)

    stored = st.session_state.get("valuation_result")
    if not stored:
        st.info(
            "Lancez la valorisation pour afficher les scénarios, la décomposition de la valeur "
            "et les tables de sensibilité."
        )
        return
    _render_valuation_result(stored, profile, dataset)


def _serialise_bundle(bundle: workflows.ValuationBundle) -> Dict[str, Any]:
    """Turn a bundle into a JSON-safe dict kept in the session."""
    results = []
    for result in bundle.results:
        results.append(
            {
                "scenario": result.scenario,
                "label": result.label,
                "currency": result.currency,
                "wacc": result.wacc,
                "terminal_growth": result.terminal_growth,
                "tax_rate": result.tax_rate,
                "pv_explicit": result.pv_explicit,
                "terminal_value": result.terminal_value,
                "pv_terminal_value": result.pv_terminal_value,
                "enterprise_value": result.enterprise_value,
                "net_debt": result.net_debt,
                "other_senior_claims": result.other_senior_claims,
                "non_operating_assets": result.non_operating_assets,
                "equity_value": result.equity_value,
                "shares_outstanding": result.shares_outstanding,
                "value_per_share": result.value_per_share,
                "current_price": result.current_price,
                "warnings": list(result.warnings),
                "years": [
                    {
                        "index": year.index,
                        "fiscal_year": year.fiscal_year,
                        "revenue": year.revenue,
                        "ebit": year.ebit,
                        "nopat": year.nopat,
                        "da": year.da,
                        "capex": year.capex,
                        "nwc": year.nwc,
                        "change_nwc": year.change_nwc,
                        "fcff": year.fcff,
                        "discount_factor": year.discount_factor,
                        "pv_fcff": year.pv_fcff,
                        "ebitda": year.ebitda,
                    }
                    for year in result.years
                ],
            }
        )
    grid = None
    if bundle.sensitivity is not None:
        grid = {
            "row_label": bundle.sensitivity.row_label,
            "col_label": bundle.sensitivity.col_label,
            "row_values": list(bundle.sensitivity.row_values),
            "col_values": list(bundle.sensitivity.col_values),
            "values": [list(row) for row in bundle.sensitivity.values],
        }
    tornado = None
    if bundle.tornado is not None:
        tornado = {
            "base": bundle.tornado.base.value_per_share,
            "currency": bundle.tornado.base.currency,
            "bars": [
                {
                    "field_name": bar.field_name,
                    "label": bar.label,
                    "low_input": bar.low_input,
                    "high_input": bar.high_input,
                    "low_value": bar.low_value,
                    "base_value": bar.base_value,
                    "high_value": bar.high_value,
                }
                for bar in bundle.tornado.bars
            ],
        }
    return {
        "results": results,
        "base": results[0] if results else None,
        "grid": grid,
        "tornado": tornado,
        "assumptions": {k: (v if not isinstance(v, list) else list(v)) for k, v in bundle.assumptions.items()},
        "provenance": dict(bundle.provenance),
        "warnings": list(bundle.warnings),
        "missing": list(bundle.setup.missing),
    }


def _save_valuation(bundle: workflows.ValuationBundle, profile: CompanyProfile) -> None:
    base = bundle.base
    if base is None:
        return
    run_id = f"{profile.company_id}-{base.scenario}-{_dt.datetime.now():%Y%m%d%H%M%S}"
    try:
        repo.save_valuation_run(
            db(),
            run_id=run_id,
            company_id=profile.company_id,
            method="DCF FCFF",
            scenario=base.scenario,
            currency=base.currency,
            assumptions={k: (v if not isinstance(v, list) else list(v)) for k, v in bundle.assumptions.items()},
            result={
                "value_per_share": base.value_per_share,
                "enterprise_value": base.enterprise_value,
                "equity_value": base.equity_value,
                "wacc": base.wacc,
                "terminal_growth": base.terminal_growth,
                "scenarios": {r.scenario: r.value_per_share for r in bundle.results},
            },
            value_per_share=base.value_per_share,
            simulated=base.inputs.simulated if base.inputs else False,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Enregistrement de la valorisation impossible : %s", exc)


def _render_assumption_provenance(setup: workflows.DCFSetup) -> None:
    with st.expander("Origine de chaque hypothèse", expanded=False):
        rows = [
            {
                "Hypothèse": key,
                "Valeur": (
                    ", ".join(f"{v:.3f}" for v in value)
                    if isinstance(value, (list, tuple))
                    else f"{value:,.3f}" if isinstance(value, float) else str(value)
                ),
                "Origine": setup.provenance_of(key),
            }
            for key, value in setup.assumptions.items()
        ]
        show_frame(pd.DataFrame(rows))
        for note in setup.notes:
            st.caption(note)


def _render_valuation_result(stored: Dict[str, Any], profile: CompanyProfile, dataset: FinancialDataSet) -> None:
    st.divider()
    st.markdown("### Résultat de la valorisation")
    base = stored.get("base")
    if not base:
        st.error("Aucun résultat enregistré.")
        return

    current_price = base.get("current_price")
    upside = (
        (base["value_per_share"] / current_price - 1) * 100
        if current_price and base.get("value_per_share")
        else None
    )
    kpi_row(
        [
            ("Valeur par action (base)", C.fmt_number(base["value_per_share"], 2), base["currency"]),
            ("Valeur d'entreprise", C.fmt_number(base["enterprise_value"]), f"millions {base['currency']}"),
            ("Fonds propres", C.fmt_number(base["equity_value"]), "après dette nette"),
            (
                "Part de la valeur terminale",
                C.fmt_pct(base["pv_terminal_value"] / base["enterprise_value"] if base["enterprise_value"] else None),
                "au-delà de l'horizon explicite",
            ),
            (
                "Potentiel vs cours",
                C.fmt_pct(upside / 100) if upside is not None else C.MISSING_GLYPH,
                f"cours de référence {C.fmt_number(current_price, 2)}" if current_price else "cours non renseigné",
            ),
        ]
    )

    if base["pv_terminal_value"] and base["enterprise_value"]:
        share = base["pv_terminal_value"] / base["enterprise_value"]
        if share > 0.75:
            st.markdown(
                '<div class="erai-warn"><strong>Valeur terminale dominante :</strong> '
                f"{share:.0%} de la valeur d'entreprise provient de l'horizon au-delà de la "
                "projection explicite. Le résultat est très sensible aux deux hypothèses "
                "WACC et croissance perpétuelle — lisez la table de sensibilité avant toute "
                "conclusion.</div>",
                unsafe_allow_html=True,
            )
    for warning in stored.get("warnings", [])[:6]:
        st.warning(warning)

    plot_col, bridge_col = st.columns(2)
    with plot_col:
        chart(_figure_from_results(stored["results"], scenarios=True,
                                   current_price=current_price))
    with bridge_col:
        chart(_figure_from_results(stored["results"], scenarios=False,
                                   current_price=current_price))

    chart(_figure_from_sensitivity(stored.get("grid"), stored.get("tornado")))
    _render_tornado(stored.get("tornado"))

    st.markdown("### Détail du scénario de base")
    years_frame = pd.DataFrame(base["years"])
    if not years_frame.empty:
        years_frame = years_frame.rename(
            columns={
                "fiscal_year": "Exercice", "revenue": "CA", "ebit": "EBIT", "nopat": "NOPAT",
                "da": "Amort.", "capex": "Invest.", "change_nwc": "Δ BFR", "fcff": "FCFF",
                "discount_factor": "Facteur", "pv_fcff": "VA", "ebitda": "EBITDA",
            }
        )
        show_frame(years_frame, column_config={
            "Exercice": st.column_config.NumberColumn(format="%d"),
            **{c: st.column_config.NumberColumn(format="%.0f")
               for c in ["CA", "EBIT", "NOPAT", "Amort.", "Invest.", "Δ BFR", "FCFF", "VA", "EBITDA"]},
            "Facteur": st.column_config.NumberColumn(format="%.3f"),
        })
    st.caption(
        "FCFF = résultat opérationnel après impôt + amortissements − investissements − "
        "variation du BFR. La dette nette n'est soustraite qu'une fois, au niveau de la "
        "valeur d'entreprise : il n'y a pas de double déduction."
    )

    with st.expander("Hypothèses retenues pour ce calcul"):
        rows = [
            {"Hypothèse": key, "Valeur": _fmt_assumption(value), "Origine": stored["provenance"].get(key, "—")}
            for key, value in stored["assumptions"].items()
        ]
        show_frame(pd.DataFrame(rows))

    st.divider()
    _report_section(profile, dataset, extra_title="Note de recherche avec cette valorisation")


def _fmt_assumption(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(f"{v:.2%}" if abs(v) < 2 else f"{v:,.1f}" for v in value) + "]"
    if isinstance(value, float):
        return f"{value:.2%}" if abs(value) < 2 else f"{value:,.2f}"
    return str(value)


def _figure_from_results(results: List[Dict[str, Any]], *, scenarios: bool, current_price: Optional[float]):
    if scenarios:
        fig = charts.scenario_comparison(_results_to_objects(results), current_price)
    else:
        fig = charts.dcf_bridge(_results_to_objects(results)[0])
    return fig


def _results_to_objects(results: List[Dict[str, Any]]):
    """Rebuild lightweight objects the chart functions understand."""
    from financial_engine.dcf import DCFResult, DCFYear

    objects = []
    for row in results:
        years = []
        for i, year in enumerate(row.get("years", [])):
            payload = dict(year)
            # Older session payloads may predate these fields; charts do not
            # depend on them, so fill neutral defaults instead of crashing.
            payload.setdefault("index", i + 1)
            payload.setdefault("nwc", 0.0)
            years.append(DCFYear(**payload))
        objects.append(
            DCFResult(
                currency=row["currency"],
                units="millions",
                scenario=row["scenario"],
                label=row.get("label", ""),
                years=years,
                wacc=row["wacc"],
                terminal_growth=row["terminal_growth"],
                tax_rate=row["tax_rate"],
                pv_explicit=row["pv_explicit"],
                terminal_value=row["terminal_value"],
                pv_terminal_value=row["pv_terminal_value"],
                enterprise_value=row["enterprise_value"],
                net_debt=row["net_debt"],
                other_senior_claims=row.get("other_senior_claims", 0.0),
                non_operating_assets=row.get("non_operating_assets", 0.0),
                equity_value=row["equity_value"],
                shares_outstanding=row["shares_outstanding"],
                value_per_share=row["value_per_share"],
                current_price=row.get("current_price"),
                warnings=list(row.get("warnings", [])),
            )
        )
    return objects


def _figure_from_sensitivity(grid: Optional[Dict[str, Any]], tornado: Optional[Dict[str, Any]]):
    if grid is None:
        return charts._empty_figure(
            "Sensibilité non disponible : relancez la valorisation avec WACC > croissance "
            "perpétuelle pour obtenir la table."
        )
    from financial_engine.sensitivity import SensitivityGrid

    sensitivity = SensitivityGrid(
        row_label=grid["row_label"],
        col_label=grid["col_label"],
        row_values=list(grid["row_values"]),
        col_values=list(grid["col_values"]),
        values=[list(row) for row in grid["values"]],
        enterprise_values=[],
        current_price=None,
    )
    return charts.sensitivity_heatmap(sensitivity)


def _render_tornado(tornado: Optional[Dict[str, Any]]) -> None:
    if not tornado or not tornado.get("bars"):
        return
    from financial_engine.sensitivity import TornadoBar, TornadoResult
    from financial_engine.dcf import DCFResult

    base = DCFResult(
        currency=tornado["currency"], units="millions", scenario="base", label="",
        years=[], wacc=0.0, terminal_growth=0.0, tax_rate=0.0, pv_explicit=0.0,
        terminal_value=0.0, pv_terminal_value=0.0, enterprise_value=0.0, net_debt=0.0,
        other_senior_claims=0.0, non_operating_assets=0.0,
        equity_value=0.0, shares_outstanding=0.0, value_per_share=tornado["base"],
    )
    bars = []
    for bar in tornado["bars"]:
        payload = dict(bar)
        # Older session payloads may lack the input levels; the chart only
        # uses the value levels, so neutral defaults keep it renderable.
        payload.setdefault("low_input", 0.0)
        payload.setdefault("high_input", 0.0)
        bars.append(TornadoBar(**payload))
    chart(charts.tornado_chart(TornadoResult(base=base, bars=bars)))


# ==========================================================================
# Page 5 — Comparables
# ==========================================================================

def page_comparables() -> None:
    profile = current_profile()
    if profile is None:
        st.info("Sélectionnez une société dans la barre latérale.")
        return
    dataset = get_dataset(profile.company_id)
    st.markdown(f"## Comparables — {profile.name}")
    simulated_banner(dataset)

    if not dataset.observations:
        st.warning("Chargez d'abord les états financiers : les multiples ont besoin de l'EBITDA, du résultat net et des fonds propres.")
        return

    target_inputs, missing = workflows.build_target_inputs(dataset, profile)
    st.markdown("### Métriques de la société cible")
    if missing:
        missing_banner(missing, "Éléments manquants pour convertir un multiple en valeur par action")
    if target_inputs is not None:
        kpi_row(
            [
                ("Chiffre d'affaires", C.fmt_number(target_inputs.revenue), "millions"),
                ("EBITDA", C.fmt_number(target_inputs.ebitda), "millions"),
                ("Résultat net", C.fmt_number(target_inputs.net_income), "millions"),
                ("Fonds propres", C.fmt_number(target_inputs.book_value), "millions"),
                ("Dette nette", C.fmt_number(target_inputs.net_debt), "positive = endettement"),
                ("Actions diluées", C.fmt_number(target_inputs.diluted_shares, 1), "millions"),
            ]
        )
    if target_inputs and target_inputs.current_price:
        st.caption(f"Cours de référence utilisé : {C.fmt_number(target_inputs.current_price, 2)} {profile.currency.value}")

    st.divider()
    st.markdown("### Pairs comparables")
    st.caption(
        "Les multiples sont des rapports sans dimension, mais le régime de marché qui les "
        "produit ne l'est pas : les pairs d'une devise différente sont écartés par défaut, "
        "sauf si vous déclarez explicitement un taux de change."
    )

    peers = repo.list_peers(db(), profile.company_id)
    if peers:
        show_frame(
            pd.DataFrame(
                [
                    {
                        "Nom": peer.name,
                        "Ticker": peer.ticker or "",
                        "Devise": peer.currency.value,
                        "Capitalisation": peer.market_cap,
                        "VE/EBITDA (déclaré)": peer.ev_ebitda,
                        "P/E (déclaré)": peer.pe,
                        "P/B (déclaré)": peer.pb,
                        "Secteur": peer.sector or "",
                        "Inclus": "oui" if peer.include else "non",
                    }
                    for peer in peers
                ]
            )
        )
    else:
        st.info("Aucun comparable enregistré pour cette société.")

    with st.expander("Ajouter / modifier un comparable", expanded=not peers):
        _peer_form(profile)
    if peers:
        with st.expander("Supprimer un comparable"):
            to_delete = st.selectbox("Comparable", [p.name for p in peers], key="peer_delete")
            if st.button("Supprimer", key="peer_delete_btn"):
                repo.delete_peer(db(), profile.company_id, to_delete)
                st.success(f"{to_delete} supprimé.")
                st.rerun()

    currencies = sorted({p.currency.value for p in peers if p.currency.value != profile.currency.value})
    fx_rates: Dict[Currency, float] = {}
    if currencies:
        st.markdown("#### Taux de change déclarés")
        st.caption(
            "Indiquez combien vaut **1 unité de la devise du pair** dans la devise de la "
            "société cible. Un taux saisi est un choix explicite de l'analyste : il est "
            "enregistré avec le résultat."
        )
        columns = st.columns(len(currencies))
        for index, code in enumerate(currencies):
            with columns[index]:
                rate = st.number_input(
                    f"1 {code} = ? {profile.currency.value}", min_value=0.0,
                    value=0.0, step=0.01, key=f"fx_{code}",
                )
                if rate > 0:
                    fx_rates[Currency(code)] = rate

    if not peers:
        return

    if st.button("Lancer la valorisation par comparables", type="primary", key="run_multiples"):
        with st.spinner("Calcul des multiples et des valeurs implicites…"):
            bundle = workflows.run_multiples(
                dataset, peers, profile,
                fx_rates=fx_rates or None,
                current_price=target_inputs.current_price if target_inputs else None,
            )
        if not bundle.ok:
            for error in bundle.errors:
                st.error(error)
        else:
            st.session_state["multiples_result"] = _serialise_multiples(bundle)
            st.rerun()

    stored = st.session_state.get("multiples_result")
    if not stored:
        return
    _render_multiples(stored, profile, peers)


def _peer_form(profile: CompanyProfile) -> None:
    with st.form("peer_form", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            name = st.text_input("Nom *")
            ticker = st.text_input("Ticker")
        with col2:
            currency = st.selectbox("Devise", [c.value for c in Currency],
                                    index=[c.value for c in Currency].index(profile.currency.value))
            sector = st.text_input("Secteur")
        with col3:
            as_of = st.date_input("Date des données", value=_dt.date.today())
            include = st.checkbox("Inclure dans les statistiques", value=True)
        st.markdown("**Données (au moins l'une des colonnes de droite est nécessaire)**")
        col4, col5, col6, col7 = st.columns(4)
        with col4:
            market_cap = st.text_input("Capitalisation (millions)")
            book_value = st.text_input("Capitaux propres (millions)")
        with col5:
            revenue = st.text_input("Chiffre d'affaires (millions)")
            net_income = st.text_input("Résultat net (millions)")
        with col6:
            ebitda = st.text_input("EBITDA (millions)")
            enterprise_value = st.text_input("Valeur d'entreprise (millions)")
        with col7:
            pe = st.text_input("P/E déclaré")
            ev_ebitda = st.text_input("VE/EBITDA déclaré")
        notes = st.text_input("Note (source des chiffres)")
        submitted = st.form_submit_button("Enregistrer le comparable", type="primary")

    if not submitted:
        return
    if not name.strip():
        st.error("Le nom est obligatoire.")
        return
    peer = ComparableCompany(
        name=name.strip(),
        ticker=ticker.strip() or None,
        sector=sector.strip() or None,
        currency=Currency(currency),
        as_of=as_of,
        market_cap=_to_float(market_cap),
        enterprise_value=_to_float(enterprise_value),
        revenue=_to_float(revenue),
        ebitda=_to_float(ebitda),
        net_income=_to_float(net_income),
        book_value=_to_float(book_value),
        pe=_to_float(pe),
        ev_ebitda=_to_float(ev_ebitda),
        include=include,
        notes=notes.strip() or None,
    )
    repo.upsert_peers(db(), profile.company_id, [peer])
    st.success(f"{peer.name} enregistré.")
    st.rerun()


def _to_float(raw: str) -> Optional[float]:
    from data_sources.parsers import parse_number

    if raw is None or not str(raw).strip():
        return None
    return parse_number(raw)


def _serialise_multiples(bundle: workflows.MultiplesBundle) -> Dict[str, Any]:
    result = bundle.result
    return {
        "peers_included": [
            {
                "name": peer.name, "ticker": peer.ticker,
                "currency": peer.currency.value, "pe": peer.pe,
                "ev_ebitda": peer.ev_ebitda, "pb": peer.pb,
                "market_cap": peer.market_cap, "notes": peer.notes,
            }
            for peer in result.peers_included
        ],
        "stats": [
            {
                "label": stats.label, "count": stats.count, "median": stats.median,
                "mean": stats.mean, "minimum": stats.minimum, "maximum": stats.maximum,
                "sufficient": stats.sufficient, "note": stats.note,
                "peers_used": list(stats.peers_used),
            }
            for stats in result.stats.values()
        ],
        "implied": [
            {
                "metric": iv.metric, "statistic": iv.statistic, "multiple": iv.multiple,
                "implied_enterprise_value": iv.implied_enterprise_value,
                "implied_equity_value": iv.implied_equity_value,
                "implied_value_per_share": iv.implied_value_per_share,
                "basis": iv.basis, "formula": iv.formula,
            }
            for iv in result.implied
        ],
        "peers_excluded": [{"name": peer.name, "reason": reason} for peer, reason in result.peers_excluded],
        "valuation_range": list(result.valuation_range) if result.valuation_range else None,
        "median_implied_value": result.median_implied_value,
        "currency": result.target.currency.value,
        "current_price": result.target.current_price,
        "method_note": result.method_note,
        "warnings": list(result.warnings),
    }


def _render_multiples(stored: Dict[str, Any], profile: CompanyProfile, peers: Sequence[ComparableCompany]) -> None:
    st.divider()
    st.markdown("### Résultat des comparables")
    currency = stored["currency"]
    kpi_row(
        [
            (
                "Médiane implicite",
                C.fmt_number(stored["median_implied_value"], 2),
                f"{currency} par action",
            ),
            (
                "Fourchette",
                (
                    f"{C.fmt_number(stored['valuation_range'][0], 2)} – "
                    f"{C.fmt_number(stored['valuation_range'][1], 2)}"
                    if stored.get("valuation_range")
                    else C.MISSING_GLYPH
                ),
                "min – max des multiples appliqués",
            ),
            ("Pairs retenus", str(len(stored["peers_included"])), "utilisés dans la médiane"),
            ("Pairs écartés", str(len(stored["peers_excluded"])), "devise, données manquantes ou valeur non significative"),
        ]
    )

    if stored["stats"]:
        st.markdown("#### Statistiques par multiple")
        show_frame(
            pd.DataFrame(
                [
                    {
                        "Multiple": stats["label"],
                        "N": stats["count"],
                        "Min": stats["minimum"],
                        "Médiane": stats["median"],
                        "Moyenne": stats["mean"],
                        "Max": stats["maximum"],
                        "Robustesse": "suffisante" if stats["sufficient"] else (stats["note"] or "insuffisante"),
                    }
                    for stats in stored["stats"]
                ]
            )
        )
    if stored["implied"]:
        st.markdown("#### Valeur implicite par action")
        show_frame(
            pd.DataFrame(
                [
                    {
                        "Base": f"{row['metric']} · {row['statistic']}",
                        "Multiple appliqué": row["multiple"],
                        "Valeur d'entreprise": row["implied_enterprise_value"],
                        "Fonds propres": row["implied_equity_value"],
                        "Par action": row["implied_value_per_share"],
                        "Assiette": row["basis"],
                    }
                    for row in stored["implied"]
                ]
            )
        )
        chart(_multiples_figure(stored))
    if stored["peers_excluded"]:
        st.markdown("#### Pairs écartés et raison")
        show_frame(pd.DataFrame([{"Pair": p["name"], "Raison": p["reason"]} for p in stored["peers_excluded"]]))
    for warning in stored["warnings"][:5]:
        st.warning(warning)
    if stored.get("method_note"):
        st.caption(stored["method_note"])

    st.divider()
    _report_section(profile, get_dataset(profile.company_id))


def _multiples_figure(stored: Dict[str, Any]):
    from financial_engine.multiples import ImpliedValue, MultiplesResult, TargetValuationInputs

    implied = [
        ImpliedValue(
            metric=row["metric"], statistic=row["statistic"], multiple=row["multiple"],
            implied_enterprise_value=row["implied_enterprise_value"],
            implied_equity_value=row["implied_equity_value"],
            implied_value_per_share=row["implied_value_per_share"],
            basis=row["basis"], formula=row["formula"],
        )
        for row in stored["implied"]
    ]
    target = TargetValuationInputs(
        name="cible", currency=Currency(stored["currency"]), net_debt=None,
        diluted_shares=None, current_price=stored.get("current_price"),
    )
    result = MultiplesResult(
        target=target, peers_included=[], peers_excluded=[], stats={},
        implied=implied, valuation_range=tuple(stored["valuation_range"]) if stored.get("valuation_range") else None,
        median_implied_value=stored.get("median_implied_value"),
    )
    return charts.valuation_range(result)


# ==========================================================================
# Page 6 — AI assistant
# ==========================================================================

def page_assistant() -> None:
    profile = current_profile()
    st.markdown("## Assistant IA")
    app_settings = settings()

    if not app_settings.ai_enabled:
        st.markdown(
            f'<div class="erai-warn"><strong>{C.AI_DISABLED_NOTE}</strong></div>',
            unsafe_allow_html=True,
        )
        st.markdown("#### Ce que fait l'assistant quand il est activé")
        st.markdown(
            "L'assistant ne répond jamais de mémoire : il appelle des fonctions déterministes "
            "de l'application et cite leurs résultats. Les fonctions disponibles sont :"
        )
        from agent.tools import TOOL_DEFINITIONS

        show_frame(
            pd.DataFrame(
                [
                    {
                        "Fonction": definition.name,
                        "Rôle": definition.description.split(".")[0] + ".",
                        "Société requise": "oui" if definition.requires_company else "non",
                    }
                    for definition in TOOL_DEFINITIONS
                ]
            )
        )
        with st.expander("Comment activer l'assistant"):
            st.code(
                "# fichier .env\n"
                "ERAI_LLM_BASE_URL=https://api.groq.com/openai/v1\n"
                "ERAI_LLM_API_KEY=votre_cle\n"
                "ERAI_LLM_MODEL=llama-3.3-70b-versatile\n",
                language="bash",
            )
            st.caption(
                "Tout point d'accès compatible OpenAI convient (Groq, Gemini compat, "
                "OpenRouter, Mistral, Ollama local…). Redémarrez Streamlit après modification."
            )
        return

    if profile is None:
        st.info("Sélectionnez une société dans la barre latérale pour contextualiser les réponses.")
        return

    from agent.financial_agent import FinancialAgent
    from agent.llm_provider import ChatMessage, get_provider
    from agent.prompts import suggested_questions
    from agent.tools import ToolContext

    dataset = get_dataset(profile.company_id)
    simulated_banner(dataset)

    provider = get_provider(app_settings)
    st.caption(
        f"Modèle : `{app_settings.llm_model}` via `{app_settings.llm_base_url}` · "
        f"société contextualisée : {profile.name} · les appels d'outils sont journalisés "
        "sous chaque réponse."
    )

    ctx = ToolContext(
        db=db(),
        settings=app_settings,
        reports_dir=app_settings.reports_dir,
        default_company_id=profile.company_id,
    )
    _restore_assistant_cache(ctx)

    history: List[Dict[str, Any]] = st.session_state.setdefault("assistant_history", [])

    if not history:
        st.markdown("#### Questions types")
        columns = st.columns(2)
        for index, question in enumerate(suggested_questions(get_lang())):
            with columns[index % 2]:
                if st.button(question, key=f"suggested_{index}", width="stretch"):
                    st.session_state["assistant_pending"] = question
                    st.rerun()

    for entry in history:
        with st.chat_message("user"):
            st.markdown(entry["question"])
        with st.chat_message("assistant", avatar="✦"):
            st.markdown(entry["answer"])
            if entry.get("error"):
                st.error(entry["error"])
            invocations = entry.get("invocations", [])
            if invocations:
                with st.expander(f"🔧 {len(invocations)} appel(s) de fonction", expanded=False):
                    for invocation in invocations:
                        status = "✅" if invocation["ok"] else "⚠️"
                        st.markdown(
                            f'<div class="erai-tool">{status} {invocation["tool"]}'
                            f'({", ".join(f"{k}={v!r}" for k, v in invocation["arguments"].items())})'
                            f" → {invocation['summary']}</div>",
                            unsafe_allow_html=True,
                        )
                        with st.popover("Détail"):
                            st.json(invocation["payload"], expanded=False)
            if entry.get("tools"):
                st.caption("Fonctions utilisées : " + ", ".join(entry["tools"]))

    pending = st.session_state.pop("assistant_pending", None)
    question = st.chat_input("Posez une question sur les données chargées…")
    question = question or pending
    if not question:
        if st.button("Effacer la conversation", key="clear_chat"):
            st.session_state["assistant_history"] = []
            st.rerun()
        return

    with st.chat_message("user"):
        st.markdown(question)

    agent = FinancialAgent(ctx, provider, lang=get_lang())
    api_history = []
    for entry in history[-4:]:
        api_history.append(ChatMessage("user", entry["question"]))
        api_history.append(ChatMessage("assistant", entry["answer"]))

    with st.chat_message("assistant", avatar="✦"):
        with st.spinner("Appel des fonctions et rédaction…"):
            run = agent.run(question, history=api_history)
        st.markdown(run.answer)
        if run.error:
            st.error(run.error)
        if run.invocations:
            with st.expander(f"🔧 {len(run.invocations)} appel(s) de fonction", expanded=False):
                for invocation in run.invocations:
                    status = "✅" if invocation.ok else "⚠️"
                    st.markdown(
                        f'<div class="erai-tool">{status} {invocation.name}'
                        f'({", ".join(f"{k}={v!r}" for k, v in invocation.arguments.items())})'
                        f" → {invocation.summary}</div>",
                        unsafe_allow_html=True,
                    )
                    with st.popover("Détail"):
                        st.json(invocation.payload, expanded=False)
        st.caption(
            f"Modèle : {run.model or app_settings.llm_model} · tours d'outils : {run.iterations}"
            + (" · ⚠️ limite atteinte" if run.stopped_early else "")
        )

    history.append(
        {
            "question": question,
            "answer": run.answer,
            "error": run.error,
            "tools": run.tools_used(),
            "invocations": [invocation.as_dict() for invocation in run.invocations],
        }
    )
    st.session_state["assistant_history"] = history
    _persist_assistant_cache(ctx)
    st.rerun()


def _restore_assistant_cache(ctx) -> None:
    """Keep the last computed valuation available to the tools between reruns."""
    stored = st.session_state.get("valuation_result")
    if not stored:
        return
    try:
        from workflows import ValuationBundle, DCFSetup

        results = _results_to_objects(stored["results"])
        ctx.cache["valuation"] = ValuationBundle(
            setup=DCFSetup(inputs=None, assumptions=stored.get("assumptions", {}),
                           provenance=stored.get("provenance", {})),
            results=results,
            assumptions=stored.get("assumptions", {}),
            provenance=stored.get("provenance", {}),
            warnings=stored.get("warnings", []),
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Cache de valorisation non restauré : %s", exc)


def _persist_assistant_cache(ctx) -> None:
    """Write back anything the tools computed, so the UI stays consistent."""
    bundle = ctx.cache.get("valuation")
    if bundle is not None and getattr(bundle, "ok", False):
        st.session_state["valuation_result"] = _serialise_bundle(bundle)
    multiples = ctx.cache.get("multiples")
    if multiples is not None and getattr(multiples, "ok", False):
        st.session_state["multiples_result"] = _serialise_multiples(multiples)


# ==========================================================================
# Page 7 — Sources & data quality
# ==========================================================================

def page_sources() -> None:
    profile = current_profile()
    if profile is None:
        st.info("Sélectionnez une société dans la barre latérale.")
        return
    dataset = get_dataset(profile.company_id)
    st.markdown(f"## Sources et qualité des données — {profile.name}")
    simulated_banner(dataset)

    sources = repo.list_sources(db(), profile.company_id)
    quality = workflows.assess(dataset) if dataset.observations else None

    tab_sources, tab_quality, tab_coverage, tab_overrides, tab_web = st.tabs(
        ["Documents sources", "Contrôles", "Couverture", "Saisies manuelles", "Recherche web"]
    )

    with tab_sources:
        if sources:
            st.caption(
                "Conservation systématique : nom, éditeur, date de publication, horodatage de "
                "récupération, devise et échelle. Ces métadonnées accompagnent chaque valeur."
            )
            show_frame(C.sources_frame(sources))
            for source in sources:
                if source.notes:
                    st.caption(f"**{source.source_id}** — {source.notes}")
        else:
            st.info(
                "Aucun document source enregistré. Les chiffres présents ne sont donc pas "
                "traçables : importez les états financiers ou renseignez la source de votre "
                "saisie manuelle."
            )
        with st.expander("Enregistrer un document source manuellement"):
            _source_form(profile)

    with tab_quality:
        if quality is None:
            st.info("Aucune donnée à contrôler.")
        else:
            kpi_row(
                [
                    ("Erreurs", str(len(quality.errors)), "à corriger en priorité"),
                    ("Avertissements", str(len(quality.warnings)), "à examiner"),
                    ("Informations", str(len(quality.infos)), "contexte"),
                    ("Score", f"{getattr(quality, 'score', 0):.0f}/100" if getattr(quality, "score", None) is not None else C.MISSING_GLYPH, "heuristique interne"),
                ]
            )
            if quality.issues:
                show_frame(C.issues_frame(quality))
            else:
                st.success("Aucune anomalie détectée par les contrôles automatiques.")
            with st.expander("Ce que vérifient les contrôles"):
                st.markdown(
                    "- Identités comptables (actif = passif, résultat brut = CA − coût des ventes)\n"
                    "- Cohérence des signes (investissements, impôt, dividendes)\n"
                    "- Devises et échelles mélangées\n"
                    "- Doublons et retraitements entre exercices\n"
                    "- Somme des trimestres vs exercice\n"
                    "- Valeurs aberrantes et division par zéro"
                )

    with tab_coverage:
        if not dataset.observations:
            st.info("Aucune donnée chargée.")
        else:
            st.caption(
                "Statut de chaque poste et de chaque exercice. Une cellule vide signifie "
                "« donnée absente », jamais « zéro »."
            )
            show_frame(C.coverage_frame(dataset))
            absent = C.missing_items(dataset, list(C.LINE_ITEM_LABELS_FR.keys()))
            missing_banner(absent, "Postes absents à la dernière période")

    with tab_overrides:
        overrides = repo.get_manual_overrides(db(), profile.company_id)
        if overrides:
            show_frame(pd.DataFrame([{"Champ": k, "Valeur": v} for k, v in overrides.items()]))
        else:
            st.caption(
                "Aucune surcharge manuelle enregistrée. Les surcharges servent à fixer une "
                "valeur de référence (par exemple un cours ou un nombre d'actions vérifié) "
                "sans modifier les états financiers."
            )
        with st.form("override_form"):
            col1, col2, col3 = st.columns([2, 2, 3])
            with col1:
                field = st.selectbox(
                    "Champ",
                    ["reference_price", "shares_outstanding", "net_debt", "market_cap"],
                )
            with col2:
                value = st.text_input("Valeur")
            with col3:
                note = st.text_input("Note / justification")
            if st.form_submit_button("Enregistrer la surcharge"):
                parsed = _to_float(value)
                if parsed is None:
                    st.error("Valeur numérique invalide.")
                else:
                    repo.set_manual_override(db(), profile.company_id, field, parsed, note)
                    st.success(f"{field} = {parsed:,.4f} enregistré.")
                    st.rerun()

    with tab_web:
        _web_search_section(profile)

    st.divider()
    _report_section(profile, dataset)


def _web_search_section(profile: CompanyProfile) -> None:
    """Tier-2 web leads: titles, URLs and raw snippets — never figures."""
    from data_sources.web_search import get_search_provider, research_brief

    app_settings = settings()
    if not app_settings.search_enabled:
        st.info(
            "La recherche web est désactivée. Renseignez `ERAI_SEARCH_PROVIDER=duckduckgo` "
            "(gratuit, sans clé) dans le fichier `.env` en local, ou dans les **secrets** "
            "Streamlit (Manage app → Settings → Secrets) sur Streamlit Cloud, puis "
            "redémarrez l'application. Les autres fournisseurs sont décrits dans la page "
            "Paramètres."
        )
        return

    st.caption(
        "Pistes issues de moteurs publics (Tier 2) : titres, URL et extraits bruts. "
        "Ce ne sont pas des chiffres officiels — vérifiez la valeur dans le document "
        "source avant toute utilisation. Les résultats ne sont pas enregistrés en base."
    )
    default_query = f"{profile.name} résultats financiers {_dt.date.today().year}"
    configured_max = int(getattr(app_settings, "search_max_results", 8) or 8)
    with st.form("web_search_form"):
        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_input("Requête de recherche", value=default_query)
        with col2:
            max_results = int(
                st.number_input(
                    "Résultats max",
                    min_value=3,
                    max_value=15,
                    value=min(max(configured_max, 3), 15),
                    step=1,
                )
            )
        brief = st.checkbox(
            "Lancer le brief complet (4 requêtes standard : résultats, guidance, "
            "marché, risques)",
            value=False,
        )
        submitted = st.form_submit_button("Rechercher")

    state_key = f"web_search_results_{profile.company_id}"
    if submitted:
        provider = get_search_provider(app_settings)
        try:
            with st.spinner("Recherche en cours…"):
                if brief:
                    responses = research_brief(
                        profile.name, getattr(profile, "sector", None) or "", settings=app_settings
                    )
                else:
                    responses = [provider.search(query, max_results=max_results)]
        except Exception as exc:  # provider/network errors are surfaced, never hidden
            logger.warning("Recherche web échouée : %s", exc)
            st.error(f"Recherche impossible : {exc}")
            return
        st.session_state[state_key] = [
            {
                "query": response.query,
                "provider": response.provider,
                "error": response.error,
                "note": response.note,
                "results": list(response.results),
            }
            for response in responses
        ]

    stored = st.session_state.get(state_key)
    if not stored:
        st.caption(
            "Saisissez une requête puis cliquez sur « Rechercher ». Les pistes "
            "s'affichent ici avec leur URL, sans être versées dans la base."
        )
        return
    for response in stored:
        if len(stored) > 1:
            st.markdown(f"**Requête : {response['query']}**")
        if response["error"]:
            st.warning(f"Fournisseur « {response['provider']} » : {response['error']}")
            continue
        if not response["results"]:
            st.warning(response["note"] or "Aucun résultat.")
            continue
        show_frame(C.web_results_frame(response["results"]))
        st.caption(
            f"Fournisseur : {response['provider']} — pistes non vérifiées. Pour "
            "utiliser un chiffre repéré ici : ouvrez le document, vérifiez la valeur, "
            "puis saisissez-la (onglet « Saisies manuelles ») en conservant sa source."
        )


def _source_form(profile: CompanyProfile) -> None:
    with st.form("source_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Nom du document *")
            url = st.text_input("URL")
            publisher = st.text_input("Éditeur")
        with col2:
            document_type = st.text_input("Type (rapport annuel, communiqué, 10-K…)")
            published = st.date_input("Date de publication", value=_dt.date.today())
            currency = st.selectbox("Devise du document", [c.value for c in Currency])
        scale = st.selectbox("Échelle des montants du document", [s.value for s in Scale], index=2)
        notes = st.text_area("Notes", height=80)
        submitted = st.form_submit_button("Enregistrer la source")

    if not submitted or not name.strip():
        if submitted:
            st.error("Le nom du document est obligatoire.")
        return
    source_id = f"manual:{abs(hash((name.strip(), str(published), profile.company_id))) % 10**10}"
    repo.upsert_source(
        db(),
        repo.SourceRecord(
            source_id=source_id,
            name=name.strip(),
            url=url.strip() or None,
            publisher=publisher.strip() or None,
            publication_date=published,
            document_type=document_type.strip() or None,
            currency=Currency(currency),
            scale=Scale(scale),
            notes=notes.strip() or None,
        ),
        profile.company_id,
    )
    st.success(f"Source « {name} » enregistrée sous l'identifiant {source_id}.")
    st.rerun()


# ==========================================================================
# Page 8 — Settings
# ==========================================================================

def page_settings() -> None:
    st.markdown("## Paramètres")
    app_settings = settings()

    st.markdown("### État de la configuration")
    yfinance_ok = _module_available("yfinance")
    rows = [
        ("Assistant IA", "activé" if app_settings.ai_enabled else "désactivé",
         f"{app_settings.llm_model} via {app_settings.llm_base_url}" if app_settings.ai_enabled else "renseignez ERAI_LLM_API_KEY"),
        ("Recherche web", app_settings.search_provider if app_settings.search_enabled else "désactivée",
         "via .env ou secrets Streamlit : ERAI_SEARCH_PROVIDER" if not app_settings.search_enabled else f"maximum {app_settings.search_max_results} résultats"),
        ("Données de marché US (yfinance)", "disponible" if yfinance_ok else "non installé",
         "source non officielle, indicative" if yfinance_ok else "pip install -r requirements-optional.txt"),
        ("User-Agent SEC", "configuré" if "example.com" not in app_settings.sec_user_agent else "valeur d'exemple",
         app_settings.sec_user_agent),
        ("Langue", get_lang(), "modifiable dans la barre latérale ou via ERAI_LANG"),
        ("Base de données", app_settings.db_path.name, str(app_settings.db_path.parent)),
    ]
    show_frame(pd.DataFrame(rows, columns=["Élément", "État", "Détail"]))
    st.caption(
        "Configuration : en local, fichier `.env` (non versionné). Sur Streamlit Cloud : "
        "Manage app → Settings → Secrets, au format TOML — par exemple "
        "`ERAI_SEARCH_PROVIDER = \"duckduckgo\"`. Les variables déjà présentes dans "
        "l'environnement restent prioritaires."
    )

    if app_settings.ai_enabled:
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Tester la connexion au modèle"):
                from agent.llm_provider import check_connection, get_provider

                with st.spinner("Appel de test…"):
                    outcome = check_connection(get_provider(app_settings))
                if outcome.get("ok"):
                    st.success(f"Connexion réussie · modèle {outcome.get('model')} · réponse « {outcome.get('reply')} »")
                else:
                    st.error(f"Échec : {outcome.get('error')}")
        with col2:
            st.caption(
                "Le test envoie un unique message court. La clé d'API n'est jamais écrite en "
                "base ni affichée : elle est lue depuis l'environnement à chaque appel."
            )

    st.divider()
    st.markdown("### Stockage")
    database = db()
    info = database.info()
    counts = database.table_counts()
    kpi_row(
        [
            ("Taille", f"{info.size_bytes / 1024:.0f} Kio", info.path.name),
            ("Tables", str(info.tables), f"schéma v{info.schema_version}"),
            ("Observations", str(counts.get("observations", 0)), "lignes de données"),
            ("Sources", str(counts.get("sources", 0)), "documents référencés"),
            ("Valorisations", str(counts.get("valuation_runs", 0)), "calculs conservés"),
            ("Cache HTTP", str(counts.get("http_cache", 0)), "réponses mises en cache"),
        ]
    )
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button("Vider le cache HTTP"):
            with database.connect() as conn:
                deleted = conn.execute("DELETE FROM http_cache").rowcount
            st.success(f"{deleted} entrée(s) supprimée(s).")
    with col2:
        if st.button("Recharger les données"):
            bump_data_version()
            st.session_state.pop("valuation_result", None)
            st.session_state.pop("multiples_result", None)
            st.success("Cache applicatif vidé.")
            st.rerun()
    with col3:
        st.caption(
            "Le cache HTTP évite de solliciter à nouveau les serveurs publics. Videz-le si "
            "vous suspectez des données obsolètes."
        )

    st.divider()
    st.markdown("### Données en base")
    companies = repo.list_companies(database)
    if companies:
        show_frame(
            pd.DataFrame(
                [
                    {
                        "Société": company.name,
                        "Identifiant": company.company_id,
                        "Ticker": company.ticker,
                        "Devise": company.currency.value,
                        "Observations": len(get_dataset(company.company_id).observations),
                        "Rapports": len(repo.list_reports(database, company.company_id)),
                    }
                    for company in companies
                ]
            )
        )

    with st.expander("⚠️ Zone sensible — supprimer des données"):
        target = st.selectbox("Société", [c.company_id for c in companies] or ["—"], key="danger_company")
        action = st.radio(
            "Action",
            [
                "Supprimer les observations",
                "Supprimer les comparables",
                "Supprimer les sources",
                "Réinitialiser toute la base",
            ],
            key="danger_action",
        )
        confirm = st.text_input("Tapez SUPPRIMER pour confirmer", key="danger_confirm")
        if st.button("Exécuter la suppression", type="primary", disabled=confirm != "SUPPRIMER"):
            if action == "Supprimer les observations":
                count = repo.delete_observations(database, target)
                st.success(f"{count} observation(s) supprimée(s).")
            elif action == "Supprimer les comparables":
                peers = repo.list_peers(database, target)
                for peer in peers:
                    repo.delete_peer(database, target, peer.name)
                st.success(f"{len(peers)} comparable(s) supprimé(s).")
            elif action == "Supprimer les sources":
                with database.connect() as conn:
                    conn.execute("DELETE FROM sources WHERE company_id = ?", (target,))
                st.success("Sources supprimées.")
            else:
                database.reset()
                st.success("Base réinitialisée.")
            bump_data_version()
            st.rerun()

    st.divider()
    st.markdown("### Rapports générés")
    reports = repo.list_reports(database)
    if reports:
        show_frame(
            pd.DataFrame(
                [
                    {
                        "Titre": row.get("title"),
                        "Société": row.get("company_id"),
                        "Créé le": row.get("created_at"),
                        "Fichier": Path(row.get("path", "")).name,
                    }
                    for row in reports
                ]
            )
        )
        st.caption(
            "Les rapports sont écrits dans `data/reports`. Utilisez le bouton de la section "
            "« Note de recherche » pour en produire un nouveau."
        )
    else:
        st.info("Aucun rapport généré pour le moment.")

    st.divider()
    st.markdown("### Valorisations enregistrées")
    runs = repo.list_valuation_runs(database, limit=20)
    if runs:
        show_frame(
            pd.DataFrame(
                [
                    {
                        "Date": row.get("created_at"),
                        "Société": row.get("company_id"),
                        "Méthode": row.get("method"),
                        "Scénario": row.get("scenario"),
                        "Par action": row.get("value_per_share"),
                        "Devise": row.get("currency"),
                        "Simulé": "oui" if row.get("simulated") else "non",
                    }
                    for row in runs
                ]
            )
        )
    else:
        st.info("Aucune valorisation enregistrée.")

    st.divider()
    sales_pitch_note()


def _module_available(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


# ==========================================================================
# Report section (shared)
# ==========================================================================

def _report_section(profile: CompanyProfile, dataset: FinancialDataSet, extra_title: str = "") -> None:
    st.markdown(f"### {extra_title or 'Note de recherche'}")
    st.caption(
        "Le rapport est un document HTML imprimable et un PDF : dix sections, faits et "
        "hypothèses séparés, chaque chiffre accompagné de son statut et de sa source."
    )
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        want_html = st.checkbox("HTML", value=True, key=f"rep_html_{extra_title}")
    with col2:
        want_pdf = st.checkbox("PDF", value=True, key=f"rep_pdf_{extra_title}")
    with col3:
        include_ai = st.checkbox(
            "Inclure la dernière réponse de l'assistant",
            value=False,
            key=f"rep_ai_{extra_title}",
            help="Le texte généré est inséré dans une section étiquetée « à revérifier ».",
        )

    if st.button("Générer le rapport", type="primary", key=f"rep_go_{extra_title}"):
        formats = [f for f, wanted in (("html", want_html), ("pdf", want_pdf)) if wanted]
        if not formats:
            st.warning("Choisissez au moins un format.")
            return
        narrative = None
        if include_ai:
            history = st.session_state.get("assistant_history") or []
            if history:
                narrative = history[-1]["answer"]
            else:
                st.info("Aucune réponse de l'assistant à inclure.")
        with st.spinner("Rédaction du rapport…"):
            try:
                valuation = _bundle_from_session()
                multiples = _multiples_from_session()
                written = workflows.generate_company_report(
                    profile=profile,
                    dataset=dataset,
                    sources=repo.list_sources(db(), profile.company_id),
                    reports_dir=settings().reports_dir,
                    valuation=valuation,
                    multiples=multiples,
                    quality=workflows.assess(dataset) if dataset.observations else None,
                    current_price=(valuation.current_price if valuation else None),
                    ai_narrative=narrative,
                    formats=formats,
                )
            except Exception as exc:
                st.error(f"Génération impossible : {exc}")
                logger.exception("Report generation failed")
                return
        for fmt, path in written.items():
            repo.save_report(
                db(),
                report_id=f"{profile.company_id}-{fmt}-{_dt.datetime.now():%Y%m%d%H%M%S}",
                company_id=profile.company_id,
                title=f"Note de recherche {profile.name} ({fmt.upper()})",
                path=str(path),
                meta={"format": fmt, "generated_at": _dt.datetime.now().isoformat()},
            )
            if fmt == "html":
                content = path.read_text(encoding="utf-8")
                st.download_button(
                    "⬇ Télécharger le rapport HTML",
                    data=content.encode("utf-8"),
                    file_name=path.name,
                    mime="text/html",
                    key=f"dl_html_{extra_title}",
                )
            else:
                st.download_button(
                    "⬇ Télécharger le rapport PDF",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime="application/pdf",
                    key=f"dl_pdf_{extra_title}",
                )
            st.caption(f"Fichier écrit : {path}")


def _bundle_from_session() -> Optional[workflows.ValuationBundle]:
    stored = st.session_state.get("valuation_result")
    if not stored:
        return None
    bundle = workflows.ValuationBundle(
        setup=workflows.DCFSetup(inputs=None, assumptions=stored.get("assumptions", {}),
                                 provenance=stored.get("provenance", {}), missing=stored.get("missing", [])),
        results=_results_to_objects(stored["results"]),
        current_price=stored["base"].get("current_price") if stored.get("base") else None,
        warnings=stored.get("warnings", []),
        assumptions=stored.get("assumptions", {}),
        provenance=stored.get("provenance", {}),
    )
    grid = stored.get("grid")
    if grid:
        from financial_engine.sensitivity import SensitivityGrid

        bundle.sensitivity = SensitivityGrid(
            row_label=grid["row_label"], col_label=grid["col_label"],
            row_values=list(grid["row_values"]), col_values=list(grid["col_values"]),
            values=[list(row) for row in grid["values"]], enterprise_values=[], current_price=None,
        )
    tornado = stored.get("tornado")
    if tornado:
        from financial_engine.dcf import DCFResult
        from financial_engine.sensitivity import TornadoBar, TornadoResult

        base = DCFResult(
            currency=tornado["currency"], units="millions", scenario="base", label="",
            years=[], wacc=0.0, terminal_growth=0.0, tax_rate=0.0, pv_explicit=0.0,
            terminal_value=0.0, pv_terminal_value=0.0, enterprise_value=0.0, net_debt=0.0,
            other_senior_claims=0.0, non_operating_assets=0.0,
            equity_value=0.0, shares_outstanding=0.0, value_per_share=tornado["base"],
        )
        bars = []
        for bar in tornado["bars"]:
            payload = dict(bar)
            payload.setdefault("low_input", 0.0)
            payload.setdefault("high_input", 0.0)
            bars.append(TornadoBar(**payload))
        bundle.tornado = TornadoResult(base=base, bars=bars)
    return bundle


def _multiples_from_session() -> Optional[workflows.MultiplesBundle]:
    stored = st.session_state.get("multiples_result")
    if not stored:
        return None
    from financial_engine.multiples import ImpliedValue, MultipleStats, MultiplesResult, TargetValuationInputs
    from financial_engine.schemas import ExtractionStatus

    stats = {
        row["label"]: MultipleStats(
            key=row["label"].lower().replace("/", "_"), label=row["label"], count=row["count"],
            median=row["median"], mean=row["mean"], minimum=row["minimum"], maximum=row["maximum"],
            q1=None, q3=None, peers_used=list(row.get("peers_used", [])), excluded=[],
            sufficient=row["sufficient"], note=row.get("note"),
        )
        for row in stored.get("stats", [])
    }
    implied = [
        ImpliedValue(
            metric=row["metric"], statistic=row["statistic"], multiple=row["multiple"],
            implied_enterprise_value=row["implied_enterprise_value"],
            implied_equity_value=row["implied_equity_value"],
            implied_value_per_share=row["implied_value_per_share"],
            basis=row["basis"], formula=row["formula"],
        )
        for row in stored.get("implied", [])
    ]
    result = MultiplesResult(
        target=TargetValuationInputs(
            name="cible", currency=Currency(stored["currency"]), net_debt=None,
            diluted_shares=None, current_price=stored.get("current_price"),
        ),
        peers_included=[], peers_excluded=[], stats=stats, implied=implied,
        valuation_range=tuple(stored["valuation_range"]) if stored.get("valuation_range") else None,
        median_implied_value=stored.get("median_implied_value"),
        warnings=list(stored.get("warnings", [])), method_note=stored.get("method_note"),
    )
    return workflows.MultiplesBundle(result=result)


# ==========================================================================
# Router
# ==========================================================================

RENDERERS = {
    "dashboard": page_dashboard,
    "research": page_research,
    "statements": page_statements,
    "valuation": page_valuation,
    "comparables": page_comparables,
    "assistant": page_assistant,
    "sources": page_sources,
    "settings": page_settings,
}


def main() -> None:
    app_settings = settings()
    app_settings.ensure_dirs()
    seed_registry_companies()

    if "company_id" not in st.session_state:
        companies = repo.list_companies(db())
        if companies:
            st.session_state["company_id"] = companies[0].company_id

    page = render_sidebar()
    renderer = RENDERERS.get(page, page_dashboard)
    try:
        renderer()
    except Exception as exc:  # pragma: no cover - last-resort guard
        logger.exception("Page %s failed", page)
        st.error(
            "Une erreur est survenue pendant l'affichage de cette page. Les détails sont dans "
            f"la console. Détail technique : {type(exc).__name__}: {exc}"
        )
        with st.expander("Trace complète"):
            import traceback

            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
