"""Internationalisation.

French is the default interface language; the catalogue is keyed so that
English can be added without touching the page code.  Technical documentation
and code stay in English.
"""

from __future__ import annotations

import streamlit as st

DEFAULT_LANG = "fr"
SUPPORTED = ("fr", "en")

TRANSLATIONS: dict[str, dict[str, str]] = {
    "fr": {
        # ---- navigation -------------------------------------------------
        "nav.title": "Recherche Actions",
        "nav.dashboard": "Tableau de bord",
        "nav.research": "Recherche société",
        "nav.statements": "Etats financiers",
        "nav.valuation": "Valorisation",
        "nav.comparables": "Comparables",
        "nav.assistant": "Assistant IA",
        "nav.sources": "Sources et qualité",
        "nav.settings": "Paramètres",
        "nav.company": "Société",
        "nav.language": "Langue",
        # ---- common -----------------------------------------------------
        "common.company": "Société",
        "common.ticker": "Ticker",
        "common.exchange": "Place",
        "common.currency": "Devise",
        "common.period": "Période",
        "common.source": "Source",
        "common.value": "Valeur",
        "common.formula": "Formule",
        "common.status": "Statut",
        "common.unavailable": "Indisponible",
        "common.not_applicable": "Non applicable",
        "common.manual_entry": "Saisie manuelle",
        "common.simulated": "Données SIMULEES",
        "common.actual": "Données réelles",
        "common.estimated": "Estimation",
        "common.loading": "Chargement...",
        "common.no_data": "Aucune donnée disponible",
        "common.warning": "Avertissement",
        "common.error": "Erreur",
        "common.info": "Information",
        "common.download": "Telecharger",
        "common.millions": "millions",
        # ---- statuses ---------------------------------------------------
        "status.reported": "Publié",
        "status.derived": "Calcule (derive)",
        "status.manual": "Saisie manuelle",
        "status.estimated": "Estimation analyste",
        "status.unverified": "Non vérifié",
        "status.missing": "Manquant",
        # ---- pages ------------------------------------------------------
        "dashboard.title": "Tableau de bord",
        "dashboard.subtitle": "Synthèse de la société sélectionnée",
        "dashboard.market_cap": "Capitalisation",
        "dashboard.price": "Cours",
        "dashboard.price_time": "Horodatage du cours",
        "dashboard.latest_period": "Dernière période publiée",
        "dashboard.data_quality": "Qualité des données",
        "dashboard.no_price": "Cours indisponible pour cette société",
        "valuation.no_dcf": "Le DCF ne peut pas être calculé : des données critiques manquent.",
        "valuation.intrinsic": "Valeur intrinseque estimee",
        "valuation.upside": "Potentiel",
        "valuation.reliability": "Fiabilité du modèle",
        "sources.disclaimer": (
            "Les données de marché et les estimations sont distinctes des comptes "
            "publiés. Vérifiez toujours les chiffres contre les documents officiels."
        ),
        # ---- disclaimer --------------------------------------------------
        "disclaimer": (
            "Outil d'analyse et de recherche. Il ne constitue pas un conseil en "
            "investissement ni une recommandation d'achat ou de vente."
        ),
    },
    "en": {
        "nav.title": "Equity Research",
        "nav.dashboard": "Dashboard",
        "nav.research": "Company research",
        "nav.statements": "Financial statements",
        "nav.valuation": "Valuation",
        "nav.comparables": "Comparables",
        "nav.assistant": "AI assistant",
        "nav.sources": "Sources & quality",
        "nav.settings": "Settings",
        "nav.company": "Company",
        "nav.language": "Language",
        "common.company": "Company",
        "common.ticker": "Ticker",
        "common.exchange": "Exchange",
        "common.currency": "Currency",
        "common.period": "Period",
        "common.source": "Source",
        "common.value": "Value",
        "common.formula": "Formula",
        "common.status": "Status",
        "common.unavailable": "Unavailable",
        "common.not_applicable": "Not applicable",
        "common.manual_entry": "Manual entry",
        "common.simulated": "SIMULATED DATA",
        "common.actual": "Actual data",
        "common.estimated": "Estimate",
        "common.loading": "Loading...",
        "common.no_data": "No data available",
        "common.warning": "Warning",
        "common.error": "Error",
        "common.info": "Information",
        "common.download": "Download",
        "common.millions": "millions",
        "status.reported": "Reported",
        "status.derived": "Calculated (derived)",
        "status.manual": "Manual entry",
        "status.estimated": "Analyst estimate",
        "status.unverified": "Unverified",
        "status.missing": "Missing",
        "dashboard.title": "Dashboard",
        "dashboard.subtitle": "Overview of the selected company",
        "dashboard.market_cap": "Market capitalisation",
        "dashboard.price": "Price",
        "dashboard.price_time": "Price timestamp",
        "dashboard.latest_period": "Latest reported period",
        "dashboard.data_quality": "Data quality",
        "dashboard.no_price": "No price available for this company",
        "valuation.no_dcf": "The DCF cannot be computed: critical inputs are missing.",
        "valuation.intrinsic": "Estimated intrinsic value",
        "valuation.upside": "Upside",
        "valuation.reliability": "Model reliability",
        "sources.disclaimer": (
            "Market data and estimates are kept separate from reported statements. "
            "Always verify figures against the official documents."
        ),
        "disclaimer": (
            "Research and analysis tool. Not investment advice and not a "
            "recommendation to buy or sell any security."
        ),
    },
}


def get_lang() -> str:
    lang = st.session_state.get("lang")
    if lang in SUPPORTED:
        return lang
    try:
        from config import get_settings

        configured = get_settings().lang
    except Exception:
        configured = DEFAULT_LANG
    return configured if configured in SUPPORTED else DEFAULT_LANG


def set_lang(lang: str) -> None:
    if lang in SUPPORTED:
        st.session_state["lang"] = lang


def t(key: str, lang: str | None = None, **kwargs) -> str:
    """Translate ``key``; unknown keys fall back to the key itself."""
    lang = lang or get_lang()
    text = TRANSLATIONS.get(lang, {}).get(key) or TRANSLATIONS[DEFAULT_LANG].get(key) or key
    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, IndexError):
            return text
    return text
