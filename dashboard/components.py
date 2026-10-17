"""Presentation helpers shared by every page of the terminal.

Nothing here computes financial results; these are pure formatting/UI helpers so
that the same disclaimer, the same provenance badge, and the same missing-data
banner appear everywhere.  Keeping them in one module is what stops a figure
from being displayed on one page without its source and on another with it.
"""

from __future__ import annotations

import datetime as _dt
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from financial_engine.schemas import (
    ExtractionStatus,
    FinancialDataSet,
    LineItem,
    Observation,
    SourceRecord,
    SourceTier,
)
from financial_engine.validation import DataQualityReport, Severity

# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------

MISSING_GLYPH = "—"

STATUS_LABELS = {
    ExtractionStatus.REPORTED: "Publié",
    ExtractionStatus.DERIVED: "Calculé",
    ExtractionStatus.MANUAL: "Saisie manuelle",
    ExtractionStatus.ESTIMATED: "Estimation",
    ExtractionStatus.UNVERIFIED: "Non vérifié",
    ExtractionStatus.MISSING: "Absent",
}

STATUS_ORDER = [
    ExtractionStatus.REPORTED,
    ExtractionStatus.DERIVED,
    ExtractionStatus.MANUAL,
    ExtractionStatus.ESTIMATED,
    ExtractionStatus.UNVERIFIED,
    ExtractionStatus.MISSING,
]

LINE_ITEM_LABELS_FR: Dict[LineItem, str] = {
    LineItem.REVENUE: "Chiffre d'affaires",
    LineItem.COST_OF_REVENUE: "Coût des ventes",
    LineItem.GROSS_PROFIT: "Résultat brut",
    LineItem.OPERATING_EXPENSES: "Charges opérationnelles",
    LineItem.EBIT: "Résultat d'exploitation (EBIT)",
    LineItem.OPERATING_INCOME: "Résultat opérationnel",
    LineItem.DEPRECIATION_AMORTIZATION: "Amortissements",
    LineItem.EBITDA: "EBITDA",
    LineItem.INTEREST_EXPENSE: "Charges d'intérêts",
    LineItem.PRETAX_INCOME: "Résultat avant impôt",
    LineItem.INCOME_TAX_EXPENSE: "Impôt sur les sociétés",
    LineItem.NET_INCOME: "Résultat net",
    LineItem.EPS_DILUTED: "BPA dilué",
    LineItem.CASH_AND_EQUIVALENTS: "Trésorerie et équivalents",
    LineItem.SHORT_TERM_INVESTMENTS: "Placements court terme",
    LineItem.ACCOUNTS_RECEIVABLE: "Créances clients",
    LineItem.INVENTORY: "Stocks",
    LineItem.CURRENT_ASSETS: "Actif circulant",
    LineItem.TOTAL_ASSETS: "Total actif",
    LineItem.ACCOUNTS_PAYABLE: "Dettes fournisseurs",
    LineItem.CURRENT_LIABILITIES: "Passif circulant",
    LineItem.SHORT_TERM_DEBT: "Dette court terme",
    LineItem.LONG_TERM_DEBT: "Dette long terme",
    LineItem.TOTAL_DEBT: "Dette financière totale",
    LineItem.TOTAL_LIABILITIES: "Total passif",
    LineItem.TOTAL_EQUITY: "Capitaux propres",
    LineItem.NET_WORKING_CAPITAL: "Besoin en fonds de roulement",
    LineItem.MINORITY_INTEREST: "Intérêts minoritaires",
    LineItem.OPERATING_CASH_FLOW: "Flux de trésorerie d'exploitation",
    LineItem.CAPITAL_EXPENDITURE: "Investissements",
    LineItem.FREE_CASH_FLOW: "Free cash flow",
    LineItem.DIVIDENDS_PAID: "Dividendes versés",
    LineItem.SHARES_OUTSTANDING: "Actions en circulation",
    LineItem.DILUTED_SHARES: "Actions diluées",
    LineItem.MARKET_PRICE: "Cours",
    LineItem.MARKET_CAP: "Capitalisation boursière",
}


def line_item_label(item: LineItem) -> str:
    return LINE_ITEM_LABELS_FR.get(item, item.value.replace("_", " ").title())


def fmt_number(value: Optional[float], digits: int = 0, missing: str = MISSING_GLYPH) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return missing
    return f"{value:,.{digits}f}".replace(",", " ")


def fmt_money(value: Optional[float], currency: str = "", digits: int = 0) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return MISSING_GLYPH
    text = fmt_number(value, digits)
    return f"{text} {currency}".strip()


def fmt_pct(value: Optional[float], digits: int = 1, already_pct: bool = False) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return MISSING_GLYPH
    pct = value if already_pct else value * 100
    return f"{pct:,.{digits}f} %".replace(",", " ")


def fmt_multiple(value: Optional[float], digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return MISSING_GLYPH
    return f"{value:,.{digits}f}×"


def fmt_date(value: Optional[_dt.date]) -> str:
    if value is None:
        return MISSING_GLYPH
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return MISSING_GLYPH
        try:
            value = _dt.date.fromisoformat(text[:10])
        except ValueError:
            return text  # keep the raw value rather than hide it
    if isinstance(value, _dt.datetime):
        return value.strftime("%d/%m/%Y %H:%M")
    return value.strftime("%d/%m/%Y")


def fmt_timestamp(value: Optional[_dt.datetime]) -> str:
    if value is None:
        return MISSING_GLYPH
    return value.strftime("%d/%m/%Y %H:%M")


def status_label(status: ExtractionStatus) -> str:
    return STATUS_LABELS.get(status, str(status))


# --------------------------------------------------------------------------
# Data-quality presentation
# --------------------------------------------------------------------------

TIER_LABELS = {
    SourceTier.OFFICIAL: "Tier 1 — document officiel (émetteur ou régulateur)",
    SourceTier.SUPPLEMENTARY: "Tier 2 — source complémentaire (bourse, fournisseur)",
    SourceTier.DERIVED: "Tier 3 — dérivé / recalculé en interne",
}


def tier_label(tier: SourceTier) -> str:
    return TIER_LABELS.get(tier, str(tier))


def severity_icon(severity: Severity) -> str:
    return {
        Severity.ERROR: "🔴",
        Severity.WARNING: "🟠",
        Severity.INFO: "🔵",
    }.get(severity, "⚪")


def quality_badge(report: DataQualityReport) -> str:
    """A single short string summarising a data-quality report."""
    errors = len(report.errors)
    warnings = len(report.warnings)
    if errors:
        return f"🔴 {errors} erreur(s), {warnings} avertissement(s)"
    if warnings:
        return f"🟠 {warnings} avertissement(s)"
    return "🟢 Aucune anomalie détectée"


def issues_frame(report: DataQualityReport) -> pd.DataFrame:
    rows = [
        {
            "Sévérité": severity_icon(issue.severity) + " " + issue.severity.value,
            "Code": issue.code,
            "Message": issue.message,
        }
        for issue in report.issues
    ]
    return pd.DataFrame(rows)


def coverage_frame(dataset: FinancialDataSet, items: Optional[Sequence[LineItem]] = None) -> pd.DataFrame:
    """Presence of each line item per annual period, with its extraction status."""
    items = list(items or LINE_ITEM_LABELS_FR.keys())
    periods = dataset.periods()
    rows: List[Dict[str, object]] = []
    for item in items:
        row: Dict[str, object] = {"Poste": line_item_label(item)}
        for period in periods:
            observation = dataset.get(item, period)
            if observation is None or observation.is_missing:
                row[period] = MISSING_GLYPH
            else:
                row[period] = status_label(observation.status)
        present = sum(1 for p in periods if row.get(p, MISSING_GLYPH) != MISSING_GLYPH)
        row["Couverture"] = f"{present}/{len(periods)}" if periods else "0/0"
        rows.append(row)
    return pd.DataFrame(rows)


def observations_frame(
    dataset: FinancialDataSet,
    items: Optional[Iterable[LineItem]] = None,
    period_type=None,
) -> pd.DataFrame:
    """One row per observation, with full provenance, sorted for display."""
    wanted = set(items) if items else None
    rows: List[Dict[str, object]] = []
    for observation in dataset.observations:
        if wanted is not None and observation.line_item not in wanted:
            continue
        if period_type is not None and observation.period.period_type != period_type:
            continue
        rows.append(
            {
                "Poste": line_item_label(observation.line_item),
                "Période": observation.period.label,
                "Valeur": observation.value,
                "Unité": "millions",
                "Statut": status_label(observation.status),
                "Source": observation.source_id or MISSING_GLYPH,
                "Méthode d'extraction": observation.extraction_method or MISSING_GLYPH,
                "Note": observation.notes or "",
                "Fiabilité": assignment_rank(observation),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["Poste", "Période"]).reset_index(drop=True)
    return frame


def assignment_rank(observation: Observation) -> str:
    """How much this figure can be trusted, in plain words."""
    if observation.status == ExtractionStatus.REPORTED and observation.source_id:
        return "Publié, sourcé et rattaché à un document"
    if observation.status == ExtractionStatus.REPORTED:
        return "Publié, source à compléter"
    if observation.status == ExtractionStatus.DERIVED:
        return "Calculé à partir de données publiées"
    if observation.status == ExtractionStatus.MANUAL:
        return "Saisi à la main par l'analyste — à revérifier"
    if observation.status == ExtractionStatus.ESTIMATED:
        return "Estimation (hypothèse) — ne pas traiter comme un fait"
    if observation.status == ExtractionStatus.UNVERIFIED:
        return "Non vérifié — à confirmer avant utilisation"
    return "Absent"


def sources_frame(sources: Sequence[SourceRecord]) -> pd.DataFrame:
    columns = [
        "Identifiant", "Nom", "Éditeur", "Type", "Tier", "Publié le",
        "Récupéré le", "Devise", "Échelle", "URL",
    ]
    rows = [
        {
            "Identifiant": source.source_id,
            "Nom": source.name,
            "Éditeur": source.publisher or MISSING_GLYPH,
            "Type": source.document_type or MISSING_GLYPH,
            "Tier": tier_label(source.tier) if source.tier else MISSING_GLYPH,
            "Publié le": fmt_date(source.publication_date),
            "Récupéré le": fmt_timestamp(source.retrieved_at),
            "Devise": source.currency.value if source.currency else MISSING_GLYPH,
            "Échelle": source.scale.value if source.scale else MISSING_GLYPH,
            "URL": source.url or MISSING_GLYPH,
        }
        for source in sources
    ]
    return pd.DataFrame(rows, columns=columns)


def missing_items(dataset: FinancialDataSet, items: Sequence[LineItem]) -> List[str]:
    """Human-readable list of the requested items with no value at the latest period."""
    latest = dataset.latest_period_label()
    absent: List[str] = []
    for item in items:
        if latest is None:
            absent.append(line_item_label(item))
            continue
        observation = dataset.get(item, latest)
        if observation is None or observation.is_missing:
            absent.append(line_item_label(item))
    return absent


# --------------------------------------------------------------------------
# Text blocks
# --------------------------------------------------------------------------

DISCLAIMER = (
    "**Avertissement.** Cet outil est un support d'analyse, pas un conseil en "
    "investissement. Les chiffres publiés proviennent des documents cités ; les "
    "projections, scénarios et notes d'IA sont des hypothèses explicitement "
    "étiquetées. Aucune donnée n'est inventée : un chiffre absent reste absent et "
    "est signalé comme tel."
)

AI_DISABLED_NOTE = (
    "Assistant IA désactivé. Pour l'activer, renseignez `ERAI_LLM_API_KEY` (et "
    "éventuellement `ERAI_LLM_BASE_URL`, `ERAI_LLM_MODEL`) dans le fichier `.env`, "
    "puis redémarrez l'application. Toutes les autres pages fonctionnent sans clé."
)

FACTS_VS_ASSUMPTIONS = (
    "**Faits vs hypothèses.** Les tableaux marqués « Publié » ou « Calculé » "
    "proviennent des états financiers. Les blocs « Hypothèses », « Scénarios » et "
    "« Estimation » sont des choix d'analyste, modifiables et réversibles."
)

REPORTED_FIGURES_NOTE = (
    "Chiffres tels que chargés (publiés ou saisis), en millions de la devise de "
    "reporting, sans retraitement ni lissage cyclique."
)


def provenance_caption(observation: Optional[Observation], source: Optional[SourceRecord]) -> str:
    """One-line provenance string for a single figure."""
    if observation is None:
        return "Donnée absente."
    parts = [f"Statut : {status_label(observation.status)}"]
    if source is not None:
        parts.append(f"Source : {source.name}")
        if source.publication_date:
            parts.append(f"publié le {fmt_date(source.publication_date)}")
        if source.url:
            parts.append(source.url)
    elif observation.source_id:
        parts.append(f"Source : {observation.source_id}")
    if observation.extraction_method:
        parts.append(f"extraction : {observation.extraction_method}")
    return " · ".join(parts)
