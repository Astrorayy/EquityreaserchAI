"""Application workflows: the glue between stored data and the engines.

Both the Streamlit UI and the AI assistant's tools call **this** module, so a
number produced by the assistant is the same number the analyst sees on screen.
Nothing here invents data: when a required input cannot be derived from the
stored financial statements it is reported in ``missing``/``errors`` and left
for the analyst to fill in explicitly.

Conventions
-----------
* All monetary values are in **millions** of the reporting currency; share
  counts are in **millions of shares**; per-share outputs are currency units.
* Every assumption carries a provenance string (where it came from) so the UI
  can show it next to the number.
* Series assumptions accept either a scalar (broadcast to the horizon) or a
  sequence whose length matches ``forecast_years``.
"""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from financial_engine.dcf import DCFInputError, DCFInputs, DCFResult, run_dcf
from financial_engine.multiples import (
    MultiplesEngine,
    MultiplesResult,
    TargetValuationInputs,
)
from financial_engine.ratios import Metric, RatioEngine
from financial_engine.schemas import (
    CompanyProfile,
    ComparableCompany,
    Currency,
    FinancialDataSet,
    LineItem,
    PeriodType,
)
from financial_engine.sector_models import (
    RECOMMENDED_METHODS,
    SECTOR_RISKS,
    derive_baseline_assumptions,
    normalise_memory_cycle,
)
from financial_engine.sensitivity import (
    SensitivityGrid,
    TornadoResult,
    build_scenarios,
    default_bear_bull,
    tornado_analysis,
    wacc_growth_grid,
)
from financial_engine.validation import DataQualityReport, derive_missing_items, validate_dataset

logger = logging.getLogger(__name__)

#: Source id used by every synthetic (simulated) observation.
SIMULATION_SOURCE_ID = "simulated:demo"
MANUAL_SOURCE_ID = "manual:analyst"

#: Generic analyst placeholders. They are *not* data: the UI shows them as
#: assumptions and they must be replaced before any figure is used.
DEFAULT_WACC = 0.10
DEFAULT_TERMINAL_GROWTH = 0.02
DEFAULT_TAX_RATE = 0.25
DEFAULT_EBIT_MARGIN = 0.10
MAX_FORECAST_YEARS = 15

ASSUMPTION_KEYS = (
    "revenue_growth",
    "ebit_margin",
    "da_pct_revenue",
    "capex_pct_revenue",
    "nwc_pct_revenue",
    "tax_rate",
    "net_debt",
    "shares_outstanding",
    "base_revenue",
    "other_senior_claims",
    "non_operating_assets",
)

#: Assumptions without which a DCF cannot be produced at all.
ESSENTIAL_ASSUMPTIONS = ("base_revenue", "ebit_margin", "shares_outstanding", "wacc")

SCALAR_ASSUMPTION_KEYS = (
    "tax_rate", "net_debt", "shares_outstanding", "base_revenue",
    "other_senior_claims", "non_operating_assets",
)
SERIES_ASSUMPTION_KEYS = (
    "revenue_growth", "ebit_margin", "da_pct_revenue",
    "capex_pct_revenue", "nwc_pct_revenue",
)

PROVENANCE_MANUAL = "hypothèse saisie par l'analyste"
PROVENANCE_PLACEHOLDER = "valeur générique par défaut — à remplacer impérativement"


# --------------------------------------------------------------------------
# Dataset helpers
# --------------------------------------------------------------------------

def is_simulated(dataset: FinancialDataSet) -> bool:
    return SIMULATION_SOURCE_ID in set(dataset.sources())


def has_manual_entries(dataset: FinancialDataSet) -> bool:
    return any(
        (observation.source_id or "").startswith("manual:")
        for observation in dataset.observations
    )


def latest_periods(dataset: FinancialDataSet, count: int = 2) -> List[str]:
    periods = dataset.periods(PeriodType.ANNUAL)
    return periods[-count:]


def dataset_currency(dataset: FinancialDataSet, fallback: Currency = Currency.USD) -> Currency:
    return dataset.currency or fallback


# --------------------------------------------------------------------------
# DCF setup
# --------------------------------------------------------------------------

@dataclass
class DCFSetup:
    """A ready-to-run DCF input set plus the provenance of every assumption."""

    inputs: Optional[DCFInputs]
    assumptions: Dict[str, object] = field(default_factory=dict)
    provenance: Dict[str, str] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.inputs is not None and not self.errors

    def provenance_of(self, key: str) -> str:
        return self.provenance.get(key, "origine inconnue")


def _broadcast(value: object, years: int, field_name: str) -> List[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)] * years
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = [float(v) for v in value]
        if len(values) == years:
            return values
        if not values:
            raise ValueError(f"{field_name}: série vide")
        if len(values) < years:
            # Repeat the last value rather than inventing a trend.
            return values + [values[-1]] * (years - len(values))
        return values[:years]
    raise ValueError(f"{field_name}: valeur inattendue {value!r}")


def build_dcf_inputs(
    dataset: FinancialDataSet,
    *,
    forecast_years: int = 5,
    overrides: Optional[Dict[str, object]] = None,
    wacc: Optional[float] = None,
    terminal_growth: Optional[float] = None,
    scenario: str = "base",
    label: str = "",
) -> DCFSetup:
    """Assemble ``DCFInputs`` from stored statements plus explicit overrides.

    The stored-statements part is *historical*: medians of realised margins and
    intensities, proposed as a starting point.  Anything supplied in
    ``overrides`` (or ``wacc``/``terminal_growth``) is labelled as an analyst
    assumption.  Anything that cannot be sourced at all is returned in
    ``missing`` and, when it is essential, in ``errors`` — never silently
    replaced by a plausible-looking number.
    """
    setup = DCFSetup(inputs=None)
    overrides = dict(overrides or {})
    if forecast_years < 1 or forecast_years > MAX_FORECAST_YEARS:
        setup.errors.append(
            f"Horizon de projection invalide ({forecast_years} ans) : "
            f"attendu entre 1 et {MAX_FORECAST_YEARS} ans."
        )
        return setup

    baseline = derive_baseline_assumptions(dataset, forecast_years=forecast_years)
    suggested: Dict[str, object] = dict(baseline.suggested)
    for key, reason in baseline.provenance.items():
        setup.provenance[key] = reason
    setup.missing.extend(baseline.missing)
    setup.notes.extend(baseline.notes)

    # Historical statistics first, then analyst overrides on top.
    for key, value in suggested.items():
        if key in ASSUMPTION_KEYS and value is not None:
            setup.assumptions[key] = value

    manual: List[str] = []
    for key, value in overrides.items():
        if key not in ASSUMPTION_KEYS:
            setup.warnings.append(f"Hypothèse ignorée (inconnue) : {key}")
            continue
        if value is None:
            continue
        setup.assumptions[key] = value
        setup.provenance[key] = PROVENANCE_MANUAL
        manual.append(key)
    if wacc is not None:
        setup.assumptions["wacc"] = float(wacc)
        setup.provenance["wacc"] = PROVENANCE_MANUAL
    if terminal_growth is not None:
        setup.assumptions["terminal_growth"] = float(terminal_growth)
        setup.provenance["terminal_growth"] = PROVENANCE_MANUAL

    # --- placeholders for what the data cannot provide ------------------
    def placeholder(key: str, value: object, why: str) -> None:
        setup.assumptions.setdefault(key, value)
        setup.provenance[key] = why
        if key not in setup.missing:
            setup.missing.append(key)

    placeholder("wacc", DEFAULT_WACC, PROVENANCE_PLACEHOLDER + " (coût moyen du capital)")
    placeholder(
        "terminal_growth",
        DEFAULT_TERMINAL_GROWTH,
        PROVENANCE_PLACEHOLDER + " (croissance perpétuelle)",
    )
    setup.assumptions.setdefault("tax_rate", DEFAULT_TAX_RATE)
    setup.provenance.setdefault("tax_rate", PROVENANCE_PLACEHOLDER + " (taux d'impôt)")
    if "shares_outstanding" not in setup.assumptions:
        setup.assumptions["shares_outstanding"] = 0.0
        setup.provenance["shares_outstanding"] = "indisponible : nombre d'actions absents"
    setup.assumptions.setdefault("net_debt", 0.0)
    setup.provenance.setdefault(
        "net_debt", "indisponible : dette nette supposée nulle (vérifier le bilan)"
    )

    # A missing margin cannot be guessed: refuse rather than invent.
    if setup.assumptions.get("ebit_margin") in (None,):
        setup.assumptions["ebit_margin"] = DEFAULT_EBIT_MARGIN
        setup.provenance["ebit_margin"] = PROVENANCE_PLACEHOLDER + " (marge opérationnelle)"
        setup.errors.append(
            "Marge opérationnelle introuvable dans les états financiers. Une valeur "
            "générique de 10 % est proposée : remplacez-la par votre hypothèse avant "
            "d'interpréter le résultat."
        )

    for optional_key, consequence in (
        ("da_pct_revenue", "les amortissements sont considérés comme nuls"),
        ("capex_pct_revenue", "les investissements sont considérés comme nuls"),
        ("nwc_pct_revenue", "la variation du besoin en fonds de roulement est ignorée"),
    ):
        if setup.assumptions.get(optional_key) is None:
            setup.assumptions[optional_key] = 0.0
            setup.provenance[optional_key] = f"indisponible : {consequence}"
            setup.warnings.append(
                f"Hypothèse « {optional_key} » indisponible : {consequence}. "
                "Le flux de trésorerie disponible est donc approximatif."
            )

    growth = setup.assumptions.get("revenue_growth")
    if growth is None:
        setup.assumptions["revenue_growth"] = 0.0
        setup.provenance["revenue_growth"] = "indisponible : croissance nulle par défaut"
        setup.warnings.append(
            "Croissance du chiffre d'affaires indisponible : le modèle suppose une "
            "croissance nulle. Remplacez cette hypothèse."
        )
    elif isinstance(growth, (int, float)) and growth < -0.5:
        setup.warnings.append(
            f"Croissance historique de {growth:.0%} par an : vérifiez qu'il ne s'agit pas "
            "d'un exercice atypique avant de l'utiliser telle quelle."
        )

    base_revenue = setup.assumptions.get("base_revenue")
    if base_revenue is None or float(base_revenue) <= 0:
        setup.errors.append(
            "Chiffre d'affaires de référence absent ou non positif : impossible de lancer un "
            "DCF. Saisissez le chiffre d'affaires du dernier exercice ou chargez les comptes."
        )
        return setup

    shares = setup.assumptions.get("shares_outstanding")
    if shares is None or float(shares) <= 0:
        setup.errors.append(
            "Nombre d'actions diluées absent : la valeur par action ne peut pas être calculée. "
            "Renseignez-le dans la page Valorisation."
        )
        return setup

    years = int(forecast_years)
    try:
        series_kwargs = {
            key: _broadcast(setup.assumptions[key], years, key)
            for key in SERIES_ASSUMPTION_KEYS
        }
    except (TypeError, ValueError) as exc:
        setup.errors.append(f"Série d'hypothèses invalide : {exc}")
        return setup

    base_nwc = dataset.value(LineItem.NET_WORKING_CAPITAL, dataset.latest_period_label() or "") if dataset.periods() else None

    try:
        setup.inputs = DCFInputs(
            base_revenue=float(base_revenue),
            forecast_years=years,
            revenue_growth=series_kwargs["revenue_growth"],
            ebit_margin=series_kwargs["ebit_margin"],
            tax_rate=float(setup.assumptions.get("tax_rate", DEFAULT_TAX_RATE)),
            da_pct_revenue=series_kwargs["da_pct_revenue"],
            capex_pct_revenue=series_kwargs["capex_pct_revenue"],
            nwc_pct_revenue=series_kwargs["nwc_pct_revenue"],
            wacc=float(setup.assumptions.get("wacc", DEFAULT_WACC)),
            terminal_growth=float(setup.assumptions.get("terminal_growth", DEFAULT_TERMINAL_GROWTH)),
            net_debt=float(setup.assumptions.get("net_debt", 0.0) or 0.0),
            other_senior_claims=float(setup.assumptions.get("other_senior_claims", 0.0) or 0.0),
            non_operating_assets=float(setup.assumptions.get("non_operating_assets", 0.0) or 0.0),
            shares_outstanding=float(shares),
            base_nwc=base_nwc,
            currency=dataset_currency(dataset).value,
            scenario=scenario,
            label=label,
            manually_entered=tuple(dict.fromkeys(manual)),
            simulated=is_simulated(dataset),
        )
    except DCFInputError as exc:
        setup.errors.append(str(exc))
        return setup

    if not setup.errors:
        try:
            run_dcf(setup.inputs, inspect_only=False)
        except DCFInputError as exc:
            setup.errors.append(str(exc))

    if is_simulated(dataset):
        setup.notes.append(
            "Jeu de données SIMULÉ : ce calcul ne porte pas sur une société réelle."
        )
    return setup


# --------------------------------------------------------------------------
# Valuation bundle
# --------------------------------------------------------------------------

@dataclass
class ValuationBundle:
    setup: DCFSetup
    results: List[DCFResult] = field(default_factory=list)
    sensitivity: Optional[SensitivityGrid] = None
    tornado: Optional[TornadoResult] = None
    current_price: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    assumptions: Dict[str, object] = field(default_factory=dict)
    provenance: Dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.results)

    @property
    def base(self) -> Optional[DCFResult]:
        for result in self.results:
            if result.scenario == "base":
                return result
        return self.results[0] if self.results else None

    def by_scenario(self, name: str) -> Optional[DCFResult]:
        for result in self.results:
            if result.scenario == name:
                return result
        return None


def run_valuation(
    dataset: FinancialDataSet,
    *,
    wacc: Optional[float] = None,
    terminal_growth: Optional[float] = None,
    forecast_years: int = 5,
    overrides: Optional[Dict[str, object]] = None,
    current_price: Optional[float] = None,
    base_year: Optional[int] = None,
    with_scenarios: bool = True,
    with_sensitivity: bool = True,
    base_label: str = "",
) -> ValuationBundle:
    """Run the base DCF plus, optionally, scenarios and the sensitivity grid."""
    setup = build_dcf_inputs(
        dataset,
        forecast_years=forecast_years,
        overrides=overrides,
        wacc=wacc,
        terminal_growth=terminal_growth,
        label=base_label,
    )
    bundle = ValuationBundle(
        setup=setup,
        assumptions=setup.assumptions,
        provenance=setup.provenance,
        current_price=current_price,
        warnings=list(setup.warnings),
    )
    if setup.inputs is None:
        return bundle

    try:
        if with_scenarios:
            bear, bull = default_bear_bull(setup.inputs)
            bundle.results = build_scenarios(setup.inputs, bear=bear, bull=bull)
        else:
            bundle.results = [run_dcf(setup.inputs, current_price=current_price, base_year=base_year)]
    except DCFInputError as exc:
        bundle.warnings.append(str(exc))
        return bundle

    for result in bundle.results:
        bundle.warnings.extend(result.warnings)

    if with_sensitivity and setup.inputs.wacc > setup.inputs.terminal_growth:
        try:
            bundle.sensitivity = wacc_growth_grid(
                setup.inputs, current_price=current_price, base_year=base_year
            )
            bundle.tornado = tornado_analysis(
                setup.inputs, current_price=current_price, base_year=base_year
            )
        except DCFInputError as exc:
            bundle.warnings.append(f"Sensibilité non calculée : {exc}")
    elif with_sensitivity:
        bundle.warnings.append(
            "Sensibilité non calculée : le WACC est inférieur ou égal à la croissance "
            "perpétuelle, ce qui rend la valeur terminale inexploitable."
        )
    return bundle


# --------------------------------------------------------------------------
# Comparables
# --------------------------------------------------------------------------

@dataclass
class MultiplesBundle:
    result: Optional[MultiplesResult] = None
    target: Optional[TargetValuationInputs] = None
    peers: List[ComparableCompany] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.result is not None


def build_target_inputs(
    dataset: FinancialDataSet,
    profile: Optional[CompanyProfile] = None,
    current_price: Optional[float] = None,
    period: Optional[str] = None,
) -> Tuple[Optional[TargetValuationInputs], List[str]]:
    """Assemble the target's metrics for the comparables engine."""
    missing: List[str] = []
    if not dataset.periods():
        return None, ["aucun exercice financier chargé"]
    period = period or dataset.latest_period_label()
    if period is None:
        return None, ["aucune période disponible"]

    revenue = dataset.value(LineItem.REVENUE, period)
    net_income = dataset.value(LineItem.NET_INCOME, period)
    equity = dataset.value(LineItem.TOTAL_EQUITY, period)
    ebit = dataset.value(LineItem.EBIT, period) or dataset.value(LineItem.OPERATING_INCOME, period)
    da = dataset.value(LineItem.DEPRECIATION_AMORTIZATION, period)
    ebitda = dataset.value(LineItem.EBITDA, period)
    if ebitda is None and ebit is not None and da is not None:
        ebitda = ebit + abs(da)
    shares = dataset.value(LineItem.DILUTED_SHARES, period) or dataset.value(LineItem.SHARES_OUTSTANDING, period)

    engine = RatioEngine(dataset)
    net_debt, nd_reason = engine.net_debt(period)
    if net_debt is None:
        missing.append(f"dette nette ({nd_reason})")
    minority = dataset.value(LineItem.MINORITY_INTEREST, period) or 0.0

    for label, value in (
        ("chiffre d'affaires", revenue),
        ("EBITDA", ebitda),
        ("résultat net", net_income),
        ("capitaux propres", equity),
        ("actions diluées", shares),
    ):
        if value is None:
            missing.append(label)

    if shares is None or shares <= 0:
        return None, missing + ["nombre d'actions non exploitable"]

    price = current_price or dataset.value(LineItem.MARKET_PRICE, period)
    target = TargetValuationInputs(
        name=profile.name if profile else dataset.company_id,
        currency=dataset_currency(dataset),
        net_debt=net_debt or 0.0,
        diluted_shares=shares,
        ebitda=ebitda,
        revenue=revenue,
        net_income=net_income,
        book_value=equity,
        current_price=price,
        minority_interest=minority,
    )
    return target, missing


def run_multiples(
    dataset: FinancialDataSet,
    peers: Sequence[ComparableCompany],
    profile: Optional[CompanyProfile] = None,
    fx_rates: Optional[Dict[Currency, float]] = None,
    metrics: Optional[Sequence[str]] = None,
    min_peers: int = 3,
    current_price: Optional[float] = None,
) -> MultiplesBundle:
    bundle = MultiplesBundle(peers=list(peers))
    target, missing = build_target_inputs(dataset, profile, current_price=current_price)
    bundle.missing = missing
    bundle.target = target
    if target is None:
        bundle.errors.append(
            "Valorisation par comparables impossible : " + ", ".join(missing or ["données manquantes"])
        )
        return bundle
    if not peers:
        bundle.errors.append(
            "Aucun comparable enregistré : ajoutez au moins trois sociétés aux multiples "
            "comparables avant d'interpréter le résultat."
        )
        return bundle
    engine = MultiplesEngine(target, fxto_target=fx_rates)
    try:
        bundle.result = engine.run(peers, metrics=metrics, min_peers=min_peers)
    except Exception as exc:  # pragma: no cover - engine is defensive already
        bundle.errors.append(f"Calcul des multiples impossible : {exc}")
        return bundle
    bundle.warnings.extend(bundle.result.warnings)
    return bundle


# --------------------------------------------------------------------------
# Ratios / sector context
# --------------------------------------------------------------------------

def ratio_metrics(dataset: FinancialDataSet, period: Optional[str] = None) -> Dict[str, Metric]:
    if not dataset.periods():
        return {}
    period = period or dataset.latest_period_label()
    if period is None:
        return {}
    previous_periods = [p for p in dataset.periods() if p < period]
    previous = previous_periods[-1] if previous_periods else None
    engine = RatioEngine(dataset)
    try:
        return engine.compute_all(period, previous)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Calcul des ratios impossible pour %s : %s", period, exc)
        return {}


def sector_context(company_id: str, profile=None) -> Tuple[List[str], List[str]]:
    """Sector risks and recommended methods, for registry or user-added companies."""
    if profile is None:
        from data_sources.company_registry import get_profile

        profile = get_profile(company_id)
    sector = " ".join(
        filter(None, [getattr(profile, "sector", None), getattr(profile, "industry", None)])
    ) or "generic"
    key = _sector_key(sector)
    risks = list(SECTOR_RISKS.get(key, SECTOR_RISKS.get("generic", [])))
    methods = list(RECOMMENDED_METHODS.get(key, RECOMMENDED_METHODS.get("generic", [])))
    return risks, methods


_MINING_HINTS = ("mine", "minier", "miniere", "mining", "metal", "métal", "gold", "copper", "cuivre")
_SEMI_HINTS = ("semi", "memory", "mémoire", "memoire", "storage", "stockage")


def _sector_key(sector: str) -> str:
    lowered = (sector or "").lower()
    if any(hint in lowered for hint in _MINING_HINTS):
        return "mining"
    if any(hint in lowered for hint in _SEMI_HINTS):
        return "semiconductors"
    return "generic"


def cycle_analysis(dataset: FinancialDataSet):
    """Normalised-cycle statistics for cyclical sectors (may return partials)."""
    try:
        return normalise_memory_cycle(dataset)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Normalisation cyclique impossible : %s", exc)
        return None


def ensure_derived(dataset: FinancialDataSet) -> Tuple[FinancialDataSet, List[str]]:
    """Add derived items (gross profit, EBITDA, FCF…) without overwriting facts."""
    report = derive_missing_items(dataset)
    created = [
        entry if isinstance(entry, str) else getattr(entry, "line_item", str(entry))
        for entry in (report.created or [])
    ]
    return dataset, [str(c) for c in created]


def assess(dataset: FinancialDataSet, previous: Optional[FinancialDataSet] = None) -> DataQualityReport:
    return validate_dataset(dataset, previous)


# --------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------

def build_report_context(
    *,
    profile: CompanyProfile,
    dataset: FinancialDataSet,
    sources: Sequence,
    valuation: Optional[ValuationBundle] = None,
    multiples: Optional[MultiplesBundle] = None,
    quality: Optional[DataQualityReport] = None,
    current_price: Optional[float] = None,
    ai_narrative: Optional[str] = None,
    extra_notes: Optional[Sequence[str]] = None,
    generated_at: Optional[_dt.datetime] = None,
):
    """Build a :class:`reports.generator.ReportContext` from workflow objects."""
    from reports.generator import ReportContext

    ratio_period = dataset.latest_period_label()
    metrics = list(ratio_metrics(dataset, ratio_period).values()) if ratio_period else []

    assumptions: Dict[str, object] = {}
    provenance: Dict[str, str] = {}
    if valuation is not None:
        assumptions = dict(valuation.assumptions)
        provenance = dict(valuation.provenance)

    return ReportContext(
        profile=profile,
        dataset=dataset,
        quality=quality if quality is not None else (validate_dataset(dataset) if dataset.observations else None),
        sources=list(sources),
        ratio_metrics=metrics,
        assumptions=assumptions,
        assumption_provenance=provenance,
        dcf_results=list(valuation.results) if valuation else [],
        sensitivity=valuation.sensitivity if valuation else None,
        tornado=valuation.tornado if valuation else None,
        multiples=multiples.result if multiples else None,
        current_price=current_price,
        ai_narrative=ai_narrative,
        extra_notes=list(extra_notes or []),
        generated_at=generated_at or _dt.datetime.now(),
        simulated=is_simulated(dataset),
    )


def generate_company_report(
    *,
    profile: CompanyProfile,
    dataset: FinancialDataSet,
    sources: Sequence,
    reports_dir: Path,
    valuation: Optional[ValuationBundle] = None,
    multiples: Optional[MultiplesBundle] = None,
    quality: Optional[DataQualityReport] = None,
    current_price: Optional[float] = None,
    ai_narrative: Optional[str] = None,
    extra_notes: Optional[Sequence[str]] = None,
    formats: Sequence[str] = ("html", "pdf"),
) -> Dict[str, Path]:
    """Write the report and return the paths keyed by format."""
    from reports.generator import generate_report

    ctx = build_report_context(
        profile=profile,
        dataset=dataset,
        sources=sources,
        valuation=valuation,
        multiples=multiples,
        quality=quality,
        current_price=current_price,
        ai_narrative=ai_narrative,
        extra_notes=extra_notes,
    )
    risks, methods = sector_context(profile.company_id)
    return generate_report(
        ctx,
        reports_dir=reports_dir,
        formats=formats,
        sector_risks=risks,
        method_notes=methods,
    )


def relabel_dataset(
    dataset: FinancialDataSet,
    company_id: str,
    currency: Optional[Currency] = None,
) -> FinancialDataSet:
    """Re-point a dataset (and every observation in it) at another company.

    Used when a synthetic or imported dataset is attached to a company, so the
    stored rows carry the company the analyst chose — never the placeholder the
    generator happened to write.  ``currency`` re-stamps the reporting currency
    too, which is only appropriate for simulated or manually entered datasets:
    figures extracted from a real document keep the currency printed on it.
    """
    target_currency = currency or dataset.currency
    if (
        dataset.company_id == company_id
        and target_currency == dataset.currency
        and all(
            observation.company_id == company_id for observation in dataset.observations
        )
    ):
        return dataset
    relabelled = FinancialDataSet(
        company_id=company_id,
        currency=target_currency,
        observations=[
            observation.model_copy(
                update={
                    "company_id": company_id,
                    "currency": target_currency,
                    "reported_currency": observation.reported_currency or target_currency,
                }
            )
            for observation in dataset.observations
        ],
    )
    return relabelled


def dataset_summary(dataset: FinancialDataSet) -> Dict[str, object]:
    """Small dict used by the assistant to answer "what do you know?" questions."""
    periods = dataset.periods()
    items = [item for item in LineItem]
    filled = 0
    for item in items:
        for period in periods:
            observation = dataset.get(item, period)
            if observation is not None and not observation.is_missing:
                filled += 1
    return {
        "company_id": dataset.company_id,
        "currency": dataset_currency(dataset).value,
        "periods_annual": periods,
        "latest_period": dataset.latest_period_label(),
        "items_available": filled,
        "items_total": len(items) * len(periods),
        "sources": dataset.sources(),
        "simulated": is_simulated(dataset),
        "manual_entries": has_manual_entries(dataset),
    }
