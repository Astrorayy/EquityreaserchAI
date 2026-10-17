"""Sector-specific valuation helpers.

Two industries are addressed:

* **Mining & metals** (Managem): a project-level NPV model and a
  commodity-price sensitivity that translate a price move into an EBIT-margin
  impact, plus sector risk factors.
* **Semiconductors / NAND flash** (SanDisk): a mid-cycle normalisation tool that
  stops peak-margin or trough-margin years from being treated as sustainable.

A third, general tool derives a *baseline* assumption set from the company's own
reported history.  That is the only way the application is allowed to propose
starting assumptions: everything is traced back to a reported figure, and
anything that cannot be derived is returned as an explicit gap rather than as an
invented number.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from financial_engine.dcf import DCFInputError
from financial_engine.ratios import RatioEngine, safe_div
from financial_engine.schemas import (
    CompanyProfile,
    ExtractionStatus,
    FinancialDataSet,
    LineItem,
    PeriodType,
)

# --------------------------------------------------------------------------
# Baseline assumptions derived from history
# --------------------------------------------------------------------------


@dataclass
class BaselineAssumptions:
    """Starting point suggested by the company's own reported history."""

    suggested: Dict[str, float] = field(default_factory=dict)
    provenance: Dict[str, str] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def as_records(self) -> List[Dict[str, object]]:
        keys = sorted(set(self.suggested) | set(self.provenance) | set(self.missing))
        rows: List[Dict[str, object]] = []
        for k in keys:
            rows.append(
                {
                    "assumption": k,
                    "value": self.suggested.get(k),
                    "provenance": self.provenance.get(k, "indisponible"),
                }
            )
        return rows


def _median_of(values: Sequence[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None and v == v]
    return statistics.median(clean) if clean else None


def derive_baseline_assumptions(
    dataset: FinancialDataSet,
    forecast_years: int = 5,
    history_window: int = 5,
) -> BaselineAssumptions:
    """Derive neutral starting assumptions from reported annual history.

    Nothing here is a forecast: each value is a *historical* statistic
    (median margin, realised CAGR) proposed as a placeholder that the analyst
    must review.  Provenance strings are shown in the UI next to every number.
    """
    out = BaselineAssumptions()
    engine = RatioEngine(dataset)

    labels = dataset.periods(PeriodType.ANNUAL)[-history_window:]
    if not labels:
        out.missing.extend(
            [
                "revenue_growth",
                "ebit_margin",
                "da_pct_revenue",
                "capex_pct_revenue",
                "nwc_pct_revenue",
                "tax_rate",
                "net_debt",
                "shares_outstanding",
            ]
        )
        for key in out.missing:
            out.provenance[key] = "indisponible : aucun exercice annuel en base"
        out.notes.append(
            "Aucun exercice annuel disponible : impossible de proposer des hypothèses "
            "de départ. Saisissez les données manuellement ou importez des comptes officiels."
        )
        return out

    # --- revenue growth: realised CAGR over the window -------------------
    revenues = [dataset.value(LineItem.REVENUE, label) for label in labels]
    usable = [(label, r) for label, r in zip(labels, revenues) if r is not None and r > 0]
    if len(usable) >= 2:
        (first_label, first), (last_label, last) = usable[0], usable[-1]
        periods = max(1, len(usable) - 1)
        cagr = (last / first) ** (1.0 / periods) - 1.0
        out.suggested["revenue_growth"] = cagr
        out.provenance["revenue_growth"] = (
            f"TCAM realise de {first_label} a {last_label} "
            f"({first:,.0f} -> {last:,.0f} m)"
        )
        if len(usable) < 3:
            out.notes.append(
                "Le TCAM repose sur moins de trois exercices : il est peu representatif."
            )
    else:
        out.missing.append("revenue_growth")
        out.provenance["revenue_growth"] = "historique de chiffre d'affaires insuffisant"

    # --- margins: median of the reported history -------------------------
    op_margins = [engine.operating_margin(lbl).value for lbl in labels]
    derived = _median_of([None if m is None else m / 100.0 for m in op_margins])
    if derived is not None:
        out.suggested["ebit_margin"] = derived
        out.provenance["ebit_margin"] = (
            f"médiane des marges opérationnelles {labels[0]}..{labels[-1]}"
        )
        spanning = [m for m in op_margins if m is not None]
        if len(spanning) >= 2 and (max(spanning) - min(spanning)) > 15:
            out.notes.append(
                f"Les marges opérationnelles varient de {min(spanning):.1f}% à "
                f"{max(spanning):.1f}% : le secteur est très cyclique. La médiane "
                "est proposée mais une analyse de milieu de cycle est indispensable."
            )
    else:
        out.missing.append("ebit_margin")
        out.provenance["ebit_margin"] = "résultat opérationnel indisponible"

    # --- capex and D&A intensity ----------------------------------------
    capex_intensity = [engine.capex_intensity(lbl).value for lbl in labels]
    capex_med = _median_of([None if v is None else v / 100.0 for v in capex_intensity])
    if capex_med is not None:
        out.suggested["capex_pct_revenue"] = abs(capex_med)
        out.provenance["capex_pct_revenue"] = f"médiane des investissements / CA sur {labels[0]}..{labels[-1]}"
    else:
        out.missing.append("capex_pct_revenue")
        out.provenance["capex_pct_revenue"] = "investissements indisponibles"

    da_ratios: List[Optional[float]] = []
    for lbl in labels:
        da = dataset.value(LineItem.DEPRECIATION_AMORTIZATION, lbl)
        rev = dataset.value(LineItem.REVENUE, lbl)
        value, _ = safe_div(None if da is None else abs(da), rev, label="D&A intensity")
        da_ratios.append(value)
    da_med = _median_of(da_ratios)
    if da_med is not None:
        out.suggested["da_pct_revenue"] = da_med
        out.provenance["da_pct_revenue"] = "médiane des amortissements / CA (ou EBIT+DA/CA)"
    else:
        out.missing.append("da_pct_revenue")
        out.provenance["da_pct_revenue"] = "amortissements indisponibles"

    # --- working capital intensity ---------------------------------------
    nwc_ratios: List[Optional[float]] = []
    for lbl in labels:
        nwc = dataset.value(LineItem.NET_WORKING_CAPITAL, lbl)
        rev = dataset.value(LineItem.REVENUE, lbl)
        value, _ = safe_div(nwc, rev, label="NWC intensity")
        nwc_ratios.append(value)
    nwc_med = _median_of(nwc_ratios)
    if nwc_med is not None:
        out.suggested["nwc_pct_revenue"] = nwc_med
        out.provenance["nwc_pct_revenue"] = "médiane du BFR / CA"
    else:
        out.missing.append("nwc_pct_revenue")
        out.provenance["nwc_pct_revenue"] = "BFR non calculable (actif/passif courant indisponible)"

    # --- effective tax rate ---------------------------------------------
    rates = []
    for lbl in labels:
        tax = dataset.value(LineItem.INCOME_TAX_EXPENSE, lbl)
        pretax = dataset.value(LineItem.PRETAX_INCOME, lbl)
        if tax is not None and pretax not in (None, 0):
            r = abs(tax) / pretax
            if 0.0 <= r <= 0.6:
                rates.append(r)
    tax_med = _median_of(rates)
    if tax_med is not None:
        out.suggested["tax_rate"] = tax_med
        out.provenance["tax_rate"] = "médiane du taux d'imposition effectif publié"
    else:
        out.missing.append("tax_rate")
        out.provenance["tax_rate"] = "charge d'impôt / résultat avant impôt indisponible"

    # --- balance-sheet anchors at the latest period -----------------------
    last = labels[-1]
    out.suggested["base_revenue"] = dataset.value(LineItem.REVENUE, last) or 0.0
    out.provenance["base_revenue"] = f"chiffre d'affaires publié {last}"

    net_debt, nd_reason = engine.net_debt(last)
    if net_debt is not None:
        out.suggested["net_debt"] = net_debt
        out.provenance["net_debt"] = f"dette totale - trésorerie, au {last}"
    else:
        out.missing.append("net_debt")
        out.provenance["net_debt"] = nd_reason or "indisponible"

    shares = dataset.value(LineItem.DILUTED_SHARES, last) or dataset.value(LineItem.SHARES_OUTSTANDING, last)
    if shares:
        out.suggested["shares_outstanding"] = shares
        out.provenance["shares_outstanding"] = f"actions en circulation au {last}"
    else:
        out.missing.append("shares_outstanding")
        out.provenance["shares_outstanding"] = "nombre d'actions indisponible"

    out.notes.append(
        "Ces valeurs sont des statistiques HISTORIQUES proposées comme point de départ. "
        "Elles ne constituent pas une prévision et doivent être revues par l'analyste."
    )
    return out


# --------------------------------------------------------------------------
# Mining: project-level NPV and commodity sensitivity
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MineProjectInputs:
    """A single mine or project, using disclosed technical parameters only."""

    name: str
    annual_production: float          # in `unit`
    unit: str                         # "oz", "t", "kt", "lb"...
    commodity_price: float            # per unit, in `currency`
    cash_cost: float                  # per unit, in `currency`
    annual_sustaining_capex: float    # absolute, in millions
    tax_rate: float
    discount_rate: float
    mine_life_years: int
    currency: str = "USD"
    #: country / project risk premium added to the discount rate, decimal
    country_risk_premium: float = 0.0
    #: all parameters verified against a technical report?
    verified: bool = False
    source_note: str = ""

    def __post_init__(self) -> None:
        if self.annual_production <= 0:
            raise DCFInputError("annual_production doit être strictement positif")
        if self.mine_life_years < 1:
            raise DCFInputError("mine_life_years doit valoir au moins 1")
        if not (0.0 <= self.tax_rate < 1.0):
            raise DCFInputError("tax_rate doit être compris entre 0 et 1")
        if self.discount_rate + self.country_risk_premium <= 0:
            raise DCFInputError("discount_rate + country_risk_premium doit être positif")


@dataclass
class MineProjectResult:
    name: str
    unit_cash_margin: float
    annual_revenue: float              # millions
    annual_ebitda: float               # millions
    annual_fcf: float                  # millions
    npv: float                         # millions
    discount_rate_used: float
    years: List[Dict[str, float]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def mine_project_npv(inputs: MineProjectInputs) -> MineProjectResult:
    """Transparent project NPV for a single mine.

    ``annual revenue = production x price`` (millions is the caller's
    responsibility for the unit of ``production``), ``EBITDA = revenue - cash
    cost x production - sustaining capex``, ``FCF = EBITDA x (1 - tax) + tax
    shield on capex`` approximated as ``EBITDA x (1 - tax)`` when no D&A is
    given.  The model is intentionally simple: it is a sensitivity tool, not a
    feasibility study.
    """
    rate = inputs.discount_rate + inputs.country_risk_premium
    unit_margin = inputs.commodity_price - inputs.cash_cost
    annual_revenue = inputs.annual_production * inputs.commodity_price / 1e6
    annual_cash_cost = inputs.annual_production * inputs.cash_cost / 1e6
    annual_ebitda = annual_revenue - annual_cash_cost - inputs.annual_sustaining_capex
    annual_fcf = annual_ebitda * (1.0 - inputs.tax_rate)

    warnings: List[str] = []
    if unit_margin <= 0:
        warnings.append(
            "Le prix retenu est inférieur au coût de production unitaire : la mine "
            "est en perte au niveau du coût de trésorerie."
        )
    if not inputs.verified:
        warnings.append(
            "Les paramètres techniques (production, coûts, durée de vie) n'ont pas "
            "été vérifiés contre un rapport technique officiel. Résultat indicatif."
        )
    if inputs.country_risk_premium > 0:
        warnings.append(
            f"Une prime de risque pays de {inputs.country_risk_premium:.1%} a été "
            "ajoutée au taux d'actualisation."
        )

    rows: List[Dict[str, float]] = []
    npv = 0.0
    for t in range(1, inputs.mine_life_years + 1):
        df = 1.0 / ((1.0 + rate) ** t)
        pv = annual_fcf * df
        npv += pv
        rows.append({"year": t, "fcf": annual_fcf, "discount_factor": df, "pv": pv})

    return MineProjectResult(
        name=inputs.name,
        unit_cash_margin=unit_margin,
        annual_revenue=annual_revenue,
        annual_ebitda=annual_ebitda,
        annual_fcf=annual_fcf,
        npv=npv,
        discount_rate_used=rate,
        years=rows,
        warnings=warnings,
    )


def commodity_margin_impact(
    revenue: float,
    ebit: float,
    commodity_revenue_share: float,
    price_change: float,
    variable_cost_share: float,
    currency: str = "USD",
) -> Dict[str, float]:
    """Translate a commodity-price move into an EBIT and margin impact.

    ``delta revenue = revenue x share x price_change`` and the cost base
    absorbs ``variable_cost_share`` of the revenue change (royalties, energy,
    consumables that move with throughput).  Everything is linear: this is a
    first-order sensitivity, not a forecast.
    """
    if not (0.0 <= commodity_revenue_share <= 1.0):
        raise DCFInputError("commodity_revenue_share doit être compris entre 0 et 1")
    if not (0.0 <= variable_cost_share <= 1.0):
        raise DCFInputError("variable_cost_share doit être compris entre 0 et 1")

    exposed_revenue = revenue * commodity_revenue_share
    delta_revenue = exposed_revenue * price_change
    delta_cost = delta_revenue * variable_cost_share
    delta_ebit = delta_revenue - delta_cost
    new_revenue = revenue + delta_revenue
    new_ebit = ebit + delta_ebit
    return {
        "exposed_revenue": exposed_revenue,
        "revenue_change": delta_revenue,
        "cost_change": delta_cost,
        "ebit_change": delta_ebit,
        "new_revenue": new_revenue,
        "new_ebit": new_ebit,
        "current_ebit_margin": ebit / revenue if revenue else float("nan"),
        "new_ebit_margin": new_ebit / new_revenue if new_revenue else float("nan"),
        "currency": currency,
    }


# --------------------------------------------------------------------------
# Semiconductors / memory: cycle normalisation
# --------------------------------------------------------------------------


@dataclass
class CycleNormalisation:
    """Mid-cycle view of a cyclical company's economics."""

    periods: List[str] = field(default_factory=list)
    margins: Dict[str, Optional[float]] = field(default_factory=dict)
    revenues: Dict[str, Optional[float]] = field(default_factory=dict)
    peak_margin: Optional[float] = None
    trough_margin: Optional[float] = None
    mid_cycle_margin: Optional[float] = None
    latest_margin: Optional[float] = None
    mid_cycle_revenue: Optional[float] = None
    normalised_ebit: Optional[float] = None
    cycle_spread_pp: Optional[float] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def latest_vs_mid_cycle_pp(self) -> Optional[float]:
        if self.latest_margin is None or self.mid_cycle_margin is None:
            return None
        return self.latest_margin - self.mid_cycle_margin


def normalise_memory_cycle(
    dataset: FinancialDataSet,
    history_window: int = 8,
    use_ebitda: bool = False,
) -> CycleNormalisation:
    """Build a mid-cycle EBIT margin from the reported history.

    The mid-cycle margin is the **median** of the observed margins - deliberately
    not the latest one, and not a hand-picked "normal" year.  The spread between
    the peak and the trough is reported so the analyst can judge how dangerous a
    single-year assumption would be.
    """
    engine = RatioEngine(dataset)
    labels = dataset.periods(PeriodType.ANNUAL)[-history_window:]

    out = CycleNormalisation(periods=list(labels))
    for label in labels:
        metric = engine.ebitda_margin(label) if use_ebitda else engine.operating_margin(label)
        out.margins[label] = None if metric.value is None else metric.value / 100.0
        out.revenues[label] = dataset.value(LineItem.REVENUE, label)

    valid = [m for m in out.margins.values() if m is not None]
    if not valid:
        out.warnings.append(
            "Aucune marge opérationnelle exploitable : impossible de construire une "
            "vue de milieu de cycle. Saisissez les données historiques."
        )
        return out

    out.peak_margin = max(valid)
    out.trough_margin = min(valid)
    out.mid_cycle_margin = statistics.median(valid)
    out.latest_margin = next(
        (out.margins[lbl] for lbl in reversed(labels) if out.margins.get(lbl) is not None), None
    )
    out.cycle_spread_pp = (out.peak_margin - out.trough_margin) * 100.0

    valid_revenues = [r for r in out.revenues.values() if r is not None]
    if valid_revenues:
        out.mid_cycle_revenue = statistics.median(valid_revenues)
    if out.mid_cycle_margin is not None and out.mid_cycle_revenue is not None:
        out.normalised_ebit = out.mid_cycle_revenue * out.mid_cycle_margin

    if len(valid) < 4:
        out.warnings.append(
            f"Seulement {len(valid)} exercice(s) disponible(s) : la normalisation de "
            "cycle n'est pas fiable. Un cycle mémoire complet couvre généralement 3 à "
            "5 ans de haut et de bas."
        )
    if out.cycle_spread_pp is not None and out.cycle_spread_pp > 25:
        out.warnings.append(
            f"L'amplitude des marges atteint {out.cycle_spread_pp:.0f} points de "
            "pourcentage : utiliser une seule année comme référence serait trompeur."
        )
    if out.latest_margin is not None and out.mid_cycle_margin is not None:
        gap = out.latest_vs_mid_cycle_pp
        if gap is not None and abs(gap) > 8:
            direction = "au-dessus" if gap > 0 else "en dessous"
            out.warnings.append(
                f"La dernière marge publiée est {abs(gap):.1f} points {direction} de la "
                "marge de milieu de cycle : le résultat courant n'est pas representatif "
                "d'une année normale."
            )
    return out


# --------------------------------------------------------------------------
# Risk factors (generic, sector-level - never presented as company facts)
# --------------------------------------------------------------------------


SECTOR_RISKS: Dict[str, List[str]] = {
    "mining": [
        "Volatilité des prix des métaux (cobalt, cuivre, or, argent, zinc) : le chiffre d'affaires et l'EBITDA sont directement exposés.",
        "Risque pays et risque politique dans les juridictions d'exploitation (Afrique de l'Ouest et Afrique centrale pour Managem).",
        "Risque de change : coûts en dirhams et dans les devises locales, revenus souvent libellés en dollars.",
        "Risque d'exécution des projets et de dépassement de capex.",
        "Risque de ressource et de réserve : la durée de vie des mines doit être vérifiée dans les rapports techniques.",
        "Risque réglementaire et fiscal minier (codes miniers, redevances, participation de l'Etat).",
        "Risque environnemental et social : autorisations, réhabilitation, conflits avec les communautés locales.",
    ],
    "semiconductors": [
        "Cyclicité du marché mémoire : les prix NAND peuvent varier de façon extrême en quelques trimestres.",
        "Surcapacité sectorielle : les décisions d'investissement des concurrents déterminent l'équilibre offre/demande.",
        "Concentration client : une part importante du chiffre d'affaires peut dépendre d'un petit nombre d'acheteurs.",
        "Intensité capitalistique : le capex en fabs et en R&D pèse sur le flux de trésorerie disponible.",
        "Risque d'inventaire et d'obsolescence technologique.",
        "Dépendance à l'écosystème de production (fonderies, substrats, équipementiers) et à la géographie des chaînes d'approvisionnement.",
        "Exposition aux restrictions commerciales et au contrôle des exportations.",
        "Historique de restructuration/cession : les données antérieures à une séparation sont établies sur un périmètre différent.",
    ],
    "financials": [
        "Qualité du portefeuille de crédits et coût du risque.",
        "Sensibilité aux taux d'intérêt et au spread de taux.",
        "Adéquation des fonds propres réglementaires (ratios Bâle).",
        "Risque de liquidité et de refinancement.",
    ],
}


def sector_risks(sector_key: str) -> List[str]:
    """Generic, sector-level risk factors - clearly not company-specific facts."""
    return list(SECTOR_RISKS.get(sector_key, []))


def sector_key_for(profile: CompanyProfile) -> str:
    """Best-effort mapping from a company profile to a risk/method family."""
    text = f"{profile.sector or ''} {profile.industry or ''}".lower()
    if any(k in text for k in ("mining", "metal", "mine", "extraction")):
        return "mining"
    if any(k in text for k in ("semiconductor", "memory", "flash", "storage", "technolog")):
        return "semiconductors"
    if any(k in text for k in ("bank", "insurance", "financ")):
        return "financials"
    return "generic"


#: Which valuation methods are appropriate for each sector family.
RECOMMENDED_METHODS: Dict[str, List[str]] = {
    "mining": [
        "DCF consolidé (FCFF) avec sensibilité aux prix des métaux",
        "VE/EBITDA sur des pairs miniers comparables",
        "Actif net réévalué / somme des projets (développement futur, données projet vérifiées requises)",
        "ANAV / valeur des réserves (développement futur)",
    ],
    "semiconductors": [
        "DCF sur des marges de milieu de cycle normalisées",
        "VE/EBITDA et VE/CA sur des pairs mémoire et semi-conducteurs",
        "P/E normalisé (bénéfice de milieu de cycle)",
        "Sensibilité explicite à la cyclicité des prix NAND",
    ],
    "financials": [
        "Cours / actif net comptable (P/B) avec ROE justificatif",
        "Revenu résiduel (Edwards-Bell-Ohlson)",
        "Modèle de dividende actualisé (DDM)",
        "Le VE/EBITDA n'est PAS pertinent pour une institution financière",
    ],
    "generic": [
        "DCF (FCFF)",
        "Multiples de comparables (VE/EBITDA, P/E)",
    ],
}


def recommended_methods(profile: CompanyProfile) -> List[str]:
    return list(RECOMMENDED_METHODS.get(sector_key_for(profile), RECOMMENDED_METHODS["generic"]))
