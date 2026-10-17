"""Company registry.

Every field below was checked against an official source; anything that could
not be verified is left ``None`` and explained in ``identity_notes`` rather than
guessed.  The verification date of each profile is recorded explicitly.

Managem
    Identity and the investor-relations document URLs come from
    https://www.managemgroup.com (investor pages, checked 2026-10-17).
    The Casablanca Stock Exchange website could not be reached from the test
    machine because its TLS certificate chain is not trusted locally - that is
    documented, not worked around.

SanDisk
    Identity, CIK, SIC code, exchange and fiscal year end come from the SEC
    EDGAR submissions API for CIK 0002023554 (checked 2026-10-17).  Financial
    figures are deliberately NOT stored here: they are downloaded from EDGAR at
    run time so they can never go stale.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

from financial_engine.schemas import CompanyProfile, Currency, Scale

#: When the identity fields below were last verified against an official source.
PROFILES_VERIFIED_ON = date(2026, 10, 17)

_MANAGEM = CompanyProfile(
    company_id="MANAGEM",
    name="Managem",
    short_name="Managem",
    ticker="MNG",
    exchange="Bourse de Casablanca",
    country="Maroc",
    currency=Currency.MAD,
    sector="Mines et metaux",
    industry="Extraction et production de métaux non ferreux",
    description=(
        "Groupe minier marocain intégré (extraction, hydrométallurgie et "
        "commercialisation) dont le portefeuille couvre notamment le cobalt, le "
        "cuivre, l'or, l'argent, le zinc, le plomb et la fluorine, au Maroc et en "
        "Afrique (Gabon, Guinée, RDC, Soudan, Mali, Côte d'Ivoire, Éthiopie). "
        "Filiale du groupe Al Mada. La société détient des participations dans des "
        "sociétés également cotées à Casablanca, notamment la Société Métallurgique "
        "d'Imiter (SMI)."
    ),
    fiscal_year_end="12-31",
    website="https://www.managemgroup.com",
    ir_url="https://www.managemgroup.com/investisseurs/managem/rapports-annuels",
    reporting_scale=Scale.MILLIONS,
    sec_cik=None,           # not an SEC registrant
    isin=None,              # not verified from an official source
    identity_notes=(
        "Périmètre de consolidation : les comptes consolidés de Managem incluent "
        "les filiales cotées (SMI notamment). Ne comparez pas les comptes sociaux "
        "d'une filiale avec les comptes consolidés du groupe. "
        "Le régime de change marocain (dirham géré par Bank Al-Maghrib) doit être "
        "pris en compte pour toute conversion MAD/USD."
    ),
    data_sources=[
        "Rapports financiers annuels et semestriels publiés par Managem",
        "Communiqués de presse du groupe",
        "Bulletin de la Bourse de Casablanca",
        "Publications de l'AMMC (régulateur du marché marocain)",
    ],
)

_SANDISK = CompanyProfile(
    company_id="SANDISK",
    name="SanDisk Corporation",
    short_name="SanDisk",
    ticker="SNDK",
    exchange="Nasdaq",
    country="États-Unis",
    currency=Currency.USD,
    sector="Semi-conducteurs",
    industry="Mémoire flash NAND et stockage (SSD, cartes mémoire, solutions embarquées)",
    description=(
        "Fabricant de mémoires flash NAND et de solutions de stockage (SSD, cartes "
        "mémoire, produits embarqués). La société résulte de la séparation de "
        "l'activité flash de Western Digital, finalisée en février 2025, et est "
        "cotée sur le Nasdaq sous le ticker SNDK. L'activité est fortement "
        "cyclique : les prix de la mémoire NAND déterminent l'essentiel de la "
        "rentabilité d'une année sur l'autre."
    ),
    # Verified via the SEC submissions API: fiscalYearEnd = "0628".
    fiscal_year_end="06-28",
    website="https://www.sandisk.com",
    ir_url="https://investor.sandisk.com",
    reporting_scale=Scale.MILLIONS,
    sec_cik="0002023554",
    identity_notes=(
        "IDENTITÉ — point critique. SanDisk Corporation (CIK 0002023554) est une "
        "entité juridique distincte de Western Digital Corporation "
        "(CIK 0000106040, ancien ticker SNDK). Les états financiers antérieurs à "
        "la séparation décrivent le périmètre flash de Western Digital en comptes "
        "carve-out ; ils ne sont pas comparables sans retraitement à ceux de "
        "l'entité autonome. Les tables ci-dessous n'excluent jamais silencieusement "
        "ces périodes : elles sont étiquetées et signalées."
    ),
    listing_date=None,  # exact spin-off completion day not verified from a filing
    history_cutoff=date(2024, 12, 27),
    history_cutoff_note=(
        "Premier dépôt autonome : formulaire 10-Q pour le trimestre clos le "
        "27/12/2024, déposé le 07/03/2025. Le formulaire 10 d'enregistrement "
        "avait été déposé le 25/11/2024. Toute donnée antérieure provient du "
        "périmètre Western Digital et doit être traitée comme une information "
        "pro forma / carve-out, pas comme un historique de la société cotée."
    ),
    data_sources=[
        "SEC EDGAR (formulaires 10-K, 10-Q, 8-K)",
        "API SEC company submissions et XBRL company facts",
        "Communiqués de résultats et présentations investisseurs de SanDisk",
    ],
)

#: The registry. Adding a company here is all that is needed for the UI to
#: offer it: the DCF, ratio, validation and report layers are company-agnostic.
REGISTRY: Dict[str, CompanyProfile] = {
    "MANAGEM": _MANAGEM,
    "SANDISK": _SANDISK,
}

#: Display order in the sidebar.
DEFAULT_ORDER: List[str] = ["MANAGEM", "SANDISK"]


def list_profiles() -> List[CompanyProfile]:
    return [REGISTRY[k] for k in DEFAULT_ORDER if k in REGISTRY]


def get_profile(company_id: str) -> Optional[CompanyProfile]:
    return REGISTRY.get(company_id.upper())


def profile_by_ticker(ticker: str) -> Optional[CompanyProfile]:
    needle = ticker.strip().upper()
    for profile in REGISTRY.values():
        if profile.ticker.upper() == needle:
            return profile
    return None


def search(query: str) -> List[CompanyProfile]:
    """Free-text search over name, ticker, sector and country."""
    needle = query.strip().lower()
    if not needle:
        return list_profiles()
    out: List[CompanyProfile] = []
    for p in REGISTRY.values():
        haystack = " ".join(
            filter(None, [p.name, p.short_name, p.ticker, p.exchange, p.country,
                          p.sector, p.industry, p.description])
        ).lower()
        if needle in haystack:
            out.append(p)
    return out


def currency_for(company_id: str) -> Currency:
    profile = get_profile(company_id)
    return profile.currency if profile else Currency.USD


def fiscal_year_end_for(company_id: str) -> Optional[str]:
    profile = get_profile(company_id)
    return profile.fiscal_year_end if profile else None


def fiscal_period_end_month(company_id: str) -> Optional[int]:
    """Month number of the fiscal year end, used to infer the fiscal year."""
    fy = fiscal_year_end_for(company_id)
    if not fy or "-" not in fy:
        return None
    try:
        return int(fy.split("-")[0])
    except ValueError:
        return None


def parse_fiscal_year_end(raw: Optional[str]) -> Optional[tuple[int, int]]:
    """Parse a fiscal year end in either of the two formats in use here.

    * SEC EDGAR / ``CompanyProfile.fiscal_year_end``: ``"0628"`` (MMDD)
    * narrative / ISO style: ``"06-28"``

    Returns ``(month, day)`` or ``None`` when the input is unusable.
    """
    if not raw:
        return None
    text = str(raw).strip()
    if "-" in text:
        parts = text.split("-", 1)
    elif len(text) == 4 and text.isdigit():
        parts = [text[:2], text[2:]]
    else:
        parts = []
    if len(parts) != 2:
        return None
    try:
        month, day = int(parts[0]), int(parts[1])
    except (TypeError, ValueError):
        return None
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    return month, day


def fiscal_year_for(period_end: date, fiscal_year_end: Optional[str]) -> int:
    """Label a period end date with its fiscal year.

    The fiscal year is named after the **calendar year of the period end**, but
    a 52/53-week filer's actual year end can fall a few days after the nominal
    date (SanDisk's nominal 06-28 year end became 2026-07-03).  The rule below
    therefore snaps to the nominal date with a one-week tolerance:

    * ``nominal = date(period_end.year, M, D)``
    * fiscal year = ``period_end.year`` if ``period_end <= nominal + 7 days``
    * otherwise ``period_end.year + 1``

    Verified against SEC data for SanDisk (FY end 06-28):

    ====================  ===========  ==================
    period end            result       reality (SEC)
    ====================  ===========  ==================
    2026-07-03            FY2026       FY2026 (10-K)
    2025-06-27            FY2025       FY2025 (10-K)
    2026-01-02            FY2026       Q2 FY2026 (10-Q)
    2024-12-27            FY2025       Q2 FY2025 (10-Q)
    2023-09-29            FY2024       Q1 FY2024 (pro forma)
    ====================  ===========  ==================
    """
    parsed = parse_fiscal_year_end(fiscal_year_end)
    if parsed is None:
        return period_end.year
    month, day = parsed

    try:
        nominal = date(period_end.year, month, day)
    except ValueError:
        nominal = date(period_end.year, month, 28)

    # Seven-day tolerance covers 52/53-week retail/semiconductor calendars.
    if (period_end - nominal).days <= 7:
        return period_end.year
    return period_end.year + 1


def infer_fiscal_year(company_id: str, period_end: date) -> int:
    """Fiscal year label for a period end date of a registered company."""
    return fiscal_year_for(period_end, fiscal_year_end_for(company_id))
