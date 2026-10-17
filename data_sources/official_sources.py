"""Tier-1 official-source catalogue and discovery.

Two mechanisms are provided:

Static catalogue
    Documents that were actually retrieved and verified on the date recorded in
    :data:`CATALOGUE_VERIFIED_ON`.  These are real URLs - nothing is invented.

Live discovery
    Parsing the *official* investor-relations pages (Managem) and the SEC filing
    index (SanDisk) to find the current documents.  Public HTML only, robots.txt
    respected, rate limited by :mod:`data_sources.http_client`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Dict, List, Mapping, Optional
from urllib.parse import urljoin

from data_sources.http_client import HttpClient
from data_sources.parsers import clean_text, normalise_label
from financial_engine.schemas import Currency, Scale, SourceRecord, SourceTier

#: The date the static catalogue below was retrieved and checked.
CATALOGUE_VERIFIED_ON = date(2026, 10, 17)


@dataclass(frozen=True)
class OfficialDocument:
    company_id: str
    name: str
    url: str
    document_type: str
    tier: SourceTier = SourceTier.OFFICIAL
    fiscal_year: Optional[int] = None
    period: Optional[str] = None          # "FY", "S1", "T1"...
    language: str = "fr"
    verified_on: Optional[date] = None
    notes: Optional[str] = None

    def to_source(self) -> SourceRecord:
        return SourceRecord(
            source_id=f"official:{self.company_id}:{_slug(self.name)}",
            name=self.name,
            url=self.url,
            publisher=self.company_id,
            publication_date=None,
            document_type=self.document_type,
            tier=self.tier,
            retrieved_at=datetime.now(timezone.utc),
            notes=self.notes,
        )


def _slug(text: str) -> str:
    return normalise_label(text).replace(" ", "-")[:70]


# --------------------------------------------------------------------------
# Managem - verified official documents (managemgroup.com)
# --------------------------------------------------------------------------

_MANAGEM_BASE = "https://www.managemgroup.com"

MANAGEM_DOCUMENTS: List[OfficialDocument] = [
    OfficialDocument(
        "MANAGEM", "Managem - Rapport Financier Semestriel S1 2026",
        f"{_MANAGEM_BASE}/sites/default/files/2026-09/MANAGEM%20RFS%202026.pdf",
        "Rapport financier semestriel", fiscal_year=2026, period="S1", language="fr",
        verified_on=CATALOGUE_VERIFIED_ON,
        notes="Comptes consolidés semestriels.",
    ),
    OfficialDocument(
        "MANAGEM", "Managem - Publication résultats S1 2026",
        f"{_MANAGEM_BASE}/sites/default/files/2026-09/MANAGEM%20Publication%20S1%202026.pdf",
        "Publication de résultats", fiscal_year=2026, period="S1", language="fr",
        verified_on=CATALOGUE_VERIFIED_ON,
    ),
    OfficialDocument(
        "MANAGEM", "Managem - Communique résultats S1 2026",
        f"{_MANAGEM_BASE}/sites/default/files/2026-09/CP%20MNG%20S1_2026%20V2.pdf",
        "Communique de presse", fiscal_year=2026, period="S1", language="fr",
        verified_on=CATALOGUE_VERIFIED_ON,
    ),
    OfficialDocument(
        "MANAGEM", "Managem - Rapport Financier Annuel 2025",
        f"{_MANAGEM_BASE}/sites/default/files/2026-04/RFA%20Managem%202025%2023H20.pdf",
        "Rapport financier annuel", fiscal_year=2025, period="FY", language="fr",
        verified_on=CATALOGUE_VERIFIED_ON,
        notes="Comptes consolidés annuels de l'exercice 2025.",
    ),
    OfficialDocument(
        "MANAGEM", "Managem - Rapport Financier Semestriel 2025",
        f"{_MANAGEM_BASE}/sites/default/files/2025-09/RFS%202025%20MANAGEM%203009.pdf",
        "Rapport financier semestriel", fiscal_year=2025, period="S1", language="fr",
        verified_on=CATALOGUE_VERIFIED_ON,
    ),
    OfficialDocument(
        "MANAGEM", "Managem - Rapport Financier Annuel 2024",
        f"{_MANAGEM_BASE}/sites/default/files/2025-05/MANAGEM%20RFA%202024%2030042025%2023H30.pdf",
        "Rapport financier annuel", fiscal_year=2024, period="FY", language="fr",
        verified_on=CATALOGUE_VERIFIED_ON,
    ),
    OfficialDocument(
        "MANAGEM", "Managem - Rapport Financier Annuel 2023",
        f"{_MANAGEM_BASE}/sites/default/files/2024-04/MANAGEM%20RFA%202023%2030-04-2024%2021H.pdf",
        "Rapport financier annuel", fiscal_year=2023, period="FY", language="fr",
        verified_on=CATALOGUE_VERIFIED_ON,
    ),
    OfficialDocument(
        "MANAGEM", "Managem - Communique résultats T1 2026",
        f"{_MANAGEM_BASE}/sites/default/files/2026-05/CP%20MNG%20T1%202026.pdf",
        "Communique de presse", fiscal_year=2026, period="T1", language="fr",
        verified_on=CATALOGUE_VERIFIED_ON,
    ),
    OfficialDocument(
        "MANAGEM", "Managem - Communique résultats T2 2026",
        f"{_MANAGEM_BASE}/sites/default/files/2026-08/CP%20MNG%20T2%202026_0.pdf",
        "Communique de presse", fiscal_year=2026, period="T2", language="fr",
        verified_on=CATALOGUE_VERIFIED_ON,
    ),
]

#: Official investor-relations landing pages used for live discovery.
MANAGEM_IR_PAGES: Dict[str, str] = {
    "rapports-annuels": f"{_MANAGEM_BASE}/investisseurs/managem/rapports-annuels",
    "communiques-et-publications": f"{_MANAGEM_BASE}/Investisseurs/Managem/communiques-et-publications",
    "actionnariat": f"{_MANAGEM_BASE}/investisseurs/managem/actionnariat",
}

#: Other regulators / market operators worth quoting for Moroccan companies.
MOROCCAN_SOURCES: Dict[str, str] = {
    "Bourse de Casablanca": "https://www.casablanca-bourse.com",
    "AMMC (Autorité Marocaine du Marché des Capitaux)": "https://www.ammc.ma",
}


def managem_documents() -> List[OfficialDocument]:
    return list(MANAGEM_DOCUMENTS)


# --------------------------------------------------------------------------
# Live discovery
# --------------------------------------------------------------------------

#: Words that identify a financial document link on an IR page.  Accent-free:
#: both the hint and the link text are folded through ``normalise_label``.
_DOC_HINTS = (
    "rapport", "rfa", "rfs", "comptes", "resultat", "publication", "communique",
    "annual report", "financial", "earnings", "10-k", "10-q", "presentation",
)

_FINAL_DOC_RE = None


def discover_links(http: HttpClient, page_url: str, same_host_only: bool = True) -> List[Dict[str, str]]:
    """Every link on a page, with absolute URLs and a short label.

    This is a *discovery* helper: it returns pointers, never financial values.
    """
    response = http.get(page_url)
    if not response.ok:
        return []
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    from urllib.parse import urlparse

    base_host = urlparse(page_url).netloc
    out: List[Dict[str, str]] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if not href or href.startswith(("#", "mailto:", "javascript:")):
            continue
        absolute = urljoin(page_url, href)
        parsed = urlparse(absolute)
        if parsed.scheme not in ("http", "https"):
            continue
        if same_host_only and parsed.netloc and parsed.netloc != base_host:
            continue
        if absolute in seen:
            continue
        seen.add(absolute)
        out.append(
            {
                "url": absolute,
                "label": clean_text(anchor.get_text(" ", strip=True))[:160],
                "parent": page_url,
            }
        )
    return _dedupe_and_rank(out)


def _dedupe_and_rank(links: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """De-duplicate, drop anchors pointing at the page itself, rank documents first."""
    def score(link: Dict[str, str]) -> tuple[int, int]:
        raw = f"{link['url']} {link.get('label', '')}".lower()
        folded = normalise_label(raw)
        is_doc = 1 if ".pdf" in raw else 0
        hinted = sum(1 for h in _DOC_HINTS if normalise_label(h) in folded)
        return (-(is_doc * 10 + hinted), len(link["url"]))

    filtered = [l for l in links if l["url"].rstrip("/") != l.get("parent", "").rstrip("/")]
    return sorted(filtered, key=score)


def discover_managem_documents(http: Optional[HttpClient] = None) -> List[Dict[str, str]]:
    """Live discovery over Managem's official IR pages."""
    http = http or HttpClient()
    found: List[Dict[str, str]] = []
    for label, url in MANAGEM_IR_PAGES.items():
        for link in discover_links(http, url):
            folded = normalise_label(f"{link['url']} {link.get('label','')}")
            if any(normalise_label(h) in folded for h in _DOC_HINTS):
                link["page"] = label
                found.append(link)
    return found


def discover_sec_filings(http: Optional[HttpClient] = None, cik: str = "0002023554", limit: int = 40) -> List[Dict[str, object]]:
    """Recent SEC filings for a CIK, via the official submissions API."""
    from data_sources.sec_edgar import SecEdgarClient, fetch_recent_filings

    client = SecEdgarClient(http=http)
    return fetch_recent_filings(client, cik, forms=("10-K", "10-Q", "8-K", "DEF 14A"), limit=limit)


# --------------------------------------------------------------------------
# Unified entry point
# --------------------------------------------------------------------------


@dataclass
class SourceDiscoveryResult:
    company_id: str
    official_documents: List[Dict[str, object]] = field(default_factory=list)
    filings: List[Dict[str, object]] = field(default_factory=list)
    secondary_sources: List[Dict[str, str]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def official_count(self) -> int:
        return len(self.official_documents) + len(self.filings)


def _field(document: object, key: str, default: object = None) -> object:
    """Read ``key`` from either a mapping or an object with attributes."""
    if isinstance(document, Mapping):
        return document.get(key, default)
    return getattr(document, key, default)


def document_rows(documents: Optional[List[object]]) -> List[Dict[str, object]]:
    """Normalise a document list to plain dictionaries.

    ``official_documents`` mixes ``OfficialDocument`` dataclasses (static
    catalogue) with plain mappings (live discovery built from fetched links),
    so every consumer must go through one reader rather than guessing which
    accessor works.  ``verified_on`` is flattened to an ISO string (or None).
    """
    rows: List[Dict[str, object]] = []
    for document in documents or []:
        verified_on = _field(document, "verified_on")
        if isinstance(verified_on, (date, datetime)):
            verified_on = verified_on.isoformat()
        tier = _field(document, "tier")
        if tier is None:
            tier = ""
        elif isinstance(tier, SourceTier):
            tier = tier.name.lower()
        elif isinstance(tier, int):
            try:
                tier = SourceTier(tier).name.lower()
            except ValueError:
                tier = str(tier)
        else:
            tier = str(tier)
        rows.append(
            {
                "name": _field(document, "name") or "",
                "url": _field(document, "url") or "",
                "document_type": _field(document, "document_type") or "",
                "tier": tier if tier else "official",
                "fiscal_year": _field(document, "fiscal_year"),
                "period": _field(document, "period"),
                "verified_on": verified_on or None,
                "origin": _field(document, "origin") or "",
                "notes": _field(document, "notes") or "",
            }
        )
    return rows


def discover_official_sources(company_id: str, http: Optional[HttpClient] = None, live: bool = True) -> SourceDiscoveryResult:
    """Tier-1 sources for a company: static catalogue + optional live discovery."""
    http = http or HttpClient()
    result = SourceDiscoveryResult(company_id=company_id.upper())

    if company_id.upper() == "MANAGEM":
        result.official_documents = [
            {
                "name": d.name,
                "url": d.url,
                "document_type": d.document_type,
                "fiscal_year": d.fiscal_year,
                "period": d.period,
                "verified_on": d.verified_on.isoformat() if d.verified_on else None,
                "origin": "static catalogue",
            }
            for d in managem_documents()
        ]
        result.notes.append(
            "Documents vérifiés le "
            f"{CATALOGUE_VERIFIED_ON.isoformat()} sur managemgroup.com. "
            "La Bourse de Casablanca n'a pas pu être interrogée depuis cette machine : "
            "son certificat TLS n'est pas reconnu localement. Téléchargez le bulletin "
            "de la Bourse manuellement si nécessaire."
        )
        if live:
            try:
                live_links = discover_managem_documents(http)
                known = {d["url"] for d in result.official_documents}
                for link in live_links:
                    if link["url"] not in known:
                        result.official_documents.append(
                            {
                                "name": link.get("label") or link["url"].rsplit("/", 1)[-1],
                                "url": link["url"],
                                "document_type": "lien découvert (à vérifier)",
                                "fiscal_year": None,
                                "period": None,
                                "verified_on": None,
                                "origin": f"découverte live ({link.get('page','')})",
                            }
                        )
                result.notes.append(
                    f"{len(live_links)} lien(s) supplémentaire(s) découvert(s) en direct."
                )
            except Exception as exc:  # pragma: no cover - network dependent
                result.errors.append(f"Découverte live Managem indisponible : {exc}")

        result.secondary_sources = [
            {"name": name, "url": url} for name, url in MOROCCAN_SOURCES.items()
        ]

    elif company_id.upper() == "SANDISK":
        if live:
            try:
                result.filings = discover_sec_filings(http)
            except Exception as exc:  # pragma: no cover - network dependent
                result.errors.append(f"Index SEC indisponible : {exc}")
        result.notes.append(
            "Les documents officiels de SanDisk sont récupérés a la demande via les "
            "API SEC EDGAR (submissions, companyfacts). Aucun chiffre n'est mis en "
            "cache en dur dans le code."
        )
    else:
        result.errors.append(f"Société inconnue du catalogue officiel : {company_id}")

    return result
