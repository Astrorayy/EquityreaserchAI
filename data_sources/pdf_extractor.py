"""PDF extraction (PyMuPDF with a pdfplumber fallback).

Design rules
------------
* Extraction is **assisted, never authoritative**: the result carries a page
  number, the extractor used, and a confidence flag.  Nothing is written to the
  dataset without an explicit confirmation step in the UI.
* Scanned/image-only pages are reported as such instead of being silently
  returned as empty text (no OCR is bundled, and pretending otherwise would
  produce fabricated numbers).
* Tables come back with the header row and a scale guess taken from the text
  surrounding the table, because "1 234" means very different things depending
  on whether the header says "en milliers" or "en millions".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from data_sources.parsers import clean_text, guess_scale, normalise_label, parse_number

#: Documents above this size are refused: they are almost never financial
#: statements, and 8 GB of RAM must stay available for the dashboard.
MAX_PDF_BYTES = 60 * 1024 * 1024


@dataclass
class PdfPage:
    number: int              # 1-based
    text: str
    is_empty: bool = False
    likely_scanned: bool = False
    char_count: int = 0

    def __post_init__(self) -> None:
        self.char_count = len(self.text or "")
        self.is_empty = self.char_count == 0


@dataclass
class PdfTable:
    page: int
    index: int
    rows: List[List[str]]
    scale: Optional[float] = None
    scale_evidence: str = ""
    currency: Optional[str] = None
    extractor: str = "pymupdf"
    confidence: str = "low"
    title_hint: Optional[str] = None

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return max((len(r) for r in self.rows), default=0)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame()
        width = self.n_cols
        padded = [row + [""] * (width - len(row)) for row in self.rows]
        frame = pd.DataFrame(padded)
        if not frame.empty:
            header = [clean_text(c) for c in frame.iloc[0].tolist()]
            if any(header):
                frame.columns = [
                    header[i] if header[i] else f"col_{i}" for i in range(len(header))
                ]
                frame = frame.iloc[1:].reset_index(drop=True)
        return frame

    def numeric_preview(self, max_rows: int = 8) -> List[Dict[str, object]]:
        """First column plus parsed numbers, so the analyst can sanity-check."""
        out: List[Dict[str, object]] = []
        for row in self.rows[:max_rows]:
            if not row:
                continue
            out.append(
                {
                    "label": clean_text(row[0]),
                    "values": [
                        parse_number(cell) for cell in row[1: max_rows + 1]
                    ],
                }
            )
        return out


@dataclass
class PdfDocument:
    path: Path
    pages: List[PdfPage] = field(default_factory=list)
    tables: List[PdfTable] = field(default_factory=list)
    extractor: str = ""
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def scanned_page_count(self) -> int:
        return sum(1 for p in self.pages if p.likely_scanned)

    @property
    def full_text(self) -> str:
        return "\n".join(p.text for p in self.pages)

    @property
    def text_per_page(self) -> float:
        if not self.pages:
            return 0.0
        return sum(p.char_count for p in self.pages) / len(self.pages)

    def tables_on_page(self, page: int) -> List[PdfTable]:
        return [t for t in self.tables if t.page == page]

    def search(self, needle: str, context: int = 120) -> List[Dict[str, object]]:
        """Find a phrase and return a snippet with its page number."""
        needle_norm = clean_text(needle).lower()
        if not needle_norm:
            return []
        out: List[Dict[str, object]] = []
        for page in self.pages:
            haystack = (page.text or "").lower()
            start = 0
            while True:
                found = haystack.find(needle_norm, start)
                if found < 0:
                    break
                out.append(
                    {
                        "page": page.number,
                        "snippet": clean_text(
                            page.text[max(0, found - context): found + len(needle) + context]
                        ),
                    }
                )
                start = found + len(needle_norm)
                if len(out) >= 40:
                    return out
        return out

    def summary(self) -> Dict[str, object]:
        return {
            "file": self.path.name,
            "pages": self.page_count,
            "tables": len(self.tables),
            "scanned_pages": self.scanned_page_count,
            "avg_chars_per_page": round(self.text_per_page),
            "extractor": self.extractor,
            "error": self.error,
        }


# --------------------------------------------------------------------------
# Heuristics
# --------------------------------------------------------------------------

#: Financial tables are dense in these words.  Kept accent-free: both the hint
#: and the page text are folded through ``normalise_label`` before comparing,
#: so "resultat" also matches "Résultat".
_FINANCIAL_HINTS = (
    "revenue", "chiffre d'affaires", "chiffre d affaires", "turnover", "sales",
    "net income", "resultat net", "resultat", "ebitda", "ebit", "operating income",
    "resultat d'exploitation", "gross profit", "marge brute", "total assets",
    "actif", "passif", "liabilities", "dettes", "cash flow", "tresorerie",
    "flux de tresorerie", "capital expenditure", "investissements", "equity",
    "capitaux propres",
)


def _looks_financial(text: str) -> tuple[bool, int]:
    lowered = normalise_label(text)
    hits = sum(1 for hint in _FINANCIAL_HINTS if normalise_label(hint) in lowered)
    return hits > 0, hits


def _has_enough_numbers(text: str) -> bool:
    if not text:
        return False
    tokens = clean_text(text).split()
    if not tokens:
        return False
    numeric = sum(1 for t in tokens[:400] if parse_number(t) is not None)
    return numeric >= 5


# --------------------------------------------------------------------------
# PyMuPDF backend
# --------------------------------------------------------------------------

def _extract_with_pymupdf(path: Path) -> PdfDocument:
    doc = PdfDocument(path=path, extractor="pymupdf")
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        doc.error = f"PyMuPDF non installe : {exc}"
        return doc

    try:
        handle = fitz.open(str(path))
    except Exception as exc:  # pragma: no cover - corrupt file dependent
        doc.error = f"Ouverture du PDF impossible : {type(exc).__name__}: {exc}"
        return doc

    try:
        doc.metadata = {
            "title": (handle.metadata or {}).get("title"),
            "author": (handle.metadata or {}).get("author"),
            "creator": (handle.metadata or {}).get("creator"),
            "creation_date": (handle.metadata or {}).get("creationDate"),
            "page_count": handle.page_count,
        }

        for page_index in range(handle.page_count):
            page = handle.load_page(page_index)
            try:
                text = page.get_text("text") or ""
            except Exception:  # pragma: no cover
                text = ""
            page_obj = PdfPage(number=page_index + 1, text=text)
            # A page with pictures but no text is almost certainly a scan.
            if page_obj.is_empty:
                try:
                    images = page.get_images(full=True) or []
                    page_obj.likely_scanned = len(images) > 0
                except Exception:  # pragma: no cover
                    page_obj.likely_scanned = False
            doc.pages.append(page_obj)

            # Table detection: PyMuPDF's find_tables is available on recent builds.
            try:
                finder = page.find_tables()
                raw_tables = getattr(finder, "tables", []) or []
            except Exception:
                raw_tables = []

            for t_index, table in enumerate(raw_tables):
                try:
                    rows = [[clean_text(c) for c in row] for row in (table.extract() or [])]
                except Exception:  # pragma: no cover
                    continue
                rows = [r for r in rows if any(c for c in r)]
                if len(rows) < 2:
                    continue
                scale_guess = guess_scale(text)
                financial, _hits = _looks_financial(text)
                confidence = "medium" if (financial and _has_enough_numbers(text)) else "low"
                doc.tables.append(
                    PdfTable(
                        page=page_index + 1,
                        index=t_index,
                        rows=rows,
                        scale=scale_guess.scale,
                        scale_evidence=scale_guess.evidence,
                        currency=scale_guess.currency,
                        extractor="pymupdf",
                        confidence=confidence,
                        title_hint=_nearest_heading(text, rows[0]),
                    )
                )
    finally:
        try:
            handle.close()
        except Exception:  # pragma: no cover
            pass

    if not doc.pages:
        doc.error = "Le PDF ne contient aucune page lisible."
    elif doc.scanned_page_count == doc.page_count:
        doc.warnings.append(
            "Toutes les pages semblent être des images scannées : aucun texte "
            "extractible. L'application n'embarque pas d'OCR, donc aucune donnée ne "
            "peut être extraite automatiquement. Utilisez la saisie manuelle ou une "
            "version texte du document."
        )
    elif doc.scanned_page_count:
        doc.warnings.append(
            f"{doc.scanned_page_count} page(s) sur {doc.page_count} semblent scannées "
            "et n'ont pas pu être lues."
        )

    if not doc.tables and doc.page_count:
        doc.warnings.append(
            "Aucun tableau structure detecte. PyMuPDF ne reconnait que les tableaux "
            "vectorises ; un PDF mis en page sans bordures peut necessiter une saisie "
            "manuelle."
        )
    return doc


# --------------------------------------------------------------------------
# pdfplumber fallback backend
# --------------------------------------------------------------------------

def _extract_with_pdfplumber(path: Path) -> PdfDocument:
    doc = PdfDocument(path=path, extractor="pdfplumber")
    try:
        import pdfplumber
    except ImportError as exc:
        doc.error = f"pdfplumber non installe : {exc}"
        return doc

    try:
        with pdfplumber.open(str(path)) as handle:
            for page_index, page in enumerate(handle.pages, start=1):
                try:
                    text = page.extract_text() or ""
                except Exception:  # pragma: no cover
                    text = ""
                page_obj = PdfPage(number=page_index, text=text)
                if page_obj.is_empty:
                    page_obj.likely_scanned = bool(getattr(page, "images", []))
                doc.pages.append(page_obj)

                try:
                    raw_tables = page.extract_tables() or []
                except Exception:  # pragma: no cover
                    raw_tables = []

                for t_index, table in enumerate(raw_tables):
                    rows = [
                        [clean_text(cell) for cell in (row or [])]
                        for row in table
                    ]
                    rows = [r for r in rows if any(c for c in r)]
                    if len(rows) < 2:
                        continue
                    scale_guess = guess_scale(text)
                    financial, _hits = _looks_financial(text)
                    doc.tables.append(
                        PdfTable(
                            page=page_index,
                            index=t_index,
                            rows=rows,
                            scale=scale_guess.scale,
                            scale_evidence=scale_guess.evidence,
                            currency=scale_guess.currency,
                            extractor="pdfplumber",
                            confidence="medium" if financial else "low",
                            title_hint=_nearest_heading(text, rows[0]),
                        )
                    )
    except Exception as exc:  # pragma: no cover
        doc.error = f"pdfplumber a échoué : {type(exc).__name__}: {exc}"
    return doc


def _nearest_heading(page_text: str, header_row: Sequence[str]) -> Optional[str]:
    """Best-effort caption for a table: the last short line above it."""
    if not page_text:
        return None
    first_cell = clean_text(header_row[0]) if header_row else ""
    lines = [clean_text(l) for l in page_text.splitlines() if clean_text(l)]
    if first_cell:
        needle = first_cell[:12].lower()
        for i, line in enumerate(lines):
            if needle and needle in line.lower() and i > 0:
                return lines[i - 1][:160]
    # Fall back to the last short line of the page.
    short = [l for l in lines if 4 <= len(l) <= 90]
    return short[-1][:160] if short else None


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def extract_pdf(path: Path | str, prefer_pdfplumber: bool = False) -> PdfDocument:
    """Extract text and tables from a PDF.

    PyMuPDF is used first because it is markedly faster; pdfplumber is used when
    PyMuPDF finds no tables (it is slower but more tolerant of borderless
    layouts).
    """
    pdf_path = Path(path)
    if not pdf_path.exists():
        return PdfDocument(path=pdf_path, error=f"Fichier introuvable : {pdf_path}")
    size = pdf_path.stat().st_size
    if size == 0:
        return PdfDocument(path=pdf_path, error="Le fichier est vide.")
    if size > MAX_PDF_BYTES:
        return PdfDocument(
            path=pdf_path,
            error=(
                f"PDF de {size / 1e6:.0f} Mo refuse (limite {MAX_PDF_BYTES / 1e6:.0f} Mo) : "
                "les documents de cette taille ne sont pas des etats financiers et "
                "satureraient la mémoire disponible."
            ),
        )

    document = (
        _extract_with_pdfplumber(pdf_path)
        if prefer_pdfplumber
        else _extract_with_pymupdf(pdf_path)
    )

    if document.ok and not document.tables and not prefer_pdfplumber:
        fallback = _extract_with_pdfplumber(pdf_path)
        if fallback.ok and fallback.tables:
            fallback.warnings.append(
                "Tableaux extraits avec pdfplumber (PyMuPDF n'a rien detecte)."
            )
            return fallback

    if not document.ok and not prefer_pdfplumber:
        fallback = _extract_with_pdfplumber(pdf_path)
        if fallback.ok:
            fallback.warnings.append(
                f"PyMuPDF a échoué ({document.error}) ; pdfplumber a été utilisé."
            )
            return fallback

    return document


def candidate_tables(
    document: PdfDocument, min_rows: int = 3, min_numbers: int = 4
) -> List[PdfTable]:
    """Tables that look like financial statements, best candidates first."""
    scored: List[tuple[int, PdfTable]] = []
    for table in document.tables:
        if table.n_rows < min_rows:
            continue
        numeric_cells = 0
        for row in table.rows[:30]:
            for cell in row[1:]:
                if parse_number(cell) is not None:
                    numeric_cells += 1
        if numeric_cells < min_numbers:
            continue
        blob = " ".join(" ".join(r[:2]) for r in table.rows[:25])
        financial, hits = _looks_financial(blob)
        score = numeric_cells + hits * 5 + (10 if financial else 0)
        if table.confidence == "medium":
            score += 5
        scored.append((score, table))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [t for _, t in scored]


def extract_labeled_values(
    document: PdfDocument, label: str, max_hits: int = 6
) -> List[Dict[str, object]]:
    """Find a label in the tables and return the numbers on the same row.

    This is deliberately conservative: it reports what it found, with the page
    and the raw cells, and never decides on its own which column is the right
    period.  A human confirms before anything enters the dataset.
    """
    from data_sources.parsers import normalise_label

    target = normalise_label(label)
    if not target:
        return []
    out: List[Dict[str, object]] = []
    for table in document.tables:
        for row_index, row in enumerate(table.rows):
            if not row:
                continue
            candidate = normalise_label(row[0])
            if not candidate:
                continue
            if target in candidate or candidate in target:
                numbers = [parse_number(cell) for cell in row[1:]]
                numbers = [n for n in numbers if n is not None]
                if not numbers:
                    continue
                out.append(
                    {
                        "page": table.page,
                        "table_index": table.index,
                        "row_index": row_index,
                        "label_found": clean_text(row[0]),
                        "raw_cells": row,
                        "numbers": numbers,
                        "scale": table.scale,
                        "scale_evidence": table.scale_evidence,
                        "currency": table.currency,
                        "extractor": table.extractor,
                        "confidence": table.confidence,
                    }
                )
                if len(out) >= max_hits:
                    return out
    return out
