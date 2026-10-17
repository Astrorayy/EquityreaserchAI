"""HTML table extraction for permitted public pages.

Only public HTML is parsed, via :mod:`data_sources.http_client` (which honours
robots.txt and rate limits).  The output mirrors the PDF extractor so that the
UI can treat both the same way: rows of cleaned strings, a scale guess, and a
confidence flag.  No value is ever written to the dataset automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from data_sources.http_client import HttpClient
from data_sources.parsers import clean_text, guess_scale, normalise_label, parse_number


@dataclass
class HtmlTable:
    source_url: str
    index: int
    rows: List[List[str]]
    caption: Optional[str] = None
    scale: Optional[float] = None
    scale_evidence: str = ""
    currency: Optional[str] = None
    confidence: str = "low"
    context: str = ""

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
        header = (
            [clean_text(c) for c in frame.iloc[0].tolist()] if not frame.empty else []
        )
        if any(header):
            frame.columns = [header[i] or f"col_{i}" for i in range(len(header))]
            frame = frame.iloc[1:].reset_index(drop=True)
        return frame

    def numeric_rows(self, max_rows: int = 40) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for row in self.rows[:max_rows]:
            if not row:
                continue
            numbers = [parse_number(cell) for cell in row[1:]]
            if any(n is not None for n in numbers):
                out.append({"label": clean_text(row[0]), "values": numbers, "raw": row})
        return out


@dataclass
class HtmlExtractionResult:
    url: str
    tables: List[HtmlTable] = field(default_factory=list)
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None


def _table_rows(node) -> List[List[str]]:
    """Rows of a <table> as lists of cleaned strings, handling colspan."""
    rows: List[List[str]] = []
    for tr in node.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        row: List[str] = []
        for cell in cells:
            text = clean_text(cell.get_text(" ", strip=True))
            try:
                colspan = int(cell.get("colspan", 1) or 1)
            except (TypeError, ValueError):
                colspan = 1
            row.append(text)
            for _ in range(max(0, min(colspan, 6) - 1)):
                row.append("")
        if any(c for c in row):
            rows.append(row)
    return rows


def _caption_for(node) -> Optional[str]:
    for parent in node.parents:
        if getattr(parent, "name", None) == "caption":
            return clean_text(parent.get_text(" ", strip=True))
    previous = node.find_previous(["h1", "h2", "h3", "h4", "p", "strong"])
    if previous is not None:
        text = clean_text(previous.get_text(" ", strip=True))
        if 4 <= len(text) <= 200:
            return text
    return None


#: Accent-free on purpose: hint and page context are both folded through
#: ``normalise_label`` before comparison.
_FINANCIAL_HINTS = (
    "revenue", "chiffre", "affaires", "resultat", "income", "ebitda", "ebit",
    "assets", "actif", "liabilities", "dettes", "cash", "tresorerie", "equity",
    "capitaux", "marge", "margin", "profit",
)


def _score_table(table: HtmlTable) -> int:
    if table.n_rows < 2:
        return -1
    numbers = 0
    for row in table.rows[:40]:
        for cell in row[1:]:
            if parse_number(cell) is not None:
                numbers += 1
    if numbers < 3:
        return -1
    blob = normalise_label(table.context or "")
    hits = sum(1 for h in _FINANCIAL_HINTS if normalise_label(h) in blob)
    return numbers + hits * 8 + (15 if hits else 0)


def extract_tables_from_html(
    html: str,
    source_url: str = "",
    min_rows: int = 2,
    context_chars: int = 1200,
) -> List[HtmlTable]:
    """Parse every table in an HTML document, scored and scale-annotated."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []

    soup = BeautifulSoup(html, "html.parser")
    page_text = clean_text(soup.get_text(" ", strip=True))

    tables: List[HtmlTable] = []
    for index, node in enumerate(soup.find_all("table")):
        rows = _table_rows(node)
        if len(rows) < min_rows:
            continue
        context = clean_text(node.get_text(" ", strip=True))[:context_chars]
        scale_guess = guess_scale(context, page_text[:400])
        table = HtmlTable(
            source_url=source_url,
            index=index,
            rows=rows,
            caption=_caption_for(node),
            scale=scale_guess.scale,
            scale_evidence=scale_guess.evidence,
            currency=scale_guess.currency,
            context=context,
        )
        score = _score_table(table)
        if score < 0:
            continue
        table.confidence = "medium" if score >= 25 else "low"
        tables.append(table)

    tables.sort(key=_score_table, reverse=True)
    return tables


def fetch_and_extract(
    url: str,
    http: Optional[HttpClient] = None,
    min_rows: int = 2,
) -> HtmlExtractionResult:
    """Download a permitted public page and extract its tables."""
    http = http or HttpClient()
    response = http.get(url)
    if not response.ok:
        return HtmlExtractionResult(
            url=url,
            error=response.error or f"HTTP {response.status_code}",
        )
    content_type = (response.content_type or "").lower()
    if "html" not in content_type and "xml" not in content_type:
        return HtmlExtractionResult(
            url=url,
            error=(
                f"Le contenu n'est pas du HTML (Content-Type: {content_type or 'inconnu'}). "
                "Utilisez l'extracteur PDF pour un document PDF."
            ),
        )
    result = HtmlExtractionResult(url=url)
    result.tables = extract_tables_from_html(response.text, url, min_rows=min_rows)
    if not result.tables:
        result.warnings.append(
            "Aucun tableau financier exploitable trouvé sur cette page. La page a "
            "peut-être ete générée en JavaScript, ou ne contient pas de tableau de "
            "données."
        )
    return result


def find_financial_table_html(result: HtmlExtractionResult) -> Optional[HtmlTable]:
    """Best-scoring table of a page, or ``None``."""
    return result.tables[0] if result.tables else None
