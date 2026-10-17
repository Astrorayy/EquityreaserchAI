"""Number and unit parsing helpers shared by the PDF and HTML extractors.

Financial documents use dozens of number conventions.  Getting them wrong is
how a "1 234" (thousands separator) becomes one million.  Every parser here is
therefore explicit about what it accepts, and returns ``None`` rather than a
guess when it is unsure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

#: Placeholders that mean "no value" in a financial table.
NULL_TOKENS = {
    "", "-", "--", "---", "n/a", "na", "n.d.", "nd", "nil", "none", "nan",
    "\u2013", "\u2014", "s.o.", "so", "non disponible", "\u2014\u2014",
}

#: Scale words, with the factor that converts the document figure to units.
SCALE_WORDS: dict[str, float] = {
    "trillion": 1e12, "tn": 1e12, "t": 1e12,
    "billion": 1e9, "bn": 1e9, "b": 1e9, "md": 1e9, "milliard": 1e9, "milliards": 1e9,
    "million": 1e6, "mn": 1e6, "m": 1e6, "mm": 1e6, "millions": 1e6, "million(s)": 1e6,
    "thousand": 1e3, "thousands": 1e3, "k": 1e3, "milliers": 1e3, "millier": 1e3,
    "unit": 1.0, "units": 1.0,
}

#: French unit words that are not English
FRENCH_SCALE_RE = re.compile(
    r"\b(en\s+)?(milliers?|millions?|milliards?|MDH|MMDH|K MAD|M MAD)\b", re.IGNORECASE
)
EN_SCALE_RE = re.compile(
    r"\b(in\s+)?(thousands|millions|billions|trillions)\b", re.IGNORECASE
)
CURRENCY_RE = re.compile(
    r"\b(MAD|DH|dirhams?|USD|US\$|\$|EUR|€|GBP|£|CHF|JPY|ZAR)\b", re.IGNORECASE
)


def is_null_token(text: object) -> bool:
    if text is None:
        return True
    if isinstance(text, float) and np.isnan(text):
        return True
    cleaned = str(text).strip().lower()
    return cleaned in NULL_TOKENS


def parse_number(raw: object, allow_parentheses_negative: bool = True) -> Optional[float]:
    """Parse a single financial figure.

    Handles, and documents, the conventions actually found in filings:

    ==========================  ============
    input                       result
    ==========================  ============
    ``"1 234.5"``               1234.5
    ``"1,234.5"``               1234.5
    ``"1 234,5"``               1234.5   (French decimal comma)
    ``"1.234,5"``               1234.5   (European grouping)
    ``"(1 234)"``               -1234.0  (accounting negative)
    ``"1 234-"``                -1234.0  (French trailing minus)
    ``"-1 234"``                -1234.0
    ``"12.5%"``                 0.125    (percentage converted to a fraction)
    ``"n/a"`` / ``"-"``         None
    ==========================  ============

    Returns ``None`` instead of guessing when the token is empty, a placeholder,
    or contains letters that are not a recognised unit.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        value = float(raw)
        return None if np.isnan(value) else value
    if is_null_token(raw):
        return None

    text = str(raw).strip()
    text = text.replace("\u00a0", " ").replace("\u202f", " ")

    negative = False
    if allow_parentheses_negative and text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1].strip()

    is_percent = "%" in text
    text = text.replace("%", "")

    # currency symbols and codes
    text = re.sub(r"(USD|EUR|GBP|CHF|JPY|MAD|DH|ZAR|CAD|AUD)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[$\u20ac\u00a3\u00a5]", "", text)
    text = text.strip()

    if text.endswith("-") and not text.startswith("-"):
        negative = True
        text = text[:-1].strip()
    if text.startswith("-"):
        negative = not negative
        text = text[1:].strip()
    if text.startswith("+"):
        text = text[1:].strip()

    # Decide the decimal separator.
    has_comma = "," in text
    has_dot = "." in text
    if has_comma and has_dot:
        # the LAST separator is the decimal one
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif has_comma:
        parts = text.split(",")
        if len(parts) == 2 and len(parts[1]) in (1, 2):
            text = text.replace(",", ".")          # 1234,5 -> decimal comma
        else:
            text = text.replace(",", "")           # 1,234 -> grouping
    elif has_dot:
        parts = text.split(".")
        # "1.234" with 3 decimals is almost certainly a thousands separator
        if len(parts) == 2 and len(parts[1]) == 3 and len(parts[0]) <= 3 and int(parts[1]) != 0:
            text = text.replace(".", "")

    text = text.replace(" ", "")
    if not text or not re.fullmatch(r"\d*\.?\d*", text):
        return None
    try:
        value = float(text)
    except ValueError:
        return None

    if is_percent:
        value = value / 100.0
    return -value if negative else value


@dataclass(frozen=True)
class ScaleGuess:
    scale: Optional[float]
    currency: Optional[str]
    evidence: str

    @property
    def confident(self) -> bool:
        return self.scale is not None


_SCALE_LOOKUP = [
    (re.compile(r"\b(milliards?|billions?|bn|MDH)\b", re.IGNORECASE), 1e9, "milliards/billions"),
    (re.compile(r"\b(millions?|mn|MMDH)\b", re.IGNORECASE), 1e6, "millions"),
    (re.compile(r"\b(milliers?|thousands?)\b", re.IGNORECASE), 1e3, "milliers/thousands"),
]


def guess_scale(*texts: Optional[str], default: Optional[float] = None) -> ScaleGuess:
    """Detect the reporting scale from surrounding text.

    Financial statements announce their units in a header line such as
    "En milliers de dirhams" or "(in millions, except per share data)".  This
    function looks for the *last* such announcement before the table.
    """
    haystack = " ".join(t for t in texts if t)
    if not haystack:
        return ScaleGuess(None, None, "aucun texte d'entete disponible")

    currency = None
    m = CURRENCY_RE.search(haystack)
    if m:
        token = m.group(1).upper()
        currency = {"DH": "MAD", "DIRHAM": "MAD", "DIRHAMS": "MAD", "$": "USD", "US$": "USD",
                    "\u20ac": "EUR", "\u00a3": "GBP"}.get(token, token)

    # Prefer the occurrence closest to the end of the text (nearest the table).
    best: Optional[tuple[int, float, str]] = None
    for pattern, factor, label in _SCALE_LOOKUP:
        for match in pattern.finditer(haystack):
            if best is None or match.start() > best[0]:
                best = (match.start(), factor, label)
    if best is None:
        return ScaleGuess(None, currency, "aucune mention d'unite trouvee")
    return ScaleGuess(best[1], currency, f"mention '{best[2]}' detectee dans l'entete")


def to_millions(value: float, scale: float) -> float:
    """Convert a figure expressed in ``scale`` units into millions."""
    return value * scale / 1e6


def clean_text(text: Optional[str]) -> str:
    """Collapse whitespace and normalise the common ligatures/oddities."""
    if not text:
        return ""
    out = str(text)
    for src, dst in (("\u00a0", " "), ("\u202f", " "), ("\u2019", "'"), ("\u2018", "'"),
                     ("\u201c", '"'), ("\u201d", '"'), ("\ufffd", " ")):
        out = out.replace(src, dst)
    return re.sub(r"\s+", " ", out).strip()


def normalise_label(text: Optional[str]) -> str:
    """Lower-case, accent-free label used for fuzzy line-item matching."""
    import unicodedata

    base = unicodedata.normalize("NFKD", clean_text(text))
    base = "".join(ch for ch in base if not unicodedata.combining(ch))
    base = base.lower()
    base = re.sub(r"[^a-z0-9 ]+", " ", base)
    return re.sub(r"\s+", " ", base).strip()
