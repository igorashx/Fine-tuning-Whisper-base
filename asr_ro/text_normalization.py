from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
_DASH_TRANSLATION = str.maketrans(
    {
        "„": '"',
        "”": '"',
        "“": '"',
        "’": "'",
        "‘": "'",
        "‚": ",",
        "‐": "-",
        "‑": "-",
        "–": "-",
        "—": "-",
        "−": "-",
        "…": "...",
    }
)


def normalize_transcript(text: str, lowercase: bool = False) -> str:
    """Normalizează conservator transcrierile, păstrând diacriticele."""
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.translate(_DASH_TRANSLATION)
    normalized = normalized.replace("\u00a0", " ").replace("\u200b", "")
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    if lowercase:
        normalized = normalized.lower()
    return normalized


def strip_punctuation(text: str) -> str:
    cleaned = []
    for character in text:
        category = unicodedata.category(character)
        if category.startswith("P"):
            cleaned.append(" ")
        else:
            cleaned.append(character)
    return _WHITESPACE_RE.sub(" ", "".join(cleaned)).strip()


def normalize_for_metric(text: str) -> str:
    normalized = normalize_transcript(text, lowercase=True)
    normalized = strip_punctuation(normalized)
    return _WHITESPACE_RE.sub(" ", normalized).strip()
