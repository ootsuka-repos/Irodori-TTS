"""Shared text normalization and identifier sanitization for data preparation.

Two sanitizers are kept deliberately: they encode different historical rules
(``-`` vs ``_`` collapsing) and changing either would silently rename existing
speaker IDs / source IDs and orphan cached artifacts keyed by them.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any


def normalize_transcript(text: str) -> str:
    """NFC-normalize and collapse inline whitespace/newlines to single spaces."""
    value = unicodedata.normalize("NFC", str(text)).strip()
    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    value = re.sub(r"\n+", " ", value)
    return value.strip()


def meaningful_text(text: str) -> str:
    """Keep letters, marks and numbers (drops punctuation/symbols/space/control)."""
    return "".join(
        character for character in text if unicodedata.category(character)[0] in {"L", "M", "N"}
    )


def meaningful_character_count(text: str) -> int:
    return len(meaningful_text(text))


def compact_for_similarity(text: str) -> str:
    """Casefolded NFKC content characters only, for edit-distance style checks."""
    return "".join(
        character.casefold()
        for character in unicodedata.normalize("NFKC", text)
        if not unicodedata.category(character).startswith(("P", "S", "Z", "C"))
    )


def sanitize_manifest_id_component(value: Any, *, fallback: str) -> str:
    """prepare_manifest-style ID sanitizer (collapses ``-`` runs, keeps Unicode)."""
    if value is None:
        raw = ""
    elif isinstance(value, str):
        raw = value
    elif isinstance(value, (list, tuple)):
        raw = " ".join(str(x) for x in value)
    else:
        raw = str(value)
    raw = raw.strip()
    if not raw:
        return fallback
    s = unicodedata.normalize("NFKC", raw)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[:/\\\\]+", "-", s)
    s = re.sub(r"[\x00-\x1f\x7f]", "", s)
    s = re.sub(r"[^\w.\-]+", "-", s, flags=re.UNICODE)
    s = re.sub(r"-{2,}", "-", s)
    s = s.strip("-_.")
    if not s:
        s = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    if len(s) > 96:
        digest = hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
        s = f"{s[:80]}-{digest}"
    return s


def sanitize_source_component(value: str, *, fallback: str, max_length: int) -> str:
    """Pipeline path component sanitizer (collapses ``-_`` runs to ``_``)."""
    normalized = unicodedata.normalize("NFKC", value).strip()
    normalized = re.sub(r"\s+", "_", normalized)
    normalized = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "-", normalized)
    normalized = re.sub(r"[^\w.\-]+", "-", normalized, flags=re.UNICODE)
    normalized = re.sub(r"[-_]{2,}", "_", normalized).strip("-_.")
    if not normalized:
        normalized = fallback
    if len(normalized) > max_length:
        normalized = normalized[:max_length].rstrip("-_.")
    return normalized or fallback
