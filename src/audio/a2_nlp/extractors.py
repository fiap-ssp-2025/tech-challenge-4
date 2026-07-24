"""Rule-based classification and extraction helpers for A2 NLP."""

from __future__ import annotations

import re
import unicodedata

from src.contracts import TipoRelato

from .rules import (
    EMOTIONAL_DISTRESS_TERMS,
    LOCAL_PATTERNS,
    NEGATION_TERMS,
    TIME_PATTERNS,
    VIOLENCE_TERMS,
)


def normalize_text(text: str) -> str:
    """Normalize text for rule matching while preserving readable spacing."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _is_negated(text: str, start_index: int, window: int = 35) -> bool:
    """
    Check whether a matched expression is preceded by a nearby negation.

    Example:
        "ele nunca me bateu" -> violence match is considered negated.
    """
    prefix = text[max(0, start_index - window) : start_index]
    words = re.findall(r"\b[\wÀ-ÿ]+\b", prefix)
    recent_words = words[-5:]

    return any(term in recent_words for term in NEGATION_TERMS)


def _contains_non_negated_term(text: str, terms: tuple[str, ...]) -> bool:
    for term in terms:
        for match in re.finditer(re.escape(term), text):
            if not _is_negated(text, match.start()):
                return True
    return False


def classify_report(text: str) -> TipoRelato:
    """Classify the report using deterministic rules."""
    normalized = normalize_text(text)

    if not normalized:
        return "outro"

    if _contains_non_negated_term(normalized, VIOLENCE_TERMS):
        return "violencia_domestica"

    if _contains_non_negated_term(normalized, EMOTIONAL_DISTRESS_TERMS):
        return "sofrimento_emocional"

    return "outro"


def _extract_first_pattern(text: str, patterns: tuple[str, ...]) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return ""


def extract_location(text: str) -> str:
    """Extract the first known location expression."""
    return _extract_first_pattern(text, LOCAL_PATTERNS)


def extract_time(text: str) -> str:
    """Extract the first known time expression."""
    return _extract_first_pattern(text, TIME_PATTERNS)