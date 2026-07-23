"""Weighted priority score for C/D fusion.

Weights are documented rules (not learned). Clarified in /clarify — last schema change.
"""

from __future__ import annotations

from typing import Any

# Soma = 1.00 — ajustar só neste dict.
SCORE_WEIGHTS: dict[str, float] = {
    "relato": 0.25,  # tipo_relato grave (agressao / ameaca)
    "sofrimento": 0.25,  # A3
    "violencia": 0.25,  # V3
    "postura": 0.15,  # V2 (proxy mais fraco)
    "corroboracao": 0.10,  # match espaçotemporal
}

GRAVE_TIPOS = frozenset({"agressao", "ameaca"})


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def relato_signal(tipo_relato: str) -> float:
    """Normalize tipo_relato to [0, 1]: grave types → 1.0, else → 0.0."""
    return 1.0 if tipo_relato in GRAVE_TIPOS else 0.0


def compute_score(
    *,
    tipo_relato: str,
    sofrimento: float,
    violencia: float,
    postura_defensiva: float,
    corroborado: bool,
    weights: dict[str, float] | None = None,
) -> float:
    """Weighted sum of normalized signals in [0, 1]."""
    w = weights or SCORE_WEIGHTS
    signals = {
        "relato": relato_signal(tipo_relato),
        "sofrimento": _clip01(sofrimento),
        "violencia": _clip01(violencia),
        "postura": _clip01(postura_defensiva),
        "corroboracao": 1.0 if corroborado else 0.0,
    }
    total = sum(w[k] * signals[k] for k in w)
    return _clip01(total)


def score_from_modules(
    a12: dict[str, Any],
    a3: dict[str, Any],
    v2: dict[str, Any],
    v3: dict[str, Any],
    *,
    corroborado: bool,
    weights: dict[str, float] | None = None,
) -> float:
    return compute_score(
        tipo_relato=str(a12["tipo_relato"]),
        sofrimento=float(a3["sofrimento"]),
        violencia=float(v3["violencia"]),
        postura_defensiva=float(v2["postura_defensiva"]),
        corroborado=corroborado,
        weights=weights,
    )
