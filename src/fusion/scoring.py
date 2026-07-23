"""Weighted triage score for C/D fusion (feature 002 — consulta).

Weights are documented rules (not learned). Sum = 1.00 — adjust only here.
"""

from __future__ import annotations

from typing import Any

SCORE_WEIGHTS: dict[str, float] = {
    "relato": 0.25,  # indicador extraído da fala (A2)
    "sofrimento": 0.25,  # voz (A3)
    "desconforto_facial": 0.20,  # face (V3)
    "postura": 0.20,  # corpo (V2)
    "corroboracao": 0.10,  # mesma sessão dentro da janela
}

# Sinal normalizado do relato (documentado na spec 002):
RELATO_SIGNAL: dict[str, float] = {
    "violencia_domestica": 1.0,
    "sofrimento_emocional": 0.6,
    "outro": 0.0,
}


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def relato_signal(tipo_relato: str) -> float:
    """Normalize tipo_relato to [0, 1] per RELATO_SIGNAL (unknown → 0.0)."""
    return RELATO_SIGNAL.get(tipo_relato, 0.0)


def compute_score(
    *,
    tipo_relato: str,
    sofrimento: float,
    desconforto_facial: float,
    postura_defensiva: float,
    corroborado: bool,
    weights: dict[str, float] | None = None,
) -> float:
    """Weighted sum of normalized signals in [0, 1]."""
    w = weights or SCORE_WEIGHTS
    signals = {
        "relato": relato_signal(tipo_relato),
        "sofrimento": _clip01(sofrimento),
        "desconforto_facial": _clip01(desconforto_facial),
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
        desconforto_facial=float(v3["desconforto_facial"]),
        postura_defensiva=float(v2["postura_defensiva"]),
        corroborado=corroborado,
        weights=weights,
    )
