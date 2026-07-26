"""Weighted triage score for C/D fusion (feature 002 — consulta).

Weights are documented rules (not learned). Sum = 1.00 — adjust only here.

Por que a corroboração NÃO pesa no escore (mudança de 26/07/2026): na 002 áudio e
vídeo vêm sempre da mesma consulta, então `corroborado` é verdadeiro por construção.
Como termo de escore ele virava uma constante somada a todo caso — no teste da
simulação respondia por 72% de um escore de 0,138, sem discriminar nada. O flag
continua no contrato C/D (RF-07) e na nota, como informação de proveniência; o que
saiu foi o peso. Os 0,10 foram redistribuídos proporcionalmente entre os quatro
sinais medidos, preservando a razão 0,25/0,25/0,20/0,20 herdada da 001.
"""

from __future__ import annotations

from typing import Any

SCORE_WEIGHTS: dict[str, float] = {
    "relato": 0.28,  # indicador extraído da fala (A2)
    "sofrimento": 0.28,  # voz (A3)
    "desconforto_facial": 0.22,  # face (V3)
    "postura": 0.22,  # corpo (V2)
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
    corroborado: bool = True,
    weights: dict[str, float] | None = None,
) -> float:
    """Weighted sum of normalized signals in [0, 1].

    `corroborado` é aceito para compatibilidade e continua no contrato C/D, mas não
    entra no escore — ver o cabeçalho do módulo.
    """
    w = weights or SCORE_WEIGHTS
    signals = {
        "relato": relato_signal(tipo_relato),
        "sofrimento": _clip01(sofrimento),
        "desconforto_facial": _clip01(desconforto_facial),
        "postura": _clip01(postura_defensiva),
        "corroboracao": 1.0 if corroborado else 0.0,
    }
    total = sum(weight * signals[k] for k, weight in w.items())
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
