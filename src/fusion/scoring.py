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


# Fronteira de decisão de cada modelo — MEDIDA na validação, não escolhida a dedo.
# Cada uma veio do mesmo procedimento: varrer todos os cortes e ficar com o de melhor
# F1 no split de validação. Sem calibrar, a fusão soma números com significados
# diferentes (ver o cabeçalho do módulo).
#   sofrimento         → models/a3_threshold_metrics.json  (threshold_val_selected)
#   desconforto_facial → models/v3_clip_metrics.json       (clip.threshold_val)
#   postura            → o V2 não teve limiar calibrado; 0,50 é o padrão do argmax,
#                        e a calibração vira identidade — nada muda até haver medida.
# `tests/test_scoring.py::test_limiares_batem_com_as_metricas` guarda a sincronia
# entre estas constantes e os arquivos versionados.
DECISION_THRESHOLDS: dict[str, float] = {
    "sofrimento": 0.17,
    "desconforto_facial": 0.70,
    "postura": 0.50,
}


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def calibrate(raw: float, threshold: float) -> float:
    """Recoloca o escore numa escala em que 0,5 É a fronteira de decisão do modelo.

    Regra de três em dois trechos: o que está abaixo do limiar é comprimido em
    [0, 0.5]; o que está acima é esticado em [0.5, 1]. Assim 0,5 passa a significar
    a mesma coisa — "no limite do que este modelo chama de positivo" — para todos os
    sinais, que é o mínimo para somá-los com peso.

    Exemplo real: o A3 decide em 0,17. Um escore de 0,119 (acima de 90% dos áudios
    neutros do teste) valia 0,119 na soma, como se fosse quase nada; calibrado vale
    0,350. O V3 decide em 0,70: um 0,60 dele, que a soma lia como acima da média,
    vira 0,43 — abaixo da linha, como o próprio modelo classificaria.

    Escolhemos esta calibração explicável em vez de Platt scaling ou regressão
    isotônica (estatisticamente mais rigorosas) porque usa um número já medido e
    versionado, e qualquer pessoa refaz a conta no papel. Decisão registrada na
    spec 003.
    """
    raw = _clip01(raw)
    if not 0.0 < threshold < 1.0:
        return raw
    if raw <= threshold:
        return 0.5 * raw / threshold
    return 0.5 + 0.5 * (raw - threshold) / (1.0 - threshold)


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
    calibrado: bool = True,
) -> float:
    """Weighted sum of calibrated signals in [0, 1].

    Cada saída de modelo passa por `calibrate()` antes de entrar na soma, para que
    0,5 signifique a mesma coisa em todos os sinais. `relato` NÃO é calibrado: é um
    mapa discreto e documentado (RELATO_SIGNAL), não a saída de um classificador.

    `corroborado` é aceito para compatibilidade e continua no contrato C/D, mas não
    entra no escore — ver o cabeçalho do módulo.
    `calibrado=False` reproduz o comportamento anterior; existe para comparação.
    """
    w = weights or SCORE_WEIGHTS

    def _sig(name: str, value: float) -> float:
        value = _clip01(value)
        if not calibrado:
            return value
        return calibrate(value, DECISION_THRESHOLDS.get(name, 0.5))

    signals = {
        "relato": relato_signal(tipo_relato),
        "sofrimento": _sig("sofrimento", sofrimento),
        "desconforto_facial": _sig("desconforto_facial", desconforto_facial),
        "postura": _sig("postura", postura_defensiva),
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
