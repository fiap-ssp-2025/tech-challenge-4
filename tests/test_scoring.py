"""Scoring weights, calibração de escala e report disclaimer."""

import json
from pathlib import Path

import pytest

from src.fusion.report import DISCLAIMER, build_report
from src.fusion.scoring import (
    DECISION_THRESHOLDS,
    SCORE_WEIGHTS,
    calibrate,
    compute_score,
)

ROOT = Path(__file__).resolve().parents[1]


def test_weights_sum_to_one():
    assert abs(sum(SCORE_WEIGHTS.values()) - 1.0) < 1e-9
    assert SCORE_WEIGHTS == {
        "relato": 0.28,
        "sofrimento": 0.28,
        "desconforto_facial": 0.22,
        "postura": 0.22,
    }


def test_corroboracao_nao_altera_o_escore():
    """Na 002 `corroborado` é sempre verdadeiro; se pesasse, seria constante somada.

    Guarda a decisão de 26/07: o flag segue no contrato, mas fora do escore. Se
    alguém reintroduzir o peso, este teste quebra.
    """
    args = dict(
        tipo_relato="sofrimento_emocional",
        sofrimento=0.4,
        desconforto_facial=0.3,
        postura_defensiva=0.2,
    )
    assert compute_score(**args, corroborado=True) == compute_score(**args, corroborado=False)
    assert "corroboracao" not in SCORE_WEIGHTS


def test_score_and_report():
    escore = compute_score(
        tipo_relato="violencia_domestica",
        sofrimento=1.0,
        desconforto_facial=1.0,
        postura_defensiva=1.0,
        corroborado=True,
    )
    assert abs(escore - 1.0) < 1e-9
    cd = build_report(
        a12={
            "transcricao": "x",
            "tipo_relato": "violencia_domestica",
            "local": "DF",
            "tempo": "agora",
        },
        a3={"sofrimento": 0.8},
        v1={"n_pessoas": 2, "tracks": []},
        v2={"postura_defensiva": 0.7},
        v3={"desconforto_facial": 0.6},
        corroborado=True,
    )
    assert DISCLAIMER in cd["nota_ocorrencia"]
    assert 0.0 <= cd["escore"] <= 1.0


# --- calibração de escala (spec 003) -----------------------------------------


def test_calibrate_ancora_o_limiar_em_meio():
    """A propriedade que dá sentido a tudo: o limiar do modelo vira 0,5."""
    for limiar in (0.17, 0.5, 0.7, 0.9):
        assert calibrate(limiar, limiar) == pytest.approx(0.5)


def test_calibrate_preserva_extremos_e_ordem():
    assert calibrate(0.0, 0.17) == pytest.approx(0.0)
    assert calibrate(1.0, 0.17) == pytest.approx(1.0)
    # monotônica: nunca inverte a ordem de dois escores
    valores = [0.0, 0.05, 0.119, 0.17, 0.3, 0.7, 0.99, 1.0]
    calibrados = [calibrate(v, 0.17) for v in valores]
    assert calibrados == sorted(calibrados)


def test_calibrate_com_limiar_meio_e_identidade():
    """O V2 não tem limiar medido; com 0,5 a calibração não pode mexer no valor."""
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert calibrate(v, 0.5) == pytest.approx(v)


def test_calibrate_corrige_nas_duas_direcoes():
    # A3 decide em 0,17: escore baixo mas acima de 90% dos neutros sobe
    assert calibrate(0.119, 0.17) == pytest.approx(0.35, abs=0.01)
    # V3 decide em 0,70: escore que parecia "acima da média" desce para baixo da linha
    assert calibrate(0.60, 0.70) == pytest.approx(0.43, abs=0.01)


def test_limiares_batem_com_as_metricas():
    """Guarda contra as constantes e os arquivos versionados divergirem.

    Se alguém recalibrar um modelo e esquecer de atualizar DECISION_THRESHOLDS, a
    fusão passa a somar em escala errada — silenciosamente. Este teste quebra antes.
    """
    a3 = json.loads((ROOT / "models" / "a3_threshold_metrics.json").read_text())
    assert DECISION_THRESHOLDS["sofrimento"] == pytest.approx(a3["threshold_val_selected"])

    v3 = json.loads((ROOT / "models" / "v3_clip_metrics.json").read_text())
    assert DECISION_THRESHOLDS["desconforto_facial"] == pytest.approx(v3["clip"]["threshold_val"])


def test_relato_nao_e_calibrado():
    """`relato` é mapa discreto documentado, não saída de classificador."""
    assert "relato" not in DECISION_THRESHOLDS
    so_relato = compute_score(
        tipo_relato="violencia_domestica",
        sofrimento=0.0, desconforto_facial=0.0, postura_defensiva=0.0,
    )
    assert so_relato == pytest.approx(SCORE_WEIGHTS["relato"])


def test_calibrado_false_reproduz_o_comportamento_anterior():
    args = dict(tipo_relato="outro", sofrimento=0.119,
                desconforto_facial=0.0, postura_defensiva=0.10)
    antigo = compute_score(**args, calibrado=False)
    novo = compute_score(**args, calibrado=True)
    esperado_antigo = SCORE_WEIGHTS["sofrimento"] * 0.119 + SCORE_WEIGHTS["postura"] * 0.10
    assert antigo == pytest.approx(esperado_antigo)
    assert novo > antigo  # o caso real deixa de ser subestimado
