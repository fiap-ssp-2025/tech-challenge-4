"""Scoring weights and report disclaimer."""

from src.fusion.report import DISCLAIMER, build_report
from src.fusion.scoring import SCORE_WEIGHTS, compute_score


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
