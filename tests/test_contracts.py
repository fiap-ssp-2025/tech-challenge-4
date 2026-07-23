"""Contract validation tests."""

import pytest

from src.contracts import (
    ContractError,
    validate_a12,
    validate_a3,
    validate_cd,
    validate_v1,
    validate_v2,
    validate_v3,
)


def test_a12_ok():
    out = validate_a12(
        {
            "transcricao": "ajuda",
            "tipo_relato": "ameaca",
            "local": "Asa Norte",
            "tempo": "agora",
        }
    )
    assert out["tipo_relato"] == "ameaca"


def test_a12_rejects_unknown_tipo():
    with pytest.raises(ContractError):
        validate_a12(
            {
                "transcricao": "x",
                "tipo_relato": "roubo",
                "local": "y",
                "tempo": "z",
            }
        )


def test_a3_bounds():
    assert validate_a3({"sofrimento": 0.5})["sofrimento"] == 0.5
    with pytest.raises(ContractError):
        validate_a3({"sofrimento": 1.5})


def test_v1_tracks_shape():
    out = validate_v1(
        {
            "n_pessoas": 1,
            "tracks": [{"id": 1, "n_frames": 10, "bbox_media": [1, 2, 3, 4]}],
        }
    )
    assert out["tracks"][0]["bbox_media"] == [1, 2, 3, 4]
    with pytest.raises(ContractError):
        validate_v1(
            {
                "n_pessoas": 1,
                "tracks": [{"id": 1, "n_frames": 10, "bbox_media": [1, 2, 3]}],
            }
        )


def test_v2_v3_cd():
    validate_v2({"postura_defensiva": 0.0})
    validate_v3({"violencia": 1.0})
    validate_cd(
        {
            "escore": 0.4,
            "corroborado": True,
            "nota_ocorrencia": "texto",
        }
    )
    with pytest.raises(ContractError):
        validate_cd({"escore": 0.4, "corroborado": "sim", "nota_ocorrencia": "x"})
