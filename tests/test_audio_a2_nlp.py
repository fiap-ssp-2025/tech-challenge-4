"""Tests for the real rule-based A2 NLP implementation."""

from src.audio.a2_nlp.infer import infer
from src.audio.a2_nlp.extractors import (
    classify_report,
    extract_location,
    extract_time,
)


def test_classifies_domestic_violence() -> None:
    result = infer(
        "dummy.wav",
        transcricao="Meu marido me bateu ontem em casa.",
    )

    assert result["tipo_relato"] == "violencia_domestica"
    assert result["tempo"] == "ontem"
    assert result["local"] == "em casa"
    assert "stub" not in result


def test_classifies_emotional_distress() -> None:
    result = infer(
        "dummy.wav",
        transcricao="Estou muito triste e ansiosa desde a semana passada.",
    )

    assert result["tipo_relato"] == "sofrimento_emocional"
    assert result["tempo"] == "semana passada"


def test_classifies_other() -> None:
    result = infer(
        "dummy.wav",
        transcricao="Vim apenas renovar a minha receita.",
    )

    assert result["tipo_relato"] == "outro"
    assert result["local"] == ""
    assert result["tempo"] == ""


def test_ignores_negated_violence_statement() -> None:
    assert classify_report("Ele nunca me bateu.") == "outro"


def test_extracts_consultation_location() -> None:
    assert (
        extract_location("Estou no Consultório 3 aguardando atendimento.")
        == "Consultório 3"
    )


def test_extracts_time_expression() -> None:
    assert extract_time("Isso aconteceu há 3 dias.") == "há 3 dias"


def test_empty_transcription_returns_valid_contract() -> None:
    result = infer("dummy.wav", transcricao="")

    assert result == {
        "transcricao": "",
        "tipo_relato": "outro",
        "local": "",
        "tempo": "",
    }