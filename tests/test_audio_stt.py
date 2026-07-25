"""Testes unitários do módulo de Speech-to-Text."""

from pathlib import Path
from unittest.mock import patch

import pytest

from audio.azure_stt import AzureSTTError
from audio.stt import STTError, transcrever_audio
from audio.whisper_stt import WhisperSTTError


def test_usa_azure_quando_funciona(tmp_path: Path):
    """Deve utilizar o Azure quando disponível."""

    audio = tmp_path / "teste.wav"
    audio.touch()

    with patch(
        "audio.stt.transcrever_azure",
        return_value="Transcrição Azure",
    ):
        resultado = transcrever_audio(audio)

    assert resultado["provedor"] == "azure"
    assert resultado["transcricao"] == "Transcrição Azure"


def test_usa_whisper_quando_azure_falha(tmp_path: Path):
    """Deve utilizar o Whisper como fallback."""

    audio = tmp_path / "teste.wav"
    audio.touch()

    with (
        patch(
            "audio.stt.transcrever_azure",
            side_effect=AzureSTTError("Falha Azure"),
        ),
        patch(
            "audio.stt.transcrever_whisper",
            return_value="Transcrição Whisper",
        ),
    ):
        resultado = transcrever_audio(audio)

    assert resultado["provedor"] == "faster-whisper"
    assert resultado["transcricao"] == "Transcrição Whisper"


def test_lanca_erro_se_os_dois_falharem(tmp_path: Path):
    """Deve lançar STTError quando nenhum provedor funcionar."""

    audio = tmp_path / "teste.wav"
    audio.touch()

    with (
        patch(
            "audio.stt.transcrever_azure",
            side_effect=AzureSTTError("Azure indisponível"),
        ),
        patch(
            "audio.stt.transcrever_whisper",
            side_effect=WhisperSTTError("Whisper indisponível"),
        ),
    ):
        with pytest.raises(STTError):
            transcrever_audio(audio)


def test_arquivo_inexistente():
    """Deve lançar FileNotFoundError para arquivo inexistente."""

    with pytest.raises(FileNotFoundError):
        transcrever_audio("arquivo_inexistente.wav")