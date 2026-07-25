"""Orquestrador de Speech-to-Text.

Tenta utilizar o Azure Speech primeiro e, em caso de falha,
utiliza o faster-whisper como fallback local.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict

from .azure_stt import (
    AzureSTTError,
    transcrever_audio as transcrever_azure,
)
from .whisper_stt import (
    WhisperSTTError,
    transcrever_audio as transcrever_whisper,
)


class STTResult(TypedDict):
    """Resultado da transcrição de áudio."""

    transcricao: str
    provedor: Literal["azure", "faster-whisper"]


class STTError(RuntimeError):
    """Erro quando nenhum provedor consegue transcrever o áudio."""


def transcrever_audio(path: str | Path) -> STTResult:
    """Transcreve um arquivo usando Azure e Whisper como fallback."""

    audio_path = Path(path)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Áudio não encontrado: {audio_path}")

    try:
        texto = transcrever_azure(audio_path)

        return {
            "transcricao": texto,
            "provedor": "azure",
        }

    except AzureSTTError as azure_error:
        print(f"[STT] Azure indisponível: {azure_error}")
        print("[STT] Tentando faster-whisper...")

        try:
            texto = transcrever_whisper(audio_path)

            return {
                "transcricao": texto,
                "provedor": "faster-whisper",
            }

        except WhisperSTTError as whisper_error:
            raise STTError(
                "Nenhum provedor conseguiu transcrever o áudio. "
                f"Azure: {azure_error}. "
                f"faster-whisper: {whisper_error}"
            ) from whisper_error


def infer(audio_path: str | Path) -> dict[str, str]:
    """
    Interface simplificada para uso pelo pipeline.

    Retorna apenas a transcrição, no formato esperado pelas
    etapas seguintes da aplicação.
    """

    resultado = transcrever_audio(audio_path)

    return {
        "transcricao": resultado["transcricao"],
        "provedor": resultado["provedor"],
    }