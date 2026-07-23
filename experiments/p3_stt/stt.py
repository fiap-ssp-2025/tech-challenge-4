"""Orquestrador de STT: Azure com fallback para faster-whisper."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict

from azure_stt import (
    AzureSTTError,
    transcrever_audio as transcrever_azure,
)

from whisper_stt import (
    WhisperSTTError,
    transcrever_audio as transcrever_whisper,
)

class STTResult(TypedDict):
    transcricao: str
    provedor: Literal["azure", "faster-whisper"]


def transcrever_audio(path: str | Path) -> STTResult:
    """Transcreve o áudio usando Azure e Whisper como fallback."""

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
            raise RuntimeError(
                "Não foi possível transcrever o áudio. "
                f"Erro Azure: {azure_error}. "
                f"Erro faster-whisper: {whisper_error}"
            ) from whisper_error


if __name__ == "__main__":
    caminho = input("Arquivo de áudio: ").strip()

    try:
        resultado = transcrever_audio(caminho)

        print("\n===== RESULTADO STT =====\n")
        print(f"Provedor: {resultado['provedor']}")
        print(f"Transcrição: {resultado['transcricao']}")

    except Exception as exc:
        print(f"\nErro: {exc}")