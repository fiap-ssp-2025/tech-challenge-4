"""Fallback offline de transcrição usando faster-whisper."""

from __future__ import annotations

from pathlib import Path

from faster_whisper import WhisperModel


class WhisperSTTError(RuntimeError):
    """Erro ao transcrever áudio com faster-whisper."""


_MODEL: WhisperModel | None = None


def _get_model(model_size: str = "small") -> WhisperModel:
    """Carrega o modelo apenas uma vez durante a execução."""
    global _MODEL

    if _MODEL is None:
        _MODEL = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
        )

    return _MODEL


def transcrever_audio(
    path: str | Path,
    *,
    idioma: str = "pt",
    modelo: str = "small",
) -> str:
    """Transcreve um arquivo de áudio localmente com faster-whisper."""
    audio_path = Path(path)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Áudio não encontrado: {audio_path}")

    try:
        whisper = _get_model(modelo)

        segments, _info = whisper.transcribe(
            str(audio_path),
            language=idioma,
            vad_filter=True,
        )

        partes = [segment.text.strip() for segment in segments if segment.text.strip()]
        texto = " ".join(partes).strip()

    except Exception as exc:
        raise WhisperSTTError(
            f"Falha ao transcrever com faster-whisper: {exc}"
        ) from exc

    if not texto:
        raise WhisperSTTError("O faster-whisper retornou uma transcrição vazia.")

    return texto