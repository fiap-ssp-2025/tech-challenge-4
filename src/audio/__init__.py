"""Serviços de processamento e transcrição de áudio."""

from .stt import STTError, STTResult, infer, transcrever_audio

__all__ = [
    "STTError",
    "STTResult",
    "infer",
    "transcrever_audio",
]