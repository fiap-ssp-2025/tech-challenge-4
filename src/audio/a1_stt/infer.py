"""A1 — Speech-to-Text para áudio de consultas."""

from __future__ import annotations

from pathlib import Path

from src.audio.stt import infer as infer_stt


def infer(audio_path: str | Path) -> dict[str, str]:
    """
    Transcreve o áudio da consulta.

    Usa Azure Speech como provedor principal e
    faster-whisper como fallback offline.
    """
    return infer_stt(audio_path)


__all__ = ["infer"]