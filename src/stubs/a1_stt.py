"""A1 STT stub — does not call Azure; requires credentials to acknowledge cloud path."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from src.contracts import A12Result, validate_a12

load_dotenv()

_FALLBACK_MSG = (
    "AZURE_SPEECH_KEY ausente. Configure .env (veja .env.example) para o STT Azure. "
    "Fallback offline faster-whisper ainda não está disponível (TODO P3)."
)


def infer(path: str | Path) -> A12Result:
    """Return a fixed A1/A2-shaped payload (transcription only; A2 stub fills structure).

    Raises RuntimeError if Azure credentials are missing — never calls the cloud API
    from this stub. Use A2 stub for structured fields in the pipeline.
    """
    _ = Path(path)
    if not os.getenv("AZURE_SPEECH_KEY", "").strip():
        raise RuntimeError(_FALLBACK_MSG)

    # Plausible fixed transcription; A2 stub owns tipo_relato/local/tempo.
    return validate_a12(
        {
            "transcricao": (
                "Tenho medo de voltar para casa, ele fica muito nervoso comigo."
            ),
            "tipo_relato": "violencia_domestica",
            "local": "Consultório 3 — Ambulatório de Ginecologia",
            "tempo": "durante a consulta",
            "stub": True,
        }
    )
