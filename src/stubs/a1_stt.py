"""Stub A1 STT — não chama Azure; exige credenciais para reconhecer o caminho em nuvem."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from src.contracts import A12Result, validate_a12

load_dotenv()

_FALLBACK_MSG = (
    "AZURE_SPEECH_KEY ausente. Configure .env (veja .env.example) para o STT Azure. "
    "Fallback offline faster-whisper ainda não está disponível."
)


def infer(path: str | Path) -> A12Result:
    """Retorna um payload fixo no formato A1/A2 (só transcrição; o stub A2 preenche a estrutura).

    Levanta RuntimeError se as credenciais Azure estiverem ausentes — nunca chama a API
    na nuvem a partir deste stub. Use o stub A2 para os campos estruturados no pipeline.
    """
    _ = Path(path)
    if not os.getenv("AZURE_SPEECH_KEY", "").strip():
        raise RuntimeError(_FALLBACK_MSG)

    # Transcrição fixa plausível; o stub A2 define tipo_relato/local/tempo.
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
