"""Stub A2 NLP — campos estruturados a partir do contexto da transcrição (stub)."""

from __future__ import annotations

from pathlib import Path

from src.contracts import A12Result, validate_a12


def infer(path: str | Path, transcricao: str | None = None) -> A12Result:
    """Retorna contrato A1/A2 estruturado fixo. O parâmetro opcional transcricao sobrescreve o texto."""
    _ = Path(path)
    text = transcricao or (
        "Tenho medo de voltar para casa, ele fica muito nervoso comigo."
    )
    return validate_a12(
        {
            "transcricao": text,
            "tipo_relato": "violencia_domestica",
            "local": "Consultório 3 — Ambulatório de Ginecologia",
            "tempo": "durante a consulta",
            "stub": True,
        }
    )
