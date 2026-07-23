"""A2 NLP stub — structured fields from (stub) transcription context."""

from __future__ import annotations

from pathlib import Path

from src.contracts import A12Result, validate_a12


def infer(path: str | Path, transcricao: str | None = None) -> A12Result:
    """Return fixed structured A1/A2 contract. Optional transcricao overrides text."""
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
