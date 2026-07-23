"""A2 NLP stub — structured fields from (stub) transcription context."""

from __future__ import annotations

from pathlib import Path

from src.contracts import A12Result, validate_a12


def infer(path: str | Path, transcricao: str | None = None) -> A12Result:
    """Return fixed structured A1/A2 contract. Optional transcricao overrides text."""
    _ = Path(path)
    text = transcricao or (
        "Preciso de ajuda, tem um homem me agredindo na praça do Cruzeiro."
    )
    return validate_a12(
        {
            "transcricao": text,
            "tipo_relato": "agressao",
            "local": "Praça do Cruzeiro, Brasília-DF",
            "tempo": "agora",
            "stub": True,
        }
    )
