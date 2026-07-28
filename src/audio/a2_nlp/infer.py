"""A2 NLP — estruturação por regras para triagem em consulta."""

from __future__ import annotations

from pathlib import Path

from src.contracts import A12Result, validate_a12

from .extractors import classify_report, extract_location, extract_time


def infer(path: str | Path, transcricao: str | None = None) -> A12Result:
    """
    Estrutura uma transcrição A1 com regras determinísticas de NLP.

    Args:
        path:
            Caminho do áudio mantido por compatibilidade com o pipeline A1/A2.
        transcricao:
            Texto gerado pelo A1. Quando ausente, processa string vazia.

    Returns:
        Contrato A1/A2 validado.
    """
    _ = Path(path)
    text = transcricao or ""

    result: A12Result = {
        "transcricao": text,
        "tipo_relato": classify_report(text),
        "local": extract_location(text),
        "tempo": extract_time(text),
    }

    return validate_a12(result)


__all__ = ["infer"]