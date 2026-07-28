"""Stub V3 de desconforto facial — escore fixo até o modelo FER estar disponível."""

from __future__ import annotations

from pathlib import Path

from src.contracts import V3Result, validate_v3


def infer(path: str | Path) -> V3Result:
    _ = Path(path)
    return validate_v3({"desconforto_facial": 0.68, "stub": True})
