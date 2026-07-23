"""V3 violence stub — fixed violence score."""

from __future__ import annotations

from pathlib import Path

from src.contracts import V3Result, validate_v3


def infer(path: str | Path) -> V3Result:
    _ = Path(path)
    return validate_v3({"violencia": 0.76, "stub": True})
