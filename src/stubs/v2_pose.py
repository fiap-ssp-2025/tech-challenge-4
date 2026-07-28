"""Stub V2 pose — escore de postura defensiva fixo."""

from __future__ import annotations

from pathlib import Path

from src.contracts import V2Result, validate_v2


def infer(path: str | Path) -> V2Result:
    _ = Path(path)
    return validate_v2({"postura_defensiva": 0.71, "stub": True})
