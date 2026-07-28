"""Stub A3 emotion — escore de sofrimento fixo."""

from __future__ import annotations

from pathlib import Path

from src.contracts import A3Result, validate_a3


def infer(path: str | Path) -> A3Result:
    _ = Path(path)
    return validate_a3({"sofrimento": 0.82, "stub": True})
