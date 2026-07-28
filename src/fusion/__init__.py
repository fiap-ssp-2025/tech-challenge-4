"""Pacote de fusão (C/D): correlação, escore, nota de ocorrência."""

from src.fusion.correlate import correlate, haversine_m, within_correlation
from src.fusion.report import DISCLAIMER, build_report
from src.fusion.scoring import SCORE_WEIGHTS, compute_score, score_from_modules

__all__ = [
    "SCORE_WEIGHTS",
    "DISCLAIMER",
    "correlate",
    "haversine_m",
    "within_correlation",
    "compute_score",
    "score_from_modules",
    "build_report",
]
