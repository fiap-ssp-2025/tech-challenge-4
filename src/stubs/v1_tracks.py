"""V1 tracks stub — fixed person count and tracks."""

from __future__ import annotations

from pathlib import Path

from src.contracts import V1Result, validate_v1


def infer(path: str | Path) -> V1Result:
    _ = Path(path)
    return validate_v1(
        {
            "n_pessoas": 2,
            "tracks": [
                {"id": 1, "n_frames": 48, "bbox_media": [120.0, 80.0, 64.0, 160.0]},
                {"id": 2, "n_frames": 40, "bbox_media": [300.0, 90.0, 70.0, 170.0]},
            ],
            "stub": True,
        }
    )
