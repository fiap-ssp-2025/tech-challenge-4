"""Generate synthetic fusion events for correlation tests (Brasília-DF)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Pontos plausíveis no DF (WGS84)
CRUZEIRO = (-15.7942, -47.8822)
ASA_NORTE = (-15.7640, -47.8825)
TAGUATINGA = (-15.8339, -48.0567)

OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "fusion_synthetic" / "events.jsonl"


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def build_events(base: datetime | None = None) -> list[dict]:
    """6 paired + 2 audio-only + 2 video-only + 2 time-divergent (>10 min)."""
    t0 = base or datetime(2026, 7, 23, 18, 0, 0, tzinfo=timezone.utc)
    events: list[dict] = []

    # --- 6 pairs (same lat/lon/timestamp) ---
    pair_points = [
        (CRUZEIRO, "pair-01"),
        (CRUZEIRO, "pair-02"),
        (ASA_NORTE, "pair-03"),
        (ASA_NORTE, "pair-04"),
        (TAGUATINGA, "pair-05"),
        (TAGUATINGA, "pair-06"),
    ]
    for i, ((lat, lon), pid) in enumerate(pair_points):
        # 30 min between pairs at the same coords so they do not cross-match (±10 min).
        ts = _iso(t0 + timedelta(minutes=i * 30))
        events.append(
            {
                "id": f"{pid}-audio",
                "modality": "audio",
                "group": "pair",
                "pair_id": pid,
                "lat": lat,
                "lon": lon,
                "timestamp": ts,
            }
        )
        events.append(
            {
                "id": f"{pid}-video",
                "modality": "video",
                "group": "pair",
                "pair_id": pid,
                "lat": lat,
                "lon": lon,
                "timestamp": ts,
            }
        )

    # --- 2 audio-only controls ---
    for i, (lat, lon) in enumerate([CRUZEIRO, ASA_NORTE]):
        events.append(
            {
                "id": f"audio-only-{i+1}",
                "modality": "audio",
                "group": "audio_only",
                "lat": lat + 0.05,  # far from pair clusters used below
                "lon": lon + 0.05,
                "timestamp": _iso(t0 + timedelta(hours=2, minutes=i)),
            }
        )

    # --- 2 video-only controls ---
    for i, (lat, lon) in enumerate([TAGUATINGA, CRUZEIRO]):
        events.append(
            {
                "id": f"video-only-{i+1}",
                "modality": "video",
                "group": "video_only",
                "lat": lat - 0.06,
                "lon": lon - 0.06,
                "timestamp": _iso(t0 + timedelta(hours=3, minutes=i)),
            }
        )

    # --- 2 divergent-time controls (same place, >10 min apart) ---
    for i, ((lat, lon), label) in enumerate(
        [(CRUZEIRO, "div-01"), (ASA_NORTE, "div-02")]
    ):
        events.append(
            {
                "id": f"{label}-audio",
                "modality": "audio",
                "group": "time_divergent",
                "pair_id": label,
                "lat": lat,
                "lon": lon,
                "timestamp": _iso(t0 + timedelta(hours=4, minutes=i * 5)),
            }
        )
        events.append(
            {
                "id": f"{label}-video",
                "modality": "video",
                "group": "time_divergent",
                "pair_id": label,
                "lat": lat,
                "lon": lon,
                "timestamp": _iso(
                    t0 + timedelta(hours=4, minutes=i * 5 + 15)
                ),  # +15 min > 10
            }
        )

    return events


def write_events(path: Path = OUT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    events = build_events()
    with path.open("w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
    return path


def load_events(path: Path = OUT_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


if __name__ == "__main__":
    out = write_events()
    print(f"Wrote {out}")
