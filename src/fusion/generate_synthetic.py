"""Gera eventos de sessão sintéticos para testes de fusão (feature 002 — consulta)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

OUT_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "fusion_synthetic" / "events.jsonl"
)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def build_events(base: datetime | None = None) -> list[dict]:
    """6 sessões pareadas + 2 só-áudio + 2 só-vídeo + 2 divergentes no tempo (>10 min)."""
    t0 = base or datetime(2026, 7, 23, 9, 0, 0, tzinfo=timezone.utc)
    events: list[dict] = []

    def add(eid, modality, group, session_id, ts, pair_id=None):
        ev = {
            "id": eid,
            "modality": modality,
            "group": group,
            "session_id": session_id,
            "timestamp": _iso(ts),
        }
        if pair_id:
            ev["pair_id"] = pair_id
        events.append(ev)

    # 6 consultas pareadas: áudio e vídeo da MESMA sessão, intervalo de 2 min.
    for i in range(1, 7):
        sid = f"consulta-{i:02d}"
        ts = t0 + timedelta(minutes=(i - 1) * 40)
        add(f"{sid}-audio", "audio", "pair", sid, ts, pair_id=sid)
        add(f"{sid}-video", "video", "pair", sid, ts + timedelta(minutes=2), pair_id=sid)

    # 2 audio-only (sessão sem vídeo)
    for i in range(1, 3):
        sid = f"consulta-a{i:02d}"
        add(f"{sid}-audio", "audio", "audio_only", sid, t0 + timedelta(hours=5, minutes=i * 15))

    # 2 só-vídeo
    for i in range(1, 3):
        sid = f"consulta-v{i:02d}"
        add(f"{sid}-video", "video", "video_only", sid, t0 + timedelta(hours=6, minutes=i * 15))

    # 2 divergentes no tempo: mesma sessão, vídeo 30 min depois (> janela de 10 min)
    for i in range(1, 3):
        sid = f"consulta-d{i:02d}"
        ts = t0 + timedelta(hours=7, minutes=i * 60)
        add(f"{sid}-audio", "audio", "time_divergent", sid, ts)
        add(f"{sid}-video", "video", "time_divergent", sid, ts + timedelta(minutes=30))

    return events


def write_events(path: Path | None = None) -> Path:
    out = path or OUT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for ev in build_events():
            fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
    return out


if __name__ == "__main__":
    print(write_events())
