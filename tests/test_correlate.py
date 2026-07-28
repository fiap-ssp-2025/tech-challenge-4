"""Testes de correlação sobre eventos sintéticos de sessão (feature 002)."""

from pathlib import Path

from src.fusion.correlate import correlate, same_session, within_correlation
from src.fusion.generate_synthetic import build_events, write_events

EVENTS_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "fusion_synthetic" / "events.jsonl"
)


def _ensure_events():
    if not EVENTS_PATH.exists():
        write_events(EVENTS_PATH)
    return build_events()


def test_six_session_pairs_match_and_controls_rejected():
    events = _ensure_events()
    audios = [e for e in events if e["modality"] == "audio"]
    videos = [e for e in events if e["modality"] == "video"]

    pairs = correlate(audios, videos)
    pair_matches = [
        (a, v)
        for a, v in pairs
        if a.get("group") == "pair" and v.get("group") == "pair"
        and a.get("session_id") == v.get("session_id")
    ]
    assert len(pair_matches) == 6
    # Nenhum cruzamento entre sessões diferentes
    assert all(a["session_id"] == v["session_id"] for a, v in pairs)

    audio_only = [e for e in events if e["group"] == "audio_only"]
    video_only = [e for e in events if e["group"] == "video_only"]
    assert correlate(audio_only, videos) == []
    assert correlate(audios, video_only) == []

    div_a = [e for e in events if e["group"] == "time_divergent" and e["modality"] == "audio"]
    div_v = [e for e in events if e["group"] == "time_divergent" and e["modality"] == "video"]
    assert correlate(div_a, div_v) == []
    assert len(audio_only) == 2 and len(video_only) == 2
    assert len(div_a) == 2 and len(div_v) == 2


def test_same_session_window():
    a = {"session_id": "consulta-01", "timestamp": "2026-07-23T09:00:00Z"}
    v_ok = {"session_id": "consulta-01", "timestamp": "2026-07-23T09:05:00Z"}
    v_other = {"session_id": "consulta-02", "timestamp": "2026-07-23T09:05:00Z"}
    v_late = {"session_id": "consulta-01", "timestamp": "2026-07-23T09:30:00Z"}
    assert same_session(a, v_ok)
    assert not same_session(a, v_other)
    assert not same_session(a, v_late)
    # within_correlation usa sessão como caminho primário
    assert within_correlation(a, v_ok)
    assert not within_correlation(a, v_late)


def test_geo_legacy_fallback_kept():
    a = {"lat": -15.7942, "lon": -47.8822, "timestamp": "2026-07-23T18:00:00Z"}
    v_ok = {"lat": -15.7942, "lon": -47.8822, "timestamp": "2026-07-23T18:05:00Z"}
    v_far = {"lat": -15.9000, "lon": -47.8822, "timestamp": "2026-07-23T18:05:00Z"}
    assert within_correlation(a, v_ok)
    assert not within_correlation(a, v_far)
