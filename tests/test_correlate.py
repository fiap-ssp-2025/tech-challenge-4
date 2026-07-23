"""Correlation tests against synthetic fusion events."""

from pathlib import Path

from src.fusion.correlate import correlate, within_correlation
from src.fusion.generate_synthetic import build_events, write_events

EVENTS_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "fusion_synthetic" / "events.jsonl"
)


def _ensure_events():
    if not EVENTS_PATH.exists():
        write_events(EVENTS_PATH)
    return build_events()


def test_six_pairs_match_and_six_controls_rejected():
    events = _ensure_events()
    audios = [e for e in events if e["modality"] == "audio"]
    videos = [e for e in events if e["modality"] == "video"]

    pairs = correlate(audios, videos)
    # Exact pair_id matches among group=pair
    pair_matches = [
        (a, v)
        for a, v in pairs
        if a.get("group") == "pair"
        and v.get("group") == "pair"
        and a.get("pair_id") == v.get("pair_id")
    ]
    assert len(pair_matches) == 6

    # Controls: audio-only never correlates; video-only never; divergent never
    audio_only = [e for e in events if e["group"] == "audio_only"]
    video_only = [e for e in events if e["group"] == "video_only"]
    assert correlate(audio_only, videos) == []
    assert correlate(audios, video_only) == []

    div_a = [e for e in events if e["group"] == "time_divergent" and e["modality"] == "audio"]
    div_v = [e for e in events if e["group"] == "time_divergent" and e["modality"] == "video"]
    assert correlate(div_a, div_v) == []
    # 2 audio-only + 2 video-only + 2 divergent pairs = 6 control scenarios rejected
    assert len(audio_only) == 2
    assert len(video_only) == 2
    assert len(div_a) == 2 and len(div_v) == 2


def test_within_correlation_params():
    a = {"lat": -15.7942, "lon": -47.8822, "timestamp": "2026-07-23T18:00:00Z"}
    v_ok = {"lat": -15.7942, "lon": -47.8822, "timestamp": "2026-07-23T18:05:00Z"}
    v_far = {"lat": -15.9000, "lon": -47.8822, "timestamp": "2026-07-23T18:05:00Z"}
    assert within_correlation(a, v_ok)
    assert not within_correlation(a, v_far)
    assert not within_correlation(a, v_ok, window_minutes=1)
