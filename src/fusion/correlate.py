"""Spatiotemporal correlation for audio↔video events (C/D)."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Iterable

EARTH_RADIUS_M = 6_371_000.0
DEFAULT_RADIUS_M = 300.0
DEFAULT_WINDOW_MINUTES = 10.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters between two WGS84 points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def _parse_ts(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    # Accept ISO-8601 with optional Z
    text = value.replace("Z", "+00:00") if isinstance(value, str) else value
    return datetime.fromisoformat(text)


def _event_coords(event: dict[str, Any]) -> tuple[float, float]:
    if "lat" in event and "lon" in event:
        return float(event["lat"]), float(event["lon"])
    local = event.get("local") or {}
    if isinstance(local, dict) and "lat" in local and "lon" in local:
        return float(local["lat"]), float(local["lon"])
    raise KeyError("event needs lat/lon (top-level or under local)")


def _event_ts(event: dict[str, Any]) -> datetime:
    for key in ("timestamp", "tempo", "ts"):
        if key in event:
            return _parse_ts(event[key])
    raise KeyError("event needs timestamp/tempo/ts")


def within_correlation(
    audio_event: dict[str, Any],
    video_event: dict[str, Any],
    *,
    radius_m: float = DEFAULT_RADIUS_M,
    window_minutes: float = DEFAULT_WINDOW_MINUTES,
) -> bool:
    """True if audio and video events share location/time windows."""
    alat, alon = _event_coords(audio_event)
    vlat, vlon = _event_coords(video_event)
    if haversine_m(alat, alon, vlat, vlon) > radius_m:
        return False
    delta = abs((_event_ts(audio_event) - _event_ts(video_event)).total_seconds())
    return delta <= window_minutes * 60.0


def correlate(
    audio_events: Iterable[dict[str, Any]],
    video_events: Iterable[dict[str, Any]],
    *,
    radius_m: float = DEFAULT_RADIUS_M,
    window_minutes: float = DEFAULT_WINDOW_MINUTES,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Return all (audio, video) pairs that match radius + time window."""
    audios = list(audio_events)
    videos = list(video_events)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for a in audios:
        for v in videos:
            if within_correlation(
                a, v, radius_m=radius_m, window_minutes=window_minutes
            ):
                pairs.append((a, v))
    return pairs
