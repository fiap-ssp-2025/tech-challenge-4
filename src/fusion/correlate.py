"""Correlação de sessão para eventos áudio↔vídeo (C/D, feature 002).

Caminho primário: mesmo ``session_id`` dentro da janela de tempo (áudio e vídeo
da mesma consulta). O caminho geográfico (haversine) fica como utilitário legado
da feature 001 e só é usado quando ambos os eventos não têm ``session_id``.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Iterable

EARTH_RADIUS_M = 6_371_000.0
DEFAULT_RADIUS_M = 300.0
DEFAULT_WINDOW_MINUTES = 10.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distância ortodrômica em metros entre dois pontos WGS84."""
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
    # Aceita ISO-8601 com Z opcional
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


def same_session(
    audio_event: dict[str, Any],
    video_event: dict[str, Any],
    *,
    window_minutes: float = DEFAULT_WINDOW_MINUTES,
) -> bool:
    """Verdadeiro se ambos os eventos tiverem o mesmo session_id dentro da janela de tempo."""
    sid_a = audio_event.get("session_id")
    sid_v = video_event.get("session_id")
    if not sid_a or not sid_v or sid_a != sid_v:
        return False
    delta = abs((_event_ts(audio_event) - _event_ts(video_event)).total_seconds())
    return delta <= window_minutes * 60.0


def within_correlation(
    audio_event: dict[str, Any],
    video_event: dict[str, Any],
    *,
    radius_m: float = DEFAULT_RADIUS_M,
    window_minutes: float = DEFAULT_WINDOW_MINUTES,
) -> bool:
    """Primário: match por sessão. Fallback (legado 001): raio geográfico + janela."""
    if audio_event.get("session_id") or video_event.get("session_id"):
        return same_session(audio_event, video_event, window_minutes=window_minutes)
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
    """Retorna todos os pares (áudio, vídeo) que batem raio + janela de tempo."""
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
