"""JSON contracts for audio/video/fusion inference modules (immutable from Etapa 2)."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

TipoRelato = Literal["agressao", "ameaca", "perseguicao", "outro"]
TIPO_RELATO_VALUES: frozenset[str] = frozenset(
    {"agressao", "ameaca", "perseguicao", "outro"}
)


class A12Result(TypedDict, total=False):
    transcricao: str
    tipo_relato: TipoRelato
    local: str
    tempo: str
    stub: bool


class A3Result(TypedDict, total=False):
    sofrimento: float
    stub: bool


class Track(TypedDict):
    id: int
    n_frames: int
    bbox_media: list[float]  # [x, y, w, h]


class V1Result(TypedDict, total=False):
    n_pessoas: int
    tracks: list[Track]
    stub: bool


class V2Result(TypedDict, total=False):
    postura_defensiva: float
    stub: bool


class V3Result(TypedDict, total=False):
    violencia: float
    stub: bool


class CDResult(TypedDict, total=False):
    escore: float
    corroborado: bool
    nota_ocorrencia: str
    stub: bool


class ContractError(ValueError):
    """Raised when a module output violates its JSON contract."""


def _require_keys(data: dict[str, Any], keys: tuple[str, ...], name: str) -> None:
    missing = [k for k in keys if k not in data]
    if missing:
        raise ContractError(f"{name}: missing keys {missing}")


def _unit_interval(value: Any, field: str, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ContractError(f"{name}.{field} must be a number")
    f = float(value)
    if not 0.0 <= f <= 1.0:
        raise ContractError(f"{name}.{field} must be in [0, 1], got {f}")
    return f


def validate_a12(data: dict[str, Any]) -> A12Result:
    _require_keys(data, ("transcricao", "tipo_relato", "local", "tempo"), "A1/A2")
    if not isinstance(data["transcricao"], str):
        raise ContractError("A1/A2.transcricao must be str")
    if data["tipo_relato"] not in TIPO_RELATO_VALUES:
        raise ContractError(
            f"A1/A2.tipo_relato must be one of {sorted(TIPO_RELATO_VALUES)}, "
            f"got {data['tipo_relato']!r}"
        )
    if not isinstance(data["local"], str):
        raise ContractError("A1/A2.local must be str")
    if not isinstance(data["tempo"], str):
        raise ContractError("A1/A2.tempo must be str")
    return data  # type: ignore[return-value]


def validate_a3(data: dict[str, Any]) -> A3Result:
    _require_keys(data, ("sofrimento",), "A3")
    _unit_interval(data["sofrimento"], "sofrimento", "A3")
    return data  # type: ignore[return-value]


def validate_v1(data: dict[str, Any]) -> V1Result:
    _require_keys(data, ("n_pessoas", "tracks"), "V1")
    if not isinstance(data["n_pessoas"], int) or isinstance(data["n_pessoas"], bool):
        raise ContractError("V1.n_pessoas must be int")
    if data["n_pessoas"] < 0:
        raise ContractError("V1.n_pessoas must be >= 0")
    tracks = data["tracks"]
    if not isinstance(tracks, list):
        raise ContractError("V1.tracks must be a list")
    for i, t in enumerate(tracks):
        if not isinstance(t, dict):
            raise ContractError(f"V1.tracks[{i}] must be an object")
        for key in ("id", "n_frames", "bbox_media"):
            if key not in t:
                raise ContractError(f"V1.tracks[{i}] missing '{key}'")
        if not isinstance(t["id"], int) or isinstance(t["id"], bool):
            raise ContractError(f"V1.tracks[{i}].id must be int")
        if not isinstance(t["n_frames"], int) or isinstance(t["n_frames"], bool):
            raise ContractError(f"V1.tracks[{i}].n_frames must be int")
        bbox = t["bbox_media"]
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ContractError(f"V1.tracks[{i}].bbox_media must be [x, y, w, h]")
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in bbox):
            raise ContractError(f"V1.tracks[{i}].bbox_media values must be numbers")
    return data  # type: ignore[return-value]


def validate_v2(data: dict[str, Any]) -> V2Result:
    _require_keys(data, ("postura_defensiva",), "V2")
    _unit_interval(data["postura_defensiva"], "postura_defensiva", "V2")
    return data  # type: ignore[return-value]


def validate_v3(data: dict[str, Any]) -> V3Result:
    _require_keys(data, ("violencia",), "V3")
    _unit_interval(data["violencia"], "violencia", "V3")
    return data  # type: ignore[return-value]


def validate_cd(data: dict[str, Any]) -> CDResult:
    _require_keys(data, ("escore", "corroborado", "nota_ocorrencia"), "C/D")
    _unit_interval(data["escore"], "escore", "C/D")
    if not isinstance(data["corroborado"], bool):
        raise ContractError("C/D.corroborado must be bool")
    if not isinstance(data["nota_ocorrencia"], str):
        raise ContractError("C/D.nota_ocorrencia must be str")
    return data  # type: ignore[return-value]
