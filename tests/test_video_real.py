"""V1/V2 reais sobre vídeo — pulados quando pesos/artefatos não estão na máquina.

Um clone limpo não tem `models/v2_posture_head.pkl` (gitignored) nem os `.pt` do
YOLO em cache, então estes testes são `skipif` por desenho: quem clonou limpo
roda a suíte verde; quem treinou localmente exercita o caminho real.
"""

from pathlib import Path

import pytest

from src import resolve
from src.contracts import validate_v1, validate_v2
from src.resolve import ENV_FORCE_STUBS, ENV_REQUIRE_REAL, ResolvedPipeline
from src.run_event import make_dummy_media

ROOT = Path(__file__).resolve().parents[1]
YOLO_DET = ROOT / "yolov8n.pt"
YOLO_POSE = ROOT / "yolov8n-pose.pt"
RAVDESS = ROOT / "data" / "video_consulta" / "raw" / "ravdess"


def _missing(*paths: Path) -> bool:
    return not resolve._has_package("ultralytics") or any(not p.exists() for p in paths)


needs_v1 = pytest.mark.skipif(
    _missing(YOLO_DET), reason="ultralytics ou yolov8n.pt ausente (sem download em teste)"
)
needs_v2 = pytest.mark.skipif(
    _missing(YOLO_POSE, resolve.V2_POSTURE_HEAD),
    reason="ultralytics, yolov8n-pose.pt ou models/v2_posture_head.pkl ausente",
)


@pytest.fixture(autouse=True)
def real_modules(monkeypatch):
    monkeypatch.delenv(ENV_FORCE_STUBS, raising=False)
    monkeypatch.delenv(ENV_REQUIRE_REAL, raising=False)


@pytest.fixture(scope="module")
def dummy_mp4(tmp_path_factory):
    root = tmp_path_factory.mktemp("media")
    _, mp4 = make_dummy_media(root / "dummy.wav", root / "dummy.mp4")
    return mp4


def _first_ravdess_clip() -> Path | None:
    if not RAVDESS.is_dir():
        return None
    return next(iter(sorted(RAVDESS.rglob("*.mp4"))), None)


@needs_v1
def test_v1_real_respects_the_contract(dummy_mp4):
    pipeline = ResolvedPipeline(["v1_tracks"], verbose=False)
    assert pipeline.origin("v1_tracks") == "real"

    out = pipeline.call("v1_tracks", dummy_mp4)

    validate_v1(out)
    assert out.get("stub") is None  # veio do modelo, não do stub
    assert out["n_pessoas"] == len(out["tracks"])
    assert pipeline.timings_ms()["v1"] > 0.0


@needs_v2
def test_v2_real_respects_the_contract(dummy_mp4):
    """Vídeo preto: sem keypoints → 0.0, ainda dentro do contrato."""
    pipeline = ResolvedPipeline(["v2_pose"], verbose=False)
    assert pipeline.origin("v2_pose") == "real"

    out = pipeline.call("v2_pose", dummy_mp4)

    validate_v2(out)
    assert out.get("stub") is None


@needs_v2
@pytest.mark.skipif(_first_ravdess_clip() is None, reason="RAVDESS não baixado")
def test_v2_real_scores_a_ravdess_clip():
    """Com pessoa em quadro, a cabeça de postura roda de fato (predict_proba)."""
    pipeline = ResolvedPipeline(["v2_pose"], verbose=False)

    out = pipeline.call("v2_pose", _first_ravdess_clip())

    validate_v2(out)
    assert 0.0 < out["postura_defensiva"] <= 1.0


@needs_v1
@needs_v2
def test_require_real_accepts_the_video_modules(monkeypatch):
    """Contrato da CI futura: TC4_REQUIRE_REAL não pode falhar com V1/V2 prontos."""
    monkeypatch.setenv(ENV_REQUIRE_REAL, "v1_tracks,v2_pose")

    pipeline = ResolvedPipeline(["v1_tracks", "v2_pose"], verbose=False)

    assert pipeline.origins() == {"v1": "real", "v2": "real"}
