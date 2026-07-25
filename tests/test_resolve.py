"""Resolution layer: real when available, stub otherwise — with an explicit reason."""

import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest

from src import resolve
from src.resolve import (
    ENV_FORCE_STUBS,
    ENV_REQUIRE_REAL,
    MODULE_NAMES,
    REGISTRY,
    ModuleResolutionError,
    ResolvedPipeline,
    canonical_name,
    get_module,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Cada teste decide o modo; nunca herda o ambiente do dev."""
    monkeypatch.delenv(ENV_FORCE_STUBS, raising=False)
    monkeypatch.delenv(ENV_REQUIRE_REAL, raising=False)


def _ultralytics_missing() -> bool:
    return not resolve._has_package("ultralytics")


# --- nomes e aliases ---------------------------------------------------------


def test_canonical_name_accepts_full_name_and_short_key():
    assert canonical_name("v2_pose") == "v2_pose"
    assert canonical_name("v2") == "v2_pose"


def test_unknown_module_name_raises():
    with pytest.raises(ModuleResolutionError, match="desconhecido"):
        get_module("v9_teleport")


def test_registry_covers_the_six_pipeline_modules():
    assert set(MODULE_NAMES) == {
        "a1_stt",
        "a2_nlp",
        "a3_emotion",
        "v1_tracks",
        "v2_pose",
        "v3_face",
    }


# --- fallback: artefato ausente ---------------------------------------------


def test_missing_artifact_falls_back_to_stub_with_reason(monkeypatch, tmp_path, capsys):
    spec = replace(REGISTRY["v2_pose"], artifacts=(tmp_path / "nao_existe.pkl",))
    monkeypatch.setitem(REGISTRY, "v2_pose", spec)

    resolved = get_module("v2_pose", verbose=True)

    assert resolved.origin == "stub"
    assert "artefato ausente" in resolved.reason
    assert "train_v2_posture" in resolved.reason  # a dica de regeneração
    assert "v2_pose" in capsys.readouterr().out  # o motivo é logado, não silencioso
    assert resolved.infer(Path("qualquer.mp4"))["stub"] is True


def test_missing_stt_package_falls_back_to_stub(monkeypatch):
    """Sem azure-speech / faster-whisper o A1 cai para stub (Azure é opcional em runtime)."""
    spec = replace(
        REGISTRY["a1_stt"],
        packages=("pacote_stt_inexistente_xyz",),
        env_vars=(),
    )
    monkeypatch.setitem(REGISTRY, "a1_stt", spec)

    resolved = get_module("a1_stt")
    assert resolved.origin == "stub"
    assert "pacote ausente" in resolved.reason


def test_real_that_only_reexports_the_stub_is_reported_as_stub(monkeypatch):
    """Enquanto T110–T112 não chegam, `infer.py` real é passthrough do stub."""
    spec = replace(REGISTRY["v3_face"], real="src.stubs.v3_face", packages=())
    monkeypatch.setitem(REGISTRY, "v3_face", spec)

    resolved = get_module("v3_face")

    assert resolved.origin == "stub"
    assert "re-exporta o stub" in resolved.reason


def test_broken_real_import_falls_back_to_stub(monkeypatch):
    spec = replace(REGISTRY["a3_emotion"], real="src.audio.a3_emotion.nao_existe")
    monkeypatch.setitem(REGISTRY, "a3_emotion", spec)

    resolved = get_module("a3_emotion")

    assert resolved.origin == "stub"
    assert "import falhou" in resolved.reason


# --- módulos reais presentes -------------------------------------------------


@pytest.mark.skipif(_ultralytics_missing(), reason="ultralytics não instalado")
def test_v1_resolves_to_real_when_ultralytics_is_available():
    resolved = get_module("v1_tracks")
    assert resolved.origin == "real"
    assert resolved.reason == ""
    assert resolved.module.__name__ == "src.video.v1_tracks.infer"


@pytest.mark.skipif(
    _ultralytics_missing() or not resolve.V2_POSTURE_HEAD.exists(),
    reason="ultralytics ou models/v2_posture_head.pkl ausente",
)
def test_v2_resolves_to_real_when_the_posture_head_exists():
    resolved = get_module("v2_pose")
    assert resolved.origin == "real"


def test_registry_artifact_matches_the_path_the_v2_module_loads():
    """Guarda contra o registry e o módulo do P5 divergirem sobre o .pkl."""
    from src.video.v2_pose.infer import MODEL_PATH

    assert MODEL_PATH == resolve.V2_POSTURE_HEAD


# --- chaves de ambiente ------------------------------------------------------


def test_force_stubs_makes_every_module_a_stub(monkeypatch):
    monkeypatch.setenv(ENV_FORCE_STUBS, "1")
    pipeline = ResolvedPipeline(verbose=False)
    assert set(pipeline.origins().values()) == {"stub"}
    assert all(m.key in pipeline.origins() for m in REGISTRY.values())


def test_require_real_raises_when_the_module_is_not_available(monkeypatch, tmp_path):
    spec = replace(REGISTRY["v2_pose"], artifacts=(tmp_path / "nao_existe.pkl",))
    monkeypatch.setitem(REGISTRY, "v2_pose", spec)
    monkeypatch.setenv(ENV_REQUIRE_REAL, "v2_pose")

    with pytest.raises(ModuleResolutionError, match="exige v2_pose real"):
        get_module("v2_pose")


def test_require_real_and_force_stubs_are_mutually_exclusive(monkeypatch):
    monkeypatch.setenv(ENV_FORCE_STUBS, "1")
    monkeypatch.setenv(ENV_REQUIRE_REAL, "v1_tracks")
    with pytest.raises(ModuleResolutionError, match="mutuamente exclusivos"):
        get_module("v1_tracks")


def test_require_real_rejects_unknown_module(monkeypatch):
    monkeypatch.setenv(ENV_REQUIRE_REAL, "v1_tracks,v9_teleport")
    with pytest.raises(ModuleResolutionError, match="desconhecido"):
        get_module("v1_tracks")


# --- execução ----------------------------------------------------------------


def test_pipeline_times_every_call(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_FORCE_STUBS, "1")
    pipeline = ResolvedPipeline(verbose=False)

    pipeline.call("a3_emotion", tmp_path / "x.wav")

    assert pipeline.timings_ms()["a3"] >= 0.0
    assert set(pipeline.timings_ms()) == {"a3"}


def test_real_module_that_blows_up_degrades_to_stub(monkeypatch, capsys, tmp_path):
    """O treino pode chegar quebrado; a demo não pode morrer por causa disso."""

    def boom(*_args, **_kwargs):
        raise RuntimeError("peso corrompido")

    fake = ModuleType("tc4_fake_real_v2")
    fake.infer = boom
    monkeypatch.setitem(sys.modules, "tc4_fake_real_v2", fake)
    spec = replace(
        REGISTRY["v2_pose"], real="tc4_fake_real_v2", packages=(), artifacts=()
    )
    monkeypatch.setitem(REGISTRY, "v2_pose", spec)

    pipeline = ResolvedPipeline(["v2_pose"], verbose=False)
    assert pipeline.origin("v2_pose") == "real"

    payload = pipeline.call("v2_pose", tmp_path / "x.mp4")

    assert payload["stub"] is True
    assert pipeline.origin("v2_pose") == "stub"
    assert "falhou em execução" in capsys.readouterr().out


def test_stub_errors_propagate(monkeypatch, tmp_path):
    """Sem chave Azure o stub A1 levanta — o runner é quem trata."""
    monkeypatch.setenv(ENV_FORCE_STUBS, "1")
    monkeypatch.delenv("AZURE_SPEECH_KEY", raising=False)
    pipeline = ResolvedPipeline(["a1_stt"], verbose=False)

    with pytest.raises(RuntimeError, match="faster-whisper"):
        pipeline.call("a1_stt", tmp_path / "x.wav")
