"""End-to-end pipeline with stubs."""

from pathlib import Path

import pytest

from src.resolve import ENV_FORCE_STUBS, ENV_REQUIRE_REAL
from src.run_event import make_dummy_media, run_pipeline
from src.stubs import a1_stt


@pytest.fixture(scope="module")
def dummies(tmp_path_factory):
    root = tmp_path_factory.mktemp("media")
    wav = root / "dummy.wav"
    mp4 = root / "dummy.mp4"
    make_dummy_media(wav, mp4)
    return wav, mp4


@pytest.fixture
def force_stubs(monkeypatch):
    """E2E da Etapa 1: 100% stub, independente de pesos treinados na máquina."""
    monkeypatch.delenv(ENV_REQUIRE_REAL, raising=False)
    monkeypatch.setenv(ENV_FORCE_STUBS, "1")


def test_a1_stub_fails_without_azure_key(monkeypatch, dummies):
    monkeypatch.delenv("AZURE_SPEECH_KEY", raising=False)
    with pytest.raises(RuntimeError, match="faster-whisper"):
        a1_stt.infer(dummies[0])


def test_e2e_pipeline_with_stubs(monkeypatch, dummies, force_stubs):
    monkeypatch.delenv("AZURE_SPEECH_KEY", raising=False)
    wav, mp4 = dummies
    result = run_pipeline(wav, mp4)
    cd = result["cd"]
    assert "escore" in cd
    assert isinstance(cd["corroborado"], bool)
    assert cd["corroborado"] is True
    assert "indicativo, não veredito" in cd["nota_ocorrencia"]
    assert result["a12"]["tipo_relato"] in {
        "violencia_domestica",
        "sofrimento_emocional",
        "outro",
    }
    assert result["a3"].get("stub") is True
    assert result["v1"].get("stub") is True


def test_e2e_reports_resolution_map_and_timings(dummies, force_stubs):
    wav, mp4 = dummies
    result = run_pipeline(wav, mp4)

    assert result["resolved"] == {k: "stub" for k in ("a1", "a2", "a3", "v1", "v2", "v3")}
    assert set(result["tempos_ms"]) == {"a1", "a2", "a3", "v1", "v2", "v3"}
    assert all(ms >= 0.0 for ms in result["tempos_ms"].values())
    assert result["total_ms"] >= 0.0
