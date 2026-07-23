"""End-to-end pipeline with stubs."""

from pathlib import Path

import pytest

from src.run_event import make_dummy_media, run_pipeline
from src.stubs import a1_stt


@pytest.fixture(scope="module")
def dummies(tmp_path_factory):
    root = tmp_path_factory.mktemp("media")
    wav = root / "dummy.wav"
    mp4 = root / "dummy.mp4"
    make_dummy_media(wav, mp4)
    return wav, mp4


def test_a1_stub_fails_without_azure_key(monkeypatch, dummies):
    monkeypatch.delenv("AZURE_SPEECH_KEY", raising=False)
    with pytest.raises(RuntimeError, match="faster-whisper"):
        a1_stt.infer(dummies[0])


def test_e2e_pipeline_with_stubs(monkeypatch, dummies):
    monkeypatch.delenv("AZURE_SPEECH_KEY", raising=False)
    wav, mp4 = dummies
    result = run_pipeline(wav, mp4)
    cd = result["cd"]
    assert "escore" in cd
    assert isinstance(cd["corroborado"], bool)
    assert cd["corroborado"] is True
    assert "indicativo, não veredito" in cd["nota_ocorrencia"]
    assert result["a12"]["tipo_relato"] in {
        "agressao",
        "ameaca",
        "perseguicao",
        "outro",
    }
    assert result["a3"].get("stub") is True
    assert result["v1"].get("stub") is True
