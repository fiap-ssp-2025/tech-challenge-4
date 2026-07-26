"""A3 emotion (P2) — real module contract smoke test."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.audio.a3_emotion.infer import MODEL_DIR
from src.audio.a3_emotion import infer as a3_infer

pytestmark = pytest.mark.skipif(
    not MODEL_DIR.exists(),
    reason="A3 model not trained yet; run scripts/train_a3_emotion.py",
)


def test_infer_returns_valid_contract(tmp_path: Path):
    wav_path = tmp_path / "silence.wav"
    sf.write(wav_path, np.zeros(16_000, dtype=np.float32), 16_000)

    result = a3_infer.infer(wav_path)

    assert "sofrimento" in result
    assert 0.0 <= result["sofrimento"] <= 1.0


def test_infer_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        a3_infer.infer(tmp_path / "does_not_exist.wav")


def _fala_sintetica(seconds: float, freq: float) -> np.ndarray:
    t = np.linspace(0, seconds, int(16_000 * seconds), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_dois_locutores_pontuam_separado_e_vale_o_maximo(tmp_path: Path, monkeypatch, capsys):
    """Feature 003: cada locutor é pontuado sozinho; o escore final é o maior deles."""
    from src.audio import diarize
    from src.audio.diarize import Segment

    wav = tmp_path / "consulta.wav"
    sf.write(wav, np.concatenate([_fala_sintetica(4, 180), _fala_sintetica(4, 320)]), 16_000)

    monkeypatch.setattr(diarize, "_cache", {})
    monkeypatch.setattr(
        diarize,
        "_diarize_azure",
        lambda p: [Segment("Convidado-1", 0.0, 4.0), Segment("Convidado-2", 4.0, 8.0)],
    )

    result = a3_infer.infer(wav)

    assert 0.0 <= result["sofrimento"] <= 1.0
    saida = capsys.readouterr().out
    assert "por locutor" in saida and "Convidado-1" in saida and "Convidado-2" in saida


def test_sem_diarizacao_resultado_e_identico_ao_caminho_antigo(tmp_path: Path, monkeypatch):
    """RNF-20: sem diarização, o comportamento tem de ser bit-a-bit o anterior."""
    from src.audio import diarize

    wav = tmp_path / "monologo.wav"
    sf.write(wav, _fala_sintetica(8, 220), 16_000)

    monkeypatch.setattr(diarize, "_cache", {})
    monkeypatch.setattr(diarize, "_diarize_azure", lambda p: None)
    sem_diarizacao = a3_infer.infer(wav)["sofrimento"]

    # mesmo áudio pontuado direto pelo caminho de janelas, sem passar pela diarização
    import librosa

    y, _ = librosa.load(wav, sr=16_000, mono=True)
    model, fe = a3_infer._get_model()
    direto = a3_infer._score_samples(y, model, fe)

    assert sem_diarizacao == pytest.approx(direto, abs=1e-9)


def test_um_locutor_so_nao_muda_o_caminho(tmp_path: Path, monkeypatch):
    from src.audio import diarize
    from src.audio.diarize import Segment

    wav = tmp_path / "um.wav"
    sf.write(wav, _fala_sintetica(6, 200), 16_000)

    monkeypatch.setattr(diarize, "_cache", {})
    monkeypatch.setattr(diarize, "_diarize_azure", lambda p: [Segment("Convidado-1", 0.0, 6.0)])

    result = a3_infer.infer(wav)
    assert 0.0 <= result["sofrimento"] <= 1.0
