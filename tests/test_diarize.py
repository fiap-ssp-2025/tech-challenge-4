"""Diarização (feature 003) — agrupamento por locutor e degradação sem credencial."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.audio import diarize
from src.audio.diarize import Segment, audio_by_speaker, speaker_segments


def test_audio_by_speaker_concatena_trechos_do_mesmo_locutor():
    """Trechos curtos do mesmo locutor viram um bloco só — senão o janelamento de 6 s
    receberia amostras minúsculas e ruidosas (RF-23)."""
    sr = 16_000
    y = np.arange(10 * sr, dtype=np.float32)  # 10 s, valores distintos por amostra
    segments = [
        Segment("Convidado-1", 0.0, 2.0),
        Segment("Convidado-2", 2.0, 4.0),
        Segment("Convidado-1", 4.0, 7.0),
    ]

    out = audio_by_speaker(y, segments, sr=sr)

    assert set(out) == {"Convidado-1", "Convidado-2"}
    assert len(out["Convidado-1"]) == 5 * sr  # 2 s + 3 s concatenados
    assert len(out["Convidado-2"]) == 2 * sr
    # o conteúdo tem de vir das faixas certas, na ordem
    assert out["Convidado-2"][0] == pytest.approx(2 * sr)
    assert out["Convidado-1"][2 * sr] == pytest.approx(4 * sr)


def test_locutor_com_fala_curta_demais_e_descartado():
    sr = 16_000
    y = np.zeros(10 * sr, dtype=np.float32)
    segments = [
        Segment("Convidado-1", 0.0, 5.0),
        Segment("Convidado-2", 5.0, 5.3),  # 0,3 s < MIN_SPEAKER_SECONDS
    ]

    out = audio_by_speaker(y, segments, sr=sr)

    assert set(out) == {"Convidado-1"}


def test_sem_credencial_degrada_para_none_sem_levantar(tmp_path: Path, monkeypatch, capsys):
    """RF-22: ausência de credencial não pode quebrar o pipeline."""
    monkeypatch.setattr(diarize, "_cache", {})
    monkeypatch.setenv("AZURE_SPEECH_KEY", "")
    monkeypatch.setenv("AZURE_SPEECH_REGION", "")
    wav = tmp_path / "a.wav"
    sf.write(wav, np.zeros(16_000, dtype=np.float32), 16_000)

    assert speaker_segments(wav) is None
    assert "sem AZURE_SPEECH_KEY" in capsys.readouterr().out


def test_falha_do_sdk_degrada_para_none(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(diarize, "_cache", {})
    monkeypatch.setenv("AZURE_SPEECH_KEY", "chave-de-teste")
    monkeypatch.setenv("AZURE_SPEECH_REGION", "brazilsouth")
    monkeypatch.setattr(
        diarize, "_diarize_azure", lambda p: diarize._log("boom") or None
    )
    wav = tmp_path / "b.wav"
    sf.write(wav, np.zeros(16_000, dtype=np.float32), 16_000)

    assert speaker_segments(wav) is None


def test_arquivo_inexistente_levanta(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        speaker_segments(tmp_path / "nao_existe.wav")


def test_cache_evita_segunda_chamada(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(diarize, "_cache", {})
    wav = tmp_path / "c.wav"
    sf.write(wav, np.zeros(16_000, dtype=np.float32), 16_000)

    chamadas = []

    def _fake(path):
        chamadas.append(path)
        return [Segment("Convidado-1", 0.0, 1.0)]

    monkeypatch.setattr(diarize, "_diarize_azure", _fake)

    speaker_segments(wav)
    speaker_segments(wav)

    assert len(chamadas) == 1
