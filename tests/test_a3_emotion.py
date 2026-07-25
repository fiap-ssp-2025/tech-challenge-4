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
