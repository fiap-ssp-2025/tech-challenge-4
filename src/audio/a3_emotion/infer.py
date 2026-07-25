"""A3 emotion — fine-tuned wav2vec2 PT-BR, sofrimento na voz (P2)."""

from __future__ import annotations

from pathlib import Path

from src.contracts import A3Result, validate_a3

MODEL_DIR = Path(__file__).resolve().parents[3] / "models" / "a3_emotion"
TARGET_SR = 16_000
MAX_DURATION_S = 30.0  # guard against pathological long inputs at inference time
NON_NEUTRAL_ID = 1  # must match LABEL2ID in scripts/train_a3_emotion.py

_model = None
_feature_extractor = None


def _get_model():
    global _model, _feature_extractor
    if _model is None:
        if not MODEL_DIR.exists():
            raise FileNotFoundError(
                f"A3 model not found: {MODEL_DIR}\n"
                "Run: uv run python scripts/train_a3_emotion.py"
            )
        from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

        _feature_extractor = AutoFeatureExtractor.from_pretrained(str(MODEL_DIR))
        _model = Wav2Vec2ForSequenceClassification.from_pretrained(str(MODEL_DIR))
        _model.eval()
    return _model, _feature_extractor


def infer(path: str | Path) -> A3Result:
    """Estimate sofrimento (voice distress) score [0..1] from audio."""
    import librosa
    import torch

    audio_path = Path(path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    model, feature_extractor = _get_model()

    y, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    max_samples = int(TARGET_SR * MAX_DURATION_S)
    if len(y) > max_samples:
        y = y[:max_samples]

    inputs = feature_extractor(y, sampling_rate=TARGET_SR, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    score = float(probs[NON_NEUTRAL_ID])

    return validate_a3({"sofrimento": score})


__all__ = ["infer"]
