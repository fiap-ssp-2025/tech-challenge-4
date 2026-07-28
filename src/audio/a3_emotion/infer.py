"""A3 emotion — wav2vec2 PT-BR fine-tuned, sofrimento na voz."""

from __future__ import annotations

from pathlib import Path

from src.contracts import A3Result, validate_a3

MODEL_DIR = Path(__file__).resolve().parents[3] / "models" / "a3_emotion"
TARGET_SR = 16_000
WINDOW_S = 6.0  # deve coincidir com MAX_DURATION_S em scripts/train_a3_emotion.py
MAX_DURATION_S = 30.0  # proteção contra entradas patologicamente longas na inferência
NON_NEUTRAL_ID = 1  # deve coincidir com LABEL2ID em scripts/train_a3_emotion.py

_model = None
_feature_extractor = None


def _get_model():
    global _model, _feature_extractor
    if _model is None:
        if not MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Modelo A3 não encontrado: {MODEL_DIR}\n"
                "Run: uv run python scripts/train_a3_emotion.py"
            )
        from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

        _feature_extractor = AutoFeatureExtractor.from_pretrained(str(MODEL_DIR))
        _model = Wav2Vec2ForSequenceClassification.from_pretrained(str(MODEL_DIR))
        _model.eval()
    return _model, _feature_extractor


def _score_samples(y, model, feature_extractor) -> float:
    """Pontua um trecho de áudio: janelas de WINDOW_S segundos, agregadas pelo MÁXIMO.

    Máximo e não média: sofrimento é um evento, não um estado médio da gravação. Numa
    consulta longa o sinal costuma ser um trecho curto e agudo, que a média dilui no
    resto da fala. Triagem prefere errar sinalizando — perder o pico é o erro caro.
    """
    import torch

    max_samples = int(TARGET_SR * MAX_DURATION_S)
    if len(y) > max_samples:
        y = y[:max_samples]

    # O modelo foi ajustado em clipes de até WINDOW_S segundos; áudio maior é lido em
    # janelas desse tamanho, nunca de uma vez só.
    window = int(TARGET_SR * WINDOW_S)
    chunks = [y[i : i + window] for i in range(0, len(y), window)] or [y]
    if len(chunks) > 1 and len(chunks[-1]) < int(0.5 * TARGET_SR):
        chunks.pop()  # cauda de menos de meio segundo é ruído, não sinal

    scores = []
    with torch.no_grad():
        for chunk in chunks:
            inputs = feature_extractor(chunk, sampling_rate=TARGET_SR, return_tensors="pt")
            logits = model(**inputs).logits
            scores.append(float(torch.softmax(logits, dim=-1)[0][NON_NEUTRAL_ID]))
    return max(scores)


def infer(path: str | Path) -> A3Result:
    """Estima o escore de sofrimento (distress na voz) [0..1] a partir do áudio.

    Com mais de um locutor (feature 003), cada um é pontuado em separado e vale o
    MAIOR escore — a fala neutra do profissional não pode diluir a da paciente. Não
    se tenta descobrir quem é quem: pontua-se todos e toma-se o máximo.

    Sem diarização disponível (sem credencial, sem rede, erro do serviço), o caminho
    é exatamente o anterior: o áudio inteiro em janelas.
    """
    import librosa

    audio_path = Path(path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Áudio não encontrado: {audio_path}")

    model, feature_extractor = _get_model()
    y, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)

    from src.audio.diarize import audio_by_speaker, speaker_segments

    segments = speaker_segments(audio_path)
    per_speaker = audio_by_speaker(y, segments) if segments else {}

    if len(per_speaker) > 1:
        scores = {
            speaker: _score_samples(samples, model, feature_extractor)
            for speaker, samples in per_speaker.items()
        }
        detalhe = ", ".join(f"{k}={v:.3f}" for k, v in sorted(scores.items()))
        print(f"[A3] sofrimento por locutor: {detalhe} → máximo")
        score = max(scores.values())
    else:
        score = _score_samples(y, model, feature_extractor)

    return validate_a3({"sofrimento": score})


__all__ = ["infer"]
