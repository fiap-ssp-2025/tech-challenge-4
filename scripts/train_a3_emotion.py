#!/usr/bin/env python3
"""Ajuste fino do wav2vec2 PT-BR para emoção na voz (neutral vs non_neutral) — A3.

Entrada:  data/audio_ptbr/labels.csv  (path,label,speaker,split de preprocess_audio.py)
Saída:    models/a3_emotion/          (modelo HF + feature extractor)
          models/a3_emotion/metrics.json

Checkpoint base: jonatasgrosman/wav2vec2-large-xlsr-53-portuguese — mesma linhagem
dos baselines da shared-task CORAA SER (wav2vec2-xlsr-53 ajustado em fala PT-BR).

Nota de hardware: esta máquina não tem GPU (ver torch.cuda.is_available()). Fine-tune
completo das 24 camadas transformer (315M params) é inviável em CPU, então o feature
encoder (CNN) fica congelado e só os N blocos transformer do topo + cabeça de
classificação são treinados — fine-tune parcial padrão para wav2vec2 SER com pouco
compute.

Uso:
    uv run python scripts/train_a3_emotion.py
    uv run python scripts/train_a3_emotion.py --epochs 12 --trainable-layers 6
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoFeatureExtractor,
    Trainer,
    TrainingArguments,
    Wav2Vec2ForSequenceClassification,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = ROOT / "data" / "audio_ptbr" / "labels.csv"
DEFAULT_MODEL_DIR = ROOT / "models" / "a3_emotion"

BASE_CHECKPOINT = "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese"
TARGET_SR = 16_000  # taxa nativa do modelo; wavs CORAA processados estão em 8 kHz e sobem aqui
MAX_DURATION_S = 6.0
MAX_SAMPLES = int(TARGET_SR * MAX_DURATION_S)
LABELS = ["neutral", "non_neutral"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}


def load_audio(rel_path: str) -> np.ndarray:
    y, _ = librosa.load(ROOT / rel_path, sr=TARGET_SR, mono=True)
    if len(y) > MAX_SAMPLES:
        y = y[:MAX_SAMPLES]
    return y.astype(np.float32)


def build_dataset(df: pd.DataFrame, feature_extractor: AutoFeatureExtractor) -> Dataset:
    # Mapeia label -> int no pandas: se a coluna "label" mantiver o Feature type
    # string original, o datasets.Dataset.map() abaixo converteria o int de volta
    # para string (corrupção silenciosa int->"0"/"1").
    prepared = df[["path", "label"]].copy()
    prepared["label"] = prepared["label"].map(LABEL2ID)
    ds = Dataset.from_pandas(prepared.reset_index(drop=True))

    def _prepare(batch: dict) -> dict:
        audio = load_audio(batch["path"])
        inputs = feature_extractor(audio, sampling_rate=TARGET_SR)
        batch["input_values"] = inputs["input_values"][0]
        return batch

    return ds.map(_prepare, remove_columns=["path"])


@dataclass
class DataCollator:
    feature_extractor: AutoFeatureExtractor

    def __call__(self, features: list[dict]) -> dict:
        input_values = [{"input_values": f["input_values"]} for f in features]
        labels = torch.tensor([f["label"] for f in features], dtype=torch.long)
        batch = self.feature_extractor.pad(input_values, padding=True, return_tensors="pt")
        batch["labels"] = labels
        return batch


def freeze_backbone(model: Wav2Vec2ForSequenceClassification, n_trainable_layers: int) -> None:
    model.freeze_feature_encoder()
    layers = model.wav2vec2.encoder.layers
    n_freeze = max(0, len(layers) - n_trainable_layers)
    for layer in layers[:n_freeze]:
        for p in layer.parameters():
            p.requires_grad = False
    print(
        f"[freeze] feature encoder frozen; {n_freeze}/{len(layers)} transformer "
        f"blocks frozen (top {n_trainable_layers} + classifier trainable)"
    )


class WeightedTrainer(Trainer):
    """Cross-entropy com pesos de classe — split de treino ~3,9:1 neutral:non_neutral."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss = torch.nn.functional.cross_entropy(logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"f1_macro": f1_score(labels, preds, average="macro")}


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune wav2vec2 PT-BR for A3 (sofrimento na voz)")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--base-checkpoint", type=str, default=BASE_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--trainable-layers", type=int, default=4)
    args = parser.parse_args()

    if not args.labels.exists():
        print(f"Labels not found: {args.labels}")
        print("Run scripts/download_coraa.py + scripts/preprocess_audio.py first.")
        return 1

    df = pd.read_csv(args.labels)
    train_df = df[df.split == "train"].reset_index(drop=True)
    val_df = df[df.split == "val"].reset_index(drop=True)
    test_df = df[df.split == "test"].reset_index(drop=True)
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    print(f"Loading base checkpoint: {args.base_checkpoint}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.base_checkpoint)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        args.base_checkpoint,
        num_labels=len(LABELS),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )
    freeze_backbone(model, args.trainable_layers)
    # Gradient checkpointing + feature encoder congelado faz o tensor de entrada
    # do segmento checkpointado ter requires_grad=False por padrão -> nenhum
    # gradiente flui. enable_input_require_grads() corrige isso.
    model.enable_input_require_grads()

    print("Preparing datasets (resample 8kHz -> 16kHz, extract features) ...")
    train_ds = build_dataset(train_df, feature_extractor)
    val_ds = build_dataset(val_df, feature_extractor)
    test_ds = build_dataset(test_df, feature_extractor)

    class_weights = torch.tensor(
        compute_class_weight(
            "balanced",
            classes=np.arange(len(LABELS)),
            y=train_df["label"].map(LABEL2ID).to_numpy(),
        ),
        dtype=torch.float32,
    )
    print(f"class weights (neutral, non_neutral): {class_weights.tolist()}")

    training_args = TrainingArguments(
        output_dir=str(args.model_dir / "_checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,  # troca compute por pico de RAM — CPU compartilhada, run sem supervisão
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=20,
        report_to=[],
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        seed=42,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollator(feature_extractor),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()

    print("\n=== Evaluation (held-out test speakers) ===")
    test_out = trainer.predict(test_ds)
    preds = np.argmax(test_out.predictions, axis=1)
    labels = test_out.label_ids
    print(classification_report(labels, preds, target_names=LABELS))
    f1_macro = f1_score(labels, preds, average="macro")
    print(f"F1 macro (test): {f1_macro:.4f}")

    args.model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.model_dir))
    feature_extractor.save_pretrained(str(args.model_dir))

    metrics = {
        "f1_macro_test": round(float(f1_macro), 4),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "base_checkpoint": args.base_checkpoint,
        "trainable_layers": args.trainable_layers,
    }
    metrics_path = args.model_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nModel saved -> {args.model_dir}")
    print(f"Metrics saved -> {metrics_path}")

    meta_ok = f1_macro >= 0.75
    tag = "aceito" if meta_ok else "ABAIXO da meta -- considerar baseline mais simples no go/no-go (Etapa 3)"
    print(f"\n{'OK' if meta_ok else 'FALHOU'}: A3 {tag}: F1 macro {f1_macro:.4f} (meta >= 0.75)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
