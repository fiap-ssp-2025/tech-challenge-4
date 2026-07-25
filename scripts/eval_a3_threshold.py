#!/usr/bin/env python3
"""Calibra o limiar de decisão do A3 no split de validação e reporta no teste.

Motivação: `train_a3_emotion.py` reporta F1 macro com o argmax do softmax, ou seja,
limiar fixo em 0,50. Com 20% de positivos e treino com pesos de classe, 0,50 não é o
ponto de operação certo — o modelo fica preciso e pouco sensível (recall 0,39), que é
o erro mais caro numa triagem de socorro.

Este script escolhe o limiar **na validação** e só então mede no teste, para o número
reportado não ter vazamento. O contrato A3 devolve `sofrimento` como score contínuo
[0..1], então o limiar não é aplicado aqui: ele é informação para a camada de fusão
(T114) e para o go/no-go (T120).

A pontuação delega em `a3_emotion.infer()`, de propósito: o limiar precisa ser calibrado
sobre exatamente o mesmo regime de inferência que roda em produção (janelas de 6 s com
média). Reimplementar a lógica aqui a faria divergir em silêncio.

Uso:
    uv run python scripts/eval_a3_threshold.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = ROOT / "data" / "audio_ptbr" / "labels.csv"
DEFAULT_MODEL_DIR = ROOT / "models" / "a3_emotion"
LABEL2ID = {"neutral": 0, "non_neutral": 1}


def score_split(df: pd.DataFrame, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Pontua um split chamando o `infer()` real.

    Delegar em vez de reimplementar é deliberado: o A3 pontua em janelas de 6 s com
    média, e uma cópia local dessa lógica sairia de sincronia com `infer.py` sem
    ninguém perceber — calibrando o limiar num regime e servindo noutro.
    """
    from src.audio.a3_emotion.infer import infer

    sub = df[df.split == split].reset_index(drop=True)
    scores, labels = [], []
    for _, row in sub.iterrows():
        scores.append(float(infer(ROOT / row["path"])["sofrimento"]))
        labels.append(LABEL2ID[row["label"]])
    return np.array(scores), np.array(labels)


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibra limiar do A3 na validação, reporta no teste")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()

    if not args.model_dir.exists():
        print(f"Modelo não encontrado: {args.model_dir}")
        print("Rode antes: uv run python scripts/train_a3_emotion.py")
        return 1

    df = pd.read_csv(args.labels)

    val_scores, val_y = score_split(df, "val")
    grid = np.arange(0.02, 0.99, 0.01)
    val_f1 = [f1_score(val_y, (val_scores >= t).astype(int), average="macro") for t in grid]
    best_t = float(grid[int(np.argmax(val_f1))])
    print(f"VAL   n={len(val_y)} non_neutral={int(val_y.sum())} AUC={roc_auc_score(val_y, val_scores):.4f}")
    print(f"VAL   limiar escolhido = {best_t:.2f} (F1 macro val = {max(val_f1):.4f})")

    test_scores, test_y = score_split(df, "test")
    print(f"\nTESTE n={len(test_y)} non_neutral={int(test_y.sum())} AUC={roc_auc_score(test_y, test_scores):.4f}")

    results = {}
    for name, t in [("default_0.50", 0.50), (f"val_calibrado_{best_t:.2f}", best_t)]:
        pred = (test_scores >= t).astype(int)
        f1m = f1_score(test_y, pred, average="macro")
        results[name] = round(float(f1m), 4)
        print(f"\n=== TESTE, limiar {t:.2f} -> F1 macro {f1m:.4f} ===")
        print(classification_report(test_y, pred, target_names=list(LABEL2ID), digits=2))

    out = {
        "threshold_val_selected": round(best_t, 2),
        "auc_val": round(float(roc_auc_score(val_y, val_scores)), 4),
        "auc_test": round(float(roc_auc_score(test_y, test_scores)), 4),
        "f1_macro_test": results,
    }
    path = args.model_dir / "threshold_metrics.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nSalvo -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
