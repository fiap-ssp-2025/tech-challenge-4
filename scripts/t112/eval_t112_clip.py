#!/usr/bin/env python3
"""Avalia os checkpoints do T112 na unidade em que o V3 realmente opera: o clipe.

Motivacao: o labels.csv e por FRAME, mas o contrato do V3 recebe um video e devolve
UM score `desconforto_facial`. Medir por frame responde "acertou este quadro?"; o que
importa para a triagem e "acertou esta gravacao?". Aqui os scores dos frames de um
mesmo clipe sao promediados antes da metrica — a mesma agregacao que o infer.py faz.

Protocolo mantido: limiar escolhido na validacao, teste medido uma vez, por modelo.
Reporta frame-level e clip-level lado a lado, sem escolher o que for mais bonito.

Uso:
    python eval_t112_clip.py --ckpt results_v1/v3_fer_best.pt --data-root /workspace/t112
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader

import train_t112_fer as T

CLIP_RE = re.compile(r"_f\d+$")  # ravdess_06_calm_01-01-..._f01.jpg -> tira o "_f01"


def clip_id(path: str) -> str:
    return CLIP_RE.sub("", Path(path).stem)


def metrics_at(y: np.ndarray, scores: np.ndarray, thr: float) -> dict:
    return {
        "f1_macro": round(float(f1_score(y, (scores >= thr).astype(int), average="macro")), 4),
        "auc": round(float(roc_auc_score(y, scores)), 4),
        "n": int(len(y)),
    }


def aggregate_by_clip(df: pd.DataFrame, scores: np.ndarray, y: np.ndarray):
    agg = pd.DataFrame({"clip": [clip_id(p) for p in df["path"]], "score": scores, "y": y})
    grouped = agg.groupby("clip").agg(score=("score", "mean"), y=("y", "max"))
    return grouped["score"].to_numpy(), grouped["y"].to_numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description="T112 — avaliacao por clipe (unidade do V3)")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--drop-emotions", default="", help="emoções fora do BENCHMARK (ex.: calm)")
    ap.add_argument("--tag", default="", help="sufixo do arquivo de saída")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(args.ckpt, map_location="cpu")
    model = T.build_model(blob["arch"]).to(device)
    model.load_state_dict(blob["state_dict"])
    T.log(f"checkpoint {args.ckpt.name} arch={blob['arch']} device={device} tta={args.tta}")

    faces = args.data_root / "faces"
    df = pd.read_csv(faces / "labels.csv")
    dropped = [e.strip() for e in args.drop_emotions.split(",") if e.strip()]
    if dropped:
        before = len(df)
        df = df[~df["emotion"].isin(dropped)].reset_index(drop=True)
        T.log(f"benchmark sem {dropped}: {before} -> {len(df)} frames")
    out: dict[str, dict] = {"ckpt": str(args.ckpt), "arch": blob["arch"], "tta": bool(args.tta),
                            "benchmark_dropped": dropped}
    cache = {}

    for split in ("val", "test"):
        sub = df[df.split == split].reset_index(drop=True)
        # arch é obrigatório: cada backbone espera a imagem na escala de cor em que
        # foi treinado. Omitir aqui pontua o ViT com a normalização da EfficientNet.
        loader = DataLoader(
            T.FaceFrames(sub, faces, train=False, arch=blob["arch"]),
            batch_size=args.batch_size // 2 if blob["arch"] == "vit_fer" else args.batch_size,
            shuffle=False, num_workers=args.workers, pin_memory=device == "cuda",
        )
        scores, y = T.scores_of(model, loader, device, tta=args.tta)
        cache[split] = (sub, scores, y)

    # Limiar escolhido na validacao, por unidade (frame e clipe tem escalas distintas).
    grid = np.arange(0.02, 0.99, 0.01)
    for unit in ("frame", "clip"):
        if unit == "frame":
            v_s, v_y = cache["val"][1], cache["val"][2]
            t_s, t_y = cache["test"][1], cache["test"][2]
        else:
            v_s, v_y = aggregate_by_clip(*cache["val"])
            t_s, t_y = aggregate_by_clip(*cache["test"])
        best_t = float(grid[int(np.argmax([f1_score(v_y, (v_s >= t).astype(int), average="macro") for t in grid]))])
        out[unit] = {
            "threshold_val": round(best_t, 2),
            "val": metrics_at(v_y, v_s, best_t),
            "test_at_0.50": metrics_at(t_y, t_s, 0.50),
            "test_at_val_threshold": metrics_at(t_y, t_s, best_t),
        }
        T.log(
            f"{unit:5s} limiar_val={best_t:.2f} | TESTE F1@0.50={out[unit]['test_at_0.50']['f1_macro']:.4f} "
            f"F1@{best_t:.2f}={out[unit]['test_at_val_threshold']['f1_macro']:.4f} "
            f"AUC={out[unit]['test_at_0.50']['auc']:.4f} (n={out[unit]['test_at_0.50']['n']})"
        )

    dest = args.ckpt.parent / f"clip_metrics{args.tag}.json"
    dest.write_text(json.dumps(out, indent=2))
    T.log(f"salvo -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
