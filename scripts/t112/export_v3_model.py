#!/usr/bin/env python3
"""Converte o checkpoint do T112 (.pt) no diretorio auto-contido que o V3 carrega.

Por que: o `.pt` guarda so os pesos; reconstruir a arquitetura exigiria baixar o
checkpoint FER do Hub a cada inferencia. Salvando no formato `save_pretrained`
(config.json + model.safetensors), o `infer.py` carrega **offline**, como o A3.

Uso:
    uv run python scripts/t112/export_v3_model.py \
        --ckpt models/t112_runs/results_r3_com_calm/v3_fer_best.pt
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT = ROOT / "models" / "t112_runs" / "results_r3_com_calm" / "v3_fer_best.pt"
DEFAULT_OUT = ROOT / "models" / "v3_face"
FER_CHECKPOINT = "trpakov/vit-face-expression"
LABELS = ["neutro", "desconforto"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Exporta o V3 para diretorio HF auto-contido")
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--metrics-dir", type=Path, default=None,
                    help="dir com metrics.json / clip_metrics_*.json (default: pasta do ckpt)")
    args = ap.parse_args()

    if not args.ckpt.is_file():
        print(f"Checkpoint nao encontrado: {args.ckpt}")
        return 1

    blob = torch.load(args.ckpt, map_location="cpu")
    arch = blob["arch"]
    if arch != "vit_fer":
        print(f"Este exportador cobre apenas 'vit_fer'; checkpoint e '{arch}'.")
        return 1

    from transformers import AutoModelForImageClassification

    model = AutoModelForImageClassification.from_pretrained(
        FER_CHECKPOINT,
        num_labels=len(LABELS),
        ignore_mismatched_sizes=True,
        label2id={l: i for i, l in enumerate(LABELS)},
        id2label=dict(enumerate(LABELS)),
    )
    # O treino embrulhou o modelo (HFWrapper.inner); aqui tiramos o prefixo.
    state = {k.removeprefix("inner."): v for k, v in blob["state_dict"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[aviso] missing={list(missing)[:4]} unexpected={list(unexpected)[:4]}")

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)

    src_metrics = args.metrics_dir or args.ckpt.parent
    for name in ("metrics.json", "threshold_metrics.json",
                 "clip_metrics_bench_com_calm.json", "clip_metrics_bench_sem_calm.json"):
        f = src_metrics / name
        if f.is_file():
            shutil.copy2(f, args.out / name)

    # Preprocessamento fica junto do peso: sem isso, servir difere do treino.
    (args.out / "preprocess.json").write_text(json.dumps({
        "img_size": blob.get("img_size", 224),
        "norm_mean": [0.5, 0.5, 0.5],
        "norm_std": [0.5, 0.5, 0.5],
        "labels": LABELS,
        "positive_index": 1,
        "crop": "yolov8n person -> upper 55% + margem 0.15 (ver scripts/extract_face_frames.py)",
    }, indent=2))

    total = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file())
    print(f"[ok] V3 exportado -> {args.out}  ({total / 1e6:.0f} MB)")
    print("Arquivos:", ", ".join(sorted(f.name for f in args.out.iterdir())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
