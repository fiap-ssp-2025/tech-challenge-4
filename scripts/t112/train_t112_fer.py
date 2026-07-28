#!/usr/bin/env python3
"""T112 — ajuste fino FER binário (desconforto vs neutro) sobre frames RAVDESS/CREMA-D.

Protocolo herdado do T110: treino no split `train`, selecao de checkpoint e limiar no
`val`, teste avaliado UMA vez para o vencedor da varredura. Split por ator vem pronto
do labels.csv do T104 (nao e refeito aqui).

Saidas em --out:
  results.jsonl            uma linha por epoca/por run
  metrics.json             melhor run + F1 macro teste (limiar 0,50)
  threshold_metrics.json   limiar calibrado na validacao + F1/AUC no teste
  v3_fer_best.pt           {state_dict, arch, label2id, img_size} do vencedor

Uso (pod RunPod, GPU):
    python train_t112_fer.py --data-root /workspace/t112 --out /workspace/t112/results
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

LABELS = ["neutro", "desconforto"]  # índice 1 = classe de interesse (contrato V3)
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
IMG_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]  # ImageNet
NORM_STD = [0.229, 0.224, 0.225]
VIT_NORM_MEAN = [0.5, 0.5, 0.5]  # ViT/FER checkpoints usam normalizacao simetrica
VIT_NORM_STD = [0.5, 0.5, 0.5]

# Checkpoint ja treinado em EXPRESSAO FACIAL (FER2013), nao em objetos genericos:
# a rede chega sabendo o que e um rosto e o que e uma expressao, e so reaprende a
# fronteira desconforto x neutro. E a alavanca do item 1 da Rota B.
FER_CHECKPOINT = "trpakov/vit-face-expression"

SWEEPS = {
    # Rodada 1: qual arquitetura/lr aprende a tarefa.
    "v1": [
        {"arch": "resnet18", "lr": 3e-4},
        {"arch": "resnet18", "lr": 1e-4},
        {"arch": "efficientnet_b0", "lr": 3e-4},
        {"arch": "efficientnet_b0", "lr": 1e-4},
    ],
    # Rodada 2: o v1 decorou identidades (loss de treino despenca, val cai após ~ep5).
    # Aqui todas as configs atacam isso — augment forte, label smoothing e congelamento.
    "v2": [
        {"arch": "efficientnet_b0", "lr": 3e-4, "aug": "strong", "smooth": 0.1},
        {"arch": "efficientnet_b0", "lr": 1e-4, "aug": "strong", "smooth": 0.1, "freeze": 4},
        {"arch": "efficientnet_b0", "lr": 3e-4, "aug": "strong", "smooth": 0.2, "freeze": 2},
        {"arch": "resnet18", "lr": 1e-4, "aug": "strong", "smooth": 0.1, "freeze": 5},
    ],
    # Rodada 3 (Rota B): backbone pre-treinado em FER + base sem `calm`.
    # O CONTROLE existe de proposito: e a config vencedora da rodada 1, inalterada,
    # rodando na base limpa. Sem ele nao da para saber se o ganho veio do rotulo
    # (item 2) ou do backbone (item 1) — mudariamos duas variaveis de uma vez.
    "v3": [
        {"arch": "efficientnet_b0", "lr": 3e-4, "note": "CONTROLE — receita do round 1"},
        {"arch": "vit_fer", "lr": 3e-5, "note": "backbone ja treinado em expressao facial"},
    ],
}

BASE_AUG = [
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
]
# Recorte mais agressivo + TrivialAugment + apagamento: força o modelo a usar a
# expressão e não a identidade do ator (que é o que o split por ator penaliza).
STRONG_AUG = [
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.TrivialAugmentWide(),
]


def norm_of(arch: str) -> tuple[list[float], list[float]]:
    """Cada backbone espera as cores na escala em que foi treinado."""
    return (VIT_NORM_MEAN, VIT_NORM_STD) if arch == "vit_fer" else (NORM_MEAN, NORM_STD)


def train_tf(kind: str, arch: str = "efficientnet_b0") -> transforms.Compose:
    steps = (STRONG_AUG if kind == "strong" else BASE_AUG) + [
        transforms.ToTensor(),
        transforms.Normalize(*norm_of(arch)),
    ]
    if kind == "strong":
        steps.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)))
    return transforms.Compose(steps)


def eval_tf(arch: str = "efficientnet_b0") -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(*norm_of(arch)),
    ])
EVAL_TF = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ]
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_label(cfg: dict) -> str:
    extras = "".join(
        f" {k}={cfg[k]}" for k in ("aug", "smooth", "freeze") if cfg.get(k)
    )
    return f"{cfg['arch']}@{cfg['lr']}{extras}"


class FaceFrames(Dataset):
    """Frames faciais do labels.csv; caminho resolvido pelo basename sob faces/."""

    def __init__(self, df: pd.DataFrame, faces_dir: Path, train: bool, aug: str = "base",
                 arch: str = "efficientnet_b0"):
        self.paths = [faces_dir / Path(p).name for p in df["path"]]
        self.labels = df["label"].map(LABEL2ID).to_numpy()
        self.tf = train_tf(aug, arch) if train else eval_tf(arch)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tf(img), int(self.labels[idx])


class HFWrapper(torch.nn.Module):
    """Deixa um classificador da transformers com a mesma cara de um modelo torchvision:
    entra o tensor de imagem, sai o tensor de logits."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(pixel_values=x).logits


def build_model(arch: str) -> torch.nn.Module:
    if arch == "vit_fer":
        from transformers import AutoModelForImageClassification

        inner = AutoModelForImageClassification.from_pretrained(
            FER_CHECKPOINT,
            num_labels=len(LABELS),
            ignore_mismatched_sizes=True,  # troca a cabeca de 7 emocoes pela binaria
        )
        return HFWrapper(inner)
    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = torch.nn.Linear(m.fc.in_features, len(LABELS))
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, len(LABELS))
    else:
        raise ValueError(f"arch desconhecida: {arch}")
    return m


def freeze_blocks(model: torch.nn.Module, arch: str, n: int) -> None:
    """Congela os n primeiros blocos: menos parâmetros livres = menos identidade decorada."""
    if n <= 0:
        return
    if arch == "vit_fer":
        # O caminho das camadas mudou entre versões da transformers
        # (vit.encoder.layer -> vit.layers); pega a maior ModuleList e não chuta.
        lists = [m for m in model.modules() if isinstance(m, torch.nn.ModuleList)]
        blocks = max(lists, key=len)[:n]
    elif arch == "efficientnet_b0":
        blocks = model.features[:n]
    else:
        blocks = list(model.children())[:n]
    for block in blocks:
        for p in block.parameters():
            p.requires_grad = False


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def scores_of(model, loader, device, tta: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Prob. da classe `desconforto` + labels, na ordem do loader.

    tta=True calcula também a imagem espelhada e tira a média — rosto é simétrico,
    então o espelho é uma vista legítima e reduz a variância da predição.
    """
    model.eval()
    scores, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        views = [x, torch.flip(x, dims=[3])] if tta else [x]
        probs = []
        for view in views:
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(view)
            probs.append(torch.softmax(logits.float(), dim=1)[:, 1])
        scores.append(torch.stack(probs).mean(0).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(scores), np.concatenate(labels)


def run_one(cfg, loaders, class_weights, device, epochs, patience, jsonl, make_loaders=None) -> dict:
    # Augment e normalização variam por config (ViT usa escala de cor própria),
    # então os loaders são remontados a cada run.
    if make_loaders is not None:
        loaders = make_loaders(cfg)
    seed_everything(cfg.get("seed", 42))
    model = build_model(cfg["arch"]).to(device)
    freeze_blocks(model, cfg["arch"], cfg.get("freeze", 0))
    smooth = cfg.get("smooth", 0.0)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg["lr"], weight_decay=1e-4
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    weight = class_weights.to(device)

    best = {"val_f1": -1.0, "epoch": -1, "state": None}
    bad_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0, seen, loss_sum = time.time(), 0, 0.0
        for x, y in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = F.cross_entropy(model(x), y, weight=weight, label_smoothing=smooth)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss) * len(y)
            seen += len(y)
        sched.step()

        val_scores, val_y = scores_of(model, loaders["val"], device)
        val_f1 = f1_score(val_y, (val_scores >= 0.5).astype(int), average="macro")
        row = {
            "run": run_label(cfg),
            "epoch": epoch,
            "train_loss": round(loss_sum / seen, 4),
            "val_f1_macro": round(float(val_f1), 4),
            "secs": round(time.time() - t0, 1),
        }
        jsonl.write(json.dumps(row) + "\n")
        jsonl.flush()
        log(f"{row['run']} ep{epoch:02d} loss={row['train_loss']} val_f1={row['val_f1_macro']} ({row['secs']}s)")

        if val_f1 > best["val_f1"]:
            best = {"val_f1": float(val_f1), "epoch": epoch,
                    "state": {k: v.cpu() for k, v in model.state_dict().items()}}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log(f"{row['run']}: early stop na epoca {epoch} (melhor: {best['val_f1']:.4f} @ep{best['epoch']})")
                break

    return {"cfg": cfg, **{k: best[k] for k in ("val_f1", "epoch")}, "state": best["state"]}


def main() -> int:
    parser = argparse.ArgumentParser(description="T112 — fine-tune FER desconforto facial (V3)")
    parser.add_argument("--data-root", type=Path, required=True, help="dir que contem faces/")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sweep", choices=sorted(SWEEPS), default="v1")
    parser.add_argument("--tta", action="store_true", help="média com o espelho na avaliação final")
    parser.add_argument(
        "--drop-emotions", default="",
        help="emoções a excluir, separadas por vírgula (ex.: calm). O split por ator é preservado.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device} torch={torch.__version__}")

    faces_dir = args.data_root / "faces"
    df = pd.read_csv(faces_dir / "labels.csv")

    dropped = [e.strip() for e in args.drop_emotions.split(",") if e.strip()]
    if dropped:
        before = len(df)
        df = df[~df["emotion"].isin(dropped)].reset_index(drop=True)
        log(f"emoções removidas {dropped}: {before} -> {len(df)} frames "
            f"({before - len(df)} fora); split por ator preservado")

    splits = {s: df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test")}
    log("split sizes: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    def make_loader(sub: pd.DataFrame, *, train: bool, aug: str = "base",
                    arch: str = "efficientnet_b0") -> DataLoader:
        # ViT-base ocupa mais memória por amostra que a EfficientNet — meio lote nele.
        bs = args.batch_size // 2 if arch == "vit_fer" else args.batch_size
        return DataLoader(
            FaceFrames(sub, faces_dir, train=train, aug=aug, arch=arch),
            batch_size=bs, shuffle=train,
            num_workers=args.workers, pin_memory=device == "cuda", drop_last=False,
        )

    def make_loaders(cfg: dict) -> dict[str, DataLoader]:
        arch, aug = cfg["arch"], cfg.get("aug", "base")
        return {
            name: make_loader(sub, train=name == "train",
                              aug=aug if name == "train" else "base", arch=arch)
            for name, sub in splits.items()
        }

    loaders = {name: make_loader(sub, train=name == "train") for name, sub in splits.items()}

    class_weights = torch.tensor(
        compute_class_weight(
            "balanced", classes=np.arange(len(LABELS)),
            y=splits["train"]["label"].map(LABEL2ID).to_numpy(),
        ),
        dtype=torch.float32,
    )
    log(f"class weights (neutro, desconforto): {class_weights.tolist()}")

    args.out.mkdir(parents=True, exist_ok=True)
    runs = []
    with open(args.out / "results.jsonl", "a") as jsonl:
        for cfg in SWEEPS[args.sweep]:
            try:
                runs.append(run_one(cfg, loaders, class_weights, device,
                                    args.epochs, args.patience, jsonl, make_loaders))
            except Exception as exc:  # segue a varredura; run com erro nao concorre
                log(f"RUN FALHOU {cfg}: {exc.__class__.__name__}: {exc}")

    if not runs:
        log("nenhuma run terminou — abortando")
        return 1

    winner = max(runs, key=lambda r: r["val_f1"])
    cfg = winner["cfg"]
    log(f"vencedor: {run_label(cfg)} val_f1={winner['val_f1']:.4f} ep{winner['epoch']}")

    model = build_model(cfg["arch"]).to(device)
    model.load_state_dict(winner["state"])
    loaders = make_loaders(cfg)  # normalização/lote do backbone vencedor

    # Limiar na validacao (protocolo T110), teste avaliado uma unica vez.
    val_scores, val_y = scores_of(model, loaders["val"], device, tta=args.tta)
    grid = np.arange(0.02, 0.99, 0.01)
    val_f1_grid = [f1_score(val_y, (val_scores >= t).astype(int), average="macro") for t in grid]
    best_t = float(grid[int(np.argmax(val_f1_grid))])
    log(f"limiar calibrado na val: {best_t:.2f} (F1 val {max(val_f1_grid):.4f}, AUC val {roc_auc_score(val_y, val_scores):.4f})")

    test_scores, test_y = scores_of(model, loaders["test"], device, tta=args.tta)
    f1_default = f1_score(test_y, (test_scores >= 0.5).astype(int), average="macro")
    f1_cal = f1_score(test_y, (test_scores >= best_t).astype(int), average="macro")
    auc_test = roc_auc_score(test_y, test_scores)
    log(f"TESTE: F1@0.50={f1_default:.4f}  F1@{best_t:.2f}={f1_cal:.4f}  AUC={auc_test:.4f}")
    print(classification_report(test_y, (test_scores >= 0.5).astype(int), target_names=LABELS))

    (args.out / "metrics.json").write_text(json.dumps({
        "f1_macro_test": round(float(f1_default), 4),
        "arch": cfg["arch"], "lr": cfg["lr"], "best_epoch": winner["epoch"],
        "sweep": args.sweep, "tta": bool(args.tta), "config": {k: v for k, v in cfg.items()},
        "dropped_emotions": dropped,
        "val_f1_macro": round(winner["val_f1"], 4),
        "n_train": len(splits["train"]), "n_val": len(splits["val"]), "n_test": len(splits["test"]),
        "runs": [{"run": run_label(r["cfg"]), "val_f1": round(r["val_f1"], 4)} for r in runs],
    }, indent=2))
    (args.out / "threshold_metrics.json").write_text(json.dumps({
        "threshold_val_selected": round(best_t, 2),
        "auc_val": round(float(roc_auc_score(val_y, val_scores)), 4),
        "auc_test": round(float(auc_test), 4),
        "f1_macro_test": {"default_0.50": round(float(f1_default), 4),
                          f"val_calibrado_{best_t:.2f}": round(float(f1_cal), 4)},
    }, indent=2))
    torch.save({"state_dict": winner["state"], "arch": cfg["arch"],
                "label2id": LABEL2ID, "img_size": IMG_SIZE}, args.out / "v3_fer_best.pt")

    meta_ok = max(f1_default, f1_cal) >= 0.70
    log(("OK meta atingida" if meta_ok else "ABAIXO da meta — levar ao go/no-go (RNF-04)")
        + f": melhor F1 macro teste {max(f1_default, f1_cal):.4f} (meta >= 0.70)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
