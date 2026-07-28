#!/usr/bin/env python3
"""Treina classificador de postura (defensiva vs neutra) sobre keypoints YOLOv8-pose.

Entrada:  data/pose_posture/annotations/keypoints.csv
Saída:    models/v2_posture_head.pkl  (Pipeline scikit-learn)

Features: keypoints normalizados (relativos ao centro/altura do torso) + 9 ângulos/
          distâncias geométricas das juntas do tronco superior.
Modelo:   GradientBoostingClassifier — robusto a escala, lida com não-linearidade.
Split:    por ator (sem vazamento entre train/test).

Uso:
    uv run python scripts/train_v2_posture.py
    uv run python scripts/train_v2_posture.py --test-actors 7 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "pose_posture" / "annotations" / "keypoints.csv"
DEFAULT_MODEL_DIR = ROOT / "models"

N_KP_COLS = 51  # 17 keypoints × (x, y, conf)

# Índices COCO usados na engenharia de features
_NOSE = 0
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_ELBOW,    _R_ELBOW    = 7, 8
_L_WRIST,    _R_WRIST    = 9, 10
_L_HIP,      _R_HIP      = 11, 12


def _angle_at_b(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Ângulo (radianos) no ponto b formado pelos segmentos b-a e b-c."""
    ba = a - b; bc = c - b
    cos_v = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.arccos(np.clip(cos_v, -1.0, 1.0)))


def engineer_features(raw: np.ndarray) -> np.ndarray:
    """Converte keypoints brutos (N, 51) → (N, 43) normalizados + features geométricas.

    Normalização: traduz para o centro do torso, escala pela altura do torso →
    invariante a posição e escala, para generalizar entre distâncias de câmera.
    """
    kps = raw.reshape(-1, 17, 3)           # (N, 17, 3)
    N = len(kps)
    out = np.zeros((N, 34 + 9), dtype=np.float32)

    for i, pts in enumerate(kps):
        xy   = pts[:, :2]                  # (17, 2)

        shoulder_mid = (xy[_L_SHOULDER] + xy[_R_SHOULDER]) / 2.0
        hip_mid      = (xy[_L_HIP]      + xy[_R_HIP])      / 2.0
        torso_h      = np.linalg.norm(shoulder_mid - hip_mid) + 1e-6
        torso_c      = (shoulder_mid + hip_mid) / 2.0

        xy_n = (xy - torso_c) / torso_h   # (17, 2) normalizado

        # 34 valores (x,y) normalizados
        out[i, :34] = xy_n.flatten()

        # 9 features geométricas
        head_drop      = float(xy_n[_NOSE, 1] - ((xy_n[_L_SHOULDER, 1] + xy_n[_R_SHOULDER, 1]) / 2))
        shoulder_asym  = float(xy_n[_L_SHOULDER, 1] - xy_n[_R_SHOULDER, 1])
        shoulder_width = float(abs(xy_n[_L_SHOULDER, 0] - xy_n[_R_SHOULDER, 0]))
        shoulder_elev  = float(-((xy_n[_L_SHOULDER, 1] + xy_n[_R_SHOULDER, 1]) / 2))
        l_wrist_dist   = float(np.linalg.norm(xy_n[_L_WRIST]))
        r_wrist_dist   = float(np.linalg.norm(xy_n[_R_WRIST]))
        wrist_mean_dist = (l_wrist_dist + r_wrist_dist) / 2.0
        l_elbow_angle  = _angle_at_b(xy_n[_L_SHOULDER], xy_n[_L_ELBOW], xy_n[_L_WRIST])
        r_elbow_angle  = _angle_at_b(xy_n[_R_SHOULDER], xy_n[_R_ELBOW], xy_n[_R_WRIST])

        out[i, 34:] = [
            head_drop, shoulder_asym, shoulder_width, shoulder_elev,
            l_wrist_dist, r_wrist_dist, wrist_mean_dist,
            l_elbow_angle, r_elbow_angle,
        ]

    return out


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"label", "actor"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"keypoints.csv missing columns: {missing}")
    df["actor"] = df["actor"].astype(str)
    df["label_bin"] = (df["label"] == "defensiva").astype(int)
    return df


def get_raw_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if any(
        c.endswith(f"_{s}") for s in ("x", "y", "conf")
    )][:N_KP_COLS]


def split_by_actor(
    df: pd.DataFrame, test_actors: list[str] | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    actors = sorted(df["actor"].unique())
    if test_actors:
        test_set = set(str(a) for a in test_actors)
    else:
        n_test = max(1, len(actors) // 5)
        test_set = set(actors[-n_test:])
    train_df = df[~df["actor"].isin(test_set)].copy()
    test_df  = df[ df["actor"].isin(test_set)].copy()
    print(f"Train actors: {sorted(set(actors) - test_set)}")
    print(f"Test  actors: {sorted(test_set)}")
    return train_df, test_df


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )),
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description="Train V2 posture classifier")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--test-actors", nargs="+", default=None,
        help="Actor IDs to use as test split (e.g. --test-actors 7 8)",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"Dataset not found: {args.csv}")
        print("Run scripts/extract_pose_frames.py first.")
        return 1

    print(f"Loading {args.csv} ...")
    df = load_dataset(args.csv)
    print(f"  Total rows: {len(df)}")
    print(f"  Classes:\n{df['label'].value_counts().to_string()}")
    print(f"  Actors: {sorted(df['actor'].unique())}")

    raw_cols = get_raw_cols(df)
    train_df, test_df = split_by_actor(df, args.test_actors)

    raw_train = train_df[raw_cols].fillna(0).values.astype(np.float32)
    raw_test  = test_df[raw_cols].fillna(0).values.astype(np.float32)

    print(f"\nEngineering features (normalised keypoints + angles) ...")
    X_train = engineer_features(raw_train)
    X_test  = engineer_features(raw_test)
    y_train = train_df["label_bin"].values
    y_test  = test_df["label_bin"].values

    print(f"Feature shape: {X_train.shape[1]} dims")
    print(f"Training GradientBoosting on {len(X_train)} samples ...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    print("\n=== Evaluation (test actors) ===")
    print(classification_report(y_test, y_pred, target_names=["neutra", "defensiva"]))
    print(f"F1 macro: {f1_macro:.4f}")

    # Salva o modelo
    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / "v2_posture_head.pkl"
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved → {model_path}")

    metrics = {
        "f1_macro": round(f1_macro, 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_actors": sorted(test_df["actor"].unique().tolist()),
    }
    metrics_path = args.model_dir / "v2_posture_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved → {metrics_path}")

    # V2: aceite = F1 reportado (sem threshold mínimo obrigatório)
    # O threshold F1 ≥ 0.70 é de V3 (desconforto facial).
    print(f"\n✓ V2 aceito: F1 macro {f1_macro:.4f} reportado (critério: reportar, sem mínimo)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
