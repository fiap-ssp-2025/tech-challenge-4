#!/usr/bin/env python3
"""Acceptance checks for T104 (face frames RAVDESS/CREMA-D)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.video.v3_face.face_dataset import (
    ALLOWED_LABELS,
    ALLOWED_SPLITS,
    assert_frames_match_labels,
    assert_minority_ratio,
    assert_no_actor_leakage,
)
DEFAULT_LABELS = ROOT / "data" / "video_consulta" / "processed" / "faces" / "labels.csv"
DEFAULT_FACES = ROOT / "data" / "video_consulta" / "processed" / "faces"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify face_frames dataset (T104)")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--faces-dir", type=Path, default=DEFAULT_FACES)
    parser.add_argument("--min-minority", type=float, default=0.40)
    args = parser.parse_args()

    assert args.labels.is_file(), f"Missing labels: {args.labels}"
    df = pd.read_csv(args.labels)
    required = {"path", "emotion", "actor", "dataset", "label", "split"}
    assert required.issubset(df.columns), df.columns.tolist()

    assert set(df["label"]).issubset(ALLOWED_LABELS), set(df["label"])
    assert set(df["split"]).issubset(ALLOWED_SPLITS), set(df["split"])
    assert set(df["dataset"]).issubset({"ravdess", "cremad"}), set(df["dataset"])

    assert_frames_match_labels(df, args.faces_dir, root=ROOT)
    assert_no_actor_leakage(df)
    minority = assert_minority_ratio(df, args.min_minority)

    print("[ok] frames × labels match")
    print("[ok] zero actors crossing splits")
    print(f"[ok] minority ratio = {minority:.3f} ≥ {args.min_minority}")
    print(df["label"].value_counts().to_string())
    print(df.groupby("split")["actor"].nunique().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
