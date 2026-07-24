#!/usr/bin/env python3
"""Acceptance checks for T102 (CORAA 8 kHz + labels + speaker-independent split)."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import librosa
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = ROOT / "data" / "audio_ptbr" / "labels.csv"
DEFAULT_PROCESSED = ROOT / "data" / "audio_ptbr" / "processed"
TARGET_SR = 8000
ALLOWED_LABELS = {"neutral", "non_neutral"}
ALLOWED_SPLITS = {"train", "val", "test"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify audio_ptbr dataset (T102)")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    assert args.labels.is_file(), f"Missing labels: {args.labels}"
    df = pd.read_csv(args.labels)

    required = {"path", "label", "speaker", "split"}
    assert required.issubset(df.columns), f"labels.csv schema mismatch: {df.columns.tolist()}"

    processed = sorted(args.processed_dir.rglob("*.wav"))
    assert len(processed) == len(df), (
        f"Count mismatch: {len(processed)} wavs vs {len(df)} label rows"
    )

    missing = [p for p in df["path"] if not (ROOT / p).is_file()]
    assert not missing, f"{len(missing)} paths in labels.csv missing on disk (e.g. {missing[:3]})"

    assert set(df["label"]).issubset(ALLOWED_LABELS), set(df["label"])
    assert set(df["split"]).issubset(ALLOWED_SPLITS), set(df["split"])

    # Zero speakers crossing splits.
    per_speaker = df.groupby("speaker")["split"].nunique()
    leaked = per_speaker[per_speaker > 1]
    assert leaked.empty, f"Speakers in multiple splits: {leaked.index.tolist()}"

    # Random sample: 8 kHz mono.
    rng = random.Random(args.seed)
    sample_paths = rng.sample(list(df["path"]), k=min(args.sample_size, len(df)))
    for rel in sample_paths:
        path = ROOT / rel
        info = sf.info(path)
        assert info.samplerate == TARGET_SR, f"{rel}: sr={info.samplerate}"
        assert info.channels == 1, f"{rel}: channels={info.channels}"
        y, sr = librosa.load(path, sr=None, mono=False)
        assert sr == TARGET_SR
        assert y.ndim == 1, f"{rel}: not mono after load"

    print("[ok] labels/files counts match:", len(df))
    print("[ok] no speaker crosses splits; speakers=", df["speaker"].nunique())
    print("[ok] sample of", len(sample_paths), "audios confirmed 8 kHz mono:")
    for rel in sample_paths:
        print("   ", rel)
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
