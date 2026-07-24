#!/usr/bin/env python3
"""Resample CORAA SER to 8 kHz mono, normalize amplitude, write labels.csv.

Outputs:
  data/audio_ptbr/processed/*.wav
  data/audio_ptbr/labels.csv   columns: path,label,speaker,split

Binary vocabulary: {neutral, non_neutral}
  neutral              ← neutral
  non_neutral          ← non-neutral-female | non-neutral-male

Speaker = C-ORAL-BRASIL recording id (first token of the original filename).
Split is speaker-independent (no speaker in more than one of train/val/test).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data" / "audio_ptbr" / "raw"
DEFAULT_PROCESSED = ROOT / "data" / "audio_ptbr" / "processed"
DEFAULT_LABELS = ROOT / "data" / "audio_ptbr" / "labels.csv"

TARGET_SR = 8000
LABEL_RE = re.compile(
    r"^(?P<speaker>[^_]+)_(?P<segment>segment\d+)_(?P<label>neutral|non-neutral-female|non-neutral-male)$"
)

BINARY_MAP = {
    "neutral": "neutral",
    "non-neutral-female": "non_neutral",
    "non-neutral-male": "non_neutral",
}


def parse_original_stem(stem: str) -> tuple[str, str]:
    match = LABEL_RE.match(stem)
    if not match:
        raise ValueError(f"Unexpected CORAA filename stem: {stem}")
    return match.group("speaker"), BINARY_MAP[match.group("label")]


def collect_raw_rows(raw_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    train_dir = raw_dir / "train"
    for wav in sorted(train_dir.rglob("*.wav")):
        speaker, label = parse_original_stem(wav.stem)
        rows.append(
            {
                "src": str(wav),
                "out_name": wav.name,
                "label": label,
                "speaker": speaker,
            }
        )

    meta_path = raw_dir / "test_ser_metadata.csv"
    meta = pd.read_csv(meta_path)
    test_dir = raw_dir / "test_ser"
    for _, item in meta.iterrows():
        original = Path(str(item["file"]))
        hashed = Path(str(item["wav_file"]))
        src = test_dir / hashed.name
        if not src.is_file():
            # zip may nest under test_ser/
            candidates = list(test_dir.rglob(hashed.name))
            if not candidates:
                raise FileNotFoundError(f"Missing test wav: {hashed.name}")
            src = candidates[0]
        speaker, label = parse_original_stem(original.stem)
        # Prefer original CORAA name so path encodes speaker/label.
        rows.append(
            {
                "src": str(src),
                "out_name": original.name,
                "label": label,
                "speaker": speaker,
            }
        )

    return rows


def normalize_peak(y: np.ndarray, peak: float = 0.99) -> np.ndarray:
    max_abs = float(np.max(np.abs(y))) if y.size else 0.0
    if max_abs < 1e-9:
        return y.astype(np.float32)
    return (y * (peak / max_abs)).astype(np.float32)


def process_one(src: Path, dest: Path) -> None:
    y, _ = librosa.load(src, sr=TARGET_SR, mono=True)
    y = normalize_peak(y)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dest, y, TARGET_SR, subtype="PCM_16")


def speaker_independent_split(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> pd.Series:
    """Assign train/val/test so each speaker appears in exactly one split."""
    speakers = df[["speaker", "label"]].drop_duplicates(subset=["speaker"]).copy()
    # Stratify on whether the speaker has any non_neutral segment.
    has_nn = (
        df.assign(is_nn=(df["label"] == "non_neutral").astype(int))
        .groupby("speaker")["is_nn"]
        .max()
    )
    speakers["strata"] = speakers["speaker"].map(has_nn).astype(int)
    speaker_ids = speakers["speaker"].to_numpy()
    strata = speakers["strata"].to_numpy()

    # First peel off test, then peel val from the remainder.
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(gss_test.split(speaker_ids, strata, groups=speaker_ids))

    remaining = speaker_ids[train_val_idx]
    remaining_strata = strata[train_val_idx]
    # val_size relative to the full set → rescale against remaining
    val_rel = val_size / (1.0 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_rel, random_state=seed)
    train_idx_rel, val_idx_rel = next(
        gss_val.split(remaining, remaining_strata, groups=remaining)
    )

    split_of_speaker: dict[str, str] = {}
    for spk in remaining[train_idx_rel]:
        split_of_speaker[str(spk)] = "train"
    for spk in remaining[val_idx_rel]:
        split_of_speaker[str(spk)] = "val"
    for spk in speaker_ids[test_idx]:
        split_of_speaker[str(spk)] = "test"

    return df["speaker"].map(split_of_speaker)


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess CORAA SER to 8 kHz mono")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = collect_raw_rows(args.raw_dir)
    if not rows:
        raise SystemExit(f"No raw CORAA wavs under {args.raw_dir}. Run download_coraa.py first.")

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []

    for row in tqdm(rows, desc="preprocess"):
        dest = args.processed_dir / row["out_name"]
        process_one(Path(row["src"]), dest)
        rel_path = dest.relative_to(ROOT).as_posix()
        records.append(
            {
                "path": rel_path,
                "label": row["label"],
                "speaker": row["speaker"],
            }
        )

    labels = pd.DataFrame(records)
    labels["split"] = speaker_independent_split(labels, seed=args.seed)

    # Hard guarantee: no speaker crosses splits.
    crossed = labels.groupby("speaker")["split"].nunique()
    bad = crossed[crossed > 1]
    if len(bad):
        raise AssertionError(f"Speaker leakage across splits: {bad.index.tolist()}")

    labels = labels.sort_values(["split", "speaker", "path"]).reset_index(drop=True)
    args.labels.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(args.labels, index=False)

    print(f"[ok] wrote {len(labels)} rows → {args.labels}")
    print(labels.groupby(["split", "label"]).size().unstack(fill_value=0))
    print("speakers per split:")
    print(labels.groupby("split")["speaker"].nunique())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
