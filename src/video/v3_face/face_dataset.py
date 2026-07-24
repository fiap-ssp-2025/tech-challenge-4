"""T104 helpers: emotion parsing, binary labels, actor-wise split (no training)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import pandas as pd

# Binary target for V3 (desconforto facial).
DISCOMFORT_EMOTIONS = frozenset({"fearful", "sad"})
NEUTRAL_EMOTIONS = frozenset({"neutral", "calm"})  # calm→neutro for class balance (≥40%)

RAVDESS_EMOTION = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

CREMAD_EMOTION = {
    "NEU": "neutral",
    "FEA": "fearful",
    "SAD": "sad",
    "ANG": "angry",
    "DIS": "disgust",
    "HAP": "happy",
}

ALLOWED_LABELS = frozenset({"desconforto", "neutro"})
ALLOWED_SPLITS = frozenset({"train", "val", "test"})
LABELS_COLUMNS = ("path", "emotion", "actor", "dataset", "label", "split", "sex")


@dataclass(frozen=True)
class ActorInfo:
    actor_id: str  # e.g. "ravdess_02"
    sex: str  # "F" | "M" | "U"
    dataset: str


def emotion_to_binary(emotion: str) -> str | None:
    """Map fine emotion → {desconforto, neutro}; None if out of scope."""
    if emotion in DISCOMFORT_EMOTIONS:
        return "desconforto"
    if emotion in NEUTRAL_EMOTIONS:
        return "neutro"
    return None


def parse_ravdess_filename(name: str) -> dict[str, str] | None:
    """Parse RAVDESS stem: modality-channel-emotion-intensity-statement-rep-actor."""
    stem = name.rsplit(".", 1)[0]
    parts = stem.split("-")
    if len(parts) != 7:
        return None
    emotion = RAVDESS_EMOTION.get(parts[2])
    if emotion is None:
        return None
    actor_num = parts[6]
    sex = "F" if int(actor_num) % 2 == 0 else "M"
    return {
        "emotion": emotion,
        "actor": f"ravdess_{actor_num}",
        "dataset": "ravdess",
        "sex": sex,
        "modality": parts[0],
    }


def parse_cremad_filename(name: str) -> dict[str, str] | None:
    """Parse CREMA-D stem: ActorID_Sentence_Emotion_Intensity."""
    stem = name.rsplit(".", 1)[0]
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    code = parts[2].upper()
    emotion = CREMAD_EMOTION.get(code)
    if emotion is None:
        return None
    actor_num = parts[0]
    return {
        "emotion": emotion,
        "actor": f"crema_{actor_num}",
        "dataset": "cremad",
        "sex": "U",  # filled from demographics when available
        "modality": "video",
    }


def load_cremad_sex_map(demographics_csv) -> dict[str, str]:
    """ActorID → 'F'|'M' from VideoDemographics.csv."""
    df = pd.read_csv(demographics_csv)
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        aid = str(int(row["ActorID"])) if not isinstance(row["ActorID"], str) else str(row["ActorID"])
        sex_raw = str(row["Sex"]).strip().lower()
        out[aid] = "F" if sex_raw.startswith("f") else "M"
    return out


def assign_actor_splits(
    actors: list[ActorInfo],
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, str]:
    """Assign each actor to exactly one split; prefer female balance across splits.

    Strategy: shuffle within sex strata, then round-robin into train/val/test buckets
    sized by ratios (female first, then male/unknown).
    """
    import random

    rng = random.Random(seed)
    by_sex: dict[str, list[ActorInfo]] = defaultdict(list)
    for a in actors:
        by_sex[a.sex].append(a)
    for sex in by_sex:
        rng.shuffle(by_sex[sex])

    ordered: list[ActorInfo] = []
    # Female first (plan: prioritize female representation), then M, then U.
    for sex in ("F", "M", "U"):
        ordered.extend(by_sex.get(sex, []))

    n = len(ordered)
    if n == 0:
        return {}
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(0, int(round(n * val_ratio)))
    if n_train + n_val >= n:
        n_val = max(0, n - n_train - (1 if n - n_train > 1 else 0))
    n_test = n - n_train - n_val

    # Distribute round-robin so each split gets females early.
    buckets = {
        "train": [],
        "val": [],
        "test": [],
    }
    caps = {"train": n_train, "val": n_val, "test": n_test}
    cycle = [s for s in ("train", "val", "test") if caps[s] > 0]
    idx = 0
    for actor in ordered:
        # Find next bucket with capacity
        placed = False
        for _ in range(len(cycle)):
            split = cycle[idx % len(cycle)]
            idx += 1
            if len(buckets[split]) < caps[split]:
                buckets[split].append(actor)
                placed = True
                break
        if not placed:
            # Overflow → train
            buckets["train"].append(actor)

    mapping: dict[str, str] = {}
    for split, members in buckets.items():
        for a in members:
            mapping[a.actor_id] = split
    return mapping


def assert_no_actor_leakage(df: pd.DataFrame) -> None:
    per = df.groupby("actor")["split"].nunique()
    leaked = per[per > 1]
    assert leaked.empty, f"Actors in multiple splits: {leaked.index.tolist()}"


def assert_minority_ratio(df: pd.DataFrame, min_ratio: float = 0.40) -> float:
    counts = df["label"].value_counts(normalize=True)
    minority = float(counts.min()) if len(counts) else 0.0
    assert set(df["label"]).issubset(ALLOWED_LABELS), set(df["label"])
    assert minority >= min_ratio, (
        f"Minority class ratio {minority:.3f} < {min_ratio} (counts={df['label'].value_counts().to_dict()})"
    )
    return minority


def assert_frames_match_labels(df: pd.DataFrame, faces_dir, root=None) -> None:
    from pathlib import Path

    faces_dir = Path(faces_dir)
    root = Path(root) if root is not None else Path.cwd()
    missing = []
    for rel in df["path"]:
        p = Path(rel)
        candidates = [p, root / p, faces_dir / p.name]
        if not any(c.is_file() for c in candidates):
            missing.append(rel)
    assert not missing, f"{len(missing)} label paths missing on disk (e.g. {missing[:3]})"
    n_jpg = len(list(faces_dir.rglob("*.jpg")))
    assert n_jpg == len(df), f"Count mismatch: {n_jpg} jpgs vs {len(df)} label rows"
