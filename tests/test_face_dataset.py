"""Unit tests for T104 face dataset helpers (no heavy downloads)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.video.v3_face.face_dataset import (
    ActorInfo,
    assert_minority_ratio,
    assert_no_actor_leakage,
    assign_actor_splits,
    balance_binary_labels,
    emotion_to_binary,
    parse_cremad_filename,
    parse_ravdess_filename,
)


def test_parse_ravdess_emotion_and_sex():
    meta = parse_ravdess_filename("01-01-06-01-02-01-12.mp4")
    assert meta is not None
    assert meta["emotion"] == "fearful"
    assert meta["actor"] == "ravdess_12"
    assert meta["sex"] == "F"  # even = female
    assert meta["dataset"] == "ravdess"

    male = parse_ravdess_filename("01-01-01-01-01-01-01.mp4")
    assert male is not None
    assert male["emotion"] == "neutral"
    assert male["sex"] == "M"


def test_parse_cremad():
    meta = parse_cremad_filename("1002_DFA_FEA_XX.flv")
    assert meta is not None
    assert meta["emotion"] == "fearful"
    assert meta["actor"] == "crema_1002"
    assert meta["dataset"] == "cremad"


def test_binary_mapping_includes_calm_as_neutro():
    assert emotion_to_binary("fearful") == "desconforto"
    assert emotion_to_binary("sad") == "desconforto"
    assert emotion_to_binary("neutral") == "neutro"
    assert emotion_to_binary("calm") == "neutro"
    assert emotion_to_binary("angry") is None


def test_actor_split_no_leakage_and_covers_all():
    actors = [
        ActorInfo(f"ravdess_{i:02d}", "F" if i % 2 == 0 else "M", "ravdess")
        for i in range(1, 25)
    ]
    mapping = assign_actor_splits(actors, seed=0)
    assert len(mapping) == 24
    assert set(mapping.values()).issubset({"train", "val", "test"})
    # Build fake frame table
    rows = []
    for actor, split in mapping.items():
        for _ in range(5):
            rows.append({"actor": actor, "split": split, "label": "neutro"})
    df = pd.DataFrame(rows)
    assert_no_actor_leakage(df)


def test_minority_ratio_gate():
    ok = pd.DataFrame({"label": ["desconforto"] * 45 + ["neutro"] * 55})
    assert assert_minority_ratio(ok, 0.40) >= 0.40
    bad = pd.DataFrame({"label": ["desconforto"] * 90 + ["neutro"] * 10})
    with pytest.raises(AssertionError):
        assert_minority_ratio(bad, 0.40)


def test_balance_binary_labels_undersamples_majority():
    df = pd.DataFrame({"label": ["desconforto"] * 90 + ["neutro"] * 10, "actor": ["a"] * 100})
    with pytest.raises(AssertionError):
        assert_minority_ratio(df, 0.40)
    balanced = balance_binary_labels(df, min_ratio=0.40, seed=0)
    assert assert_minority_ratio(balanced, 0.40) >= 0.40
    assert (balanced["label"] == "neutro").sum() == 10


def test_labels_schema_fixture(tmp_path: Path):
    """Synthetic labels.csv + jpgs satisfy verify invariants."""
    faces = tmp_path / "faces"
    faces.mkdir()
    actors = ["ravdess_02", "ravdess_04", "ravdess_06", "ravdess_08", "ravdess_10"]
    split_map = assign_actor_splits(
        [ActorInfo(a, "F", "ravdess") for a in actors], seed=1
    )
    rows = []
    i = 0
    for actor in actors:
        for label, emotion in (("desconforto", "fearful"), ("neutro", "neutral")):
            for _ in range(5):
                name = f"{actor}_{label}_{i}.jpg"
                (faces / name).write_bytes(b"\xff\xd8\xff")  # minimal jpeg header-ish
                rows.append(
                    {
                        "path": str(faces / name),
                        "emotion": emotion,
                        "actor": actor,
                        "dataset": "ravdess",
                        "label": label,
                        "split": split_map[actor],
                        "sex": "F",
                    }
                )
                i += 1
    df = pd.DataFrame(rows)
    assert_no_actor_leakage(df)
    assert assert_minority_ratio(df, 0.40) >= 0.40
    assert len(list(faces.glob("*.jpg"))) == len(df)
