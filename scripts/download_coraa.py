#!/usr/bin/env python3
"""Baixa CORAA SER v1.0 em data/audio_ptbr/raw/ (ignorado pelo git).

Fonte oficial: https://github.com/rmarcacini/ser-coraa-pt-br
(TaRSila / NILC — tarefa compartilhada SE&R 2022). Os IDs do Google Drive
batem com os baselines usados pela comunidade da shared-task
(ex.: alefiury/SE-R-2022-SER-Track).
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import gdown

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data" / "audio_ptbr" / "raw"

# Artefatos públicos do CORAA SER v1.0 (rmarcacini/ser-coraa-pt-br)
FILES = {
    "data_train.zip": "1N56YOgJ_plF4K8Eyh9hqiP0_O5L8uwya",
    "test_ser.zip": "1UQOi59rFbk5bVvaF7lQWFwRaae8eaOKV",
    "test_ser_metadata.csv": "1BQCkD2gKAofIsqmcVp7bB2sauqwVietz",
}


def download_file(file_id: str, dest: Path, force: bool) -> None:
    if dest.exists() and not force:
        print(f"[skip] {dest.name} already present")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {dest.name}")
    gdown.download(id=file_id, output=str(dest), quiet=False)


def extract_zip(zip_path: Path, raw_dir: Path, expected_dir: str, force: bool) -> None:
    target = raw_dir / expected_dir
    if target.is_dir() and any(target.rglob("*.wav")) and not force:
        print(f"[skip] {expected_dir}/ already extracted")
        return
    print(f"[extract] {zip_path.name} → {raw_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download CORAA SER v1.0 (PT-BR)")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW,
        help="Destination for raw archives and extractions",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-extract even if files exist",
    )
    args = parser.parse_args()
    raw_dir: Path = args.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    for name, file_id in FILES.items():
        download_file(file_id, raw_dir / name, force=args.force)

    extract_zip(raw_dir / "data_train.zip", raw_dir, "train", force=args.force)
    extract_zip(raw_dir / "test_ser.zip", raw_dir, "test_ser", force=args.force)

    n_train = len(list((raw_dir / "train").rglob("*.wav")))
    n_test = len(list((raw_dir / "test_ser").rglob("*.wav")))
    print(f"[ok] CORAA SER raw ready: train={n_train} test={n_test} → {raw_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
