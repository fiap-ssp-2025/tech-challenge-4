#!/usr/bin/env python3
"""Download RAVDESS video subset (actors 01-08) into data/pose_posture/raw/.

Source: https://zenodo.org/record/1188976
License: Creative Commons Attribution, Non-Commercial, ShareAlike 4.0
(CC BY-NC-SA 4.0)

Only the Video_Speech_Actor_XX.zip files are downloaded (speech modality,
full video). 8 actors (~8 GB) provide enough diversity for posture training.
Increase --n-actors up to 24 for the full dataset.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data" / "pose_posture" / "raw"

ZENODO_BASE = "https://zenodo.org/record/1188976/files"


def download_actor(actor_id: int, raw_dir: Path, force: bool) -> Path:
    """Download and extract one actor zip. Returns extracted folder path."""
    name = f"Video_Speech_Actor_{actor_id:02d}.zip"
    dest_zip = raw_dir / name
    dest_dir = raw_dir / f"Actor_{actor_id:02d}"

    if dest_dir.is_dir() and any(dest_dir.glob("*.mp4")) and not force:
        print(f"[skip] Actor {actor_id:02d} already extracted")
        return dest_dir

    if not dest_zip.exists() or force:
        url = f"{ZENODO_BASE}/{name}?download=1"
        print(f"[download] {name}")
        raw_dir.mkdir(parents=True, exist_ok=True)
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(dest_zip, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=name
        ) as bar:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))

    print(f"[extract] {name}")
    with zipfile.ZipFile(dest_zip, "r") as zf:
        zf.extractall(raw_dir)

    dest_zip.unlink(missing_ok=True)
    return dest_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Download RAVDESS video subset")
    parser.add_argument(
        "--raw-dir", type=Path, default=DEFAULT_RAW,
        help="Destination folder for raw RAVDESS data",
    )
    parser.add_argument(
        "--n-actors", type=int, default=8,
        help="Number of actors to download (1-24, default 8)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if files already exist",
    )
    args = parser.parse_args()

    n = max(1, min(24, args.n_actors))
    print(f"Downloading {n} actor(s) to {args.raw_dir}")
    for i in range(1, n + 1):
        download_actor(i, args.raw_dir, args.force)

    mp4_count = len(list(args.raw_dir.rglob("*.mp4")))
    print(f"\nDone. {mp4_count} .mp4 files in {args.raw_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
