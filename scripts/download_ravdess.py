#!/usr/bin/env python3
"""Download RAVDESS Video_Speech_Actor_01..24 into data/video_consulta/raw/ravdess/.

Source: https://zenodo.org/record/1188976
License: CC BY-NC-SA 4.0

Downloads each Video_Speech_Actor_XX.zip with progress + resume (skips actors
already extracted). Zips are removed after a successful extract.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data" / "video_consulta" / "raw" / "ravdess"
ZENODO_BASE = "https://zenodo.org/record/1188976/files"
# Confirmed on Zenodo record API — Speech video zips exist for actors 01-24.
ZIP_NAME = "Video_Speech_Actor_{actor:02d}.zip"


def actor_ready(raw_dir: Path, actor_id: int) -> bool:
    dest_dir = raw_dir / f"Actor_{actor_id:02d}"
    return dest_dir.is_dir() and any(dest_dir.glob("*.mp4"))


def download_zip(url: str, dest_zip: Path) -> None:
    """Stream download with tqdm; resume via HTTP Range if a partial zip exists."""
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    headers = {}
    mode = "wb"
    existing = dest_zip.stat().st_size if dest_zip.exists() else 0
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"

    with requests.get(url, stream=True, timeout=120, headers=headers) as response:
        if response.status_code == 404:
            raise FileNotFoundError(
                f"404 for {url}. Check the real filename on "
                "https://zenodo.org/record/1188976"
            )
        # 416 = already complete / invalid range → re-download from scratch
        if response.status_code == 416:
            dest_zip.unlink(missing_ok=True)
            return download_zip(url, dest_zip)
        response.raise_for_status()

        total = response.headers.get("content-length")
        total_i = int(total) + existing if total else existing
        if response.status_code == 200 and existing > 0:
            # Server ignored Range — restart
            mode = "wb"
            existing = 0
            total_i = int(total) if total else 0

        with open(dest_zip, mode) as f, tqdm(
            total=total_i or None,
            initial=existing,
            unit="B",
            unit_scale=True,
            desc=dest_zip.name,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def download_actor(actor_id: int, raw_dir: Path, force: bool) -> Path:
    dest_dir = raw_dir / f"Actor_{actor_id:02d}"
    if actor_ready(raw_dir, actor_id) and not force:
        print(f"[skip] Actor_{actor_id:02d} already extracted")
        return dest_dir

    name = ZIP_NAME.format(actor=actor_id)
    dest_zip = raw_dir / name
    url = f"{ZENODO_BASE}/{name}?download=1"

    if force and dest_dir.exists():
        for p in dest_dir.glob("*.mp4"):
            p.unlink()

    if not dest_zip.exists() or force or dest_zip.stat().st_size == 0:
        print(f"[download] {name}")
        download_zip(url, dest_zip)
    else:
        print(f"[resume-local] {name} present ({dest_zip.stat().st_size} bytes)")

    print(f"[extract] {name}")
    with zipfile.ZipFile(dest_zip, "r") as zf:
        zf.extractall(raw_dir)

    dest_zip.unlink(missing_ok=True)
    marker = raw_dir / f".extracted_{name.removesuffix('.zip')}"
    marker.write_text("ok\n")
    return dest_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Download RAVDESS speech videos (T104)")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument(
        "--n-actors",
        type=int,
        default=24,
        help="Actors 1..N (default 24 = full speech video set)",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    n = max(1, min(24, args.n_actors))
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {n} actor(s) → {args.raw_dir}")
    for i in range(1, n + 1):
        download_actor(i, args.raw_dir, args.force)

    mp4_count = len(list(args.raw_dir.rglob("*.mp4")))
    print(f"\nDone. {mp4_count} .mp4 files in {args.raw_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
