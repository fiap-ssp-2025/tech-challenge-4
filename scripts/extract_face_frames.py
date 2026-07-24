#!/usr/bin/env python3
"""Extract face crops from RAVDESS/CREMA-D videos for V3 FER (T104 — data only).

Pipeline:
  1. Walk raw videos under data/video_consulta/raw/{ravdess,cremad}/.
  2. Keep emotions in {neutral, calm, fearful, sad} (binary desconforto|neutro).
  3. Prefer RAVDESS full-AV (modality 01); sample up to N frames per clip.
  4. Detect face (YOLOv8 person → upper-body crop, or OpenCV Haar).
  5. Save JPG crops under data/video_consulta/processed/faces/.
  6. Write labels.csv with path,emotion,actor,dataset,label,split,sex.
  7. Split POR ATOR (female-first stratification).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.video.v3_face.face_dataset import (
    ActorInfo,
    assert_minority_ratio,
    assert_no_actor_leakage,
    assign_actor_splits,
    emotion_to_binary,
    load_cremad_sex_map,
    parse_cremad_filename,
    parse_ravdess_filename,
)

DEFAULT_RAVDESS = ROOT / "data" / "video_consulta" / "raw" / "ravdess"
DEFAULT_CREMAD = ROOT / "data" / "video_consulta" / "raw" / "cremad"
DEFAULT_OUT = ROOT / "data" / "video_consulta" / "processed" / "faces"
YOLO_WEIGHTS = "yolov8n.pt"


def sample_frames(video_path: Path, max_frames: int, step: int) -> list:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    idx = 0
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def crop_face_yolo(frame, model: YOLO, conf: float, margin: float = 0.15):
    """Detect single person, crop upper region as face proxy (RAVDESS/CREMA frontal)."""
    results = model(frame, verbose=False, conf=conf)
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None
    # Largest box
    boxes = result.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    x1, y1, x2, y2 = boxes[int(areas.argmax())]
    h = y2 - y1
    # Upper 55% of person box ≈ face+shoulders for acted close-ups
    face_y2 = y1 + h * 0.55
    w = x2 - x1
    x1m = max(0, int(x1 - margin * w))
    x2m = min(frame.shape[1], int(x2 + margin * w))
    y1m = max(0, int(y1 - margin * h))
    y2m = min(frame.shape[0], int(face_y2 + margin * h * 0.2))
    if x2m - x1m < 20 or y2m - y1m < 20:
        return None
    return frame[y1m:y2m, x1m:x2m]


def crop_face_haar(frame, cascade: cv2.CascadeClassifier):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad = int(0.15 * max(w, h))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame.shape[1], x + w + pad)
    y2 = min(frame.shape[0], y + h + pad)
    return frame[y1:y2, x1:x2]


def iter_videos(ravdess_dir: Path, cremad_dir: Path):
    if ravdess_dir.is_dir():
        for mp4 in sorted(ravdess_dir.rglob("*.mp4")):
            meta = parse_ravdess_filename(mp4.name)
            if meta is None:
                continue
            if meta["modality"] != "01":
                continue  # full-AV only
            if emotion_to_binary(meta["emotion"]) is None:
                continue
            yield mp4, meta

    video_flash = cremad_dir / "VideoFlash"
    sex_map: dict[str, str] = {}
    demo = cremad_dir / "VideoDemographics.csv"
    if demo.is_file():
        raw = load_cremad_sex_map(demo)
        sex_map = {f"{int(k):04d}": v for k, v in raw.items()}

    if video_flash.is_dir():
        for vid in sorted(list(video_flash.glob("*.flv")) + list(video_flash.glob("*.mp4"))):
            # Skip LFS pointers
            if vid.stat().st_size < 2048:
                continue
            try:
                head = vid.read_bytes()[:40]
                if head.startswith(b"version https://git-lfs"):
                    continue
            except OSError:
                continue
            meta = parse_cremad_filename(vid.name)
            if meta is None:
                continue
            if emotion_to_binary(meta["emotion"]) is None:
                continue
            actor_num = meta["actor"].removeprefix("crema_")
            meta["sex"] = sex_map.get(actor_num, sex_map.get(f"{int(actor_num):04d}", "U"))
            yield vid, meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract face frames (T104)")
    parser.add_argument("--ravdess-dir", type=Path, default=DEFAULT_RAVDESS)
    parser.add_argument("--cremad-dir", type=Path, default=DEFAULT_CREMAD)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-frames-per-clip", type=int, default=6)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument(
        "--detector",
        choices=("yolo", "haar"),
        default="yolo",
        help="Face crop strategy (default: yolov8n person + upper crop)",
    )
    parser.add_argument(
        "--female-only",
        action="store_true",
        help="Keep only female actors in the exported dataset",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit-videos",
        type=int,
        default=None,
        help="Optional cap for smoke runs",
    )
    args = parser.parse_args()

    labels_path = args.out_dir / "labels.csv"
    if labels_path.exists() and not args.force:
        print(f"[skip] {labels_path} exists. Use --force to redo.")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.force:
        for old in args.out_dir.glob("*.jpg"):
            old.unlink()
        labels_path.unlink(missing_ok=True)

    detector = None
    cascade = None
    if args.detector == "yolo":
        print(f"Loading {YOLO_WEIGHTS}…")
        detector = YOLO(YOLO_WEIGHTS)
    else:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)

    rows: list[dict] = []
    skipped_detect = 0
    videos = list(iter_videos(args.ravdess_dir, args.cremad_dir))
    if args.female_only:
        videos = [(v, m) for v, m in videos if m.get("sex") == "F"]
    if args.limit_videos is not None:
        videos = videos[: args.limit_videos]

    print(f"Processing {len(videos)} clips → {args.out_dir}")
    for video_path, meta in tqdm(videos, desc="clips"):
        label = emotion_to_binary(meta["emotion"])
        assert label is not None
        frames = sample_frames(video_path, args.max_frames_per_clip, args.step)
        for fi, frame in enumerate(frames):
            if detector is not None:
                crop = crop_face_yolo(frame, detector, args.conf)
            else:
                crop = crop_face_haar(frame, cascade)
            if crop is None:
                skipped_detect += 1
                continue
            out_name = (
                f"{meta['dataset']}_{meta['actor'].split('_', 1)[-1]}_"
                f"{meta['emotion']}_{video_path.stem}_f{fi:02d}.jpg"
            )
            # sanitize
            out_name = re.sub(r"[^A-Za-z0-9_.-]", "_", out_name)
            out_path = args.out_dir / out_name
            cv2.imwrite(str(out_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            rel = out_path.relative_to(ROOT).as_posix()
            rows.append(
                {
                    "path": rel,
                    "emotion": meta["emotion"],
                    "actor": meta["actor"],
                    "dataset": meta["dataset"],
                    "label": label,
                    "sex": meta.get("sex", "U"),
                }
            )

    if not rows:
        print("No face crops written — check raw videos / detector.")
        return 1

    df = pd.DataFrame(rows)
    actors = [
        ActorInfo(actor_id=a, sex=sex, dataset=ds)
        for (a, sex, ds), _ in df.groupby(["actor", "sex", "dataset"])
    ]
    # Prefer unique actor entries
    uniq: dict[str, ActorInfo] = {}
    for a in actors:
        uniq[a.actor_id] = a
    split_map = assign_actor_splits(list(uniq.values()), seed=args.seed)
    df["split"] = df["actor"].map(split_map)
    assert df["split"].notna().all()

    cols = ["path", "emotion", "actor", "dataset", "label", "split", "sex"]
    df = df[cols]
    df.to_csv(labels_path, index=False)

    assert_no_actor_leakage(df)
    minority = assert_minority_ratio(df, 0.40)
    n_jpg = len(list(args.out_dir.glob("*.jpg")))
    assert n_jpg == len(df)

    print(f"\nDone. {len(df)} faces → {labels_path}")
    print(f"  Skipped (no detection): {skipped_detect}")
    print(f"  Minority ratio: {minority:.3f}")
    print("  Label counts:\n", df["label"].value_counts().to_string())
    print("  Split counts:\n", df["split"].value_counts().to_string())
    print("  Sex counts:\n", df["sex"].value_counts().to_string())
    print("  Actors:", sorted(df["actor"].unique()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
