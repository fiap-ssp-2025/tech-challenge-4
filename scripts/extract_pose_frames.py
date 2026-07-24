#!/usr/bin/env python3
"""Extract pose keypoints from RAVDESS videos for posture classifier training.

Pipeline:
  1. Walk data/pose_posture/raw/ for .mp4 files.
  2. Parse emotion from filename (position 3, 1-indexed, 2-digit code).
  3. Map emotion → posture label:
       fearful (06) + sad (04) → "defensiva"
       neutral (01) + calm (02) + happy (03) → "neutra"
       other emotions are skipped.
  4. Sample up to --max-frames-per-clip frames per clip.
  5. Run YOLOv8-pose on each frame → extract 17 COCO keypoints (x, y, conf).
  6. Keep only frames where exactly 1 person is detected with conf ≥ threshold.
  7. Save keypoints + labels to data/pose_posture/annotations/keypoints.csv.

Emotion codes (RAVDESS filename field 3):
  01=neutral, 02=calm, 03=happy, 04=sad,
  05=angry, 06=fearful, 07=disgust, 08=surprised
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data" / "pose_posture" / "raw"
DEFAULT_OUT = ROOT / "data" / "pose_posture" / "annotations"
MODEL_NAME = "yolov8n-pose.pt"

# RAVDESS emotion code → posture label (None = skip)
EMOTION_TO_LABEL: dict[str, str | None] = {
    "01": "neutra",    # neutral
    "02": "neutra",    # calm
    "03": "neutra",    # happy
    "04": "defensiva", # sad
    "05": None,        # angry  (ambiguous for posture)
    "06": "defensiva", # fearful
    "07": None,        # disgust (ambiguous)
    "08": None,        # surprised (ambiguous)
}

# 17 COCO keypoint names
KP_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def parse_emotion(mp4_path: Path) -> str | None:
    """Extract emotion code from RAVDESS filename (e.g. 03-01-06-...)."""
    m = re.match(r"^\d{2}-\d{2}-(\d{2})-", mp4_path.name)
    return m.group(1) if m else None


def extract_frames(video_path: Path, max_frames: int, step: int = 5) -> list[np.ndarray]:
    """Sample up to max_frames frames from video, every `step` frames."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    idx = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def keypoints_to_row(kps: np.ndarray) -> list[float]:
    """Flatten (17, 3) keypoints array [x, y, conf] → 51-element list."""
    return kps.flatten().tolist()


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract RAVDESS pose keypoints")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-frames-per-clip", type=int, default=20)
    parser.add_argument("--conf-thresh", type=float, default=0.5,
                        help="Minimum person detection confidence")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_csv = args.out_dir / "keypoints.csv"
    if out_csv.exists() and not args.force:
        existing = sum(1 for _ in open(out_csv)) - 1
        print(f"[skip] {out_csv} already exists ({existing} rows). Use --force to redo.")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)

    mp4_files = sorted(args.raw_dir.rglob("*.mp4"))
    if not mp4_files:
        print(f"No .mp4 files found in {args.raw_dir}. Run download_ravdess.py first.")
        return 1

    print(f"Found {len(mp4_files)} videos. Loading YOLOv8-pose model...")
    model = YOLO(MODEL_NAME)

    # CSV header: label, actor, emotion_code, + 51 keypoint values
    kp_cols = [f"{name}_{axis}" for name in KP_NAMES for axis in ("x", "y", "conf")]
    header = ["label", "actor", "emotion_code", "video"] + kp_cols

    rows_written = 0
    skipped_emotion = 0
    skipped_detection = 0

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for mp4 in tqdm(mp4_files, desc="Videos"):
            emotion_code = parse_emotion(mp4)
            if emotion_code is None:
                continue

            label = EMOTION_TO_LABEL.get(emotion_code)
            if label is None:
                skipped_emotion += 1
                continue

            # Actor id from parent folder name or filename last field
            actor_match = re.search(r"Actor_(\d+)", str(mp4))
            actor_id = actor_match.group(1) if actor_match else "00"

            frames = extract_frames(mp4, args.max_frames_per_clip)
            for frame in frames:
                results = model(frame, verbose=False, conf=args.conf_thresh)
                result = results[0]

                if result.keypoints is None or len(result.keypoints) == 0:
                    skipped_detection += 1
                    continue

                # Keep only single-person frames
                boxes = result.boxes
                if boxes is None or len(boxes) != 1:
                    skipped_detection += 1
                    continue

                kps_data = result.keypoints.data  # (N, 17, 3)
                kps = kps_data[0].cpu().numpy()   # (17, 3)
                row = [label, actor_id, emotion_code, mp4.name] + keypoints_to_row(kps)
                writer.writerow(row)
                rows_written += 1

    print(f"\nDone. {rows_written} keypoint rows saved to {out_csv}")
    print(f"  Skipped (ambiguous emotion): {skipped_emotion} clips")
    print(f"  Skipped (detection issues):  {skipped_detection} frames")

    # Class balance report
    import pandas as pd
    df = pd.read_csv(out_csv)
    print("\nClass balance:")
    print(df["label"].value_counts().to_string())
    print(f"\nActors: {sorted(df['actor'].unique())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
