"""Face-crop preprocessing shared by dataset extraction (T104) and inference (T112).

Train/serve skew guard: `scripts/extract_face_frames.py` (dataset build) and
`src/video/v3_face/infer.py` (real-time inference) must call the same crop
function. Resize/normalization used at train time will be added here once
the T112 training script defines them.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def crop_face_yolo(
    frame: np.ndarray, model: Any, conf: float, margin: float = 0.15
) -> np.ndarray | None:
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


def crop_face_haar(frame: np.ndarray, cascade: cv2.CascadeClassifier) -> np.ndarray | None:
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


__all__ = ["crop_face_yolo", "crop_face_haar"]
