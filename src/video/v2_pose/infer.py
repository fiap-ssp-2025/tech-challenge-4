"""V2 pose — YOLOv8n-pose + scikit-learn posture head (P5)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.contracts import V2Result, validate_v2

_pose_model = None
_clf = None

MODEL_PATH = Path(__file__).resolve().parents[4] / "models" / "v2_posture_head.pkl"

# COCO indices
_NOSE = 0
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_ELBOW,    _R_ELBOW    = 7, 8
_L_WRIST,    _R_WRIST    = 9, 10
_L_HIP,      _R_HIP      = 11, 12


def _get_pose_model():
    global _pose_model
    if _pose_model is None:
        from ultralytics import YOLO
        _pose_model = YOLO("yolov8n-pose.pt")
    return _pose_model


def _get_clf():
    global _clf
    if _clf is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Posture model not found: {MODEL_PATH}\n"
                "Run: uv run python scripts/train_v2_posture.py"
            )
        import joblib
        _clf = joblib.load(MODEL_PATH)
    return _clf


def _angle_at_b(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b; bc = c - b
    cos_v = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.arccos(np.clip(cos_v, -1.0, 1.0)))


def _engineer(kps_raw: np.ndarray) -> np.ndarray:
    """(N, 51) raw keypoints → (N, 43) normalised + geometric features.
    Must mirror scripts/train_v2_posture.py::engineer_features exactly.
    """
    kps = kps_raw.reshape(-1, 17, 3)
    N = len(kps)
    out = np.zeros((N, 43), dtype=np.float32)
    for i, pts in enumerate(kps):
        xy = pts[:, :2]
        shoulder_mid = (xy[_L_SHOULDER] + xy[_R_SHOULDER]) / 2.0
        hip_mid      = (xy[_L_HIP]      + xy[_R_HIP])      / 2.0
        torso_h      = np.linalg.norm(shoulder_mid - hip_mid) + 1e-6
        torso_c      = (shoulder_mid + hip_mid) / 2.0
        xy_n = (xy - torso_c) / torso_h
        out[i, :34] = xy_n.flatten()
        head_drop      = float(xy_n[_NOSE, 1] - ((xy_n[_L_SHOULDER, 1] + xy_n[_R_SHOULDER, 1]) / 2))
        shoulder_asym  = float(xy_n[_L_SHOULDER, 1] - xy_n[_R_SHOULDER, 1])
        shoulder_width = float(abs(xy_n[_L_SHOULDER, 0] - xy_n[_R_SHOULDER, 0]))
        shoulder_elev  = float(-((xy_n[_L_SHOULDER, 1] + xy_n[_R_SHOULDER, 1]) / 2))
        l_wrist_dist   = float(np.linalg.norm(xy_n[_L_WRIST]))
        r_wrist_dist   = float(np.linalg.norm(xy_n[_R_WRIST]))
        wrist_mean     = (l_wrist_dist + r_wrist_dist) / 2.0
        l_elbow_angle  = _angle_at_b(xy_n[_L_SHOULDER], xy_n[_L_ELBOW], xy_n[_L_WRIST])
        r_elbow_angle  = _angle_at_b(xy_n[_R_SHOULDER], xy_n[_R_ELBOW], xy_n[_R_WRIST])
        out[i, 34:] = [
            head_drop, shoulder_asym, shoulder_width, shoulder_elev,
            l_wrist_dist, r_wrist_dist, wrist_mean,
            l_elbow_angle, r_elbow_angle,
        ]
    return out


def _extract_keypoints(video_path: Path, max_frames: int = 30) -> list[np.ndarray]:
    model = _get_pose_model()
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max_frames)
    keypoints_list: list[np.ndarray] = []
    frame_idx = 0
    while len(keypoints_list) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            results = model(frame, verbose=False, conf=0.4)
            r = results[0]
            if r.keypoints is not None and len(r.keypoints) > 0 and r.boxes is not None and len(r.boxes) > 0:
                best = int(np.argmax(r.boxes.conf.cpu().numpy()))
                kps = r.keypoints.data[best].cpu().numpy()  # (17, 3)
                keypoints_list.append(kps)
        frame_idx += 1
    cap.release()
    return keypoints_list


def infer(path: str | Path) -> V2Result:
    """Estimate defensive posture score [0..1] from video."""
    video_path = Path(path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    keypoints_list = _extract_keypoints(video_path)
    if not keypoints_list:
        return validate_v2({"postura_defensiva": 0.0})

    clf = _get_clf()
    raw = np.array([kp.flatten() for kp in keypoints_list], dtype=np.float32)
    X = _engineer(raw)
    proba = clf.predict_proba(X)[:, 1]
    score = float(np.mean(proba))
    return validate_v2({"postura_defensiva": score})


__all__ = ["infer"]
