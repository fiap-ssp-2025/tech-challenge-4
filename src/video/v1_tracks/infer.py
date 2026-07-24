"""V1 tracks — YOLOv8n + ByteTrack (P5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.contracts import V1Result, validate_v1

_model = None


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO("yolov8n.pt")
    return _model


def infer(path: str | Path) -> V1Result:
    """Run YOLOv8n + ByteTrack on video. Returns person tracks."""
    video_path = Path(path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    model = _get_model()
    results = model.track(
        source=str(video_path),
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],   # class 0 = person
        verbose=False,
    )

    # Accumulate bboxes per track id
    track_data: dict[int, list[list[float]]] = {}
    for result in results:
        if result.boxes is None:
            continue
        boxes = result.boxes
        if boxes.id is None:
            continue
        ids = boxes.id.cpu().numpy().astype(int)
        xywh = boxes.xywh.cpu().numpy()   # (N, 4)
        for tid, bbox in zip(ids, xywh):
            track_data.setdefault(int(tid), []).append(bbox.tolist())

    tracks = []
    for tid, bboxes in track_data.items():
        arr = np.array(bboxes)  # (n_frames, 4)
        tracks.append({
            "id": tid,
            "n_frames": len(bboxes),
            "bbox_media": arr.mean(axis=0).tolist(),
        })

    return validate_v1({
        "n_pessoas": len(tracks),
        "tracks": tracks,
    })


__all__ = ["infer"]
