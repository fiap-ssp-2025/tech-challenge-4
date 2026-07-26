"""V3 facial discomfort — ViT pre-treinado em FER, fine-tune do T112 (P4).

O modelo pontua RECORTES DE ROSTO, não o quadro inteiro: a inferência precisa
repetir o mesmo pré-processamento do treino (detecção YOLOv8 → recorte superior
do corpo → 224×224 → normalização simétrica). Servir sem esse recorte entrega ao
modelo uma imagem que ele nunca viu no treino.

Um vídeo vira várias amostras; o score do contrato é a MÉDIA dos frames válidos —
mesma agregação usada para medir F1 0,7108 por clipe no T112.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.contracts import V3Result, validate_v3
from src.video.v3_face.preprocess import crop_face_yolo

# parents[3] = raiz do repo (src/video/v3_face/infer.py → src/video → src → raiz)
MODEL_DIR = Path(__file__).resolve().parents[3] / "models" / "v3_face"

MAX_FRAMES = 12  # teto por vídeo: além disso o ganho é marginal e o custo é linear
FRAME_STEP = 5   # 1 a cada 5 quadros, como em scripts/extract_face_frames.py
CROP_MARGIN = 0.15
UPPER_BODY_RATIO = 0.55
POSITIVE_INDEX = 1  # índice de "desconforto" — ver preprocess.json

_model = None
_cfg: dict | None = None
_yolo = None


def _get_cfg() -> dict:
    global _cfg
    if _cfg is None:
        path = MODEL_DIR / "preprocess.json"
        _cfg = json.loads(path.read_text()) if path.is_file() else {
            "img_size": 224, "norm_mean": [0.5] * 3, "norm_std": [0.5] * 3,
            "positive_index": POSITIVE_INDEX,
        }
    return _cfg


def _get_model():
    global _model
    if _model is None:
        if not MODEL_DIR.is_dir():
            raise FileNotFoundError(
                f"V3 model not found: {MODEL_DIR}\n"
                "Run: uv run python scripts/download_v3_model.py"
            )
        from transformers import AutoModelForImageClassification

        _model = AutoModelForImageClassification.from_pretrained(str(MODEL_DIR))
        _model.eval()
    return _model


def _get_yolo():
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO

        _yolo = YOLO("yolov8n.pt")
    return _yolo


def _crop_face(frame: np.ndarray, conf: float = 0.25) -> np.ndarray | None:
    """Recorte de rosto — delega ao módulo compartilhado com a extração do T104.

    Não reimplementar aqui: é a MESMA função que gerou os frames de treino. Uma cópia
    local sairia de sincronia sem ninguém perceber, e o modelo passaria a receber em
    produção um enquadramento que nunca viu treinar.
    """
    return crop_face_yolo(frame, _get_yolo(), conf=conf, margin=CROP_MARGIN)


def _sample_frames(video_path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames, idx = [], 0
    while len(frames) < MAX_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % FRAME_STEP == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def _to_tensor(crop: np.ndarray):
    import torch

    cfg = _get_cfg()
    size = int(cfg["img_size"])
    rgb = cv2.cvtColor(cv2.resize(crop, (size, size)), cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).float().div_(255.0).permute(2, 0, 1)
    mean = torch.tensor(cfg["norm_mean"]).view(3, 1, 1)
    std = torch.tensor(cfg["norm_std"]).view(3, 1, 1)
    return (x - mean) / std


def infer(path: str | Path) -> V3Result:
    """Estimate desconforto_facial [0..1] from a consultation video."""
    import torch

    video_path = Path(path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    model = _get_model()
    positive = int(_get_cfg().get("positive_index", POSITIVE_INDEX))

    crops = [c for c in (_crop_face(f) for f in _sample_frames(video_path)) if c is not None]
    if not crops:
        # Nenhum rosto detectável (vídeo preto do runner, câmera desligada). 0.0 aqui
        # significa "sem evidência visual", não "desconforto medido como ausente" —
        # a fusão pondera o caso pelos demais sinais.
        return validate_v3({"desconforto_facial": 0.0})

    batch = torch.stack([_to_tensor(c) for c in crops])
    with torch.no_grad():
        probs = torch.softmax(model(pixel_values=batch).logits, dim=-1)[:, positive]
    return validate_v3({"desconforto_facial": float(probs.mean())})


__all__ = ["infer"]
