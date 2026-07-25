"""V3 face (P4) — real module contract test + train/serve preprocessing guard."""

from pathlib import Path

import numpy as np
import pytest

from src.video.v3_face.infer import MODEL_DIR

pytestmark = pytest.mark.skipif(
    not MODEL_DIR.is_dir(),
    reason="V3 model not present; run scripts/download_v3_model.py",
)


def _write_video(path: Path, frames: int = 10, size: tuple[int, int] = (240, 320)) -> Path:
    """Vídeo sintético com um retângulo claro no lugar de uma pessoa."""
    import cv2

    h, w = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    for i in range(frames):
        frame = np.full((h, w, 3), 30, dtype=np.uint8)
        cv2.rectangle(frame, (110, 40), (210, 220), (200, 180, 170), -1)
        cv2.circle(frame, (140, 90 + i % 3), 8, (40, 40, 40), -1)
        cv2.circle(frame, (180, 90 + i % 3), 8, (40, 40, 40), -1)
        writer.write(frame)
    writer.release()
    return path


def test_infer_returns_valid_contract(tmp_path: Path):
    from src.video.v3_face import infer as v3

    video = _write_video(tmp_path / "consulta.mp4")
    result = v3.infer(video)

    assert "desconforto_facial" in result
    assert 0.0 <= result["desconforto_facial"] <= 1.0


def test_infer_missing_file_raises(tmp_path: Path):
    from src.video.v3_face import infer as v3

    with pytest.raises(FileNotFoundError):
        v3.infer(tmp_path / "nao_existe.mp4")


def test_video_without_face_scores_zero(tmp_path: Path):
    """Sem rosto detectável não há medição — o contrato ainda tem de ser respeitado."""
    import cv2

    from src.video.v3_face import infer as v3

    path = tmp_path / "preto.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64))
    for _ in range(6):
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()

    result = v3.infer(path)
    assert result["desconforto_facial"] == 0.0


def test_crop_matches_training_extractor():
    """Guarda contra o recorte do infer divergir do que gerou os dados do T104.

    Se este teste quebrar, o modelo passa a receber em produção um enquadramento
    diferente do que viu no treino — degradação silenciosa, sem erro nenhum.
    """
    import importlib.util

    from src.video.v3_face import infer as v3

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "extract_face_frames", root / "scripts" / "extract_face_frames.py"
    )
    extract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extract)

    rng = np.random.default_rng(42)
    frame = rng.integers(0, 255, (360, 480, 3), dtype=np.uint8)

    mine = v3._crop_face(frame, conf=0.25)
    theirs = extract.crop_face_yolo(frame, v3._get_yolo(), conf=0.25, margin=v3.CROP_MARGIN)

    if mine is None or theirs is None:
        assert mine is None and theirs is None, "as duas implementações têm de concordar"
    else:
        assert mine.shape == theirs.shape
        assert np.array_equal(mine, theirs)
