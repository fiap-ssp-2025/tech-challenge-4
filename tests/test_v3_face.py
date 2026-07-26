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


def test_infer_and_extraction_share_one_crop_implementation():
    """Extração (T104) e inferência (T112) têm de usar a MESMA função de recorte.

    Antes eram duas cópias e um teste comparava as saídas; agora há uma só, em
    `src.video.v3_face.preprocess`. Este teste guarda a propriedade na origem: se
    alguém reintroduzir uma cópia local, o `is` falha — não dependemos de a
    divergência aparecer num quadro de exemplo.
    """
    import importlib.util

    from src.video.v3_face import preprocess
    from src.video.v3_face import infer as v3

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "extract_face_frames", root / "scripts" / "extract_face_frames.py"
    )
    extract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extract)

    assert extract.crop_face_yolo is preprocess.crop_face_yolo
    assert v3.crop_face_yolo is preprocess.crop_face_yolo
