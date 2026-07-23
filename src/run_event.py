"""End-to-end event runner: .wav + .mp4 → prioritized occurrence note (stubs in Etapa 1)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.fusion.correlate import within_correlation
from src.fusion.report import build_report
from src.stubs import a1_stt, a2_nlp, a3_emotion, v1_tracks, v2_pose, v3_violence

# Default demo geolocation (Praça do Cruzeiro, DF) — elo simulado áudio↔vídeo.
DEFAULT_LAT = -15.7942
DEFAULT_LON = -47.8822


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run_pipeline(
    audio_path: Path,
    video_path: Path,
    *,
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    audio_ts: str | None = None,
    video_ts: str | None = None,
) -> dict:
    """A1→A2→A3→alert→correlation→V1/V2/V3→score→note."""
    audio_path = Path(audio_path)
    video_path = Path(video_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"áudio não encontrado: {audio_path}")
    if not video_path.is_file():
        raise FileNotFoundError(f"vídeo não encontrado: {video_path}")

    # --- Áudio (gatilho) ---
    transcricao: str | None = None
    try:
        a1 = a1_stt.infer(audio_path)
        transcricao = a1["transcricao"]
        print("[A1] STT stub OK (credenciais Azure presentes; API não chamada).")
    except RuntimeError as exc:
        print(f"[A1] {exc}")
        print(
            "[A1] Continuando com stub A2 até P3 entregar Azure real / faster-whisper."
        )

    a12 = a2_nlp.infer(audio_path, transcricao=transcricao)
    print(f"[A2] tipo_relato={a12['tipo_relato']} local={a12['local']}")

    a3 = a3_emotion.infer(audio_path)
    print(f"[A3] sofrimento={a3['sofrimento']:.2f}")

    print("[alerta] Ligação priorizada — solicitando vídeo da região (sob demanda).")

    ts_audio = audio_ts or _now_iso()
    ts_video = video_ts or ts_audio
    audio_event = {
        "id": "live-audio",
        "modality": "audio",
        "lat": lat,
        "lon": lon,
        "timestamp": ts_audio,
    }
    video_event = {
        "id": "live-video",
        "modality": "video",
        "lat": lat,
        "lon": lon,
        "timestamp": ts_video,
    }
    corroborado = within_correlation(audio_event, video_event)
    print(f"[C] correlação local/tempo → corroborado={corroborado}")

    # --- Vídeo (corroboração sob demanda) ---
    v1 = v1_tracks.infer(video_path)
    v2 = v2_pose.infer(video_path)
    v3 = v3_violence.infer(video_path)
    print(
        f"[V1] n_pessoas={v1['n_pessoas']} | "
        f"[V2] postura={v2['postura_defensiva']:.2f} | "
        f"[V3] violencia={v3['violencia']:.2f}"
    )

    cd = build_report(
        a12=a12, a3=a3, v1=v1, v2=v2, v3=v3, corroborado=corroborado
    )
    print(f"[D] escore={cd['escore']:.3f}")
    return {
        "a12": a12,
        "a3": a3,
        "v1": v1,
        "v2": v2,
        "v3": v3,
        "cd": cd,
        "audio_event": audio_event,
        "video_event": video_event,
    }


def make_dummy_media(
    wav_path: Path = Path("dummy.wav"),
    mp4_path: Path = Path("dummy.mp4"),
    seconds: float = 2.0,
) -> tuple[Path, Path]:
    """Create silent WAV + black MP4 for smoke tests."""
    import numpy as np
    import soundfile as sf
    import cv2

    sr = 16_000
    n = int(sr * seconds)
    sf.write(str(wav_path), np.zeros(n, dtype=np.float32), sr)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(mp4_path), fourcc, 10.0, (320, 240))
    if not writer.isOpened():
        raise RuntimeError("falha ao abrir VideoWriter para dummy.mp4")
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    for _ in range(int(10 * seconds)):
        writer.write(frame)
    writer.release()
    return wav_path, mp4_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pipeline despacho áudio–vídeo (stubs na Etapa 1)"
    )
    parser.add_argument("--audio", type=Path, required=True, help="caminho .wav")
    parser.add_argument("--video", type=Path, required=True, help="caminho .mp4")
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT)
    parser.add_argument("--lon", type=float, default=DEFAULT_LON)
    parser.add_argument(
        "--make-dummies",
        action="store_true",
        help="gera dummy.wav/dummy.mp4 se os caminhos não existirem",
    )
    parser.add_argument("--json", action="store_true", help="imprime resultado JSON")
    args = parser.parse_args(argv)

    if args.make_dummies or not args.audio.exists() or not args.video.exists():
        if args.audio.name == "dummy.wav" and args.video.name == "dummy.mp4":
            print("[setup] gerando dummy.wav e dummy.mp4 …")
            make_dummy_media(args.audio, args.video)
        elif not args.audio.exists() or not args.video.exists():
            missing = [p for p in (args.audio, args.video) if not p.exists()]
            print(f"arquivos ausentes: {missing}", file=sys.stderr)
            print("use --make-dummies com dummy.wav/dummy.mp4", file=sys.stderr)
            return 1

    result = run_pipeline(args.audio, args.video, lat=args.lat, lon=args.lon)
    print()
    print(result["cd"]["nota_ocorrencia"])
    if args.json:
        print(json.dumps(result["cd"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
