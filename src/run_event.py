"""End-to-end session runner: consulta .wav + .mp4 → triage note (feature 002)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.fusion.correlate import within_correlation
from src.fusion.report import build_report
from src.resolve import ResolvedPipeline

# Default demo session — áudio e vídeo da MESMA consulta.
DEFAULT_SESSION_ID = "consulta-demo"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run_pipeline(
    audio_path: Path,
    video_path: Path,
    *,
    session_id: str = DEFAULT_SESSION_ID,
    audio_ts: str | None = None,
    video_ts: str | None = None,
    modules: ResolvedPipeline | None = None,
) -> dict:
    """A1→A2→A3→alert→session correlation→V1/V2/V3→score→triage note."""
    audio_path = Path(audio_path)
    video_path = Path(video_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"áudio não encontrado: {audio_path}")
    if not video_path.is_file():
        raise FileNotFoundError(f"vídeo não encontrado: {video_path}")

    started = time.perf_counter()
    mods = modules if modules is not None else ResolvedPipeline()

    # --- Áudio (gatilho) ---
    transcricao: str | None = None

    try:
        a1 = mods.call("a1_stt", audio_path)
        transcricao = a1["transcricao"]
        if mods.origin("a1_stt") == "stub":
            print("[A1] STT stub OK.")
        else:
            provedor = a1.get("provedor", "não informado")
            print(f"[A1] STT real OK ({provedor}).")
    except RuntimeError as exc:
        print(f"[A1] Falha na transcrição: {exc}")
        print("[A1] Continuando o pipeline sem transcrição.")

    a12 = mods.call("a2_nlp", audio_path, transcricao=transcricao)
    print(f"[A2] tipo_relato={a12['tipo_relato']} local={a12['local']}")

    a3 = mods.call("a3_emotion", audio_path)
    print(f"[A3] sofrimento={a3['sofrimento']:.2f}")

    print("[alerta] Sinais na fala — analisando o vídeo da mesma consulta.")

    ts_audio = audio_ts or _now_iso()
    ts_video = video_ts or ts_audio
    audio_event = {
        "id": "live-audio",
        "modality": "audio",
        "session_id": session_id,
        "timestamp": ts_audio,
    }
    video_event = {
        "id": "live-video",
        "modality": "video",
        "session_id": session_id,
        "timestamp": ts_video,
    }
    corroborado = within_correlation(audio_event, video_event)
    print(f"[C] correlação por sessão → corroborado={corroborado}")

    # --- Vídeo (mesma consulta) ---
    v1 = mods.call("v1_tracks", video_path)
    v2 = mods.call("v2_pose", video_path)
    v3 = mods.call("v3_face", video_path)
    print(
        f"[V1] n_pessoas={v1['n_pessoas']} | "
        f"[V2] postura={v2['postura_defensiva']:.2f} | "
        f"[V3] desconforto_facial={v3['desconforto_facial']:.2f}"
    )

    cd = build_report(
        a12=a12, a3=a3, v1=v1, v2=v2, v3=v3, corroborado=corroborado
    )
    print(f"[D] escore={cd['escore']:.3f}")

    total_ms = (time.perf_counter() - started) * 1000
    resolved = mods.origins()
    tempos_ms = mods.timings_ms()
    _print_timings(resolved, tempos_ms, total_ms)

    return {
        "a12": a12,
        "a3": a3,
        "v1": v1,
        "v2": v2,
        "v3": v3,
        "cd": cd,
        "audio_event": audio_event,
        "video_event": video_event,
        "resolved": resolved,
        "tempos_ms": tempos_ms,
        "total_ms": round(total_ms, 1),
    }


def _print_timings(
    resolved: dict[str, str], tempos_ms: dict[str, float], total_ms: float
) -> None:
    """Per-module inference time — exigido no aceite da Etapa 3."""
    print("[tempo] inferência por módulo (ms):")
    for key, ms in tempos_ms.items():
        print(f"[tempo]   {key:<3} {ms:>9.1f} ms  ({resolved.get(key, '?')})")
    print(f"[tempo]   {'total':<3} {total_ms:>9.1f} ms (pipeline completo)")


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
        description="Pipeline de triagem em consulta (feature 002)"
    )
    parser.add_argument("--audio", type=Path, required=True, help="caminho .wav")
    parser.add_argument("--video", type=Path, required=True, help="caminho .mp4")
    parser.add_argument("--session", type=str, default=DEFAULT_SESSION_ID)
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

    result = run_pipeline(args.audio, args.video, session_id=args.session)
    print()
    print(result["cd"]["nota_ocorrencia"])
    if args.json:
        print(json.dumps(result["cd"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
