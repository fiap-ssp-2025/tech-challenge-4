#!/usr/bin/env python3
"""Testa SÓ o ramo de áudio num arquivo qualquer — foco no sofrimento na voz (A3).

Existe porque um escore solto não diz nada. "0,04" é muito ou pouco? A resposta só
aparece comparando com a distribuição do próprio modelo no conjunto de teste, que
está versionada em `models/a3_reference_scores.json`. Este script faz essa leitura:
mostra o escore, onde ele cai entre os áudios neutros e os de sofrimento do CORAA, e
o perfil janela a janela — para ver se houve um pico curto que a média esconderia.

Aceita qualquer formato (m4a, mp3, wav, mp4…): converte com ffmpeg quando preciso.

Uso:
    uv run python scripts/test_audio.py gravacao.m4a
    uv run python scripts/test_audio.py consulta.wav --sem-stt      # pula o Azure
    uv run python scripts/test_audio.py a.wav b.wav c.wav           # compara vários
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
REF_PATH = ROOT / "models" / "a3_reference_scores.json"
TARGET_SR = 16_000
WINDOW_S = 6.0


def to_wav16k(path: Path) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    """Converte para WAV 16 kHz mono se necessário. Devolve (caminho, tmpdir a manter vivo)."""
    if path.suffix.lower() == ".wav":
        import soundfile as sf

        info = sf.info(path)
        if info.samplerate == TARGET_SR and info.channels == 1:
            return path, None
    tmp = tempfile.TemporaryDirectory()
    dest = Path(tmp.name) / "audio.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(path), "-vn", "-acodec", "pcm_s16le",
         "-ar", str(TARGET_SR), "-ac", "1", str(dest)],
        check=True, capture_output=True,
    )
    return dest, tmp


def percentil_em(valor: float, amostra: list[float]) -> float:
    """% da amostra que fica ABAIXO do valor."""
    a = np.asarray(amostra)
    return float((a < valor).mean() * 100)


def barra(p: float, largura: int = 44) -> str:
    return "█" * max(0, min(largura, int(round(p * largura))))


def analisar(path: Path, *, com_stt: bool, ref: dict) -> dict:
    import librosa
    import torch

    from src.audio.a3_emotion import infer as a3
    from src.audio import diarize

    wav, _tmp = to_wav16k(path)
    y, _ = librosa.load(wav, sr=TARGET_SR, mono=True)
    model, fe = a3._get_model()
    dur = len(y) / TARGET_SR

    print(f"\n{'='*68}\n{path.name}  —  {dur:.1f}s\n{'='*68}")

    # --- perfil janela a janela (áudio inteiro, SEM o teto de 30 s do infer)
    W = int(TARGET_SR * WINDOW_S)
    perfil = []
    with torch.no_grad():
        for i in range(0, len(y), W):
            ch = y[i : i + W]
            if len(ch) < 0.5 * TARGET_SR:
                break
            inp = fe(ch, sampling_rate=TARGET_SR, return_tensors="pt")
            perfil.append((i // TARGET_SR,
                           float(torch.softmax(model(**inp).logits, -1)[0][a3.NON_NEUTRAL_ID])))

    print(f"\nPerfil de sofrimento (janelas de {WINDOW_S:.0f}s, áudio inteiro):")
    for t, p in perfil:
        marca = "  ← acima do limiar 0,17" if p >= ref["limiar_decisao"] else ""
        print(f"  {t:4d}s  {p:.3f}  {barra(p)}{marca}")

    pico = max(p for _, p in perfil)
    t_pico = max(perfil, key=lambda x: x[1])[0]

    # --- o que o pipeline de fato entrega (com teto de 30 s e diarização)
    do_pipeline = float(a3.infer(wav)["sofrimento"])

    segs = diarize.speaker_segments(wav)
    por_loc = diarize.audio_by_speaker(y, segs) if segs else {}

    print(f"\n{'Escore que o pipeline entrega:':<38s} {do_pipeline:.3f}")
    print(f"{'Pico no áudio inteiro:':<38s} {pico:.3f}  (aos {t_pico}s)")
    if do_pipeline < pico - 0.005:
        print("   ⚠ o pipeline entrega menos que o pico — teto de 30s ou reagrupamento por locutor")
    if por_loc:
        print(f"{'Locutores separados:':<38s} {len(por_loc)}")
        for k, v in sorted(por_loc.items()):
            print(f"     {k}: {len(v)/TARGET_SR:5.1f}s de fala → {a3._score_samples(v, model, fe):.3f}")

    # --- leitura: onde isso cai na distribuição conhecida do modelo
    neu, nn = ref["escores"]["neutral"], ref["escores"]["non_neutral"]
    print(f"\nLeitura (referência: teste do CORAA, {len(neu)} neutros e {len(nn)} com sofrimento):")
    for nome, val in (("pipeline", do_pipeline), ("pico", pico)):
        print(f"  {nome:<9s} {val:.3f} → acima de {percentil_em(val, neu):5.1f}% dos NEUTROS"
              f"  e de {percentil_em(val, nn):5.1f}% dos COM SOFRIMENTO")
    veredito = ("ACIMA do limiar de decisão (0,17) — o modelo chamaria de sofrimento"
                if pico >= ref["limiar_decisao"] else
                "abaixo do limiar de decisão (0,17) — o modelo não chamaria de sofrimento")
    print(f"  → {veredito}")

    if com_stt:
        from src.audio.a1_stt import infer as a1
        from src.audio.a2_nlp import infer as a2

        # A2 recebe o caminho (compatibilidade com o runner) + o texto do A1.
        transcricao = a1.infer(wav)["transcricao"]
        estrut = a2.infer(wav, transcricao=transcricao)
        print(f"\nA1 transcrição: {estrut['transcricao'][:300]}")
        print(f"A2 → tipo_relato={estrut['tipo_relato']} | local={estrut['local']!r} "
              f"| tempo={estrut['tempo']!r}")

    return {"arquivo": path.name, "duracao_s": round(dur, 1),
            "pipeline": round(do_pipeline, 4), "pico": round(pico, 4),
            "locutores": len(por_loc)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Testa o ramo de áudio (foco no A3)")
    ap.add_argument("arquivos", nargs="+", type=Path)
    ap.add_argument("--sem-stt", action="store_true", help="pula A1/A2 (não chama o Azure)")
    args = ap.parse_args()

    if not REF_PATH.is_file():
        print(f"Referência ausente: {REF_PATH}")
        return 1
    ref = json.loads(REF_PATH.read_text())

    linhas = []
    for f in args.arquivos:
        if not f.is_file():
            print(f"[pulado] não encontrado: {f}")
            continue
        linhas.append(analisar(f, com_stt=not args.sem_stt, ref=ref))

    if len(linhas) > 1:
        print(f"\n{'='*68}\nCOMPARATIVO\n{'='*68}")
        print(f"{'arquivo':<34s} {'dur':>6s} {'pipeline':>9s} {'pico':>7s} {'locut':>6s}")
        for r in linhas:
            print(f"{r['arquivo']:<34s} {r['duracao_s']:6.1f} {r['pipeline']:9.3f} "
                  f"{r['pico']:7.3f} {r['locutores']:6d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
