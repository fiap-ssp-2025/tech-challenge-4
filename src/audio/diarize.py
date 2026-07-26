"""Diarização — quem fala quando — via Azure `ConversationTranscriber` (feature 003).

Por que existe: o A3 pontua sofrimento na voz, mas uma consulta tem pelo menos duas
pessoas falando. Sem separar, a fala neutra do profissional entra na mesma conta da
paciente e puxa o escore para baixo (medido: 0,03 numa simulação de emergência).

O que este módulo NÃO faz: identificar quem é quem. Ele devolve rótulos opacos
(`Convidado-1`, `Convidado-2`) e quem consome pontua todos, sem atribuir papéis —
decisão deliberada da spec 003 (RF-24), para não tratar identidade.

Degrada em silêncio por desenho: sem credencial, sem rede ou com erro do serviço,
`speaker_segments()` devolve None e o chamador segue com o áudio inteiro. A mesma
filosofia do `src/resolve.py` — o pipeline nunca quebra por falta de um extra.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

TARGET_SR = 16_000
MIN_SPEAKER_SECONDS = 1.0  # menos que isso não dá para estimar emoção
_TICKS_PER_SECOND = 10_000_000  # o SDK reporta offset/duration em unidades de 100 ns

_cache: dict[tuple[str, float], list["Segment"] | None] = {}


@dataclass(frozen=True)
class Segment:
    """Um trecho contínuo atribuído a um locutor."""

    speaker: str
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def _log(msg: str) -> None:
    print(f"[diarize] {msg}")


def speaker_segments(path: str | Path) -> list[Segment] | None:
    """Trechos por locutor, ou None quando a diarização não está disponível.

    Nunca levanta: qualquer falha vira None + motivo logado (RF-22).
    """
    audio_path = Path(path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Áudio não encontrado: {audio_path}")

    key = (str(audio_path.resolve()), audio_path.stat().st_mtime)
    if key in _cache:
        return _cache[key]

    result = _diarize_azure(audio_path)
    _cache[key] = result
    return result


def _diarize_azure(audio_path: Path) -> list[Segment] | None:
    speech_key = os.getenv("AZURE_SPEECH_KEY", "").strip()
    speech_region = os.getenv("AZURE_SPEECH_REGION", "").strip()
    if not speech_key or not speech_region:
        _log("sem AZURE_SPEECH_KEY/REGION — pontuando o áudio inteiro (sem separação)")
        return None

    try:
        import azure.cognitiveservices.speech as speechsdk

        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
        speech_config.speech_recognition_language = "pt-BR"
        audio_config = speechsdk.audio.AudioConfig(filename=str(audio_path))
        transcriber = speechsdk.transcription.ConversationTranscriber(
            speech_config=speech_config, audio_config=audio_config
        )

        segments: list[Segment] = []
        done = False

        def _on_transcribed(evt) -> None:
            res = evt.result
            if not getattr(res, "text", "").strip():
                return
            start = res.offset / _TICKS_PER_SECOND
            segments.append(
                Segment(
                    speaker=str(getattr(res, "speaker_id", "") or "Unknown"),
                    start_s=start,
                    end_s=start + res.duration / _TICKS_PER_SECOND,
                )
            )

        def _on_stop(_evt) -> None:
            nonlocal done
            done = True

        transcriber.transcribed.connect(_on_transcribed)
        transcriber.session_stopped.connect(_on_stop)
        transcriber.canceled.connect(_on_stop)

        transcriber.start_transcribing_async().get()
        import time

        deadline = time.monotonic() + 300
        while not done and time.monotonic() < deadline:
            time.sleep(0.3)
        transcriber.stop_transcribing_async().get()
    except Exception as exc:  # noqa: BLE001 — degradar é o comportamento contratado
        _log(f"falhou ({exc.__class__.__name__}: {exc}) — pontuando o áudio inteiro")
        return None

    if not segments:
        _log("nenhum trecho reconhecido — pontuando o áudio inteiro")
        return None

    speakers = sorted({s.speaker for s in segments})
    _log(f"{len(segments)} trechos, {len(speakers)} locutor(es): {', '.join(speakers)}")
    return segments


def audio_by_speaker(
    y: np.ndarray, segments: list[Segment], sr: int = TARGET_SR
) -> dict[str, np.ndarray]:
    """Concatena, por locutor, as amostras dos seus trechos.

    Concatenar antes de janelar é essencial (RF-23): o modelo lê janelas de 6 s, e um
    "sim" de 0,4 s isolado viraria uma amostra ruidosa. Juntando a fala do locutor, as
    janelas ficam cheias.

    Locutor com menos de MIN_SPEAKER_SECONDS de fala total é descartado.
    """
    buckets: dict[str, list[np.ndarray]] = {}
    for seg in segments:
        a, b = int(seg.start_s * sr), int(seg.end_s * sr)
        chunk = y[max(0, a) : min(len(y), b)]
        if chunk.size:
            buckets.setdefault(seg.speaker, []).append(chunk)

    out: dict[str, np.ndarray] = {}
    for speaker, chunks in buckets.items():
        joined = np.concatenate(chunks)
        if len(joined) >= MIN_SPEAKER_SECONDS * sr:
            out[speaker] = joined
    return out


__all__ = ["Segment", "speaker_segments", "audio_by_speaker", "MIN_SPEAKER_SECONDS"]
