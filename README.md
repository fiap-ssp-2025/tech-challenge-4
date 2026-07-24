# Tech Challenge 4 — Despacho Inteligente Áudio–Vídeo (190/193)

Pipeline de apoio ao despacho: a **ligação 190/193** dispara STT + PLN + emoção de voz; o **vídeo da região** (sob demanda) corrobora com tracks, postura e violência. A fusão (local/tempo + escore) gera uma nota priorizada — **decisão final humana**.

Fonte da verdade: `AGENTS.md` + `specs/001-despacho-audio-video/`.

## Instalação (reproduzível)

Requer [uv](https://docs.astral.sh/uv/) e Python 3.11+.

```bash
uv sync
cp .env.example .env   # preencha AZURE_SPEECH_KEY / AZURE_SPEECH_REGION (P3)
```

## Pipeline ponta a ponta (Etapa 1 — stubs)

```bash
uv run python -m src.run_event --audio dummy.wav --video dummy.mp4
```

Se `dummy.wav` / `dummy.mp4` não existirem, o runner gera silêncio + frame preto (~2 s).

Testes:

```bash
uv run pytest
```

Eventos sintéticos de fusão:

```bash
uv run python -m src.fusion.generate_synthetic
# → data/fusion_synthetic/events.jsonl
```

## Quem substitui qual stub

| Papel | Módulo real | Stub atual |
|-------|-------------|------------|
| **P2** | `src/audio/a3_emotion/` | `src/stubs/a3_emotion.py` |
| **P3** | `src/audio/a1_stt/`, `src/audio/a2_nlp/` | `src/stubs/a1_stt.py`, `src/stubs/a2_nlp.py` |
| **P4** | `src/video/v3_violence/` | `src/stubs/v3_violence.py` |
| **P5** | `src/video/v1_tracks/`, `src/video/v2_pose/` | `src/stubs/v1_tracks.py`, `src/stubs/v2_pose.py` |
| **P1** | `src/fusion/` (já real), `src/run_event.py`, contratos | — |

Contratos JSON em `src/contracts/` — **imutáveis a partir da Etapa 2**.

## Estrutura

```text
src/
  contracts/     # A1/A2, A3, V1, V2, V3, C/D
  audio/         # a1_stt, a2_nlp, a3_emotion
  video/         # v1_tracks, v2_pose, v3_violence
  fusion/        # correlate, scoring, report
  stubs/         # infer() fixos até modelos reais
  run_event.py
data/
  audio_ptbr/ video_violence/ pose_posture/ fusion_synthetic/
specs/001-despacho-audio-video/
```

## Spec-Driven Development

Fluxo: `constitution → specify → clarify → plan → tasks → implement`.

Feature ativa: **001-despacho-audio-video** (`in-progress`). O exemplo `hello_sdd` / `specs/000-hello-sdd` permanece até a Etapa 5.

```bash
uv run hello-sdd Ada   # exemplo legado
```
