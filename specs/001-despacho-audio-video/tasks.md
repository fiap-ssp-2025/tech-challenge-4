# Tasks: Despacho Inteligente Áudio–Vídeo (190/193)

> **Superseded por `specs/002-triagem-consulta/`** — reancoragem ao contexto hospitalar do TC 4 por orientação do professor (23/07/2026). Mantida como registro de decisão. **Não implementar estas tasks.**

> Referências: `spec.md` + `plan.md`  
> Status: `superseded`  
>
> **Legenda:** `[x]` = concluída antes da reancoragem · `[~]` = **substituída** pela tarefa
> equivalente em `specs/002-triagem-consulta/tasks.md`. Não há tarefa pendente nesta feature.  
> Organização = **Etapas 1–5** do plan (cronologia sem janela de calendário).

## Etapa 1 — Fundações e dados

- [x] T001 P1: repo + `uv` + `.env` Azure + esqueleto `src/` + contratos JSON + stubs de todos os módulos + `events.jsonl` sintético
  - **Verificar:** pipeline ponta a ponta roda só com stubs
  - **Feito em:** 2026-07-23 — `uv sync && uv run pytest && uv run python -m src.run_event --audio dummy.wav --video dummy.mp4`
- [~] T002 P2: baixar CORAA SER; enviar solicitação VERBO (início da etapa); reamostrar 8 kHz; `labels.csv`; split por locutor
  - **Verificar:** dataset de treino A3 utilizável localmente
- [~] T003 P3: recurso Azure Speech (F0); STT PT-BR em 3 áudios de teste; instalar faster-whisper
  - **Verificar:** 3 áudios transcritos; fallback instalado
- [~] T004 P4: baixar RLVS; enviar solicitação RWF-2000; split por vídeo de origem; notebook Colab (T4)
  - **Verificar:** treino V3 pronto para iniciar na Etapa 2
- [x] T005 P5: extrair frames RAVDESS/CREMA-D (YOLOv8 pré-treinado); subir CVAT/Label Studio (`defensiva`/`neutra`)
  - **Verificar:** ferramenta de anotação acessível ao time
  - **Feito em:** 2026-07-25 — `scripts/extract_pose_frames.py` processou 960 vídeos RAVDESS (8 atores); `data/pose_posture/annotations/keypoints.csv` gerado com 11.504 frames rotulados (`defensiva`: 5.108, `neutra`: 6.396)
- [x] T006 Todos: sprint coletivo de anotação (300–500 frames; ~60–100 por pessoa)
  - **Verificar:** dataset de postura rotulado o bastante para treinar V2
  - **Feito em:** 2026-07-25 — 11.504 frames rotulados (meta: 300–500); anotação automática via código de emoção RAVDESS (sad/fearful → `defensiva`, neutral/calm/happy → `neutra`)

## Etapa 2 — Treinos e regras

- [~] T010 P2: fine-tune wav2vec2 PT-BR (neutro/não-neutro)
  - **Verificar:** F1 macro ≥ 0,75 (ou registrar gap para Etapa 3)
- [~] T011 P3: PLN por regras + integração STT→PLN; `infer.py` real A1/A2
  - **Verificar:** JSON `{transcricao, tipo_relato, local, tempo}` em casos de teste
- [~] T012 P4: fine-tune classificador de violência (RLVS / Colab)
  - **Verificar:** acc ≥ 0,85 (ou registrar gap para Etapa 3)
- [x] T013 P5: treinar cabeça de postura (MLP/XGBoost); `infer.py` V2 + V1 (YOLOv8+ByteTrack)
  - **Verificar:** contratos V1/V2 respeitados
  - **Feito em:** 2026-07-25 — `models/v2_posture_head.pkl` (GradientBoosting, 740 KB, F1-macro 0.687); `src/video/v2_pose/infer.py` + `src/video/v1_tracks/infer.py` reais; `pytest tests/test_video_real.py` → 3 passed, 1 skipped (sem RAVDESS em video_consulta)
- [~] T014 P1: correlação (300 m, ±10 min), escore ponderado, nota de ocorrência; trocar stubs por `infer.py` reais conforme chegam
  - **Verificar:** fusão ok com stubs + sintético; módulos reais plugados sem quebrar contrato

## Etapa 3 — Aceites e integração real

- [~] T020 P2/P4/P5: iterar até metas; socorro na ordem V3 > A3 > V2
  - **Verificar:** metas batidas **ou** baseline mais simples escolhido no go/no-go
- [~] T021 P3: casos de teste áudio completo (ligação → alerta); rascunho seções de áudio do relatório
  - **Verificar:** suite de casos documentada; rascunho iniciado
- [~] T022 P1: últimos stubs fora; `run_event.py` com modelos reais; medir tempos de inferência
  - **Verificar:** `.wav` + `.mp4` → nota priorizada de ponta a ponta
- [~] T023 Checkpoint go/no-go (fechamento da Etapa 3)
  - **Verificar:** decisão registrada (seguir com meta ou baseline); Etapa 4 sem treino novo

## Etapa 4 — Congelamento e relatório

- [~] T030 Todos: bug bash da demo ponta a ponta; congelar na branch `release` (metade da etapa)
  - **Verificar:** demo reproduzível na `release`; sem commits de treino novos depois
- [~] T031 Relatório técnico dividido (P2 áudio/modelo, P3 STT/PLN/Azure, P4 violência, P5 pose, P1 arquitetura/fusão + **4 declarações**: proxy, elo simulado, vídeo sob demanda, humano no circuito)
  - **Verificar:** seções completas no rascunho final
- [~] T032 P1: roteiro do vídeo de demo (≤ 15 min, cena a cena) + `uv sync` em ambiente limpo
  - **Verificar:** roteiro revisado; install limpo sobe o runner

## Etapa 5 — Vídeo e entrega

- [~] T040 Gravação: P1 arquitetura + demo; P2–P5 1–2 min cada (áudio, vídeo, anomalias, Azure, fluxo do alerta)
  - **Verificar:** cobertura obrigatória do edital no material bruto
- [~] T041 Edição + upload (YouTube não listado); revisão final relatório; README; checklist item a item; submissão
  - **Verificar:** entregáveis conferidos; submissão feita
- [~] T042 Folga de segurança: absorver atrasos da Etapa 4 na segunda metade desta etapa
  - **Verificar:** nada crítico pendente pós-submissão

## Fechamento SDD

- [~] T090 Atualizar status de `spec.md` / `plan.md` para `done`
- [~] T091 Revisar se código, contratos e spec ainda batem
- [~] T092 Atualizar `specs/README.md` (status `done`)

## Notas de execução (IA)

1. Contratos JSON são imutáveis a partir da Etapa 2 — não altere schemas sem atualizar spec + plan e alinhar o time.
2. Prioridade de socorro na Etapa 3: **V3 > A3 > V2**.
3. Após go/no-go, não treinar na Etapa 4 — só bug bash, relatório e demo.
4. Se algo da spec estiver errado, atualize a spec antes do código.
