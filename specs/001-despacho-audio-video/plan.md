# Plan: Despacho Inteligente Áudio–Vídeo (190/193)

> Referência: `specs/001-despacho-audio-video/spec.md`  
> **Status:** in-progress

## Resumo técnico

Pipeline modular com contratos JSON fixos por módulo (`infer.py`). A ligação 190/193 dispara STT + PLN + emoção de voz; o vídeo da região, sob demanda, aporta pose e violência. P1 orquestra fusão (correlação local/tempo, escore, nota) e integração — começando com stubs na Etapa 1 e trocando por inferências reais até a Etapa 3. Congelamento na Etapa 4; entrega e vídeo na Etapa 5.

**Regra de ouro:** contratos primeiro, integração cedo, congelamento na Etapa 4. Nada de modelo perfeito — modelo que passa o aceite.

## Papéis

| Quem | Papel | Frente |
|------|-------|--------|
| **P1** | Tech lead / integração | Repo, contratos, fusão (C/D), demo ponta a ponta, revisão |
| **P2** | Áudio-ML | Dados CORAA/VERBO + treino emoção de voz (A3) |
| **P3** | Áudio-serviços | Azure Speech STT (A1) + PLN por regras (A2) + fallback offline |
| **P4** | Vídeo-ML | Dados RLVS/RWF + classificador de violência (V3) |
| **P5** | Vídeo-pose | Frames + anotação de postura + YOLOv8-pose customizado (V2) |

**Por que P1 fica com a fusão sozinho:** correlação, escore e nota de ocorrência são regras + orquestração — uma cabeça só desenha em horas o que um par levaria uma etapa discutindo. Em troca, P1 não treina nenhum modelo: fica livre para desbloquear os outros.

## Contratos (definidos na Etapa 1, imutáveis a partir da Etapa 2)

Cada módulo entrega um `infer.py` com contrato JSON fixo — é isso que permite os 5 trabalharem em paralelo sem se esperar:

```text
A1/A2 → {transcricao, tipo_relato, local, tempo}
         tipo_relato ∈ {agressao | ameaca | perseguicao | outro}
A3    → {sofrimento: 0..1}
V1    → {n_pessoas: int,
         tracks: [ {id: int, n_frames: int, bbox_media: [x, y, w, h]} ]}
V2    → {postura_defensiva: 0..1}
V3    → {violencia: 0..1}
C/D   → {escore, corroborado: bool, nota_ocorrencia}
```

**Notas de schema (fechadas no clarify):**

- `tipo_relato`: vocabulário fixo; termos fora do léxico caem em `outro` (fallback obrigatório — PLN nunca falha o contrato inventando labels).
- `tracks` (V1): mínimo que a fusão consome (quantas pessoas, por quanto tempo, onde). Keypoints ficam **internos ao V2** e não trafegam neste contrato (evita schema pesado e acoplamento V1↔V2).
- Após a Etapa 2 estes contratos são **imutáveis**.

Enquanto os modelos não existem, P1 usa **stubs** (respostas fixas) para montar o pipeline — integração começa na Etapa 2, não no fim.

## Stack

| Camada | Escolha | Motivo |
|--------|---------|--------|
| Runtime | Python 3.11+ | Constituição |
| Empacotamento / env | `uv` + `.env` (Azure) | Reprodutibilidade e secrets fora do git |
| STT | Azure Speech (F0) + fallback `faster-whisper` | Aceite em nuvem + resiliência na demo |
| PLN (A2) | Regras (tipo de relato + local/tempo) | Entrega rápida, contrato estável |
| Emoção (A3) | Fine-tune wav2vec2 PT-BR (neutro / não-neutro) | Meta F1 macro ≥ 0,75 |
| Detecção / tracks (V1) | YOLOv8 + ByteTrack (sem treino) | Contagem e tracks prontos |
| Postura (V2) | Keypoints + cabeça MLP/XGBoost (CPU) | Dataset anotado no sprint coletivo |
| Violência (V3) | Classificador fine-tune no RLVS (Colab T4) | Meta acc ≥ 0,85 |
| Fusão (C/D) | Regras: raio 300 m, janela ±10 min, escore ponderado (dict em `fusion/scoring`) | Orquestração simples, uma cabeça (P1) |
| Dados sintéticos | `events.jsonl` | Integração e testes sem esperar modelos |
| Anotação | CVAT / Label Studio (`defensiva` / `neutra`) | Sprint coletivo de frames |

### Fusão — pesos do escore ponderado (C/D)

Pesos são **regra documentada**, não aprendidos. Ajustáveis num único dict em `fusion/scoring`.

| Sinal | Peso | Racional |
|-------|------|----------|
| `tipo_relato` grave (`agressao` / `ameaca`) | 0.25 | Sinal primário do relato |
| `sofrimento` (A3) | 0.25 | Sinal primário da voz |
| `violencia` (V3) | 0.25 | Sinal primário da cena |
| `postura_defensiva` (V2) | 0.15 | Proxy mais fraco (proxy de proxy) |
| corroboração espaçotemporal | 0.10 | Já atua como multiplicador de confiança; menor peso direto |

Soma dos pesos = **1.00**. Sinais binários / contínuos entram normalizados em `0..1` antes da ponderação (detalhe de implementação na Etapa 2, sem mudar este dict).

## Arquitetura

```text
[.wav ligação]
    → A1 STT (± fallback) → A2 PLN  → {transcricao, tipo_relato, local, tempo}
    → A3 emoção de voz              → {sofrimento}

[.mp4 região, sob demanda]
    → V1 YOLO+ByteTrack → {n_pessoas, tracks[{id, n_frames, bbox_media}]}
    → V2 pose/postura   → {postura_defensiva}   # keypoints internos ao V2
    → V3 violência      → {violencia}

[A* + V* + eventos]
    → C/D correlação (300 m, ±10 min) + escore ponderado (dict fusion/scoring) + nota
    → {escore, corroborado, nota_ocorrencia}
    → operador humano (decisão final)
```

## Estrutura de pastas (alvo)

```text
src/
  contracts/          # schemas / tipos dos JSON de inferência
  audio/
    a1_stt/
    a2_nlp/
    a3_emotion/
  video/
    v1_tracks/
    v2_pose/
    v3_violence/
  fusion/             # C/D (correlate, scoring, report)
  stubs/              # infer() fixos até modelos reais
  run_event.py        # .wav + .mp4 → nota priorizada
data/
  audio_ptbr/{raw,processed}/
  video_violence/{raw,processed}/
  pose_posture/{raw,annotations}/
  fusion_synthetic/   # events.jsonl (P1)
tests/
  ...
```

*(Detalhe fino de pacotes pode ajustar-se na Etapa 1 sem mudar contratos.)*

## Cronograma (Etapas 1–5)

Cronologia de execução **sem janela de calendário fixa**. Cada Etapa é um marco ordenado.

### Etapa 1 — Fundações e dados (tudo em paralelo)

- **P1:** repo + `uv` + `.env` Azure + esqueleto `src/` + **contratos JSON** + stubs de todos os módulos + `events.jsonl` sintético inicial. Ao fim da etapa: pipeline roda ponta a ponta com stubs.
- **P2:** baixa CORAA SER; **envia solicitação do VERBO** (início da Etapa 1 — prazo fora do nosso controle); reamostra 8 kHz; `labels.csv`; split por locutor.
- **P3:** cria recurso Azure Speech (F0); STT PT-BR funcionando em 3 áudios de teste; instala fallback faster-whisper.
- **P4:** baixa RLVS (Kaggle); **envia solicitação do RWF-2000**; split por vídeo de origem; prepara notebook Colab (GPU T4).
- **P5:** extrai frames de RAVDESS/CREMA-D com YOLOv8 pré-treinado; sobe CVAT/LabelStudio com as classes `defensiva` / `neutra`.
- **Todos (fechamento da etapa, ~1h):** *sprint de anotação* — 300–500 frames ÷ 5 pessoas = 60–100 frames cada. É o único gargalo que paraleliza perfeitamente; no bloco coletivo o dataset de postura fica pronto.

### Etapa 2 — Treinos e regras (cada um no seu módulo)

- **P2:** fine-tuning wav2vec2 PT-BR (binário neutro/não-neutro). Meta: F1 macro ≥ 0,75.
- **P3:** PLN por regras (tipo de relato + local/tempo); integra STT→PLN; entrega `infer.py` real do A1/A2.
- **P4:** fine-tune do classificador de violência no RLVS (Colab). Meta: acc ≥ 0,85.
- **P5:** treina cabeça de postura (MLP/XGBoost sobre keypoints) — roda em CPU; entrega `infer.py` do V2 + V1 (YOLOv8+ByteTrack, sem treino).
- **P1:** implementa correlação (raio 300 m, janela ±10 min), escore ponderado e gerador da nota de ocorrência; valida com stubs + sintético. Vai trocando stubs por `infer.py` reais conforme chegam.

### Etapa 3 — Aceites e integração real

- **P2/P4/P5:** iteram até bater as metas (ajuste de hiperparâmetros, balanceamento). Quem bater primeiro **ajuda o que estiver mais atrás** — prioridade de socorro: V3 > A3 > V2.
- **P3:** casos de teste do áudio completo (ligação → alerta); começa o rascunho do relatório (seções de áudio).
- **P1:** substitui os últimos stubs; `run_event.py` (`.wav` + `.mp4` → nota priorizada) rodando com modelos reais; mede tempos de inferência.
- **Checkpoint go/no-go (fechamento da Etapa 3):** se alguma meta não bateu, corta para o baseline mais simples que passe — Etapa 4 não é etapa de treinar.

### Etapa 4 — Congelamento e relatório

- **Primeira metade — todos:** *bug bash* da demo ponta a ponta; congela código na metade da etapa (branch `release`).
- **Segunda metade:** relatório técnico dividido: P2 (áudio/modelo), P3 (STT/PLN/Azure), P4 (vídeo/violência), P5 (pose/YOLO customizado), P1 (arquitetura, fusão, resultados, **as 4 declarações obrigatórias**: proxy, elo simulado, vídeo sob demanda, humano no circuito).
- **P1 (extra):** roteiro do vídeo de demonstração (cena a cena, ≤ 15 min) + teste de reprodução `uv sync` em ambiente limpo.

### Etapa 5 — Vídeo e entrega

- **Primeira metade:** gravação — P1 apresenta arquitetura e conduz a demo; P2–P5 gravam 1–2 min cada do próprio módulo (divide o esforço e mostra o grupo). Cobrir, obrigatoriamente: análise de áudio e vídeo, detecção/resposta a anomalias, integração Azure, fluxo do alerta.
- **Segunda metade:** edição e upload (YouTube não listado); revisão final do relatório; README; conferência item a item dos entregáveis do edital; submissão.
- **Folga de segurança:** a segunda metade da Etapa 5 absorve qualquer atraso da Etapa 4.

## Datasets (inalterados — todos em uso)

| Ramo | Dataset | Responsável |
|------|---------|-------------|
| Áudio PT-BR | CORAA SER (treino) + VERBO (reforço, se chegar) | P2 |
| Vídeo | RLVS (treino) + RWF-2000 (robustez, se chegar) | P4 |
| Pose | Frames RAVDESS/CREMA-D + anotação própria (sprint coletivo) | P5 |
| Fusão | Sintético (`events.jsonl`) | P1 |

## Decisões e trade-offs

| Decisão | Alternativas | Por quê esta |
|---------|--------------|--------------|
| Contratos imutáveis desde Etapa 2 | Evoluir schemas sob demanda | Paralelismo real entre P2–P5; schema fechado no clarify |
| `tipo_relato` taxonomia fechada + `outro` | Labels abertas / LLM | Léxico distinguível por regras PT-BR; contrato nunca falha |
| `tracks` mínimo (id, n_frames, bbox_media) | Expor keypoints no V1 | Fusão só precisa disso; V2 não acopla ao schema V1 |
| Pesos C/D documentados (dict) | Pesos aprendidos | Ajuste manual em um único lugar; três sinais primários iguais |
| Stubs na Etapa 1 | Esperar modelos para integrar | Integração cedo; risco de atraso de integração eliminado por desenho |
| P1 só na fusão (sem treino) | Dividir C/D em par | Menos discussão, mais desbloqueio |
| PLN por regras (A2) | LLM / NER treinado | Prazo e contrato estável |
| Go/no-go no fim da Etapa 3 | Continuar treinando na Etapa 4 | Congelamento e demo confiável |
| Núcleo CORAA + RLVS | Bloquear em VERBO/RWF | Extras viram trabalho futuro no relatório |

## Mapeamento Spec → Implementação

| Requisito / Story | Onde no código | Como verificar |
|-------------------|----------------|----------------|
| US-1 / RF-02, RF-12 | `src/audio/a1_stt`, `a2_nlp` | 3 áudios PT-BR + fallback offline |
| US-2 / RF-03 | `src/audio/a3_emotion` | F1 macro ≥ 0,75 ou baseline aceito no go/no-go |
| US-3 / RF-04–06 | `src/video/v1_*`, `v2_*`, `v3_*` | JSON de contrato; V3 acc ≥ 0,85 ou baseline |
| US-4 / RF-07–10 | `src/fusion` | Correlação 300 m / ±10 min + nota |
| US-5 / RF-01, RF-11 | `stubs/`, `run_event.py` | Pipeline E2E stubs (Etapa 1) e real (Etapa 3) |
| RNF-05–06 | branch `release`, README, relatório, vídeo | Checklist do edital na Etapa 5 |

## Riscos

| Risco | Resposta |
|-------|----------|
| VERBO/RWF-2000 não chegarem a tempo | Já previsto: núcleo fecha com CORAA + RLVS; extras viram "trabalho futuro" no relatório |
| Meta de modelo não bater na Etapa 3 | Go/no-go no fechamento: assume baseline mais simples e segue — aceite manda, não perfeição |
| Integração atrasar | Impossível por desenho: pipeline com stubs roda desde a Etapa 1 |
| Azure falhar na gravação | Fallback offline (faster-whisper) pronto desde a Etapa 1 |
| Um membro travar | P1 sem modelo próprio = socorro dedicado; ordem de prioridade definida na Etapa 3 (V3 > A3 > V2) |

## Conformidade com a constituição

- [x] Spec-first respeitado (artefatos em `specs/001-despacho-audio-video/`)
- [x] Separação what/how ok (`spec.md` sem stack de treino; este plan com stack)
- [x] Dependências mínimas justificadas (tabela Stack)
- [x] Testes previstos para critérios críticos (contratos, E2E, metas de aceite / baseline)
