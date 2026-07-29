# Plan: Triagem Multimodal em Consultas — Saúde da Mulher

> Referência: `specs/002-triagem-consulta/spec.md` · **Status:** done
> Herda da 001 tudo que não conflita: papéis P1–P5, Etapas 1–5, contratos-primeiro, stubs, go/no-go, congelamento.

## O que muda vs 001 (delta)

| Aspecto | 001 (rua/190) | 002 (consulta) |
|---|---|---|
| Gatilho | Ligação 190/193 | Gravação da consulta (presencial/teleconsulta) |
| V3 | Violência de cena (RLVS) | **Desconforto facial** (FER sobre RAVDESS/CREMA-D) |
| Correlação | Geo (300 m) + janela | **Sessão** (`session_id`) + janela ±10 min |
| Datasets vídeo | RLVS + RWF-2000 | RAVDESS/CREMA-D (face + frames de postura) |
| Saída | Nota de despacho | **Nota de triagem** à equipe de saúde |
| Declaração "elo simulado" | Necessária | Cai — áudio e vídeo são da mesma sessão |

## Papéis (P4 realocado)

| Quem | Frente |
|---|---|
| P1 | Integração, fusão C/D, demo, revisão (sem modelo próprio) |
| P2 | A3 emoção de voz PT-BR (CORAA + VERBO) |
| P3 | A1 Azure Speech (F0) + fallback; A2 PLN por regras (nova taxonomia) |
| P4 | **V3 desconforto facial** — fine-tune de FER pré-treinado em frames RAVDESS/CREMA-D (Colab T4) |
| P5 | V1 YOLOv8+ByteTrack; V2 YOLOv8-pose + cabeça de postura (sprint de anotação mantido) |

## Contratos (imutáveis a partir da Etapa 2 desta feature)

```text
A1/A2 → {transcricao, tipo_relato ∈ {violencia_domestica, sofrimento_emocional, outro}, local, tempo}
A3    → {sofrimento: 0..1}
V1    → {n_pessoas, tracks: [{id, n_frames, bbox_media:[x,y,w,h]}]}
V2    → {postura_defensiva: 0..1}
V3    → {desconforto_facial: 0..1}
C/D   → {escore, corroborado: bool, nota_ocorrencia}
Eventos → {session_id, modality, timestamp}
```

## Fusão

- `same_session`: `session_id` igual **e** |Δt| ≤ 10 min ⇒ `corroborado`.
- Pesos (dict único em `fusion/scoring`, soma 1.00): relato 0.28 · sofrimento 0.28 · desconforto_facial 0.22 · postura 0.22.
- **Corroboração não pesa no escore** (revisado em 26/07/2026). Na 002 áudio e vídeo vêm da mesma
  consulta, então `corroborado` é verdadeiro por construção — como termo de escore era uma constante
  somada a todo caso (72% de um escore de 0,138 no teste de simulação), sem discriminar nada. O flag
  permanece no contrato C/D (RF-07) e na nota, como proveniência. Os 0,10 foram redistribuídos
  proporcionalmente entre os quatro sinais medidos, preservando a razão herdada da 001.
- Sinal do relato: `violencia_domestica → 1.0`, `sofrimento_emocional → 0.6`, `outro → 0.0` (documentado).
- Compatibilidade: o caminho geográfico (haversine) permanece como utilitário legado; o primário é sessão.

## Metas de aceite (go/no-go Etapa 3)

| Módulo | Meta |
|---|---|
| A3 | F1 macro ≥ 0,75 (CORAA, split por locutor) |
| V3 | F1 macro ≥ 0,70 (fearful/sad vs neutral, split por ator) |
| V2 | F1 reportado no conjunto anotado (split por ator) |

## Datasets

| Ramo | Dataset | Responsável |
|---|---|---|
| Áudio PT-BR | CORAA SER (treino) + VERBO (reforço, se chegar) | P2 |
| Vídeo — face | RAVDESS + CREMA-D (frames; recorte feminino priorizado) | P4 |
| Vídeo — pose | Frames RAVDESS/CREMA-D + anotação própria (sprint coletivo) | P5 |
| Fusão | Sintético por sessão (`events.jsonl`) | P1 |

### Áudio PT-BR (T102) — detalhes operacionais

- Fonte CORAA SER v1.0: [rmarcacini/ser-coraa-pt-br](https://github.com/rmarcacini/ser-coraa-pt-br) (Drive público do shared-task).
- Scripts: `scripts/download_coraa.py` → `data/audio_ptbr/raw/`; `scripts/preprocess_audio.py` → 8 kHz mono + `labels.csv` (vocabulário `{neutral, non_neutral}`, split por locutor/gravação C-ORAL).
- Dep. extra: `gdown` (download dos zips do Google Drive).
- VERBO: solicitação manual por e-mail (não automatizar). O corpus não chegou a tempo; o núcleo do A3 fecha com o CORAA.
- Verificação: `scripts/verify_audio_dataset.py`.

### Vídeo — face (T104) — detalhes operacionais

- Fontes: RAVDESS speech video ([Zenodo 1188976](https://zenodo.org/record/1188976), `Video_Speech_Actor_01..24.zip`) + CREMA-D `VideoFlash` via mirror [GitLab CREMA-D-mirror](https://gitlab.com/cs-cooper-lab/crema-d-mirror) (preferir; o [GitHub original](https://github.com/CheyneyComputerScience/CREMA-D) usa git-lfs e costuma falhar).
- Scripts: `download_ravdess.py` → `data/video_consulta/raw/ravdess/`; `download_cremad.py` → `data/video_consulta/raw/cremad/` (sparse checkout do mirror); `extract_face_frames.py` → `data/video_consulta/processed/faces/` + `labels.csv`; `verify_face_dataset.py`.
- CREMA-D (decisão de peso): por padrão o pipeline usa emoções `NEU|FEA|SAD` de atrizes (`--all-actors` para incluir homens). VideoFlash completo no mirror ~2.3 GB.
- Rótulos pelo nome do arquivo (sem anotação manual). Binário V3: `desconforto` ← `{fearful,sad}`; `neutro` ← `{neutral,calm}` (`calm` entra em neutro para manter classe minoritária ≥ 40% no RAVDESS).
- Split **por ator** (ids `ravdess_XX` / `crema_YYYY`); estratificação com atrizes primeiro. Notebook esqueleto: `notebooks/t112_fer_colab.ipynb` (treino = T112).
- Detector default: YOLOv8n (pessoa) + recorte superior; alternativa `--detector haar`.

## Riscos (delta)

| Risco | Resposta |
|---|---|
| FER com poucas identidades (12–115 atores) | Split por ator; CREMA-D priorizado (diversidade); declarar limitação |
| Dado atuado ≠ consulta real | Declaração de proxy no relatório (RNF-06) |
| Demais riscos | Iguais à 001 (go/no-go, stubs, fallback Azure) |

## Conformidade com a constituição
- [x] Spec-first (este PR altera spec+plan+tasks junto com o código de realinhamento)
- [x] Separação what/how · [x] Dependências justificadas · [x] Testes para critérios críticos
