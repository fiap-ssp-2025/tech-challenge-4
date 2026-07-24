# Plan: Triagem Multimodal em Consultas — Saúde da Mulher

> Referência: `specs/002-triagem-consulta/spec.md` · **Status:** in-progress
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
- Pesos (dict único em `fusion/scoring`, soma 1.00): relato 0.25 · sofrimento 0.25 · desconforto_facial 0.20 · postura 0.20 · corroboração 0.10.
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
- VERBO: solicitação manual — texto em `docs/verbo-solicitacao.md` (não automatizar).
- Verificação: `scripts/verify_audio_dataset.py`.

## Riscos (delta)

| Risco | Resposta |
|---|---|
| FER com poucas identidades (12–115 atores) | Split por ator; CREMA-D priorizado (diversidade); declarar limitação |
| Dado atuado ≠ consulta real | Declaração de proxy no relatório (RNF-06) |
| Demais riscos | Iguais à 001 (go/no-go, stubs, fallback Azure) |

## Conformidade com a constituição
- [x] Spec-first (este PR altera spec+plan+tasks junto com o código de realinhamento)
- [x] Separação what/how · [x] Dependências justificadas · [x] Testes para critérios críticos
