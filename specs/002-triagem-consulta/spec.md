# Spec: Triagem Multimodal em Consultas — Saúde da Mulher

> **Status:** in-progress
> **Branch:** `feat/002-triagem-consulta`
> **Criado em:** 2026-07-23
> **Substitui:** `specs/001-despacho-audio-video` (superseded — reancoragem ao contexto hospitalar por orientação do professor)

## Visão

Sistema de apoio à triagem na rede hospitalar: a **gravação da consulta** (presencial ou teleconsulta, PT-BR) alimenta dois ramos — áudio (relato + sofrimento na voz) e vídeo (desconforto facial + postura defensiva + presença de acompanhante). Por virem da **mesma sessão**, os sinais são fundidos por correlação de sessão e geram um **alerta de triagem à equipe especializada**. A decisão final permanece humana.

## Contexto / Problema

O edital (TC 4) pede monitoramento multimodal para "sinais precoces de risco específicos da saúde e segurança feminina": sinais não-verbais de desconforto/medo em consultas, padrões vocais indicativos de trauma/depressão pós-parto, linguagem corporal indicativa de abuso e alertas à equipe em tempo real. Vítimas de violência raramente relatam diretamente; sinais vocais e corporais durante a consulta são o indício disponível. O sistema sinaliza casos para atenção especializada — triagem, não diagnóstico.

## User Stories

### US-1 — Transcrever e estruturar a fala da consulta
**Como** equipe de triagem, **quero** transcrição e campos estruturados (indicador de relato, contexto), **para** disparar o fluxo de alerta.
- **WHEN** chega áudio de consulta em PT-BR **THEN** A1/A2 devolve `{transcricao, tipo_relato, local, tempo}` com `tipo_relato ∈ {violencia_domestica, sofrimento_emocional, outro}`
- **WHEN** o STT em nuvem falha **THEN** fallback offline produz a saída contratada (ou falha explícita)

### US-2 — Estimar sofrimento na voz
- **WHEN** o áudio é processado pelo A3 **THEN** sai `{sofrimento: 0..1}` (modelo PT-BR)

### US-3 — Ler sinais visuais da mesma consulta
- **WHEN** o vídeo da sessão é processado **THEN** V1 devolve `{n_pessoas, tracks}` (presença/dominância de acompanhante), V2 `{postura_defensiva: 0..1}` (YOLOv8-pose customizado) e V3 `{desconforto_facial: 0..1}`

### US-4 — Fundir por sessão e alertar
**Como** profissional de saúde, **quero** escore, flag de corroboração e nota de triagem, **para** priorizar atenção especializada sem automação da decisão.
- **WHEN** áudio e vídeo compartilham `session_id` dentro da janela temporal **THEN** C/D marca `corroborado` e devolve `{escore, corroborado, nota_ocorrencia}`
- **WHEN** sessões diferem ou tempos divergem **THEN** não marca `corroborado`

### US-5 — Pipeline ponta a ponta reproduzível
- **WHEN** executo o runner com áudio+vídeo de uma sessão **THEN** obtenho a nota de triagem sem intervenção manual; stubs mantêm o pipeline vivo onde faltar modelo real

## Requisitos funcionais

| ID | Requisito | Prioridade |
|----|-----------|------------|
| RF-01 | Contrato JSON fixo por módulo via `infer` | must |
| RF-02 | A1/A2: `{transcricao, tipo_relato, local, tempo}`; taxonomia `violencia_domestica \| sofrimento_emocional \| outro` | must |
| RF-03 | A3: `{sofrimento: 0..1}` | must |
| RF-04 | V1: `{n_pessoas, tracks[{id, n_frames, bbox_media}]}` | must |
| RF-05 | V2: `{postura_defensiva: 0..1}` — YOLOv8-pose customizado | must |
| RF-06 | V3: `{desconforto_facial: 0..1}` | must |
| RF-07 | C/D: `{escore, corroborado, nota_ocorrencia}` | must |
| RF-08 | Correlação por sessão (`session_id` + janela temporal) | must |
| RF-09 | Alerta é encaminhamento à equipe; decisão humana | must |
| RF-10 | Stubs com contrato idêntico permitem integração antecipada | must |
| RF-11 | Fallback de STT offline | must |

## Requisitos não funcionais

| ID | Requisito | Prioridade |
|----|-----------|------------|
| RNF-01 | Contratos imutáveis a partir da Etapa 2 (desta feature) | must |
| RNF-02 | Pipeline E2E com stubs desde a Etapa 1 (já herdado da 001) | must |
| RNF-03 | Congelamento na Etapa 4; go/no-go na Etapa 3 | must |
| RNF-04 | Aceite > perfeição; baseline se meta não bater | must |
| RNF-05 | Reproduzível com `uv sync` | must |
| RNF-06 | Relatório/demo cobrem: análise áudio/vídeo, anomalias, Azure, fluxo do alerta e as declarações obrigatórias (dados de proxy atuados; consulta simulada; consentimento no contexto clínico/LGPD; humano no circuito) | must |

## Fora de escopo

- Cenário de via pública / despacho 190-193 (vetado pela orientação do professor — ver 001 superseded)
- Diagnóstico clínico automatizado; decisão sem humano
- Datasets de violência de cena (RLVS, RWF-2000) — removidos com o cenário
- Dependência do VERBO se não chegar (núcleo fecha com CORAA)

## Notas

- Ganho da reancoragem: áudio e vídeo agora vêm do **mesmo evento** (a consulta) — a fusão dispensa o elo sintético espaçotemporal da 001; RAVDESS (audiovisual) permite demonstrar os dois ramos sobre o mesmo clipe.
- Glossário mantido: A* = áudio; V* = vídeo; C/D = correlação/decisão (fusão).
