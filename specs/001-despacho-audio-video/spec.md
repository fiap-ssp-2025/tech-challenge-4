# Spec: Despacho Inteligente Áudio–Vídeo (190/193)

> **Status:** draft  
> **Branch sugerida:** `feat/001-despacho-audio-video`  
> **Criado em:** 2026-07-23

## Visão

Sistema de apoio ao despacho em que a **ligação 190/193** é o gatilho (relato + emoção na voz, PT-BR) e a **câmera da região** é a corroboração sob demanda (violência + postura). A correlação por local/tempo prioriza o alerta; a **decisão final permanece humana**.

## Contexto / Problema

Centrais de emergência recebem áudio com relato e carga emocional, mas a corroboração visual da região é cara e nem sempre imediata. Sem fusão local/tempo, o despacho perde priorização. O produto precisa de um pipeline ponta a ponta que produza uma nota de ocorrência priorizada — com modelos que **passem no aceite**, não modelos perfeitos.

## User Stories

### US-1 — Transcrever e estruturar a ligação

**Como** operador / sistema de despacho,  
**quero** obter transcrição e campos estruturados da ligação (tipo de relato, local, tempo),  
**para** disparar o fluxo de alerta sem depender só do áudio bruto.

#### Critérios de aceite

- **WHEN** chega um áudio de ligação válido em PT-BR  
  **THEN** o módulo A1/A2 devolve JSON com `transcricao`, `tipo_relato`, `local` e `tempo`

- **WHEN** o serviço de STT em nuvem está indisponível  
  **THEN** o sistema usa fallback offline e ainda produz a saída contratada (ou falha explícita documentada)

### US-2 — Estimar sofrimento na voz

**Como** sistema de priorização,  
**quero** um escore de sofrimento emocional na voz (`0..1`),  
**para** elevar a prioridade quando o relato verbal e a emoção reforçam urgência.

#### Critérios de aceite

- **WHEN** o áudio da ligação é processado pelo módulo A3  
  **THEN** a saída é `{sofrimento: número em 0..1}` conforme o contrato

### US-3 — Corroborar com vídeo da região

**Como** sistema de despacho,  
**quero** sob demanda analisar vídeo da região (contagem/tracks, postura defensiva, violência),  
**para** corroborar ou não o alerta da ligação.

#### Critérios de aceite

- **WHEN** um clipe de vídeo da região é solicitado  
  **THEN** V1 devolve `{n_pessoas, tracks}`, V2 `{postura_defensiva: 0..1}` e V3 `{violencia: 0..1}`

### US-4 — Fundir áudio e vídeo e priorizar

**Como** operador humano,  
**quero** um escore, flag de corroboração e nota de ocorrência,  
**para** decidir o despacho com apoio do sistema (sem automação da decisão final).

#### Critérios de aceite

- **WHEN** há evento de áudio e (opcionalmente) evidência de vídeo correlacionáveis por local/tempo  
  **THEN** C/D devolve `{escore, corroborado: bool, nota_ocorrencia}`

- **WHEN** áudio e vídeo estão dentro do raio e janela de correlação definidos no plano  
  **THEN** a fusão considera corroboração; fora disso, não marca `corroborado` indevidamente

### US-5 — Pipeline ponta a ponta reproduzível

**Como** time de entrega,  
**quero** executar um evento (`.wav` + `.mp4`) e obter a nota priorizada de ponta a ponta,  
**para** demonstrar e validar o challenge.

#### Critérios de aceite

- **WHEN** executo o runner de evento com áudio e vídeo de teste  
  **THEN** obtenho a nota priorizada sem intervenção manual nos módulos intermediários

- **WHEN** um módulo ainda não tem modelo real  
  **THEN** um stub com o mesmo contrato JSON permite o pipeline continuar

## Requisitos funcionais

| ID | Requisito | Prioridade |
|----|-----------|------------|
| RF-01 | Módulos expõem contrato JSON fixo via `infer` (A1/A2, A3, V1, V2, V3, C/D) | must |
| RF-02 | A1/A2: `{transcricao, tipo_relato, local, tempo}` | must |
| RF-03 | A3: `{sofrimento: 0..1}` | must |
| RF-04 | V1: `{n_pessoas, tracks}` | must |
| RF-05 | V2: `{postura_defensiva: 0..1}` | must |
| RF-06 | V3: `{violencia: 0..1}` | must |
| RF-07 | C/D: `{escore, corroborado: bool, nota_ocorrencia}` | must |
| RF-08 | Correlação por local/tempo prioriza o despacho | must |
| RF-09 | Ligação é o gatilho; vídeo é corroboração sob demanda | must |
| RF-10 | Decisão final do despacho é humana (sistema só apoia) | must |
| RF-11 | Stubs com o mesmo contrato permitem integração antes dos modelos reais | must |
| RF-12 | Fallback de STT offline quando o serviço em nuvem falhar | must |

## Requisitos não funcionais

| ID | Requisito | Prioridade |
|----|-----------|------------|
| RNF-01 | Contratos JSON definidos na Etapa 1 e imutáveis a partir da Etapa 2 | must |
| RNF-02 | Integração cedo: pipeline ponta a ponta com stubs desde o fim da Etapa 1 | must |
| RNF-03 | Congelamento na Etapa 4 (nada de treino novo após o go/no-go da Etapa 3) | must |
| RNF-04 | Meta de aceite > perfeição de modelo; se meta não bater, assume baseline mais simples | must |
| RNF-05 | Ambiente reproduzível (`uv sync` / instalação limpa) para demo e entrega | must |
| RNF-06 | Relatório e demo cobrem: análise áudio/vídeo, anomalias, integração nuvem, fluxo do alerta e as 4 declarações obrigatórias (proxy, elo simulado, vídeo sob demanda, humano no circuito) | must |

## Fora de escopo

- Decisão automática de despacho (sem humano)
- Monitoramento contínuo de todas as câmeras (vídeo é sob demanda)
- Modelo “perfeito”; basta passar no aceite / baseline
- Dependência de datasets de reforço (VERBO, RWF-2000) se não chegarem a tempo — núcleo fecha com CORAA + RLVS

## Perguntas em aberto

- [ ] Pesos exatos do escore ponderado da fusão C/D (detalhar no plan se ainda implícitos)
- [ ] Formato canônico de `tracks` em V1 (lista de IDs, bbox, etc.)
- [ ] Taxonomia fechada de `tipo_relato` no PLN por regras

## Notas

- Glossário: **A*** = ramo áudio; **V*** = ramo vídeo; **C/D** = correlação / despacho (fusão).
- Regra de ouro do challenge: **contratos primeiro, integração cedo, congelamento na Etapa 4**.
- Cronologia de execução: Etapas 1–5 (sem janela de calendário fixa) — ver `plan.md`.
