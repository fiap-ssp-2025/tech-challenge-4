# Spec: Melhorias de Triagem — separação de locutores no ramo de áudio

> **Status:** in-progress
> **Branch:** `feat/003-diarizacao-a3`
> **Criado em:** 2026-07-26
> **Depende de:** `specs/002-triagem-consulta` (in-progress) — esta feature **não** altera contratos.

## Visão

O escore de sofrimento na voz (A3) é calculado sobre **todo o áudio da sessão**, sem saber de
quem é cada voz. Numa consulta há pelo menos duas pessoas falando — profissional de saúde e
paciente — e o sinal que interessa é o da paciente. Esta feature separa os locutores e pontua
cada um em separado, para que a voz de quem conduz a consulta não dilua a de quem está em risco.

## Contexto / Problema

Teste com gravação de simulação de violência doméstica (26/07/2026): o A3 devolveu
**sofrimento = 0,03** num caso que é emergência de livro-texto. A análise janela a janela
mostrou que nenhum trecho passou de 0,119 — e boa parte do áudio era o **atendente**, voz
neutra e profissional, cuja presença puxa qualquer agregação para baixo.

A agregação já foi trocada de média para máximo (PR #17), o que ajuda, mas não resolve: o
máximo continua podendo vir do locutor errado, e o modelo não tem como saber.

**Diarização** — descobrir quem fala quando — é a peça que falta. O SDK do Azure Speech já
instalado (1.51.0) oferece `ConversationTranscriber`, que devolve os trechos rotulados por
locutor. Não é preciso dependência nova nem modelo adicional.

## User Stories

### US-1 — Não misturar vozes ao medir sofrimento
**Como** equipe de triagem, **quero** que o escore de sofrimento venha da voz de uma pessoa por
vez, **para** que a fala neutra do profissional não mascare o sofrimento da paciente.
- **WHEN** o áudio da sessão tem mais de um locutor **THEN** o A3 pontua cada locutor
  separadamente e devolve o **maior** escore
- **WHEN** o áudio tem um locutor só **THEN** o resultado é idêntico ao comportamento atual

### US-2 — Degradar sem quebrar
**Como** integrante do time, **quero** que o pipeline continue rodando quando a diarização não
estiver disponível, **para** que a ausência de credencial ou de rede não derrube a demo.
- **WHEN** falta `AZURE_SPEECH_KEY`, a rede falha ou o serviço erra **THEN** o A3 pontua o
  áudio inteiro como hoje, **registrando o motivo no log**
- **WHEN** a diarização devolve um único locutor **THEN** não há chamada extra de inferência

### US-3 — Não identificar pessoas
**Como** responsável pelo projeto, **quero** que o sistema **não** tente descobrir *quem* é a
paciente, **para** não introduzir decisão sobre identidade nem tratar dado biométrico além do
necessário.
- **WHEN** há N locutores **THEN** o sistema pontua os N e toma o máximo, sem rotular papéis

## Requisitos funcionais

| ID | Requisito | Prioridade |
|----|-----------|------------|
| RF-20 | `src/audio/diarize.py` devolve trechos `{locutor, inicio_s, fim_s}` via Azure `ConversationTranscriber` | must |
| RF-21 | A3 pontua cada locutor em separado e devolve o **máximo** entre eles | must |
| RF-22 | Falha ou ausência de diarização ⇒ comportamento atual, com motivo logado | must |
| RF-23 | Trechos menores que a janela do modelo (6 s) são concatenados por locutor antes de pontuar | must |
| RF-24 | Nenhuma tentativa de identificar papéis (paciente/profissional) ou identidade | must |

## Requisitos não funcionais

| ID | Requisito | Prioridade |
|----|-----------|------------|
| RNF-20 | **Contratos da 002 permanecem intactos** — A3 continua devolvendo `{sofrimento: 0..1}` e o stub segue compatível | must |
| RNF-21 | Sem dependência nova: usa o SDK Azure já instalado | must |
| RNF-22 | Métricas publicadas do A3 (F1 0,7896) permanecem válidas — o CORAA tem um locutor por clipe, então a diarização não altera a avaliação | must |
| RNF-23 | Custo adicional cabe no tier F0 (5 h/mês) — uma chamada extra por sessão | should |
| RNF-24 | Relatório declara a diarização como etapa do pipeline e o que ela **não** faz (não identifica pessoas) | must |

## Fora de escopo

- Diarização local (`pyannote.audio`) — dependência pesada e modelo com aceite manual; reavaliar
  se o Azure sair do ar.
- Refatorar o A1 para reaproveitar a mesma chamada — desejável, mas é módulo do P3 e o projeto
  está entrando em congelamento. Registrado como trabalho futuro.
- Identificar qual locutor é a paciente.
- Retreinar o A3.

## Perguntas em aberto

- O teto de 30 s do A3 (`MAX_DURATION_S`) continua valendo por locutor. Para consulta longa isso
  ainda descarta a maior parte do áudio — tratar em melhoria futura.
