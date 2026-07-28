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

### US-3 — Somar sinais que significam a mesma coisa
**Como** equipe de triagem, **quero** que os sinais dos módulos entrem na nota numa escala
comum, **para** que um sinal fraco de um módulo rigoroso não seja confundido com ausência de
sinal.
- **WHEN** um modelo tem fronteira de decisão medida diferente de 0,5 **THEN** seu escore é
  recolocado numa escala em que 0,5 é aquela fronteira, antes da soma ponderada
- **WHEN** um modelo não tem fronteira medida **THEN** assume-se 0,5 e a calibração é identidade

### US-4 — Não identificar pessoas
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
| RF-25 | A fusão calibra cada saída de modelo para que 0,5 seja a fronteira de decisão dele, antes da soma ponderada | must |
| RF-26 | Os limiares usados vêm dos arquivos versionados em `models/`, e um teste guarda a sincronia | must |
| RF-27 | `relato` **não** é calibrado — é mapa discreto documentado, não saída de classificador | must |

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
- Refatorar o A1 para reaproveitar a mesma chamada — desejável, mas o A1 é módulo separado e o projeto
  está entrando em congelamento. Registrado como trabalho futuro.
- Identificar qual locutor é a paciente.
- Retreinar o A3.

## Decisão registrada — por que esta calibração e não outra

O método canônico para isto é **Platt scaling** ou **regressão isotônica**: ajustar uma curva
sobre um conjunto de calibração para que a saída vire probabilidade de fato. É estatisticamente
mais rigoroso.

Optamos pela **ancoragem no limiar** (regra de três em dois trechos) por três razões:

1. usa um número **já medido e versionado** — o limiar da validação de cada modelo;
2. é uma conta que qualquer integrante refaz no papel, o que atende ao Princípio VII;
3. não exige conjunto de calibração novo nem artefato treinado por módulo, num momento em que
   o projeto entra em congelamento.

O custo: a saída **não** é probabilidade calibrada no sentido estatístico — 0,7 não significa
"70% dos casos assim são positivos". Ela é um **índice de risco comparável entre módulos**, que é
o que a soma ponderada precisa. Declarar isso no relatório.

## Perguntas em aberto

- O teto de 30 s do A3 (`MAX_DURATION_S`) continua valendo por locutor. Para consulta longa isso
  ainda descarta a maior parte do áudio — tratar em melhoria futura.
