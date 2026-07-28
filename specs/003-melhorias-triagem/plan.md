# Plan: Melhorias de Triagem — separação de locutores e calibração de escala

> Referência: `specs/003-melhorias-triagem/spec.md` · **Status:** in-progress

## Decisão de arquitetura

A diarização vive num módulo próprio, `src/audio/diarize.py`, e **quem a consome é o A3** —
não o runner. Motivo: o `ResolvedPipeline` chama `infer(path)` de forma uniforme para módulo real
e stub; passar trechos como argumento extra quebraria o stub e vazaria detalhe de implementação
do A3 para a camada de integração. Com o consumo dentro do A3, o contrato `{sofrimento: 0..1}`
e a assinatura `infer(path)` ficam idênticos — nada muda para `resolve.py`, para o runner ou
para os stubs.

```text
run_event → resolve.call("a3_emotion", wav)
              └→ a3_emotion.infer(wav)
                   ├→ diarize.speaker_segments(wav)      # Azure; falha → None
                   ├→ sem diarização ou 1 locutor → pontua o áudio inteiro (hoje)
                   └→ N locutores → concatena por locutor → pontua cada um → max
```

## Componentes

| Arquivo | Papel |
|---|---|
| `src/audio/diarize.py` | `speaker_segments(path)` → `list[Segment]` ou `None`; `audio_by_speaker(path)` → `dict[str, np.ndarray]` |
| `src/audio/a3_emotion/infer.py` | consome a diarização; mantém janelas de 6 s e o **máximo** (PR #17) |
| `tests/test_diarize.py` | trechos → áudio por locutor; degradação sem credencial |
| `tests/test_a3_emotion.py` | contrato preservado com e sem diarização |

## Como a agregação fica

Hoje: `max(janelas de 6 s do áudio inteiro)`.
Com diarização: `max( max(janelas do locutor A), max(janelas do locutor B), … )`.

**O resultado NÃO é monotônico** — corrigido em 26/07 após medição. A hipótese inicial era que o
máximo por locutor nunca seria menor que o máximo global, mas isso é falso: ao concatenar a fala
de cada locutor, **as fronteiras das janelas mudam**. Uma janela do áudio bruto pode conter uma
combinação de trechos que pontua alto e que simplesmente deixa de existir depois do reagrupamento.

Medido na gravação de simulação: 0,119 sem separação contra 0,016 com separação (ver T203). O
valor mais alto vinha da primeira janela do áudio bruto, que mistura a abertura do atendente com
o início da fala da vítima — ou seja, era justamente o tipo de janela contaminada que a feature
existe para eliminar. Escore menor aqui não é regressão: é a remoção de um artefato.

Consequência para o relatório: a separação torna o escore **mais fiel ao que cada voz carrega**,
não necessariamente mais alto.

## Detalhes de implementação

- **API:** `speechsdk.transcription.ConversationTranscriber` com reconhecimento contínuo; cada
  evento traz `result.speaker_id`, `result.offset` e `result.duration` (unidades de 100 ns).
- **Trechos curtos (RF-23):** o modelo foi treinado em janelas de 6 s. Cada locutor tem seus
  trechos **concatenados** antes do janelamento, senão um "sim" de 0,4 s viraria uma amostra
  isolada e ruidosa.
- **Descarte:** locutor com menos de 1 s de fala total é ignorado — não dá para estimar emoção.
- **`speaker_id` desconhecido:** o Azure devolve `Unknown` em trechos sem atribuição; esses
  entram num balde próprio e são pontuados como qualquer outro (não são descartados, para não
  perder sinal).
- **Degradação (RF-22):** qualquer exceção do SDK, ausência de chave ou retorno vazio ⇒
  `speaker_segments` devolve `None` e o A3 segue pelo caminho atual, logando o motivo — mesma
  filosofia do `resolve.py`.
- **Cache em processo:** memoiza por `(path, mtime)`, para uma segunda chamada na mesma execução
  não pagar a rede de novo.

## Calibração de escala na fusão (RF-25)

**O problema medido.** Cada modelo tem fronteira de decisão própria, escolhida na validação:
A3 em **0,17**, V3 em **0,70**, V2 sem medida (fica em 0,50). A fusão somava os três como se
todos decidissem em 0,5 — subestimando o A3 e superestimando o V3, nas duas direções ao mesmo
tempo.

Quanto isso custava, com número: no teste do CORAA, a mediana dos áudios **neutros** é 0,022 e o
p90 é 0,096. O vídeo de simulação real pontuou **0,119** — acima de 90% dos neutros, ou seja,
evidência moderada de sofrimento. A fusão lia isso como 0,119 numa escala de 0 a 1, praticamente
nada. Na faixa 0,1–0,2 do conjunto de teste há 2 áudios neutros e 4 com sofrimento: um escore ali
é ~6× mais provável de vir de voz sofrida.

**A regra.** `calibrate(bruto, limiar)` comprime `[0, limiar]` em `[0, 0.5]` e estica
`[limiar, 1]` em `[0.5, 1]`. Monotônica, preserva 0 e 1, e vira identidade quando o limiar é 0,5.

**Onde ficam os limiares.** Constantes em `fusion/scoring.DECISION_THRESHOLDS`, com a fonte
anotada; `test_limiares_batem_com_as_metricas` compara com `models/a3_threshold_metrics.json` e
`models/v3_clip_metrics.json`. Recalibrar um modelo sem atualizar a constante quebra o teste.

**Efeito medido nos três casos de demonstração:**

| Caso | A3 bruto → calibrado | V3 bruto → calibrado | Escore antes | depois |
|---|---|---|---|---|
| Pizza (real, 190) | 0,119 → 0,350 | — | 0,055 | **0,120** |
| Consulta-A (IA) | 0,024 → 0,071 | 0,950 → 0,917 | 0,637 | 0,642 |
| Denúncia (IA) | 0,015 → 0,044 | 0,990 → 0,983 | 0,515 | 0,522 |

O caso com voz humana real mais que dobra; os gerados quase não mudam. É o comportamento
esperado de uma correção de escala — conserta o que estava distorcido e deixa o resto quieto.

**Hipótese testada e refutada, para não voltar.** Suspeitamos que o A3 sofresse de descasamento
de banda: o CORAA foi processado a 8 kHz (nada acima de 4 kHz) e os vídeos entram em banda cheia.
Limitando o áudio de demonstração a 4 kHz para imitar o treino, o escore **piorou** (0,119 →
0,014). Não é banda.

## Riscos

| Risco | Resposta |
|---|---|
| Diarização erra o agrupamento e junta duas vozes | O escore pode ficar **menor** que o do áudio bruto (ver "Como a agregação fica" — não é monotônico). Aceito: o valor maior vinha de janela contaminada, que é o artefato a eliminar |
| Chamada extra ao Azure encarece/atrasa | Tier F0 cobre; medir e reportar no `[tempo]` do runner |
| Azure indisponível na hora da demo | RF-22 garante o caminho atual; ensaiar com `TC4_FORCE_STUBS=1` como plano B |
| Mexer no módulo do P2 durante congelamento | Mudança aditiva: sem diarização, o comportamento é bit-a-bit o de hoje; testes cobrem os dois caminhos |

## Conformidade com a constituição

- [x] Spec-first (esta pasta antes do código)
- [x] Separação what/how
- [x] Dependências justificadas — **nenhuma nova**
- [x] Testes para critérios críticos (degradação e contrato)
- [x] Princípio VII — o relatório declara o que a diarização faz e o que ela não faz
