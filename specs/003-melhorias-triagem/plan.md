# Plan: Melhorias de Triagem — separação de locutores

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

## Riscos

| Risco | Resposta |
|---|---|
| Diarização erra o agrupamento e junta duas vozes | O máximo por locutor continua ≥ ao de hoje; pior caso é empatar com o comportamento atual |
| Chamada extra ao Azure encarece/atrasa | Tier F0 cobre; medir e reportar no `[tempo]` do runner |
| Azure indisponível na hora da demo | RF-22 garante o caminho atual; ensaiar com `TC4_FORCE_STUBS=1` como plano B |
| Mexer no módulo do P2 durante congelamento | Mudança aditiva: sem diarização, o comportamento é bit-a-bit o de hoje; testes cobrem os dois caminhos |

## Conformidade com a constituição

- [x] Spec-first (esta pasta antes do código)
- [x] Separação what/how
- [x] Dependências justificadas — **nenhuma nova**
- [x] Testes para critérios críticos (degradação e contrato)
- [x] Princípio VII — o relatório declara o que a diarização faz e o que ela não faz
