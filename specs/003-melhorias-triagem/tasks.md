# Tasks: Melhorias de Triagem — separação de locutores

> Referências: `spec.md` + `plan.md`. Marque `[x]` só após verificar o critério.

## Implementação

- [x] T200 P1: `src/audio/diarize.py` — `speaker_segments()` via `ConversationTranscriber`,
      `audio_by_speaker()` concatenando trechos por locutor, cache por `(path, mtime)`
  - **Verificar:** devolve `None` (não levanta) sem `AZURE_SPEECH_KEY`, com motivo logado
- [x] T201 P1: A3 consome a diarização — pontua cada locutor e devolve o máximo
  - **Verificar:** contrato `{sofrimento: 0..1}` intacto; `infer(path)` sem argumento novo;
    stub segue compatível
- [x] T202 P1: testes de degradação e de contrato nos dois caminhos (com e sem diarização)
  - **Verificar:** `pytest` verde sem credencial Azure configurada
- [x] T203 P1: teste A/B na gravação de simulação — escore com e sem separação, lado a lado
  - **Verificado (gravação `simulacao-ligacao-190`, 52,7 s, 2 locutores):**

    | | Escore de sofrimento |
    |---|---|
    | Sem separação (áudio inteiro) | 0,119 |
    | Guest-1 (16,8 s de fala) | 0,016 |
    | Guest-2 (18,6 s de fala) | 0,016 |
    | **Com separação (máximo)** | **0,016** |

    **A separação BAIXOU o escore** — e isso refuta a hipótese de monotonicidade que estava no
    `plan.md` (corrigida). Motivo: concatenar por locutor **muda as fronteiras das janelas**; o
    0,119 vinha da primeira janela do áudio bruto, que mistura a abertura do atendente com o
    início da fala da vítima. Era exatamente a janela contaminada que a feature elimina.

    **Leitura honesta:** nesta gravação a feature não melhorou a detecção — o A3 não vê
    sofrimento em nenhum agrupamento (nem 0,02 por locutor). Ela removeu um artefato, o que é
    correto, mas **não há evidência de ganho de sensibilidade**. Falta testar numa consulta
    simulada de verdade (dois locutores, cenário da 002) antes de afirmar benefício.

- [x] T210 P1: calibração de escala na fusão (RF-25/26/27)
  - **Verificado:** `calibrate()` ancora o limiar de cada modelo em 0,5; limiares vindos de
    `models/*.json` com teste de sincronia; `relato` fora da calibração; `calibrado=False`
    reproduz o comportamento anterior para comparação. Efeito: pizza 0,055 → **0,120**
    (A3 0,119 → 0,350); casos gerados praticamente inalterados. `pytest` 85 passed.
  - **Refutado no caminho:** descasamento de banda 8 kHz × banda cheia não explica o escore
    baixo do A3 — limitar a 4 kHz piora (0,119 → 0,014).

## Fechamento

- [x] T290 Atualizar `specs/README.md` (índice) e declarar a diarização no RNF-24
  - **Verificado (27/07):** 003 registrada no índice; diarização e calibração de escala entram nas
    ressalvas da ata do go/no-go (`docs/go-no-go-etapa3.md`), que alimenta o RNF-06 do relatório.
- [ ] T291 Status `done` em `spec.md` / `plan.md` quando o A/B estiver registrado

## Notas de execução

1. **Contratos da 002 são imutáveis** — esta feature não os toca (RNF-20).
2. As métricas publicadas do A3 seguem válidas: o CORAA tem um locutor por clipe, então a
   diarização não altera a avaliação (RNF-22).
3. Sem diarização disponível, o comportamento tem de ser **bit-a-bit** o atual.
