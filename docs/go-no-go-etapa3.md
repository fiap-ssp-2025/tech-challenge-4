# Ata de go/no-go — fechamento da Etapa 3

> **Decisão: GO.** Seguimos para congelamento, relatório e vídeo com os modelos que temos.
> **Data:** 27/07/2026 · **Tarefas:** T023 / T120 (herdada da 001, ajustada na 002)
> **Participantes:** P1–P5 (decisão do grupo)

## Decisão

O grupo considera as métricas **aceitáveis para a entrega** e encerra a fase de treino. A partir
desta ata, pela regra do próprio plano, **não se treina mais nenhum modelo** — o trabalho restante
é bug bash, relatório e vídeo.

A decisão é consciente e leva em conta que duas das três metas foram atingidas por margem estreita,
e que dois módulos não foram validados no domínio de uso. Ambos os pontos estão declarados abaixo e
devem aparecer no relatório.

## Métricas na mesa

| Módulo | Meta do plano | Resultado | Situação |
|---|---|---|---|
| **A3** — sofrimento na voz (P2) | F1 macro ≥ 0,75 | **0,7896** (limiar 0,17 calibrado na validação); 0,7171 no limiar padrão | ✅ atingida no limiar calibrado |
| **V3** — desconforto facial (P4) | F1 macro ≥ 0,70 | **0,7108** por clipe · 0,7045 por frame; AUC 0,8076 | ✅ atingida sem ajuste de limiar |
| **V2** — postura defensiva (P5) | reportar o F1 | **0,6868** (split por ator, seed 42) | ✅ critério era reportar |
| **A1/A2** — STT e PLN (P3) | JSON estruturado em casos de teste | Azure validado em áudio real PT-BR; A2 extrai relato, local e tempo | ✅ |
| **C/D** — fusão (P1) | contratos respeitados ponta a ponta | 6 módulos reais; 85 testes verdes | ✅ |

## Ressalvas que acompanham o GO

Estas não invalidam a decisão, mas **precisam ser declaradas no relatório**:

1. **Margem estreita nas duas metas numéricas.** Os conjuntos de teste são pequenos — 101 áudios
   (23 positivos) para o A3 e 669 clipes para o V3. A incerteza é da ordem de ±0,03 a ±0,10:
   as metas foram *atingidas*, não *superadas com folga*.

2. **Rótulo por proxy no V2 e no V3.** Ambos usam emoção atuada (RAVDESS/CREMA-D) como
   aproximação de postura defensiva e desconforto clínico. Não há anotação humana de consulta real.
   A validação com dado clínico fica como trabalho futuro.

3. **A3 e V2 não foram validados no domínio de uso.** Nos testes com gravações de consulta, o A3
   ficou em torno de 0,02 e o V2 em 0,08. As amostras disponíveis eram poucas e de qualidade de
   atuação variável, **e não houve controle negativo no domínio** — então não é possível concluir
   se os módulos não transferem ou se as amostras não carregavam o sinal. Registrado como
   limitação e como próximo experimento.

4. **Mídia sintética não serve para avaliar os modelos de emoção.** Vídeos gerados por IA passaram
   pelos módulos de texto e de face, mas o A3 os lê como voz neutra (≈0,02). Modelos treinados em
   voz humana real não podem ser avaliados com voz sintetizada.

5. **O escore de triagem não é probabilidade.** Após a calibração (spec 003), é um índice de risco
   comparável entre módulos — 0,7 **não** significa "70% de chance de violência".

## O que fica como trabalho futuro (não bloqueia a entrega)

- Anotação humana de postura e de expressão em consulta real.
- Conjunto de teste de áudio no domínio clínico, **com casos neutros e casos com sofrimento**,
  para medir a transferência do A3 de forma conclusiva.
- Separação de locutores avaliada em material adequado (spec 003 implementada, sem evidência de
  ganho até agora).
- PLN além de regras: o teste com relato velado mostrou que palavras-chave não capturam divulgação
  indireta — que é a forma mais comum na prática.

## Consequências imediatas

- Etapa 3 encerrada. Congelamento autorizado.
- Nenhum treino novo, nenhuma mudança de arquitetura de modelo.
- Correções permitidas apenas para bug bash, documentação e demo.
