# Relatório Técnico — Tech Challenge 4

## Triagem Multimodal em Consultas — Saúde da Mulher

**Grupo:** P1 (integração e fusão) · P2 (emoção na voz) · P3 (STT e PLN) · P4 (expressão facial) · P5 (visão corporal)
**Repositório:** `fiap-ssp-2025/tech-challenge-4` · **Data:** julho de 2026

---

## 1. Resumo executivo

Construímos um sistema de **apoio à triagem** que analisa a gravação de uma consulta de saúde da
mulher e produz uma **nota indicativa de risco** para a equipe especializada. A gravação alimenta
dois ramos — áudio e vídeo — processados por cinco modelos independentes cujas saídas são fundidas
num escore único.

A decisão final é **sempre humana**. O sistema sinaliza; não diagnostica, não classifica pessoas e
não substitui julgamento clínico.

**Estado da entrega:** os seis módulos do pipeline rodam com modelos reais, ponta a ponta, com 77
testes automatizados verdes. As três metas de acurácia definidas no plano foram atingidas.

| Módulo | Meta | Resultado |
|---|---|---|
| A3 — sofrimento na voz | F1 macro ≥ 0,75 | **0,7896** |
| V3 — desconforto facial | F1 macro ≥ 0,70 | **0,7108** |
| V2 — postura defensiva | reportar | **0,6868** |

O resultado mais significativo não está na tabela: aplicado a uma **ocorrência real** de violência
doméstica, o módulo de voz produziu escore **0,479** — acima de 97,4% dos áudios neutros do
conjunto de referência. Voltaremos a isso na Seção 7, junto com o que ele revela sobre a diferença
entre validar um modelo e validá-lo *no domínio de uso*.

---

## 2. O problema

Violência doméstica raramente chega ao sistema de saúde como denúncia. Chega como cefaleia
persistente, insônia, dor difusa, ansiedade. A mulher frequentemente não verbaliza o que acontece —
por medo, por vergonha, ou porque o agressor está na sala.

O profissional de saúde tem poucos minutos por consulta e nem sempre formação específica para
reconhecer os sinais. A literatura registra que profissionais relatam **falta de preparo** para
abordar o tema.

**Nossa hipótese:** sinais de sofrimento distribuídos em canais diferentes — o que se diz, como se
diz, o que o rosto mostra, como o corpo se posiciona — isoladamente são fracos, mas **combinados**
podem elevar a atenção da equipe para casos que passariam despercebidos.

### O que este sistema não é

- Não é diagnóstico nem prova.
- Não identifica pessoas nem atribui papéis.
- Não decide encaminhamento — apenas informa quem decide.

---

## 3. Arquitetura

```mermaid
flowchart LR
    S(["Gravação da consulta<br/>áudio + vídeo, mesma sessão"])
    subgraph AUDIO["Ramo de áudio"]
        A1["A1 · Transcrição (STT)<br/>Azure Speech + fallback offline"]
        A2["A2 · PLN por regras<br/>tipo de relato, local, tempo"]
        A3["A3 · Sofrimento na voz<br/>wav2vec2 PT-BR"]
    end
    subgraph VIDEO["Ramo de vídeo"]
        V1["V1 · Pessoas e tracks<br/>YOLOv8 + ByteTrack"]
        V2["V2 · Postura defensiva<br/>YOLOv8-pose + GradientBoosting"]
        V3["V3 · Desconforto facial<br/>ViT pré-treinado em FER"]
    end
    CD["C/D · Fusão por sessão<br/>calibração + escore ponderado"]
    OUT(["Nota de triagem<br/>decisão final humana"])
    S --> A1 --> A2 --> CD
    S --> A3 --> CD
    S --> V1 --> CD
    S --> V2 --> CD
    S --> V3 --> CD
    CD --> OUT
```

### Decisão estruturante: contratos primeiro

Antes de qualquer modelo existir, definimos o **formato exato de saída de cada módulo** e o
congelamos. São contratos JSON validados em tempo de execução (`src/contracts/`):

| Módulo | Contrato |
|---|---|
| A1/A2 | `{transcricao, tipo_relato ∈ {violencia_domestica, sofrimento_emocional, outro}, local, tempo}` |
| A3 | `{sofrimento: 0..1}` |
| V1 | `{n_pessoas, tracks: [{id, n_frames, bbox_media}]}` |
| V2 | `{postura_defensiva: 0..1}` |
| V3 | `{desconforto_facial: 0..1}` |
| C/D | `{escore, corroborado, nota_ocorrencia}` |

Isso permitiu que cinco pessoas trabalhassem em paralelo sem se bloquear: cada uma sabia
exatamente o que receberia e o que deveria entregar, mesmo antes de o vizinho ter código.

### Decisão estruturante: stubs com degradação explícita

Cada módulo tem uma implementação falsa que devolve valor fixo **no formato correto**. A camada
`src/resolve.py` decide, a cada execução, se usa o modelo real ou o stub — e **sempre registra o
motivo**:

```
[resolve] a3_emotion  → stub  (artefato ausente: models/a3_emotion — rode scripts/download_a3_model.py)
```

Consequências práticas: o pipeline ponta a ponta funcionou desde o primeiro dia; um clone sem os
pesos treinados roda sem quebrar; e falha de credencial ou de rede degrada em vez de derrubar. Na
entrega final, todos os seis resolvem para real.

---

## 4. Dados

| Ramo | Fonte | Volume | Divisão |
|---|---|---|---|
| Áudio PT-BR | CORAA SER v1 | 933 áudios, 8 kHz mono | 666 / 166 / 101 — **por locutor** |
| Vídeo — face | RAVDESS + CREMA-D | 20.612 frames faciais | 14.388 / 2.996 / 3.228 — **por ator** (80/17/18) |
| Vídeo — postura | RAVDESS | 11.504 frames de corpo | **por ator** (8 atores) |
| Fusão | Sintético por sessão | `events.jsonl` | — |

**Divisão por pessoa, sempre.** Nenhum locutor ou ator aparece em treino e teste ao mesmo tempo.
Sem essa precaução o modelo memoriza vozes e rostos, e a métrica sai inflada — mediria capacidade
de reconhecer indivíduos, não de reconhecer emoção. Há verificação automatizada para isso
(`scripts/verify_audio_dataset.py`, `scripts/verify_face_dataset.py`).

---

## 5. Módulos

### 5.1 A1 — Transcrição · *P3*

Azure Speech como provedor primário (tier F0, região `brazilsouth`), com **faster-whisper offline**
como alternativa automática. Reconhecimento contínuo em PT-BR.

O fallback não é enfeite: se a nuvem falhar durante a demonstração ou não houver rede, o módulo
continua real. Validado com áudio do CORAA e com 2min28s de consulta simulada.

### 5.2 A2 — PLN por regras · *P3*

Extração determinística de `{tipo_relato, local, tempo}` a partir da transcrição, com 28 termos de
violência, termos de sofrimento emocional, tratamento de negação e padrões de local e tempo.

Escolhemos regras em vez de modelo treinado por dois motivos: não há corpus rotulado de relato
clínico em PT-BR ao nosso alcance, e uma regra é auditável — o profissional pode ver exatamente
qual palavra disparou a classificação. A Seção 8 discute o preço dessa escolha.

### 5.3 A3 — Sofrimento na voz · *P2*

Fine-tune do `wav2vec2-large-xlsr-53-portuguese` sobre o CORAA, tarefa binária (neutro vs
não-neutro), 8 épocas em CPU, com pesos de classe para o desbalanceamento de 3,9:1. Apenas os 4
blocos superiores e a cabeça foram treinados — o codificador convolucional permaneceu congelado.

| Ponto de operação | F1 macro (teste) |
|---|---|
| Limiar padrão 0,50 | 0,7171 |
| **Limiar 0,17 calibrado na validação** | **0,7896** |

AUC 0,9229 (validação) e 0,8456 (teste).

**Por que o limiar não é 0,5.** Treinado num conjunto majoritariamente neutro, o modelo é
conservador: atribui probabilidades baixas por padrão. A mediana dos áudios neutros no teste é
0,022 e a dos não-neutros é 0,239. Varrendo todos os cortes na validação, 0,17 maximiza o F1 —
captura 14 dos 23 casos positivos, contra 9 se usássemos 0,5. Em triagem de risco, deixar de
sinalizar é o erro caro.

**Agregação.** Áudio maior que a janela de treino (6 s) é pontuado em janelas, e vale o **máximo**,
não a média. Sofrimento é evento, não estado médio: numa gravação longa o sinal é um trecho curto
que a média dilui.

### 5.4 V1 — Pessoas e rastreamento · *P5*

YOLOv8n com ByteTrack, processando em modo `stream` para não acumular quadros em memória — sem
isso, vídeos longos estouram a RAM. Devolve contagem de pessoas e trajetória de cada uma, o que
alimenta a leitura de presença de acompanhante.

### 5.5 V2 — Postura defensiva · *P5*

YOLOv8n-pose extrai 17 pontos-chave do corpo; sobre eles, 43 características geométricas
(ângulos, distâncias normalizadas) alimentam um GradientBoosting. **F1 macro 0,6868**, divisão por
ator, semente 42.

O critério de aceite era *reportar* o F1, não atingir um alvo — decisão consciente do grupo, dado
que os rótulos são derivados por proxy (Seção 8).

### 5.6 V3 — Desconforto facial · *P4*

Ponto de partida: `trpakov/vit-face-expression`, um Vision Transformer **já treinado em expressão
facial**, reajustado para a tarefa binária desconforto vs neutro.

| Unidade | F1 macro | AUC |
|---|---|---|
| Por frame | 0,7045 | 0,7811 |
| **Por clipe** (unidade real de uso) | **0,7108** | **0,8076** |

**Por que "por clipe".** O contrato do V3 recebe um vídeo e devolve um escore. Medir quadro a
quadro responde "acertou este quadro?"; o que importa é "acertou esta gravação?". A inferência
promedia os frames do vídeo, e a métrica reflete exatamente essa agregação.

**O experimento que determinou o resultado.** Testamos dois fatores — trocar o backbone e remover
o rótulo `calm` — mas em **quatro treinos cruzados**, para saber qual produziu o efeito:

- **Trocar o backbone funcionou:** partir de uma rede que já lê expressões faciais, em vez de uma
  treinada em objetos genéricos, elevou o AUC de 0,7605 para 0,8076.
- **Remover `calm` piorou**, de forma consistente nas duas arquiteturas. Aqueles 1.152 frames são
  exemplos negativos úteis — ensinam que um rosto relaxado *não* é desconforto.

Sem o desenho cruzado, teríamos creditado todo o ganho à rede e removido o rótulo por engano.

### 5.7 C/D — Fusão · *P1*

Soma ponderada de sinais **calibrados**:

| Sinal | Peso |
|---|---|
| Relato (A2) | 0,28 |
| Sofrimento na voz (A3) | 0,28 |
| Desconforto facial (V3) | 0,22 |
| Postura defensiva (V2) | 0,22 |

**Calibração de escala.** Cada modelo tem fronteira de decisão própria, medida na validação: o A3
decide em 0,17, o V3 em 0,70. Somá-los como se ambos decidissem em 0,5 subestima o primeiro e
superestima o segundo. Antes da soma, cada escore é recolocado numa escala em que **0,5 é a
fronteira daquele modelo**.

Efeito medido: numa gravação real, o escore do A3 de 0,119 — acima de 90% dos áudios neutros de
referência — contribuía como se fosse quase zero. Calibrado, passa a valer 0,350.

Escolhemos essa calibração explicável em vez de Platt scaling ou regressão isotônica: ela usa um
número já medido e versionado, e qualquer pessoa refaz a conta no papel. O custo está declarado na
Seção 8.

**Corroboração fora do escore.** A versão original ponderava "áudio e vídeo vieram da mesma
sessão". Como nesta arquitetura isso é verdadeiro por construção, o termo era uma constante somada
a todo caso — chegava a responder por 72% de um escore. O indicador permanece na nota como
informação de proveniência, mas não pesa.

---

## 6. Reprodutibilidade

```bash
uv sync
cp .env.example .env                              # AZURE_SPEECH_KEY / AZURE_SPEECH_REGION
uv run python scripts/download_a3_model.py        # pesos do A3 (Hugging Face)
uv run python scripts/download_v3_model.py        # pesos do V3
uv run python -m src.run_event --audio consulta.wav --video consulta.mp4 --session s1
```

Pesos acima de 5 MB ficam fora do git, em repositórios privados da organização no Hugging Face,
com script de download idempotente. As **métricas** são versionadas no repositório como evidência.
Cada modelo pode ser retreinado do zero pelos scripts em `scripts/`, com semente fixa.

**77 testes automatizados**, incluindo guardas contra os erros que mais custam caro: divergência
entre o pré-processamento de treino e o de inferência, desalinhamento entre os limiares usados na
fusão e os arquivos de métricas, e regressão do comportamento de degradação.

Tempo de inferência medido numa consulta de 2min28s: A1 71 s, A3 82 s, V1 73 s, V2 2,4 s, V3 1,5 s.

---

## 7. Resultados ponta a ponta

Quatro gravações, cobrindo material sintético, atuado e real:

| Caso | Origem | Relato (A2) | Voz (A3) | Face (V3) | Postura (V2) | Escore |
|---|---|---|---|---|---|---|
| Consulta A | gerada por IA | `violencia_domestica` | 0,02 | 0,95 | 0,64 | **0,64** |
| Denúncia | gerada por IA | `sofrimento_emocional` | 0,02 | 0,99 | 0,57 | **0,52** |
| Consulta simulada | atuada | `violencia_domestica` | 0,04 | 1,00 | 0,08 | **0,53** |
| **Ocorrência real** | **ligação real** | `outro` | **0,479** | — | — | — |

### O achado central

Nas três primeiras gravações, o módulo de voz ficou em torno de 0,02–0,04, e chegamos a suspeitar
de limitação do modelo. **A suspeita estava errada.**

Aplicado a uma **ligação real** de violência doméstica, o A3 produziu **0,479** — acima de 97,4%
dos áudios neutros e de 60,9% dos áudios com sofrimento do conjunto de referência, com três janelas
de 6 s cruzando o limiar de decisão.

O que faltava não era ajuste de modelo: era **material do domínio**. Voz sintetizada por IA não
aciona um modelo treinado em fala humana espontânea, e uma simulação encenada por não-atores
carrega pouco do sinal acústico que o modelo aprendeu a reconhecer.

Registramos isso como o principal aprendizado metodológico do projeto: **um modelo pode estar
correto e ainda assim parecer inútil, se validado no material errado**. A conclusão que quase
publicamos — "o modelo não transfere para este domínio" — teria sido falsa, e teria vindo de uma
única amostra fraca.

### Limitações que o mesmo teste revelou

Na ligação real, a **diarização agrupou vítima e atendente como um único falante** — provavelmente
pela compressão GSM da telefonia, que achata diferenças de timbre. Consequência concreta: o
caminho com separação entregaria 0,102, contra 0,479 do áudio bruto.

E o **A2 classificou como `outro`** apesar de a pessoa dizer "ele está querendo me matar" — o
vocabulário cobre agressão e ameaça genérica, mas não ameaça de morte. A transcrição também sofreu
com a qualidade telefônica.

---

## 8. Declarações obrigatórias

Estas declarações são parte da entrega, não ressalvas de rodapé.

### 8.1 Rótulo por proxy

O V2 e o V3 foram treinados com **emoção atuada** (RAVDESS, CREMA-D) como aproximação do que se
quer medir. O V2 usa o proxy `fearful`/`sad` → postura defensiva; o V3 usa `fearful`/`sad` →
desconforto. Não há anotação humana de postura ou de expressão em consulta real.

O que o sistema mede, portanto, é *expressão encenada de medo e tristeza como aproximação de
sofrimento clínico* — não sofrimento anotado por profissional. A validação com dado clínico
anotado permanece como trabalho futuro.

### 8.2 Dado atuado ≠ consulta real

Os conjuntos de treino são de estúdio: atores, iluminação controlada, close frontal, fala dirigida.
Uma consulta real tem plano aberto, iluminação de consultório, ruído, e pessoas que não estão
atuando. Parte do desempenho relatado não deve ser esperada nessas condições.

### 8.3 Consentimento e proteção de dados

Todo material de demonstração é **simulado ou gerado**, produzido para este trabalho ou obtido de
fontes de conscientização pública. O áudio de ocorrência real usado na validação da Seção 7 **não
é redistribuído**: permanece fora do repositório e fora do vídeo de entrega; apenas os números
agregados que ele produziu são relatados.

Num uso real, o sistema exigiria consentimento informado, base legal específica sob a LGPD,
retenção mínima e controle de acesso — nada disso está implementado aqui, e o projeto não deve ser
tratado como pronto para operação.

### 8.4 Humano no circuito

A saída é uma **nota indicativa**, e toda nota carrega a frase *"Esta nota é indicativo, não
veredito — decisão humana."* O sistema não aciona serviço, não notifica autoridade e não registra
ocorrência. Quem decide é a equipe de saúde.

### 8.5 O escore não é probabilidade

Após a calibração da Seção 5.7, o escore é um **índice de risco comparável entre módulos**. Um
valor de 0,7 **não** significa "70% de chance de haver violência". Ele significa que os sinais
combinados estão acima do ponto em que os modelos passariam a sinalizar. Interpretá-lo como
probabilidade seria erro grave num contexto clínico.

---

## 9. Limitações e trabalho futuro

**Divulgação velada não é capturada.** Testamos o sistema com uma gravação em que a vítima sinaliza
risco sem dizer nenhuma palavra que o A2 procura — a estratégia real de quem não pode falar
abertamente. O sistema classificou como `outro` e o escore ficou baixo. É a limitação mais
importante da abordagem por regras, e a mais relevante na prática: superá-la exige modelo de
linguagem treinado em relato clínico, não mais palavras na lista.

**Vocabulário incompleto.** Ameaça de morte não está coberta pelos termos atuais. Achado concreto
da validação com ocorrência real.

**Diarização falha em áudio telefônico comprimido.** Implementada e funcional em áudio limpo, mas
agrupou dois falantes num só na ligação real.

**Conjuntos de teste pequenos.** 101 áudios (23 positivos) para o A3 e 669 clipes para o V3. As
metas foram *atingidas*, não superadas com folga — a incerteza é da ordem de ±0,03 a ±0,10.

**Falta controle negativo no domínio.** Todas as gravações de validação são casos com sofrimento.
Sem casos neutros de consulta, não é possível medir a taxa de falso alarme no uso real. É o
próximo experimento indicado.

**V2 fora de distribuição em consulta.** Treinado em corpo inteiro de atores em pé, produziu 0,08
numa paciente sentada — situação para a qual não foi treinado.

---

## 10. Conclusão

Entregamos um pipeline multimodal funcional, reproduzível e testado, que combina cinco modelos e
produz uma nota de triagem em condições realistas. As três metas de acurácia foram atingidas.

Mais importante que os números, três decisões de engenharia sustentaram o trabalho: **contratos
congelados antes dos modelos**, que permitiram paralelismo real entre cinco pessoas; **degradação
explícita com motivo logado**, que manteve o sistema executável em todos os estados intermediários;
e **medição antes de conclusão**, que nos fez descartar duas hipóteses plausíveis — a de que a
banda do áudio explicava o desempenho do A3, e a de que o modelo não transferia para o domínio.

A segunda hipótese quase virou uma afirmação no relatório. Foi desmentida por um único teste com
material adequado, e essa é a lição que levamos: **em aprendizado de máquina aplicado, a qualidade
da validação limita o que se pode afirmar — mais do que a qualidade do modelo**.

O sistema não está pronto para uso real, e a Seção 8 explica por quê. Está pronto para o que se
propôs: demonstrar que sinais fracos e distribuídos, tratados com rigor metodológico e honestidade
sobre limites, podem apoiar — nunca substituir — a atenção de quem cuida.
