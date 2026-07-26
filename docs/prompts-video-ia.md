# Prompts para gerar o vídeo de simulação com IA

> Companheiro de [`roteiro-simulacao-consulta.md`](roteiro-simulacao-consulta.md).
> Use com geradores que produzem **áudio junto com o vídeo** (Veo, Sora e similares). Em
> geradores só de imagem em movimento, o áudio terá de ser dublado depois.
>
> **Ressalva registrada:** material gerado testa o pipeline de forma limitada — os modelos de
> emoção (A3 e V3) foram treinados em voz e rosto humanos reais. Serve como ilustração da demo;
> para afirmar desempenho no relatório, o material atuado por pessoas é mais defensável.

## Como usar

1. Cole o **BLOCO FIXO** no início de **todos** os prompts — é ele que mantém as personagens
   parecidas entre um clipe e outro.
2. Gere um clipe por cena. Cada cena tem ~8–12 s.
3. Junte os clipes na ordem (`ffmpeg -f concat`) e extraia o áudio.

## Requisitos que os prompts já embutem (não remova)

| Exigência | Por quê |
|---|---|
| Rosto da paciente frontal, ocupando ~1/3 do quadro | O V3 recorta a região do rosto; plano aberto faz a detecção falhar |
| Sem música, sem trilha, sem efeitos sonoros | Trilha polui a transcrição (A1) e o escore de sofrimento (A3) |
| Falas alternadas, **sem sobreposição** | A diarização separa por turnos de fala |
| Vozes com timbres distintos | Para os locutores serem separáveis |
| Câmera fixa, sem cortes internos | Movimento atrapalha o rastreamento do V1 |
| Sem legendas nem texto na tela | Texto queimado no vídeo não serve para nada aqui |
| 16:9, 1080p, português do Brasil | A1/A2/A3 são PT-BR |

---

## BLOCO FIXO (colar antes de cada cena)

```
Cena realista de consultório médico brasileiro, iluminação neutra de ambiente clínico,
paredes claras, maca e pia ao fundo desfocados.

PERSONAGENS (mantenha idênticas em todos os clipes):
- MÉDICA: mulher, 45 anos, pele parda, cabelo escuro preso em coque baixo, óculos de
  armação fina, jaleco branco sobre blusa azul-marinho. Sentada à direita do quadro,
  atrás de uma mesa simples. Voz feminina, tom médio-grave, calma e pausada.
- PACIENTE: mulher, 32 anos, pele clara, cabelo castanho liso na altura dos ombros,
  blusa bege de manga comprida, sem maquiagem. Sentada à esquerda do quadro, de frente
  para a médica, ligeiramente voltada para a câmera. Voz feminina, tom médio-agudo,
  fala baixa e hesitante.

CÂMERA: plano fixo, altura dos olhos, sem movimento, sem corte. Enquadramento médio
que mostra as duas mulheres da cintura para cima, com o ROSTO DA PACIENTE NÍTIDO E
DE FRENTE, ocupando aproximadamente um terço da largura do quadro.

ÁUDIO: apenas as vozes, em português do Brasil, gravadas como som direto de ambiente.
SEM música, SEM trilha sonora, SEM efeitos sonoros, SEM ruído de fundo.
As falas são alternadas — nunca as duas pessoas falando ao mesmo tempo.

NÃO INCLUIR: legendas, texto na tela, marca d'água, cortes, zoom, movimento de câmera,
close extremo, plano aberto, mais de duas pessoas.

Formato: 16:9, 1080p, aparência de vídeo documental realista, sem estilização.
```

---

# VERSÃO A — relato explícito

### Cena A1
```
[BLOCO FIXO]

AÇÃO: A paciente está sentada, ombros levemente curvados, mãos no colo. A médica olha
para ela com atenção.

DIÁLOGO:
MÉDICA: "Bom dia, senhora. Pode sentar. O que a traz aqui hoje?"
PACIENTE: "Bom dia. É essa dor de cabeça, doutora. Faz umas três semanas que não passa."
MÉDICA: "Entendi. É uma dor constante ou vai e volta?"
PACIENTE: "Vai e volta. Piora à noite."
```

### Cena A2
```
[BLOCO FIXO]

AÇÃO: A paciente evita o olhar da médica ao responder sobre o sono. Expressão cansada.

DIÁLOGO:
MÉDICA: "Está dormindo bem?"
PACIENTE: "Não muito. Eu acordo várias vezes. Fico… fico atenta, sabe?"
MÉDICA: "Atenta a quê?"
PACIENTE: "A qualquer barulho."
```

### Cena A3
```
[BLOCO FIXO]

AÇÃO: A médica inclina-se levemente à frente, tom acolhedor. A paciente fica imóvel por
um instante antes de responder, olhos marejados.

DIÁLOGO:
MÉDICA: "Senhora, eu pergunto isso para todas as minhas pacientes. Está tudo bem em
casa? A senhora se sente segura?"
PACIENTE: "Não. Não está tudo bem, não."
```

### Cena A4
```
[BLOCO FIXO]

AÇÃO: A paciente fala olhando para as próprias mãos, voz trêmula, respiração curta.
Expressão de medo e vergonha.

DIÁLOGO:
MÉDICA: "A senhora pode me contar o que está acontecendo?"
PACIENTE: "É o meu marido. Ele anda muito nervoso. Semana passada ele me empurrou
contra a parede."
MÉDICA: "A senhora se machucou?"
PACIENTE: "Fiquei com o braço roxo."
```

### Cena A5
```
[BLOCO FIXO]

AÇÃO: A paciente encolhe os ombros ao falar da ameaça, olhar baixo. A médica escuta
sem interromper.

DIÁLOGO:
MÉDICA: "Isso já tinha acontecido antes?"
PACIENTE: "Já. Não é a primeira vez. Ele me ameaçou também, disse que se eu saísse de
casa ia ser pior."
```

### Cena A6
```
[BLOCO FIXO]

AÇÃO: A médica fala com firmeza tranquila. A paciente assente devagar, aliviada e
chorosa.

DIÁLOGO:
MÉDICA: "Obrigada por me contar. A senhora fez a coisa certa. Nada disso é culpa sua.
Eu vou chamar a nossa assistente social para conversar com a senhora agora."
PACIENTE: "Tudo bem. Obrigada."
```

---

# VERSÃO B — relato velado

> Acrescente ao BLOCO FIXO, **apenas nesta versão**, a terceira pessoa:
>
> ```
> TERCEIRA PESSOA: homem, 35 anos, camisa cinza, sentado ao fundo à esquerda, em
> silêncio durante toda a cena, observando. Nunca fala. Rosto parcialmente visível.
> ```
>
> Isso exercita o V1, que mede presença e dominância de acompanhante.

### Cena B1
```
[BLOCO FIXO + TERCEIRA PESSOA]

AÇÃO: A paciente responde olhando rapidamente para trás, na direção do homem sentado
ao fundo, antes de voltar o olhar para a médica.

DIÁLOGO:
MÉDICA: "Bom dia, senhora. O que a traz aqui hoje?"
PACIENTE: "Bom dia. É uma dor de cabeça que não passa. E eu ando muito cansada."
MÉDICA: "Há quanto tempo?"
PACIENTE: "Umas semanas. Mas não é nada demais, não."
```

### Cena B2
```
[BLOCO FIXO + TERCEIRA PESSOA]

AÇÃO: Respostas curtas, voz contida. A paciente mantém as mãos entrelaçadas, tensas.

DIÁLOGO:
MÉDICA: "Está dormindo bem?"
PACIENTE: "Mais ou menos. Durmo pouco."
MÉDICA: "E o apetite?"
PACIENTE: "Também não está bom. Mas é normal, né? Correria."
```

### Cena B3
```
[BLOCO FIXO + TERCEIRA PESSOA]

AÇÃO: A paciente hesita visivelmente antes de explicar o machucado. Toca o próprio
braço por reflexo.

DIÁLOGO:
MÉDICA: "A senhora tem tido dores em outros lugares?"
PACIENTE: "No braço, um pouco. Eu bati sem querer."
MÉDICA: "Bateu como?"
PACIENTE: "Na porta do armário. Foi bobeira minha."
```

### Cena B4
```
[BLOCO FIXO + TERCEIRA PESSOA]

AÇÃO: Ao responder que está tudo bem, a paciente força um meio sorriso que não alcança
os olhos. O homem ao fundo permanece imóvel.

DIÁLOGO:
MÉDICA: "Entendi. Senhora, está tudo bem em casa?"
PACIENTE: "Está. Está tudo tranquilo."
MÉDICA: "A senhora está acompanhada hoje?"
PACIENTE: "Estou, meu marido está ali. Ele me trouxe."
```

### Cena B5
```
[BLOCO FIXO + TERCEIRA PESSOA]

AÇÃO: A paciente fala mais baixo, quase sussurrando o pedido do remédio. Olhar fixo
na mesa.

DIÁLOGO:
MÉDICA: "Se a senhora quiser conversar sozinha comigo em algum momento, é só dizer."
PACIENTE: "Não precisa. Está tudo bem. Só me dá o remédio para a dor, por favor."
MÉDICA: "Vou dar. E vou deixar seu retorno marcado, tá bom?"
PACIENTE: "Tá bom. Obrigada."
```

---

## Depois de gerar

```bash
# 1. juntar os clipes na ordem
printf "file 'A1.mp4'\nfile 'A2.mp4'\nfile 'A3.mp4'\nfile 'A4.mp4'\nfile 'A5.mp4'\nfile 'A6.mp4'\n" > lista.txt
ffmpeg -f concat -safe 0 -i lista.txt -c copy consulta-A.mp4

# 2. extrair o áudio no formato que o pipeline espera
ffmpeg -i consulta-A.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 consulta-A.wav

# 3. rodar
uv run python -m src.run_event --audio consulta-A.wav --video consulta-A.mp4 --session sim-A
```

## Conferir antes de rodar

- [ ] O rosto da paciente aparece **de frente** e grande o suficiente na maior parte do tempo
- [ ] Não há música nem trilha em nenhum clipe
- [ ] As vozes das duas personagens são distinguíveis
- [ ] Ninguém fala por cima de ninguém
- [ ] Não há legenda queimada na imagem
- [ ] As personagens estão parecidas entre um clipe e outro (senão o V1 conta pessoas demais)
