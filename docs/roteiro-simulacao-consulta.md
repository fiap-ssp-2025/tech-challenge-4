# Roteiro de simulação — consulta de triagem (TC4)

> **Material atuado por integrantes do grupo.** Nenhuma pessoa retratada é paciente real e nenhum
> dado clínico real foi usado. Roteiro adaptado do caso padronizado de acesso aberto
> [*Domestic Violence Simulated Patient Case* — MedEdPORTAL](https://www.mededportal.org/doi/10.15766/mep_2374-8265.624),
> que apresenta a paciente com queixa de cefaleia e os gatilhos que devem levar o profissional a
> perguntar sobre violência.
>
> Objetivo: gerar o material de demonstração do pipeline (Etapa 5) e permitir o teste A/B da
> diarização (spec 003).

## Por que duas versões

| | O que testa | Resultado esperado |
|---|---|---|
| **Versão A — relato explícito** | O caminho feliz: A2 encontra os termos, o escore sobe | `tipo_relato = violencia_domestica`, escore alto |
| **Versão B — relato velado** | A limitação real: a vítima evita as palavras | `tipo_relato = outro`, escore baixo **mesmo havendo risco** |

Gravar as duas e mostrar as duas é mais forte que mostrar só a que funciona: demonstra a
capacidade **e** o limite, que é o que o RNF-06 e o Princípio VII pedem.

## Requisitos técnicos da gravação

Sem isso o teste não mede o que deveria:

- **Enquadramento:** paciente de frente, rosto ocupando boa parte do quadro (o V3 recorta a região
  do rosto; plano aberto demais faz a detecção falhar).
- **Resolução:** 720p ou mais. Celular na horizontal serve.
- **Áudio:** ambiente silencioso; as duas vozes audíveis e **sem falar por cima uma da outra** —
  a diarização separa por turnos.
- **Duração:** 2 a 3 minutos bastam.
- **Papéis:** vozes distintas (idealmente timbres diferentes) para a separação de locutores ficar clara.
- Extrair o áudio depois com:
  `ffmpeg -i consulta.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 consulta.wav`

---

## Versão A — relato explícito

**P = profissional de saúde · M = paciente**

**P:** Bom dia, senhora. Pode sentar. O que a traz aqui hoje?

**M:** Bom dia. É essa dor de cabeça, doutora. Faz umas três semanas que não passa.

**P:** Entendi. É uma dor constante ou vai e volta?

**M:** Vai e volta. Piora à noite. Eu tomo remédio, melhora um pouco, mas volta.

**P:** A senhora já teve dor assim antes?

**M:** Já… já sim. No ano passado também tive uma fase ruim.

**P:** Está dormindo bem?

**M:** Não muito. Eu acordo várias vezes. Fico… fico atenta, sabe?

**P:** Atenta a quê?

**M:** *(pausa)* A qualquer barulho.

**P:** Senhora, eu pergunto isso para todas as minhas pacientes, e é uma pergunta de rotina. Está
tudo bem em casa? A senhora se sente segura?

**M:** *(silêncio)* Não. Não está tudo bem, não.

**P:** A senhora pode me contar o que está acontecendo?

**M:** É o meu marido. Ele anda muito nervoso. Semana passada ele me empurrou contra a parede. Foi
por causa de uma bobagem, ele disse que eu tinha demorado no mercado.

**P:** A senhora se machucou?

**M:** Fiquei com o braço roxo. Já passou.

**P:** Isso já tinha acontecido antes?

**M:** Já. Não é a primeira vez. Ele me ameaçou também, disse que se eu saísse de casa ia ser pior.

**P:** Obrigada por me contar. A senhora fez a coisa certa. Nada disso é culpa sua, e a senhora não
está sozinha nisso. Eu vou chamar a nossa assistente social para conversar com a senhora agora,
ainda nesta consulta. Tudo bem?

**M:** Tudo bem. *(voz embargada)* Obrigada.

---

## Versão B — relato velado

**P = profissional de saúde · M = paciente · A = acompanhante (opcional, fica em silêncio no fundo)**

> Nesta versão a paciente **não usa nenhuma palavra** que o A2 procura. É o cenário real mais
> comum: a vítima não relata diretamente, por medo ou por estar acompanhada.
> Se puderem, coloquem uma terceira pessoa sentada ao fundo, calada — o V1 vai detectar duas
> pessoas no enquadramento, o que também exercita a leitura de acompanhante.

**P:** Bom dia, senhora. O que a traz aqui hoje?

**M:** Bom dia. É uma dor de cabeça que não passa. E eu ando muito cansada.

**P:** Há quanto tempo?

**M:** Umas semanas. *(olha rapidamente para trás)* Mas não é nada demais, não.

**P:** Está dormindo bem?

**M:** Mais ou menos. Durmo pouco.

**P:** E o apetite?

**M:** Também não está bom. Mas é normal, né? Correria.

**P:** A senhora tem tido dores em outros lugares?

**M:** No braço, um pouco. Eu bati sem querer.

**P:** Bateu como?

**M:** *(pausa curta)* Na porta do armário. Foi bobeira minha.

**P:** Entendi. Senhora, está tudo bem em casa?

**M:** *(pausa)* Está. Está tudo tranquilo.

**P:** A senhora está acompanhada hoje?

**M:** Estou, meu marido está ali fora. Ele me trouxe.

**P:** Certo. Se a senhora quiser conversar sozinha comigo em algum momento, é só dizer.

**M:** *(baixo)* Não precisa. Está tudo bem. *(pausa)* Só me dá o remédio para a dor, por favor.

**P:** Vou dar. E vou deixar seu retorno marcado, tá bom? Quero ver como a senhora está.

**M:** Tá bom. Obrigada.

---

## O que reportar depois de gravar

Rodar o pipeline nas duas versões e registrar lado a lado:

```bash
uv run python -m src.run_event --audio consulta-A.wav --video consulta-A.mp4 --session sim-A
uv run python -m src.run_event --audio consulta-B.wav --video consulta-B.mp4 --session sim-B
```

Anotar, para cada versão: `tipo_relato`, `sofrimento`, `postura_defensiva`, `desconforto_facial`,
`n_pessoas`, escore final, e o escore do A3 **por locutor** (a linha `[A3] sofrimento por locutor`).

A comparação A × B é o material da seção de limitações do relatório: mostra que o sistema depende
de o relato ser verbalizado, e que **a divulgação velada — a mais comum na prática — passa
despercebida pelo ramo de texto**.
