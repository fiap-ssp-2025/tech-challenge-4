# Constituição do Projeto

Princípios permanentes que toda especificação, plano e implementação devem respeitar.
Atualize este arquivo com decisões de longo prazo — não com detalhes de features.

## Princípios

### I. Spec-first
Nenhuma feature entra em código sem `spec.md` → `plan.md` → `tasks.md` aprovados (ou explicitamente aceitos pelo time). A especificação é a fonte da verdade; o código deriva dela.

### II. Separação what / how
- `spec.md` descreve **o quê** e **por quê** — sem stack, libs ou frameworks.
- `plan.md` descreve **como** — stack, arquitetura, trade-offs.
- `tasks.md` decompõe o plano em passos verificáveis.

### III. Incremental e verificável
Cada user story deve poder ser entregue e validada de forma independente. Toda tarefa em `tasks.md` tem critério de aceite claro (checkbox + como verificar).

### IV. Código legível para humanos e IAs
Preferir clareza a cleverness. Nomes explícitos, arquivos pequenos, responsabilidade única. Documentar decisões não óbvias no plano, não em comentários espalhados.

### V. Sincronia multi-agente
Qualquer pessoa (e qualquer IA) deve conseguir retomar o trabalho lendo, nesta ordem:
1. Esta constituição
2. `AGENTS.md`
3. A pasta da feature ativa em `specs/`
4. O código existente em `src/`

### VI. Mudança consciente
Se a implementação divergir da spec, **atualize a spec primeiro** (ou abra uma nova change), depois o código. Nunca deixe o código como verdade silenciosa.

### VII. Explicar como professor, não como especialista
Este é um trabalho acadêmico feito por cinco pessoas com formações diferentes: **ninguém deve
precisar aceitar um resultado que não entendeu.** Quem entrega — pessoa ou IA — tem a obrigação
de tornar o próprio trabalho compreensível, e a régua é o integrante que não trabalhou naquela
frente.

Na prática, ao relatar qualquer decisão técnica, resultado de treino ou métrica:

- **Defina o termo na primeira vez que ele aparecer** (métricas, arquiteturas, jargão de ML,
  nomes de módulo). Nota de rodapé, glossário ou parênteses — mas defina.
- **Diga o que o número significa em português**, não só o valor: "acerta ~70% dos rostos que
  nunca viu" vale mais que "F1 macro 0,6986".
- **Explique o porquê da escolha**, não só o quê: por que esta métrica, por que este dataset,
  por que este ponto de corte.
- **Separe o que é fato medido do que é interpretação ou recomendação.**
- **Declare limitações junto com o resultado**, nunca depois de perguntado — especialmente
  dados de proxy, amostras pequenas e desvios do plano.
- **Se o leitor precisa decidir algo**, apresente as opções com custo e ganho esperado de cada
  uma, e uma recomendação explícita.

Explicação boa é entregável, não cortesia. Resultado que o time não consegue explicar na
apresentação é resultado que não pode ser usado.

## Padrões técnicos (baseline)

| Área | Decisão |
|------|---------|
| Linguagem | Python 3.11+ |
| Layout | `src/` para código, `tests/` para testes, `specs/` para artefatos SDD |
| Testes | `pytest` — pelo menos um teste por critério de aceite crítico |
| Estilo | Código em inglês; specs/docs do time em português |
| Dependências | Mínimas; justificar cada adição no `plan.md` |

## Gates (antes de implementar)

- [ ] Spec cobre cenários felizes e principais erros
- [ ] Critérios de aceite são testáveis (WHEN/THEN ou checklist)
- [ ] Plan respeita esta constituição
- [ ] Tasks estão ordenadas e marcáveis
