# /plan

Crie o plano técnico da feature (fase **plan**).

## Objetivo

Produzir `specs/NNN-slug/plan.md` mapeando a spec para stack, arquitetura e verificação — respeitando a constituição.

## Passos

1. Leia constituição, `spec.md` da feature e `.specify/templates/plan-template.md`.
2. Se a spec tiver perguntas em aberto críticas, pare e sugira `/clarify`.
3. Preencha `plan.md`:
   - Stack alinhada à constituição (Python 3.11+, `src/` + `tests/`)
   - Arquitetura e estrutura de pastas
   - Decisões/trade-offs
   - Tabela Spec → código → verificação
4. Status: `draft` ou `ready` se consistente com a spec.
5. **Não** gere `tasks.md` nem código, salvo pedido explícito.
6. Ao final, indique o próximo passo: `/tasks`.

## Regras

- Justifique cada dependência nova.
- Não contradiga requisitos da spec; se precisar, atualize a spec com o usuário.
