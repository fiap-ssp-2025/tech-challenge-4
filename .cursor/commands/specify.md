# /specify

Crie ou atualize a especificação de uma feature (fase **specify** do SDD).

## Objetivo

Produzir `specs/NNN-slug/spec.md` a partir da intenção do usuário — **o quê** e **por quê**, sem stack.

## Passos

1. Leia `.specify/memory/constitution.md` e `.specify/templates/spec-template.md`.
2. Se o usuário não passou um slug, derive um (`kebab-case`) e o próximo `NNN` olhando `specs/`.
3. Crie a pasta `specs/NNN-slug/` se não existir.
4. Preencha `spec.md` com o template:
   - Visão, contexto, user stories com WHEN/THEN
   - RF/RNF em tabelas
   - Fora de escopo e perguntas em aberto
5. Status inicial: `draft`. Se estiver completa e testável, sugira `ready`.
6. **Não** crie `plan.md` nem código nesta fase, a menos que o usuário peça o fluxo completo.
7. Ao final, resuma: pasta criada, stories, e próximos passos (`/clarify` ou `/plan`).

## Entrada do usuário

A mensagem após o comando é a descrição da feature. Peça esclarecimentos só se faltar o mínimo para critérios de aceite.
