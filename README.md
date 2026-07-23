# Tech Challenge 4 — Spec-Driven Development

Repositório base para desenvolver **em sincronia com IAs** usando Spec-Driven Development (SDD): a especificação versionada é a fonte da verdade; o código deriva dela.

Funciona com Cursor, Claude Code, Copilot, Codex e qualquer agente que leia `AGENTS.md`.

## Por que SDD aqui?

Sem spec compartilhada, cada pessoa (e cada IA) “inventa” requisitos no chat. Com SDD:

1. O time alinha o **quê** em `specs/`
2. Escolhe o **como** no `plan.md`
3. Decompõe em `tasks.md`
4. Só então implementa — e qualquer agente retoma do mesmo lugar

## Estrutura

```text
AGENTS.md                 ← instruções para qualquer IA
.specify/
  memory/constitution.md  ← princípios permanentes
  templates/              ← modelos de spec / plan / tasks
.cursor/
  rules/sdd.mdc           ← regra sempre ativa (Cursor)
  commands/               ← /specify /clarify /plan /tasks /implement
specs/
  000-hello-sdd/          ← exemplo completo (done)
  NNN-slug/               ← próximas features
src/hello_sdd/            ← código do exemplo
tests/                    ← testes do exemplo
```

## Fluxo de trabalho

```text
constitution → specify → clarify → plan → tasks → implement
```

| Fase | Comando (Cursor) | Artefato |
|------|------------------|----------|
| Spec | `/specify` | `specs/NNN-slug/spec.md` |
| Clareza | `/clarify` | atualiza a spec |
| Plano | `/plan` | `plan.md` |
| Tasks | `/tasks` | `tasks.md` |
| Código | `/implement` | `src/`, `tests/` + checkboxes |

Em outras IAs: peça a mesma fase (“crie a spec seguindo `AGENTS.md` e o template”).

### Nova feature (checklist)

1. Abra o agente na raiz do repo
2. Rode `/specify Descreva a feature...` (ou peça o equivalente)
3. Feche ambiguidades com `/clarify`
4. `/plan` com restrições de stack (se houver)
5. `/tasks` → revise a lista
6. `/implement` task a task
7. Commit quando o time validar (humano pede o commit)

Numeração: próximo `NNN` livre em `specs/` (ex.: `001-minha-feature`).

## Exemplo funcional

O pacote `hello_sdd` implementa `specs/000-hello-sdd/`.

```bash
# Python 3.11+
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e ".[dev]"

python -m hello_sdd Ada
# Hello, Ada!

pytest
```

## Sincronia entre pessoas e IAs

| Situação | O que fazer |
|----------|-------------|
| Retomar trabalho | Ler `AGENTS.md` → feature ativa em `specs/` → próxima task aberta |
| Mudou o requisito | Atualizar `spec.md` (e plan/tasks) **antes** do código |
| Outra IA / outra pessoa | Mesmos artefatos no git — o chat não é a verdade |
| Feature concluída | Status `done` na spec/plan; tasks todas `[x]` |

## Convenções rápidas

- Specs e docs do time: **português**
- Código e IDs (`US-1`, `T010`): **inglês** / estáveis
- `spec.md` sem stack; `plan.md` com stack
- Constituição: decisões duráveis; features: pastas em `specs/`

## Próximos passos

1. Ajuste `.specify/memory/constitution.md` ao domínio real do challenge
2. Crie `specs/001-...` com a primeira feature de negócio
3. Mantenha `AGENTS.md` curto — detalhes vão para constituição e specs
