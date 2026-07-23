# AGENTS.md

Instruções para qualquer agente de IA (Cursor, Claude Code, Copilot, Codex, Gemini, etc.) que trabalhar neste repositório.

## Identidade do projeto

Este repositório usa **Spec-Driven Development (SDD)**. A fonte da verdade é a especificação versionada em `specs/`, não o chat.

## Antes de qualquer mudança de código

1. Leia `.specify/memory/constitution.md`.
2. Identifique a feature ativa em `specs/` (status `ready` ou `in-progress`).
3. Leia `spec.md`, `plan.md` e `tasks.md` dessa feature.
4. Só então altere código em `src/` / `tests/`.

Se não houver feature ativa e o usuário pedir implementação: **crie a spec primeiro** (fluxo specify → plan → tasks).

## Fluxo obrigatório

```text
constitution → specify → [clarify] → plan → tasks → implement
```

| Fase | Artefato | Pasta |
|------|----------|-------|
| Specify | `spec.md` | `specs/NNN-slug/` |
| Plan | `plan.md` | mesma |
| Tasks | `tasks.md` | mesma |
| Implement | código + testes | `src/`, `tests/` |

Templates: `.specify/templates/`.

## Regras de ouro

- **Não invente requisitos.** Se faltar informação, pergunte ou registre em "Perguntas em aberto" da spec.
- **Não mude stack** sem atualizar `plan.md` e checar a constituição.
- **Marque tasks** em `tasks.md` conforme conclui (`[x]`).
- **Uma feature por pasta** (`specs/NNN-slug/`). Numeração: próximo inteiro disponível (3 dígitos).
- **Código em inglês; specs em português** (exceto nomes de IDs: `US-1`, `RF-01`, `T010`).
- **Commits:** só quando o humano pedir.

## Como retomar o trabalho (handoff)

Ao iniciar uma sessão, diga o que encontrou:

1. Feature ativa e status
2. Próxima task não marcada em `tasks.md`
3. Bloqueios / perguntas em aberto

## Comandos Cursor

Neste repo existem comandos em `.cursor/commands/`:

- `/specify` — criar/atualizar spec
- `/clarify` — fechar ambiguidades
- `/plan` — plano técnico
- `/tasks` — decompor em tasks
- `/implement` — executar tasks

Em outras IAs, peça explicitamente a mesma fase (ex.: "siga o fluxo specify deste repo").

## Escopo do exemplo

`specs/000-hello-sdd/` + `src/` são o exemplo funcional mínimo. Novas features começam em `001-...`.
