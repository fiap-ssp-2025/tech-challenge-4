# /implement

Execute as tasks da feature ativa e mantenha spec/código sincronizados.

## Objetivo

Implementar `tasks.md` em ordem, marcando `[x]` somente após verificação.

## Passos

1. Leia constituição, `spec.md`, `plan.md` e `tasks.md`.
2. Atualize status da feature para `in-progress` se ainda estiver `ready`.
3. Pegue a **próxima** task não marcada e implemente só ela (ou um lote pequeno coerente, se o usuário pedir).
4. Rode a verificação indicada na task.
5. Marque `[x]` em `tasks.md` quando passar.
6. Repita até o usuário pausar ou todas as tasks terminarem.
7. No fechamento: status `done` na spec/plan; confirme que código e spec batem.

## Regras

- Se a task for impossível sem mudar o requisito: **pare**, atualize a spec/plan com o usuário, depois continue.
- Não pule testes pedidos na task.
- Não faça commit a menos que o usuário peça.
- Ao retomar, comece relatando: feature, próxima task, bloqueios.
