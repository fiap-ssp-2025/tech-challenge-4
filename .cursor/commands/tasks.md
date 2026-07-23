# /tasks

Decomponha spec + plan em checklist executável.

## Objetivo

Produzir `specs/NNN-slug/tasks.md` com tasks ordenadas, IDs estáveis e verificação explícita.

## Passos

1. Leia `spec.md`, `plan.md` e `.specify/templates/tasks-template.md`.
2. Gere fases: setup → uma fase por user story → fechamento.
3. Cada task:
   - Checkbox `- [ ]`
   - ID `T000`, `T010`, … (dezenas por fase)
   - Ação concreta + **Verificar:** (comando de teste ou critério observável)
4. Cubra todos os critérios de aceite críticos com pelo menos uma task de teste.
5. Não implemente código nesta fase.
6. Ao final, sugira `/implement` e destaque a primeira task a executar.

## Regras

- Tasks devem ser pequenas o bastante para uma sessão de agente.
- Evite tasks vagas ("melhorar código", "refatorar tudo").
