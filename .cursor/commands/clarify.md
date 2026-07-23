# /clarify

Feche ambiguidades da spec ativa antes do plano técnico.

## Objetivo

Reduzir "Perguntas em aberto" em `spec.md` e tornar critérios de aceite testáveis.

## Passos

1. Localize a feature ativa em `specs/` (ou use a indicada pelo usuário).
2. Leia `spec.md` e liste ambiguidades, contradições e critérios vagos.
3. Faça **no máximo 5 perguntas** por rodada, priorizando o que bloqueia o plan.
4. Com as respostas, atualize `spec.md` (stories, RF/RNF, fora de escopo).
5. Marque perguntas resolvidas; mantenha só o que ainda está aberto.
6. Se a spec estiver sólida, sugira status `ready` e o próximo comando `/plan`.

## Regras

- Não escolha stack aqui.
- Não implemente código.
- Prefira decisões explícitas do usuário a defaults silenciosos.
