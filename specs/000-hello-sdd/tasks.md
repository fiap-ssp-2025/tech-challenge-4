# Tasks: Hello SDD

> Referências: `spec.md` + `plan.md`  
> Marque `[x]` somente após verificar o critério.

## Fase 0 — Setup

- [x] T001 Criar `pyproject.toml` e pacote `src/hello_sdd/`
- [x] T002 Configurar `pytest` como dep de desenvolvimento

## Fase 1 — US-1 Cumprimentar pelo nome

- [x] T010 Implementar `greet(name) -> str` com validação de nome vazio
  - **Verificar:** função pura rejeita whitespace
- [x] T011 CLI em `__main__.py` com argumento posicional e exit codes
  - **Verificar:** `python -m hello_sdd Ada` imprime `Hello, Ada!`
- [x] T012 Testes de `greet` e comportamento de erro
  - **Verificar:** `pytest` passa

## Fase 2 — US-2 Ajuda

- [x] T020 Garantir `--help` via argparse
  - **Verificar:** `python -m hello_sdd --help` exit 0

## Fase N — Fechamento

- [x] T090 Status da spec/plan = `done`
- [x] T091 README documenta como rodar o exemplo
