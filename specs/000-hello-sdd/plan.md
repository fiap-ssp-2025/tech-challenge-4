# Plan: Hello SDD

> Referência: `specs/000-hello-sdd/spec.md`  
> **Status:** done

## Resumo técnico

Pacote Python padrão (`src` layout) com CLI via `argparse`, sem dependências externas. Testes com `pytest`.

## Stack

| Camada | Escolha | Motivo |
|--------|---------|--------|
| Runtime | Python 3.11+ | Constituição |
| CLI | `argparse` (stdlib) | Zero deps |
| Testes | `pytest` | Padrão do repo |
| Packaging | `pyproject.toml` | Instalação editável simples |

## Arquitetura

```text
CLI (argparse) → greet(name) → stdout / stderr + exit code
```

## Estrutura de pastas (alvo)

```text
src/hello_sdd/
  __init__.py
  __main__.py
  greet.py
tests/
  test_greet.py
pyproject.toml
```

## Decisões e trade-offs

| Decisão | Alternativas | Por quê esta |
|---------|--------------|--------------|
| stdlib argparse | click/typer | RNF-01: zero deps runtime |
| src layout | flat | Isola pacote e facilita testes |

## Mapeamento Spec → Implementação

| Requisito / Story | Onde no código | Como verificar |
|-------------------|----------------|----------------|
| US-1 / RF-01–03 | `greet.py`, `__main__.py` | `pytest` + `python -m hello_sdd Ada` |
| US-2 / RF-04 | `__main__.py` | `python -m hello_sdd --help` |

## Riscos

- Nenhum relevante para o exemplo.

## Conformidade com a constituição

- [x] Spec-first respeitado
- [x] Separação what/how ok
- [x] Dependências mínimas justificadas
- [x] Testes previstos para critérios críticos
