# Spec: Hello SDD

> **Status:** done  
> **Branch sugerida:** `feat/000-hello-sdd`  
> **Criado em:** 2026-07-23

## Visão

Uma CLI mínima que cumprimenta o usuário pelo nome, demonstrando o ciclo Spec-Driven Development ponta a ponta neste repositório.

## Contexto / Problema

O time precisa de um exemplo concreto e executável para alinhar humanos e IAs no fluxo SDD — sem domínio de negócio complexo.

## User Stories

### US-1 — Cumprimentar pelo nome

**Como** desenvolvedor explorando o repo,  
**quero** informar meu nome e receber uma saudação,  
**para** validar que o pipeline spec → código funciona.

#### Critérios de aceite

- **WHEN** executo a CLI com um nome não vazio  
  **THEN** a saída é exatamente `Hello, <nome>!` (com quebra de linha final)

- **WHEN** executo a CLI sem nome ou com nome só de espaços  
  **THEN** o processo termina com código ≠ 0 e mensagem de erro em stderr

### US-2 — Ajuda rápida

**Como** desenvolvedor,  
**quero** ver instruções de uso,  
**para** descobrir como chamar a ferramenta sem ler o código.

#### Critérios de aceite

- **WHEN** executo a CLI com `--help`  
  **THEN** a saída descreve o uso e o argumento `name`

## Requisitos funcionais

| ID | Requisito | Prioridade |
|----|-----------|------------|
| RF-01 | Aceitar um argumento posicional `name` | must |
| RF-02 | Imprimir saudação no formato fixo `Hello, {name}!` | must |
| RF-03 | Rejeitar nome vazio / whitespace | must |
| RF-04 | Expor `--help` | must |

## Requisitos não funcionais

| ID | Requisito | Prioridade |
|----|-----------|------------|
| RNF-01 | Sem dependências de terceiros no runtime | must |
| RNF-02 | Executável com `python -m hello_sdd` | must |

## Fora de escopo

- i18n / outros idiomas de saudação
- Interface web ou HTTP
- Persistência de nomes

## Perguntas em aberto

_Nenhuma — exemplo fechado._

## Notas

Feature de referência (`000`). Novas features começam em `001-`.
