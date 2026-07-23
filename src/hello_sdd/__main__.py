"""CLI entrypoint: `python -m hello_sdd <name>`."""

from __future__ import annotations

import argparse
import sys

from hello_sdd.greet import EmptyNameError, greet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hello-sdd",
        description="Saudação mínima para validar o fluxo Spec-Driven Development.",
    )
    parser.add_argument("name", help="Nome a ser cumprimentado")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        print(greet(args.name))
    except EmptyNameError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
