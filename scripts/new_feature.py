#!/usr/bin/env python3
"""Cria a pasta de uma nova feature a partir dos templates SDD."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "specs"
TEMPLATES = ROOT / ".specify" / "templates"


def next_number() -> int:
    nums: list[int] = []
    for path in SPECS.iterdir():
        if path.is_dir() and re.match(r"^\d{3}-", path.name):
            nums.append(int(path.name[:3]))
    return (max(nums) + 1) if nums else 0


def slugify(text: str) -> str:
    slug = text.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-") or "feature"


def main() -> int:
    parser = argparse.ArgumentParser(description="Scaffold de feature SDD")
    parser.add_argument("name", help="Nome ou slug da feature")
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        help="Número NNN (default: próximo livre)",
    )
    args = parser.parse_args()

    n = args.number if args.number is not None else next_number()
    slug = slugify(args.name)
    folder = SPECS / f"{n:03d}-{slug}"

    if folder.exists():
        print(f"error: already exists: {folder}", file=sys.stderr)
        return 1

    folder.mkdir(parents=True)
    mapping = {
        "spec-template.md": "spec.md",
        "plan-template.md": "plan.md",
        "tasks-template.md": "tasks.md",
    }
    for src_name, dest_name in mapping.items():
        content = (TEMPLATES / src_name).read_text(encoding="utf-8")
        content = content.replace("[NOME DA FEATURE]", slug.replace("-", " ").title())
        (folder / dest_name).write_text(content, encoding="utf-8")

    print(f"created {folder.relative_to(ROOT)}")
    print("next: preencha spec.md (ou use /specify no Cursor)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
