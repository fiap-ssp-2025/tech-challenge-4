#!/usr/bin/env python3
"""Baixa o modelo A3 (sofrimento na voz) do Hugging Face para models/a3_emotion/.

Os pesos do A3 têm ~1,2 GB e não cabem no git, então ficam num repositório **privado**
no Hub. Sem eles, `src/resolve.py` marca o A3 como "artefato ausente" e o pipeline cai
para o stub.

Autenticação: o repositório é privado, então é preciso estar logado
(`uv run hf auth login`) ou exportar `HF_TOKEN` no ambiente. Peça acesso de leitura ao
P2 se receber 401/403.

Idempotente: o `snapshot_download` reaproveita o cache e só baixa o que falta.

Uso:
    uv run python scripts/download_a3_model.py
    uv run python scripts/download_a3_model.py --repo-id outro/repo
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "marsiqueira/tc4-a3-sofrimento-voz"
DEFAULT_LOCAL_DIR = ROOT / "models" / "a3_emotion"
# training_args.bin e _checkpoints/ não são publicados; o README é do card, não do modelo.
ALLOW_PATTERNS = [
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "metrics.json",
    "threshold_metrics.json",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Baixa o modelo A3 do Hugging Face")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    try:
        path = snapshot_download(
            repo_id=args.repo_id,
            repo_type="model",
            local_dir=str(args.local_dir),
            allow_patterns=ALLOW_PATTERNS,
            # None = usa o token em cache se houver, senão tenta anônimo; True
            # exigiria login até onde ele não é necessário (LocalTokenNotFoundError).
            token=os.environ.get("HF_TOKEN") or None,
        )
    except (RepositoryNotFoundError, GatedRepoError):
        print(f"Sem acesso a {args.repo_id} (repositório privado).")
        print("Rode `uv run hf auth login` ou exporte HF_TOKEN, e peça leitura ao P2.")
        return 1

    missing = [f for f in ALLOW_PATTERNS if not (Path(path) / f).is_file()]
    if missing:
        print(f"Download incompleto, faltam: {', '.join(missing)}")
        return 1

    print(f"Modelo A3 pronto em {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
