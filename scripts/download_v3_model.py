#!/usr/bin/env python3
"""Baixa o modelo V3 (desconforto facial) do Hugging Face para models/v3_face/.

Os pesos do V3 têm ~343 MB e não cabem no git, então ficam num repositório no Hub
(mesmo arranjo do A3 — ver scripts/download_a3_model.py). Sem eles, `src/resolve.py`
marca o V3 como "artefato ausente" e o pipeline cai para o stub, com o motivo logado.

Autenticação: se o repositório for privado, é preciso estar logado
(`uv run hf auth login`) ou exportar `HF_TOKEN`. Sendo público, roda anônimo.

Idempotente: o `snapshot_download` reaproveita o cache e só baixa o que falta.

Uso:
    uv run python scripts/download_v3_model.py
    uv run python scripts/download_v3_model.py --repo-id outro/repo
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "fiap-ssp-2025/tc4-v3-desconforto-facial"
DEFAULT_LOCAL_DIR = ROOT / "models" / "v3_face"
# preprocess.json é obrigatório: guarda o recorte/normalização do treino.
ALLOW_PATTERNS = [
    "config.json",
    "model.safetensors",
    "preprocess.json",
    "metrics.json",
    "threshold_metrics.json",
    "clip_metrics_bench_com_calm.json",
]
REQUIRED = ["config.json", "model.safetensors", "preprocess.json"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Baixa o modelo V3 do Hugging Face")
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
            # None = usa o token em cache se houver, senão tenta anônimo.
            token=os.environ.get("HF_TOKEN") or None,
        )
    except (RepositoryNotFoundError, GatedRepoError):
        print(f"Sem acesso a {args.repo_id}.")
        print("Se for privado: `uv run hf auth login` (ou HF_TOKEN) e peça leitura ao P4.")
        return 1

    missing = [f for f in REQUIRED if not (Path(path) / f).is_file()]
    if missing:
        print(f"Download incompleto, faltam: {', '.join(missing)}")
        return 1

    print(f"Modelo V3 pronto em {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
