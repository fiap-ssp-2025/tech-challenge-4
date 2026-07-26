#!/usr/bin/env python3
"""Publica models/v3_face/ no Hugging Face, com model card (T112 / P4).

Espelha o arranjo do A3: pesos fora do git, num repositorio **privado** da org do
time, e `scripts/download_v3_model.py` restaurando em models/v3_face/.

Privado por padrao: o modelo deriva de RAVDESS (CC BY-NC-SA 4.0) e CREMA-D, cujas
licencas nao autorizam redistribuicao publica sem cuidado. Use --public so com
decisao explicita do grupo.

Uso:
    HF_TOKEN=... uv run python scripts/t112/publish_v3_model.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / "models" / "v3_face"
DEFAULT_REPO = "fiap-ssp-2025/tc4-v3-desconforto-facial"

CARD = """---
license: cc-by-nc-sa-4.0
base_model: trpakov/vit-face-expression
tags: [image-classification, facial-expression-recognition, vision-transformer, pt-br]
---

# TC4 · V3 — Desconforto facial

Modulo **V3** do Tech Challenge 4 (FIAP) — apoio a triagem multimodal em consultas de
saude da mulher. Recebe recortes de rosto e devolve um escore continuo
`desconforto_facial` em [0, 1]. **Nao e diagnostico**: alimenta uma nota de triagem
cuja decisao final e humana.

## Como foi treinado

- **Base:** `trpakov/vit-face-expression` (ViT-base ja treinado em expressao facial).
- **Dados:** frames faciais de RAVDESS + CREMA-D (tarefa T104), 20.612 imagens.
  Binario: `desconforto` = {fearful, sad}; `neutro` = {neutral, calm}.
- **Split por ator** (80 treino / 17 validacao / 18 teste) — nenhuma identidade
  aparece em dois splits, senao a metrica sai inflada por memorizacao de rosto.
- Selecao de arquitetura, epoca e limiar **sempre na validacao**; teste medido uma
  vez por rodada.

## Resultados (conjunto de teste, split por ator)

| Unidade | F1 macro | AUC |
|---|---|---|
| Por frame | **0,7045** | 0,7811 |
| Por clipe (unidade real de uso) | **0,7108** | **0,8076** |

Meta do projeto: F1 macro >= 0,70 — **atingida no limiar padrao 0,50**, sem calibracao.

### O que produziu o ganho

Experimento 2x2 (backbone {ImageNet, FER} x treino {com `calm`, sem `calm`}):
o **backbone pre-treinado em expressao facial** elevou o AUC de 0,7605 para 0,8076;
**remover `calm` piorou** nas duas redes — esses frames sao negativos uteis, ensinam
que rosto relaxado nao e desconforto.

## Como usar

O diretorio e auto-contido e carrega offline. `preprocess.json` acompanha os pesos
porque o modelo pontua **recortes de rosto**, nao o quadro inteiro:

1. detectar pessoa com YOLOv8n, recortar os 55% superiores da caixa + margem 0,15;
2. redimensionar para 224x224, normalizar com mean=std=0,5;
3. pontuar; para um video, **tirar a media dos frames** (foi assim que o 0,7108 foi medido).

Implementacao de referencia: `src/video/v3_face/infer.py` no repositorio do projeto.

## Limitacoes (declaracao obrigatoria — RNF-06)

- **Rotulo por proxy.** Sao atores encenando medo/tristeza em estudio, nao desconforto
  clinico anotado por profissional. O modelo mede *expressao atuada como aproximacao*
  de desconforto — a validacao com dado real e trabalho futuro.
- **Margem estreita.** Com 669 clipes de teste, a incerteza e da ordem de +/- 0,035:
  a meta foi atingida, nao superada com folga.
- **Poucas identidades** (115 atores) e dominio de estudio, nao de consultorio.
- Uso academico e nao comercial, herdado das licencas dos datasets.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Publica o V3 no Hugging Face")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--repo-id", default=DEFAULT_REPO)
    ap.add_argument("--public", action="store_true", help="publica aberto (requer decisao do grupo)")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN ausente. Exporte o token (ou rode `uv run hf auth login`).")
        return 1
    if not (args.model_dir / "model.safetensors").is_file():
        print(f"Modelo nao encontrado em {args.model_dir}. Rode scripts/t112/export_v3_model.py")
        return 1

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", private=not args.public, exist_ok=True)

    card = args.model_dir / "README.md"
    card.write_text(CARD)

    api.upload_folder(
        repo_id=args.repo_id,
        folder_path=str(args.model_dir),
        repo_type="model",
        commit_message="T112: V3 desconforto facial (ViT-FER), F1 0,7108 por clipe",
    )
    info = api.model_info(args.repo_id)
    visib = "PUBLICO" if not info.private else "privado"
    print(f"[ok] publicado em https://huggingface.co/{args.repo_id}  ({visib})")
    print("Arquivos:", ", ".join(sorted(s.rfilename for s in info.siblings)))
    print(json.dumps({"repo_id": args.repo_id, "private": info.private}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
