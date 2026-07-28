#!/usr/bin/env python3
"""Sparse-checkout do CREMA-D VideoFlash em data/video_consulta/raw/cremad/.

Prefira o mirror do GitLab (blobs normais, ~2,3 GB de VideoFlash) — o original
no GitHub usa git-lfs e costuma falhar com HTTP 502:
  https://gitlab.com/cs-cooper-lab/crema-d-mirror
  (upstream: https://github.com/CheyneyComputerScience/CREMA-D)

Decisão (documentada): por padrão verificar/baixar SÓ emoções NEU/FEA/SAD de
atores FEMININOS (plano: FER com prioridade feminina). Use --all-actors /
--emotions para ampliar. Metadados (VideoDemographics.csv, README, LICENSE)
sempre são baixados.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.video.v3_face.face_dataset import load_cremad_sex_map

DEFAULT_RAW = ROOT / "data" / "video_consulta" / "raw" / "cremad"
# O mirror GitLab traz VideoFlash como objetos git normais (sem LFS instável do GitHub).
DEFAULT_REPO_URL = "https://gitlab.com/cs-cooper-lab/crema-d-mirror.git"
DEFAULT_EMOTIONS = ("NEU", "FEA", "SAD")


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=check)


def ensure_sparse_clone(raw_dir: Path, repo_url: str) -> Path:
    """Clona com sparse checkout de VideoFlash + demografia."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    git_dir = raw_dir / ".git"
    if not git_dir.is_dir():
        # Pula smudge LFS no clone; o mirror GitLab em geral já tem blobs reais.
        env_prefix = ["git", "-c", "filter.lfs.smudge=", "-c", "filter.lfs.process="]
        run(
            env_prefix
            + [
                "clone",
                "--filter=blob:none",
                "--sparse",
                repo_url,
                str(raw_dir),
            ]
        )
    else:
        print(f"[skip] git repo already present at {raw_dir}")

    # Non-cone permite listar arquivos individuais junto com VideoFlash/.
    run(
        [
            "git",
            "sparse-checkout",
            "set",
            "--no-cone",
            "/VideoFlash/",
            "/VideoDemographics.csv",
            "/SentenceFilenames.csv",
            "/README.md",
            "/LICENSE.txt",
        ],
        cwd=raw_dir,
        check=False,
    )
    return raw_dir


def is_lfs_pointer(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size > 1024:
        return False
    try:
        head = path.read_text(errors="ignore")
    except OSError:
        return False
    return head.startswith("version https://git-lfs.github.com/spec/v1")


def select_paths(
    video_dir: Path,
    *,
    emotions: tuple[str, ...],
    female_only: bool,
    sex_map: dict[str, str],
    max_actors: int | None,
) -> list[Path]:
    female_ids = sorted(aid for aid, sex in sex_map.items() if sex == "F")
    male_ids = sorted(aid for aid, sex in sex_map.items() if sex == "M")
    ordered = female_ids + ([] if female_only else male_ids)
    if max_actors is not None:
        ordered = ordered[: max(0, max_actors)]
    actors_wanted = set(ordered)

    emo = {e.upper() for e in emotions}
    selected: list[Path] = []
    for path in sorted(video_dir.glob("*")):
        if path.suffix.lower() not in {".flv", ".mp4"}:
            continue
        parts = path.stem.split("_")
        if len(parts) < 3:
            continue
        actor, emotion = parts[0], parts[2].upper()
        if emotion not in emo:
            continue
        if actor not in actors_wanted:
            continue
        selected.append(path)
    return selected


def lfs_pull_selected(raw_dir: Path, paths: list[Path]) -> None:
    """Puxa objetos LFS em lotes (retomável) com barra de progresso."""
    if not paths:
        print("[warn] no CREMA-D videos matched the filter")
        return
    has_lfs = shutil.which("git-lfs") is not None or (
        run(["git", "lfs", "version"], cwd=raw_dir, check=False).returncode == 0
    )
    if not has_lfs:
        raise RuntimeError(
            "git-lfs is required to fetch CREMA-D VideoFlash. "
            "Install: https://git-lfs.com/ then re-run."
        )

    pending = [p for p in paths if is_lfs_pointer(p) or p.stat().st_size < 2048]
    print(f"[lfs] {len(paths)} selected; {len(pending)} still need pull")
    if not pending:
        return

    # Agrupa por id de ator para não disparar milhares de processos.
    by_actor: dict[str, list[Path]] = {}
    for path in pending:
        actor = path.name.split("_", 1)[0]
        by_actor.setdefault(actor, []).append(path)

    for actor, actor_paths in tqdm(sorted(by_actor.items()), desc="CREMA-D LFS actors"):
        # Ainda pendente neste ator?
        still = [p for p in actor_paths if is_lfs_pointer(p) or p.stat().st_size < 2048]
        if not still:
            continue
        # Inclui globs das três emoções-alvo deste ator.
        patterns = [
            f"VideoFlash/{actor}_*_NEU_*.flv",
            f"VideoFlash/{actor}_*_FEA_*.flv",
            f"VideoFlash/{actor}_*_SAD_*.flv",
        ]
        include = ",".join(patterns)
        result = run(
            ["git", "lfs", "pull", f"--include={include}"],
            cwd=raw_dir,
            check=False,
        )
        if result.returncode != 0:
            print(f"[warn] batch LFS pull failed for actor {actor}; trying per-file…")
            for path in still[:5]:  # amostra de retry; conjunto completo na próxima execução
                rel = path.relative_to(raw_dir).as_posix()
                run(["git", "lfs", "pull", f"--include={rel}"], cwd=raw_dir, check=False)
            # Segue — retomável na próxima invocação
            continue
        # Confirma que pelo menos um arquivo real chegou
        real_now = sum(1 for p in still if p.is_file() and p.stat().st_size > 2048 and not is_lfs_pointer(p))
        if real_now == 0:
            print(f"[warn] actor {actor}: LFS pull returned 0 real files (network?). Will resume later.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sparse download CREMA-D VideoFlash (T104)")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO_URL,
        help="Git remote (default: GitLab mirror; GitHub upstream needs LFS)",
    )
    parser.add_argument(
        "--emotions",
        nargs="+",
        default=list(DEFAULT_EMOTIONS),
        help="Emotion codes to pull (default: NEU FEA SAD)",
    )
    parser.add_argument(
        "--all-actors",
        action="store_true",
        help="Include male actors (default: female only — plan prioritizes female)",
    )
    parser.add_argument(
        "--max-actors",
        type=int,
        default=None,
        help="Optional cap on number of actors (female first)",
    )
    parser.add_argument(
        "--skip-lfs",
        action="store_true",
        help="Only sparse-checkout / skip LFS pull (GitLab mirror usually already has blobs)",
    )
    args = parser.parse_args()

    ensure_sparse_clone(args.raw_dir, args.repo)
    demo = args.raw_dir / "VideoDemographics.csv"
    if not demo.is_file():
        raise SystemExit(f"Missing {demo} after sparse checkout")
    sex_map = load_cremad_sex_map(demo)
    # Normaliza chaves para strings de 4 dígitos usadas nos nomes de arquivo
    sex_map_norm = {f"{int(k):04d}": v for k, v in sex_map.items()}

    video_dir = args.raw_dir / "VideoFlash"
    selected = select_paths(
        video_dir,
        emotions=tuple(args.emotions),
        female_only=not args.all_actors,
        sex_map=sex_map_norm,
        max_actors=args.max_actors,
    )
    print(
        f"[select] {len(selected)} clips "
        f"(emotions={args.emotions}, female_only={not args.all_actors}, "
        f"max_actors={args.max_actors})"
    )

    if not args.skip_lfs:
        lfs_pull_selected(args.raw_dir, selected)

    real = [
        p
        for p in selected
        if p.is_file() and p.stat().st_size > 2048 and not is_lfs_pointer(p)
    ]
    print(f"[ok] CREMA-D ready: {len(real)} real videos under {video_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
