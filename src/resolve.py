"""Module resolution: real `infer()` when available, stub otherwise — always logged.

The pipeline must survive the trainings landing one at a time (T110–T112), so no
caller imports `src.stubs.*` directly. `get_module(name)` decides, per module:

  1. TC4_FORCE_STUBS=1                       → stub (demo controlada)
  2. python package missing                  → stub
  3. artifact missing (weights / .pkl)       → stub
  4. credential missing (.env)               → stub
  5. real module fails to import             → stub
  6. real module still re-exports the stub   → stub
  7. otherwise                               → real

TC4_REQUIRE_REAL=v1_tracks,v2_pose turns a stub fallback into a hard error, so a
future CI job can assert the real modules are wired. The two env vars are
mutually exclusive.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]

ENV_FORCE_STUBS = "TC4_FORCE_STUBS"
ENV_REQUIRE_REAL = "TC4_REQUIRE_REAL"

_TRUTHY = frozenset({"1", "true", "yes", "sim", "on"})


class ModuleResolutionError(RuntimeError):
    """Raised on an unknown module name or an unmet TC4_REQUIRE_REAL demand."""


@dataclass(frozen=True)
class ModuleSpec:
    """Where the real module lives and what it needs to actually run."""

    name: str
    key: str  # short key used in the runner output (a1, a2, …)
    real: str  # dotted path of the real implementation
    stub: str  # dotted path of the stub fallback
    packages: tuple[str, ...] = ()  # imported lazily inside infer() — probe by spec
    artifacts: tuple[Path, ...] = ()  # trained weights that must exist on disk
    env_vars: tuple[str, ...] = ()  # credentials that must be set
    hint: str = ""  # how to make the real module available


V2_POSTURE_HEAD = ROOT / "models" / "v2_posture_head.pkl"
A3_EMOTION_DIR = ROOT / "models" / "a3_emotion"

REGISTRY: dict[str, ModuleSpec] = {
    "a1_stt": ModuleSpec(
        name="a1_stt",
        key="a1",
        real="src.audio.a1_stt.infer",
        stub="src.stubs.a1_stt",
        # Azure é preferencial; sem chave o infer real cai para faster-whisper.
        packages=("azure.cognitiveservices.speech", "faster_whisper"),
        hint="uv sync + AZURE_SPEECH_KEY no .env (ou só faster-whisper offline)",
    ),
    "a2_nlp": ModuleSpec(
        name="a2_nlp",
        key="a2",
        real="src.audio.a2_nlp.infer",
        stub="src.stubs.a2_nlp",
        hint="T111/P3 entrega o PLN por regras",
    ),
    "a3_emotion": ModuleSpec(
        name="a3_emotion",
        key="a3",
        real="src.audio.a3_emotion.infer",
        stub="src.stubs.a3_emotion",
        packages=("transformers",),
        artifacts=(A3_EMOTION_DIR,),
        hint="uv run python scripts/train_a3_emotion.py (ou baixe o modelo publicado — ver README)",
    ),
    "v1_tracks": ModuleSpec(
        name="v1_tracks",
        key="v1",
        real="src.video.v1_tracks.infer",
        stub="src.stubs.v1_tracks",
        packages=("ultralytics",),
        hint="uv sync (ultralytics baixa yolov8n.pt no primeiro uso)",
    ),
    "v2_pose": ModuleSpec(
        name="v2_pose",
        key="v2",
        real="src.video.v2_pose.infer",
        stub="src.stubs.v2_pose",
        packages=("ultralytics",),
        artifacts=(V2_POSTURE_HEAD,),
        hint="uv run python scripts/train_v2_posture.py",
    ),
    "v3_face": ModuleSpec(
        name="v3_face",
        key="v3",
        real="src.video.v3_face.infer",
        stub="src.stubs.v3_face",
        hint="T112/P4 entrega o FER de desconforto facial",
    ),
}

MODULE_NAMES: tuple[str, ...] = tuple(REGISTRY)

# Accept both the full name (v2_pose) and the short key (v2) everywhere.
_ALIASES: dict[str, str] = {name: name for name in REGISTRY}
_ALIASES.update({spec.key: name for name, spec in REGISTRY.items()})


@dataclass(frozen=True)
class ResolvedModule:
    """A module chosen by `get_module`, plus why it was chosen."""

    spec: ModuleSpec
    origin: str  # "real" | "stub"
    reason: str  # empty when origin == "real"
    module: ModuleType

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def key(self) -> str:
        return self.spec.key

    @property
    def infer(self) -> Callable[..., Any]:
        return self.module.infer

    def describe(self) -> str:
        suffix = f"  ({self.reason})" if self.reason else ""
        return f"{self.name:<11} → {self.origin}{suffix}"


def canonical_name(name: str) -> str:
    """Map a name or short key to the registry key. Raises on unknown names."""
    resolved = _ALIASES.get(str(name).strip().lower())
    if resolved is None:
        raise ModuleResolutionError(
            f"módulo desconhecido: {name!r}. Válidos: {', '.join(MODULE_NAMES)}"
        )
    return resolved


def _flag(var: str) -> bool:
    return os.getenv(var, "").strip().lower() in _TRUTHY


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _hint(spec: ModuleSpec) -> str:
    return f" — {spec.hint}" if spec.hint else ""


def required_real() -> frozenset[str]:
    """Modules that TC4_REQUIRE_REAL demands to be real. Raises on conflicts."""
    raw = os.getenv(ENV_REQUIRE_REAL, "").replace(";", ",")
    names = frozenset(canonical_name(tok) for tok in raw.split(",") if tok.strip())
    if names and _flag(ENV_FORCE_STUBS):
        raise ModuleResolutionError(
            f"{ENV_FORCE_STUBS} e {ENV_REQUIRE_REAL} são mutuamente exclusivos."
        )
    return names


def _has_package(package: str) -> bool:
    try:
        return importlib.util.find_spec(package) is not None
    except (ImportError, ValueError):
        return False


def _decide(spec: ModuleSpec) -> tuple[str, str, ModuleType]:
    """Return (origin, reason, module) for one spec. Never raises on a fallback."""
    stub = importlib.import_module(spec.stub)

    if _flag(ENV_FORCE_STUBS):
        return "stub", f"{ENV_FORCE_STUBS}=1 (demo controlada)", stub

    missing_pkgs = [p for p in spec.packages if not _has_package(p)]
    if missing_pkgs:
        return "stub", f"pacote ausente: {', '.join(missing_pkgs)}{_hint(spec)}", stub

    missing_art = [a for a in spec.artifacts if not a.exists()]
    if missing_art:
        listed = ", ".join(_rel(a) for a in missing_art)
        return "stub", f"artefato ausente: {listed}{_hint(spec)}", stub

    missing_env = [e for e in spec.env_vars if not os.getenv(e, "").strip()]
    if missing_env:
        return "stub", f"credencial ausente: {', '.join(missing_env)}{_hint(spec)}", stub

    try:
        real = importlib.import_module(spec.real)
    except Exception as exc:  # ImportError e qualquer falha de import-time
        return "stub", f"import falhou ({exc.__class__.__name__}: {exc})", stub

    real_infer = getattr(real, "infer", None)
    if real_infer is None:
        return "stub", f"{spec.real} não expõe infer()", stub
    if real_infer is getattr(stub, "infer", None):
        return "stub", f"{spec.real} ainda re-exporta o stub{_hint(spec)}", stub

    return "real", "", real


def get_module(name: str, *, verbose: bool = False) -> ResolvedModule:
    """Resolve one pipeline module to its real implementation or to the stub."""
    spec = REGISTRY[canonical_name(name)]
    required = required_real()  # valida as env vars mesmo quando o real está OK
    origin, reason, module = _decide(spec)

    if origin == "stub" and spec.name in required:
        raise ModuleResolutionError(
            f"{ENV_REQUIRE_REAL} exige {spec.name} real, mas caiu para stub: {reason}"
        )

    resolved = ResolvedModule(spec=spec, origin=origin, reason=reason, module=module)
    if verbose:
        print(f"[resolve] {resolved.describe()}")
    return resolved


class ResolvedPipeline:
    """The six pipeline modules resolved once, timed on every call."""

    def __init__(
        self, names: Iterable[str] | None = None, *, verbose: bool = True
    ) -> None:
        self._required = required_real()
        self._modules: dict[str, ResolvedModule] = {}
        self._timings_ms: dict[str, float] = {}
        for name in names or MODULE_NAMES:
            resolved = get_module(name)
            self._modules[resolved.name] = resolved
        if verbose:
            self.print_map()

    def __getitem__(self, name: str) -> ResolvedModule:
        return self._modules[canonical_name(name)]

    def origin(self, name: str) -> str:
        return self[name].origin

    def origins(self) -> dict[str, str]:
        """Short-key map for the runner output: {"a1": "stub", "v1": "real", …}."""
        return {m.key: m.origin for m in self._modules.values()}

    def timings_ms(self) -> dict[str, float]:
        return {k: round(v, 1) for k, v in self._timings_ms.items()}

    def print_map(self) -> None:
        print("[resolve] módulos do pipeline (real = modelo treinado, stub = fixo):")
        for resolved in self._modules.values():
            print(f"[resolve]   {resolved.describe()}")

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Run `infer()`, timing it. A real module that blows up degrades to the stub."""
        resolved = self[name]
        started = time.perf_counter()
        try:
            payload = resolved.infer(*args, **kwargs)
        except Exception as exc:
            if resolved.origin == "stub" or resolved.name in self._required:
                self._timings_ms[resolved.key] = (time.perf_counter() - started) * 1000
                raise
            print(
                f"[resolve] {resolved.name}: real falhou em execução "
                f"({exc.__class__.__name__}: {exc}) — caindo para o stub."
            )
            stub = importlib.import_module(resolved.spec.stub)
            self._modules[resolved.name] = ResolvedModule(
                spec=resolved.spec,
                origin="stub",
                reason=f"falha em execução: {exc.__class__.__name__}",
                module=stub,
            )
            payload = stub.infer(*args, **kwargs)
        self._timings_ms[resolved.key] = (time.perf_counter() - started) * 1000
        return payload


__all__ = [
    "ENV_FORCE_STUBS",
    "ENV_REQUIRE_REAL",
    "MODULE_NAMES",
    "REGISTRY",
    "ModuleResolutionError",
    "ModuleSpec",
    "ResolvedModule",
    "ResolvedPipeline",
    "canonical_name",
    "get_module",
    "required_real",
]
