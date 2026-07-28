"""Gerador de nota de triagem (C/D) — apoio à equipe de cuidado, não um veredito."""

from __future__ import annotations

from typing import Any

from src.contracts import CDResult, validate_cd
from src.fusion.scoring import score_from_modules

DISCLAIMER = (
    "Esta nota é indicativo, não veredito — decisão humana."
)


def build_nota_texto(
    *,
    a12: dict[str, Any],
    a3: dict[str, Any],
    v1: dict[str, Any] | None,
    v2: dict[str, Any] | None,
    v3: dict[str, Any] | None,
    escore: float,
    corroborado: bool,
) -> str:
    lines = [
        "NOTA DE TRIAGEM (apoio à equipe de saúde — consulta)",
        f"Tipo de relato: {a12.get('tipo_relato', 'n/d')}",
        f"Contexto: {a12.get('local', 'n/d')}",
        f"Tempo (relato): {a12.get('tempo', 'n/d')}",
        f"Transcrição: {a12.get('transcricao', '')}",
        f"Sofrimento (voz): {float(a3.get('sofrimento', 0)):.2f}",
    ]
    if v1 is not None:
        lines.append(f"Pessoas (vídeo): {v1.get('n_pessoas', 'n/d')}")
    if v2 is not None:
        lines.append(f"Postura defensiva: {float(v2.get('postura_defensiva', 0)):.2f}")
    if v3 is not None:
        lines.append(f"Desconforto facial: {float(v3.get('desconforto_facial', 0)):.2f}")
    lines.append(f"Corroborado (mesma sessão): {'sim' if corroborado else 'não'}")
    lines.append(f"Escore de triagem: {escore:.3f}")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


def build_report(
    *,
    a12: dict[str, Any],
    a3: dict[str, Any],
    v1: dict[str, Any] | None = None,
    v2: dict[str, Any] | None = None,
    v3: dict[str, Any] | None = None,
    corroborado: bool = False,
) -> CDResult:
    """Monta o contrato C/D a partir das saídas dos módulos."""
    v2_safe = v2 or {"postura_defensiva": 0.0}
    v3_safe = v3 or {"desconforto_facial": 0.0}
    escore = score_from_modules(a12, a3, v2_safe, v3_safe, corroborado=corroborado)
    nota = build_nota_texto(
        a12=a12,
        a3=a3,
        v1=v1,
        v2=v2,
        v3=v3,
        escore=escore,
        corroborado=corroborado,
    )
    return validate_cd(
        {
            "escore": escore,
            "corroborado": corroborado,
            "nota_ocorrencia": nota,
        }
    )
