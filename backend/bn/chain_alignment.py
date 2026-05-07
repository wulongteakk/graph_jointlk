from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


@dataclass
class StepAlignment:
    chain_step_index: int
    chain_step_text: str
    kg_node_id: Optional[str]
    prototype_id: Optional[str]
    bn_variable: Optional[str]
    match_type: str


@dataclass
class ChainAlignmentResult:
    rows: List[StepAlignment]
    alignment_rate: float
    unresolved_rate: float


def _norm(text: str) -> str:
    return str(text or "").strip().lower()


def align_chain(
    *,
    chain_steps: Sequence[str],
    kg_nodes: Optional[Iterable[Mapping[str, object]]] = None,
    prototype_alias: Optional[Mapping[str, str]] = None,
    bn_alias: Optional[Mapping[str, str]] = None,
) -> ChainAlignmentResult:
    """三层对齐：chain step -> KG node -> prototype -> BN variable。"""
    kg_nodes = list(kg_nodes or [])
    prototype_alias = dict(prototype_alias or {})
    bn_alias = dict(bn_alias or {})

    kg_index: Dict[str, Mapping[str, object]] = {}
    for node in kg_nodes:
        text = _norm(str(node.get("text") or node.get("name") or ""))
        if text:
            kg_index[text] = node

    rows: List[StepAlignment] = []
    aligned = 0
    for idx, step in enumerate(chain_steps):
        key = _norm(step)
        kg_hit = kg_index.get(key)
        proto = prototype_alias.get(key)
        if not proto and kg_hit:
            proto = prototype_alias.get(_norm(str(kg_hit.get("prototype") or "")))
        bn_var = bn_alias.get(_norm(proto or "")) or bn_alias.get(key)

        if kg_hit:
            match_type = "exact"
        elif proto:
            match_type = "prototype"
        elif bn_var:
            match_type = "bn_alias"
        else:
            match_type = "unresolved"

        if match_type != "unresolved":
            aligned += 1

        rows.append(
            StepAlignment(
                chain_step_index=idx,
                chain_step_text=str(step),
                kg_node_id=str(kg_hit.get("node_id")) if kg_hit and kg_hit.get("node_id") else None,
                prototype_id=proto,
                bn_variable=bn_var,
                match_type=match_type,
            )
        )

    total = max(1, len(chain_steps))
    alignment_rate = aligned / total
    return ChainAlignmentResult(rows=rows, alignment_rate=alignment_rate, unresolved_rate=1.0 - alignment_rate)


def to_dict(result: ChainAlignmentResult) -> Dict[str, object]:
    return {
        "rows": [r.__dict__ for r in result.rows],
        "alignment_rate": float(result.alignment_rate),
        "unresolved_rate": float(result.unresolved_rate),
    }