from typing import Dict, Iterable, Mapping, Sequence


def build_alignment_map(
    chain_steps: Sequence[str],
    *,
    prototype_alias: Mapping[str, str] | None = None,
    bn_alias: Mapping[str, str] | None = None,
) -> Dict[str, str]:
    """链步骤 -> BN变量 的轻量对齐器。"""
    prototype_alias = prototype_alias or {}
    bn_alias = bn_alias or {}
    out: Dict[str, str] = {}
    for step in chain_steps:
        key = str(step).strip().lower()
        proto = prototype_alias.get(key, key)
        out[str(step)] = bn_alias.get(proto, bn_alias.get(key, proto))
    return out


def alignment_rate(step_nodes: Iterable[str], align_map: Mapping[str, str]) -> float:
    nodes = [str(x) for x in step_nodes]
    if not nodes:
        return 0.0
    ok = sum(1 for n in nodes if align_map.get(n))
    return ok / len(nodes)