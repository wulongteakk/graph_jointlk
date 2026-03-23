from __future__ import annotations

from typing import Any, Dict, Optional, Set, Tuple

from src.ontology.core_ontology import DEFAULT_LAYER_BY_CORE_TYPE, normalize_legacy_node_type

PROJECTION_REL_TYPES: Set[str] = {"CAUSES", "ENABLES", "PRECEDES", "NEXT_LEVEL_CAUSES"}


def infer_event_stage(core_type: str, text: str) -> Optional[str]:
    if core_type == "SourceEvent":
        return "initiating"
    if core_type == "IntermediateEvent":
        return "intermediate"
    if core_type == "HarmEvent":
        return "harm"
    return None


def apply_hfsca_overlay_to_node(node: Any, pack) -> Dict[str, Any]:
    props = node.properties or {}
    text = props.get("name") or props.get("text") or node.id or ""
    core_type = normalize_legacy_node_type(node.type, text)

    mapping = pack.map_to_hfsca(core_type=core_type, text=text, props=props)
    event_stage = infer_event_stage(core_type, text)
    module_id = pack.match_module(text)
    hfsca_layer = mapping.get("hfsca_layer") or DEFAULT_LAYER_BY_CORE_TYPE.get(core_type)
    hfsca_category = mapping.get("hfsca_category")

    out = dict(props)
    out.update({
        "core_type": core_type,
        "hfsca_layer": hfsca_layer,
        "hfsca_category": hfsca_category,
        "module_id": module_id,
        "event_stage": event_stage,
        "hfcsa_reason": mapping.get("reason"),
        "hfcsa_confidence": mapping.get("confidence", 0.7),
        "layer_code": hfsca_layer,
        "category_code": hfsca_category,
        "is_main_chain_candidate": core_type in {
            "SourceState", "SourceEvent", "IntermediateEvent",
            "HarmEvent", "Consequence", "OrganizationAction"
        },
    })
    return out


def _project_next_level_pairs(graph_doc: Any) -> Set[Tuple[str, str]]:
    node_meta: Dict[str, Dict[str, Any]] = {}
    for node in getattr(graph_doc, "nodes", []) or []:
        props = node.properties or {}
        node_meta[node.id] = {
            "layer": props.get("hfsca_layer") or props.get("layer_code"),
            "is_main_chain_candidate": bool(props.get("is_main_chain_candidate")),
        }

    pairs: Set[Tuple[str, str]] = set()
    for rel in getattr(graph_doc, "relationships", []) or []:
        source = getattr(rel, "source", None)
        target = getattr(rel, "target", None)
        source_id = getattr(source, "id", None)
        target_id = getattr(target, "id", None)
        rel_type = (getattr(rel, "type", None) or "").upper()
        if not source_id or not target_id:
            continue
        if rel_type not in PROJECTION_REL_TYPES:
            continue

        source_meta = node_meta.get(source_id) or {}
        target_meta = node_meta.get(target_id) or {}
        if not source_meta.get("is_main_chain_candidate") or not target_meta.get("is_main_chain_candidate"):
            continue
        if not source_meta.get("layer") or not target_meta.get("layer"):
            continue
        pairs.add((source_id, target_id))
    return pairs


def apply_hfsca_overlay(graph_doc, pack):
    for node in graph_doc.nodes:
        node.properties = apply_hfsca_overlay_to_node(node, pack)

    next_level_pairs = _project_next_level_pairs(graph_doc)
    for rel in getattr(graph_doc, "relationships", []) or []:
        source_id = getattr(getattr(rel, "source", None), "id", None)
        target_id = getattr(getattr(rel, "target", None), "id", None)
        rel.properties = dict(getattr(rel, "properties", None) or {})
        rel.properties["is_next_level_projection"] = (source_id, target_id) in next_level_pairs
        if rel.properties["is_next_level_projection"]:
            rel.properties["projection_type"] = "NEXT_LEVEL_CAUSES"

    return graph_doc