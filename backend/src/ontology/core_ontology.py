CORE_NODE_TYPES = [
    "SourceState",
    "SourceEvent",
    "IntermediateEvent",
    "HarmEvent",
    "Consequence",
    "WorkActivity",
    "Equipment",
    "Material",
    "Barrier",
    "Environment",
    "Location",
    "PersonRole",
    "OrganizationAction",
    "StandardClause",
    "EvidenceUnit",
]

CORE_REL_TYPES = [
    "CAUSES",
    "ENABLES",
    "PRECEDES",
    "CO_OCCURS_IN_EVIDENCE",
    "SUPPORTED_BY",
    "VIOLATES",
    "INVOLVES",
    "LOCATED_AT",
    "SAME_AS",
]

HFCSA_LAYERS = ["O", "S", "C", "A", "E", "L"]
EVENT_STAGES = ["initiating", "intermediate", "harm"]

LEGACY_NODE_TYPE_MAP = {
    "managementaction": "OrganizationAction",
    "standardclause": "StandardClause",
    "resourcecondition": "SourceState",
    "barrier": "Barrier",
    "humanstate": "SourceState",
    "hazardsource": "SourceState",
    "unsafeact": "SourceEvent",
    "loss": "Consequence",
}

DEFAULT_LAYER_BY_CORE_TYPE = {
    "OrganizationAction": "S",
    "SourceState": "C",
    "Barrier": "C",
    "Equipment": "C",
    "Material": "C",
    "Environment": "C",
    "Location": "C",
    "PersonRole": "C",
    "StandardClause": "O",
    "SourceEvent": "A",
    "IntermediateEvent": "E",
    "HarmEvent": "E",
    "Consequence": "L",
    "EvidenceUnit": "E",
}

HARM_EVENT_HINTS = ["坠落", "侧翻", "坍塌", "打击", "触电", "爆炸", "着火"]
INTERMEDIATE_EVENT_HINTS = ["断裂", "失稳", "滑移", "受力失衡", "带电体暴露", "扩散"]


def normalize_legacy_node_type(node_type: str, text: str = "") -> str:
    nt = (node_type or "").strip().lower()
    if nt in LEGACY_NODE_TYPE_MAP:
        return LEGACY_NODE_TYPE_MAP[nt]

    if nt == "accidentevent":
        if any(k in text for k in HARM_EVENT_HINTS):
            return "HarmEvent"
        return "IntermediateEvent"

    return node_type or "SourceState"