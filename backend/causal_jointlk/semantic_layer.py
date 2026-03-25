from __future__ import annotations

from typing import Optional


LAYER_TO_CANONICAL = {
    "STATE": "SourceState",
    "CAUSE": "SourceEvent",
    "FACTOR": "SourceEvent",
    "CONDITION": "SourceEvent",
    "ACTION": "SourceEvent",
    "INTERMEDIATE": "IntermediateEvent",
    "MECHANISM": "IntermediateEvent",
    "EVENT": "HarmEvent",
    "OUTCOME": "HarmEvent",
    "CONSEQUENCE": "Consequence",
}


def to_accident_canonical_type(layer: Optional[str], core_type: Optional[str] = None) -> str:
    if core_type and core_type in {
        "SourceState", "SourceEvent", "IntermediateEvent", "HarmEvent", "Consequence"
    }:
        return core_type
    return LAYER_TO_CANONICAL.get(str(layer or "UNK").upper(), "SourceEvent")