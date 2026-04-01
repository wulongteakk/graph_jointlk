import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def load_prior_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"prior config not found: {config_path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


class CausalPrior:
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.rel_whitelist = set((self.config.get("relations") or {}).get("whitelist") or [])
        self.ctp = {str(k).upper(): [str(v).upper() for v in vs] for k, vs in (self.config.get("ctp_allowed_transitions") or {}).items()}
        self.layer_order = [str(x).upper() for x in self.config.get("layer_order") or []]
        trigger_words = self.config.get("trigger_words") or {}
        self.trigger_words = {lang: [str(x).strip().lower() for x in values] for lang, values in trigger_words.items()}
        relation_aliases = self.config.get("relation_aliases") or {}
        self.relation_aliases = {str(rel).upper(): [str(x).strip().lower() for x in aliases] for rel, aliases in relation_aliases.items()}
        self.accident_layer_mapping = {str(k).upper(): str(v) for k, v in (self.config.get("accident_layer_mapping") or {}).items()}
        self.counterfactual_rules = dict(self.config.get("counterfactual_rules") or {})
        self.first_induced = dict(self.config.get("first_induced") or {})

    def is_relation_allowed(self, rel: Optional[str]) -> bool:
        return bool(rel) and str(rel).upper() in self.rel_whitelist

    def normalize_relation(self, rel: Optional[str], text: Optional[str] = None) -> str:
        rel_up = str(rel or "").upper().strip()
        if rel_up in self.rel_whitelist:
            return rel_up
        haystack = f"{rel or ''} {text or ''}".strip().lower()
        for canonical_rel, aliases in self.relation_aliases.items():
            if any(alias in haystack for alias in aliases):
                return canonical_rel
        return rel_up if rel_up else "UNK"

    def allowed_transition(self, source_layer: Optional[str], target_layer: Optional[str]) -> bool:
        src = str(source_layer or "UNK").upper()
        tgt = str(target_layer or "UNK").upper()
        if src == "UNK" or tgt == "UNK" or src not in self.ctp:
            return True
        return tgt in self.ctp[src]

    def layer_distance(self, source_layer: Optional[str], target_layer: Optional[str]) -> int:
        src = str(source_layer or "UNK").upper()
        tgt = str(target_layer or "UNK").upper()
        if src not in self.layer_order or tgt not in self.layer_order:
            return 0
        return max(0, self.layer_order.index(tgt) - self.layer_order.index(src))

    def count_missing_layers(self, layers: Sequence[str]) -> List[str]:
        if not self.layer_order:
            return []
        normalized = [str(x).upper() for x in layers]
        seen = set(normalized)
        wanted = [x for x in self.layer_order if x in seen]
        if not wanted:
            return []
        start_idx = self.layer_order.index(wanted[0])
        end_idx = self.layer_order.index(wanted[-1])
        return [layer for layer in self.layer_order[start_idx : end_idx + 1] if layer not in seen]

    def find_trigger_words(self, text: Optional[str]) -> List[str]:
        text = (text or "").lower()
        hits: List[str] = []
        for words in self.trigger_words.values():
            for word in words:
                if word and word in text:
                    hits.append(word)
        return sorted(set(hits))

    def as_beam_params(self) -> Dict[str, Any]:
        beam_cfg = self.config.get("beam")
        if isinstance(beam_cfg, dict) and beam_cfg:
            return dict(beam_cfg)
        return dict(self.config.get("beam_search") or {})

    def as_evidence_params(self) -> Dict[str, Any]:
        return dict(self.config.get("evidence_gate") or {})

    def canonical_layer(self, raw_layer: Optional[str]) -> str:
        layer = str(raw_layer or "UNK").upper()
        return self.accident_layer_mapping.get(layer, layer)

    def as_counterfactual_params(self) -> Dict[str, Any]:
        merged = {}
        if isinstance(self.config.get("counterfactual"), dict):
            merged.update(dict(self.config.get("counterfactual") or {}))
        merged.update(dict(self.counterfactual_rules or {}))
        return merged

    def normalize_node_type(self, canonical_type: Optional[str]) -> str:
        mapping = self.first_induced.get("canonical_map") or {}
        c = str(canonical_type or "").strip()
        return mapping.get(c, c or "SourceEvent")


def normalize_name(text: Optional[str]) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text