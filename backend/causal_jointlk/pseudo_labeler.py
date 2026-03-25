from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def load_yaml_file(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to load YAML config files.")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a dict: {path}")
    return data


def merge_prior_and_pseudo_config(
    prior_cfg: Optional[Dict[str, Any]],
    pseudo_cfg: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    prior_cfg = prior_cfg or {}
    pseudo_cfg = pseudo_cfg or {}
    merged: Dict[str, Any] = dict(pseudo_cfg)

    prior_relation_whitelist = (
        prior_cfg.get("relation_type_whitelist")
        or prior_cfg.get("relation_whitelist")
        or prior_cfg.get("relation_types")
        or []
    )
    pseudo_relation_whitelist = pseudo_cfg.get("relation_whitelist") or []
    if prior_relation_whitelist:
        merged["relation_whitelist"] = sorted(
            {
                str(x).strip().upper()
                for x in list(prior_relation_whitelist) + list(pseudo_relation_whitelist)
                if str(x).strip()
            }
        )

    prior_transitions = prior_cfg.get("CTP_ALLOWED_TRANSITIONS") or prior_cfg.get("ctp_allowed_transitions") or {}
    pseudo_transitions = pseudo_cfg.get("allowed_transitions") or {}
    final_transitions: Dict[str, List[str]] = {}
    for key in set(list(prior_transitions.keys()) + list(pseudo_transitions.keys())):
        vals = list(prior_transitions.get(key, []) or []) + list(pseudo_transitions.get(key, []) or [])
        final_transitions[str(key).upper()] = sorted({str(v).upper() for v in vals if str(v).strip()})
    if final_transitions:
        merged["allowed_transitions"] = final_transitions

    for key in [
        "strong_triggers",
        "weak_triggers",
        "negation_cues",
        "uncertainty_cues",
        "max_entity_gap_chars",
        "min_relation_prop_confidence",
        "positive_min_confidence",
        "negative_min_confidence",
        "rule_weights",
        "negative_rule_weights",
        "review",
        "evidence_retrieval",
    ]:
        if key not in merged and key in prior_cfg:
            merged[key] = prior_cfg[key]
    return merged


def normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize_for_overlap(text: Optional[str]) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    # 中文按字串短片段 + 英文按词
    ascii_tokens = re.findall(r"[a-z0-9_]+", text)
    zh_tokens: List[str] = []
    for piece in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        if len(piece) <= 4:
            zh_tokens.append(piece)
        else:
            for i in range(0, len(piece) - 1):
                zh_tokens.append(piece[i : i + 2])
    return ascii_tokens + zh_tokens


def lexical_overlap(a: Optional[str], b: Optional[str]) -> float:
    ta = set(tokenize_for_overlap(a))
    tb = set(tokenize_for_overlap(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


def find_first_occurrence(text: str, needle: str) -> int:
    if not text or not needle:
        return -1
    return text.find(needle)


def coerce_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def relation_confidence_from_props(rel_props: Dict[str, Any]) -> float:
    candidates = [
        rel_props.get("confidence"),
        rel_props.get("score"),
        rel_props.get("probability"),
        rel_props.get("weight"),
        rel_props.get("llm_score"),
        rel_props.get("causal_score"),
    ]
    vals = [coerce_float(x, default=-1.0) for x in candidates if x is not None]
    vals = [x for x in vals if x >= 0.0]
    if not vals:
        return 0.0
    # 如果本身就是 0-100，则拉回 0-1。
    if max(vals) > 1.0:
        vals = [min(1.0, v / 100.0) for v in vals]
    return max(vals)


@dataclass
class PseudoLabelDecision:
    label: Optional[int]
    confidence: float
    relation_type: Optional[str]
    primary_rule: Optional[str]
    rule_hits: List[str] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)
    evidence_unit_id: Optional[str] = None
    evidence_text: Optional[str] = None

@dataclass
class MultiTaskPseudoLabelDecision:
    silver_edge_causal: int
    causal_conf: float
    silver_edge_enable: int
    enable_conf: float
    silver_causal_dir: int
    dir_conf: float
    silver_temporal_before: int
    temporal_conf: float
    silver_node_first_src: int
    src_first_conf: float
    silver_node_first_dst: int
    dst_first_conf: float
    rule_hits: Dict[str, List[str]] = field(default_factory=dict)
    sample_weight: float = 1.0
    twin_group_id: Optional[str] = None
    review_status: str = "pending"
    evidence_unit_id: Optional[str] = None
    evidence_text: Optional[str] = None
    features: Dict[str, Any] = field(default_factory=dict)


class CausalPseudoLabeler:
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.relation_whitelist = {
            str(x).strip().upper()
            for x in (self.config.get("relation_whitelist") or [])
            if str(x).strip()
        }
        self.allowed_transitions = {
            str(k).strip().upper(): {str(v).strip().upper() for v in vals}
            for k, vals in (self.config.get("allowed_transitions") or {}).items()
        }
        self.strong_triggers = [normalize_text(x) for x in self.config.get("strong_triggers", []) if normalize_text(x)]
        self.weak_triggers = [normalize_text(x) for x in self.config.get("weak_triggers", []) if normalize_text(x)]
        self.negation_cues = [normalize_text(x) for x in self.config.get("negation_cues", []) if normalize_text(x)]
        self.uncertainty_cues = [normalize_text(x) for x in self.config.get("uncertainty_cues", []) if normalize_text(x)]
        self.max_entity_gap_chars = int(self.config.get("max_entity_gap_chars", 120))
        self.min_relation_prop_confidence = float(self.config.get("min_relation_prop_confidence", 0.75))
        self.positive_min_confidence = float(self.config.get("positive_min_confidence", 0.90))
        self.negative_min_confidence = float(self.config.get("negative_min_confidence", 0.88))
        self.rule_weights = self.config.get("rule_weights") or {}
        self.negative_rule_weights = self.config.get("negative_rule_weights") or {}

    def _transition_allowed(self, source_layer: Optional[str], target_layer: Optional[str]) -> Optional[bool]:
        if not source_layer or not target_layer:
            return None
        src = str(source_layer).strip().upper()
        tgt = str(target_layer).strip().upper()
        if src not in self.allowed_transitions:
            return None
        return tgt in self.allowed_transitions.get(src, set())

    def _contains_any(self, text: str, phrases: Sequence[str]) -> List[str]:
        hits: List[str] = []
        for phrase in phrases:
            if phrase and phrase in text:
                hits.append(phrase)
        return hits

    def _evidence_features(
        self,
        source_text: str,
        target_text: str,
        evidence_units: Sequence[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        best: Optional[Dict[str, Any]] = None
        best_score = -1.0
        aggregate = {
            "shared_evidence": False,
            "strong_trigger_hit": False,
            "weak_trigger_hit": False,
            "negation_hit": False,
            "uncertainty_hit": False,
            "source_before_target": None,
            "min_gap_chars": None,
            "matched_evidence_unit_id": None,
            "matched_evidence_text": None,
            "matched_strong_triggers": [],
            "matched_weak_triggers": [],
        }
        src_n = normalize_text(source_text)
        tgt_n = normalize_text(target_text)

        for unit in evidence_units:
            text = normalize_text(unit.get("content") or unit.get("text") or "")
            if not text:
                continue

            src_idx = find_first_occurrence(text, src_n)
            tgt_idx = find_first_occurrence(text, tgt_n)
            shared = src_idx >= 0 and tgt_idx >= 0

            strong_hits = self._contains_any(text, self.strong_triggers)
            weak_hits = self._contains_any(text, self.weak_triggers)
            neg_hits = self._contains_any(text, self.negation_cues)
            uncertain_hits = self._contains_any(text, self.uncertainty_cues)

            gap = None
            order = None
            if shared:
                if src_idx <= tgt_idx:
                    gap = max(0, tgt_idx - (src_idx + len(src_n)))
                    order = True
                else:
                    gap = max(0, src_idx - (tgt_idx + len(tgt_n)))
                    order = False

            score = 0.0
            if shared:
                score += 1.0
            if strong_hits:
                score += 1.2
            if weak_hits:
                score += 0.4
            if order is True:
                score += 0.2
            if gap is not None and gap <= self.max_entity_gap_chars:
                score += 0.3
            if neg_hits:
                score -= 0.6
            if uncertain_hits:
                score -= 0.25

            if score > best_score:
                best_score = score
                best = {
                    "shared_evidence": shared,
                    "strong_trigger_hit": bool(strong_hits),
                    "weak_trigger_hit": bool(weak_hits),
                    "negation_hit": bool(neg_hits),
                    "uncertainty_hit": bool(uncertain_hits),
                    "source_before_target": order,
                    "min_gap_chars": gap,
                    "matched_evidence_unit_id": unit.get("unit_id") or unit.get("evidence_unit_id"),
                    "matched_evidence_text": unit.get("content") or unit.get("text"),
                    "matched_strong_triggers": strong_hits,
                    "matched_weak_triggers": weak_hits,
                }

        if best:
            aggregate.update(best)
        return aggregate, best

    def decide(
        self,
        *,
        source_node_id: str,
        source_text: str,
        source_layer: Optional[str],
        target_node_id: str,
        target_text: str,
        target_layer: Optional[str],
        relation_type: Optional[str],
        rel_props: Optional[Dict[str, Any]] = None,
        evidence_units: Optional[Sequence[Dict[str, Any]]] = None,
        explicit_evidence_pointer: bool = False,
        chunk_distance: Optional[int] = None,
    ) -> PseudoLabelDecision:
        rel_props = rel_props or {}
        evidence_units = list(evidence_units or [])
        relation_type_u = str(relation_type or "").strip().upper() or None
        normalized_rel = relation_type_u[len("POTENTIAL_"):] if relation_type_u and relation_type_u.startswith("POTENTIAL_") else relation_type_u
        relation_whitelist_hit = bool(
            (relation_type_u and relation_type_u in self.relation_whitelist)
            or (normalized_rel and normalized_rel in self.relation_whitelist)
        )
        transition_allowed = self._transition_allowed(source_layer, target_layer)
        rel_prop_conf = relation_confidence_from_props(rel_props)
        overlap = lexical_overlap(source_text, target_text)
        evidence_feats, _ = self._evidence_features(source_text, target_text, evidence_units)

        features: Dict[str, Any] = {
            "source_node_id": source_node_id,
            "target_node_id": target_node_id,
            "relation_type": relation_type_u,
            "relation_whitelist_hit": relation_whitelist_hit,
            "transition_allowed": transition_allowed,
            "relation_prop_conf": rel_prop_conf,
            "node_text_overlap": overlap,
            "chunk_distance": chunk_distance,
            **evidence_feats,
        }

        positive_hits: List[str] = []
        negative_hits: List[str] = []

        positive_score = 0.0
        negative_score = 0.0

        if relation_whitelist_hit:
            positive_score += float(self.rule_weights.get("rel_whitelist", 0.24))
            positive_hits.append("REL_WHITELIST")
        else:
            negative_score += float(self.negative_rule_weights.get("non_causal_relation", 0.10))
            negative_hits.append("NON_CAUSAL_RELATION")

        if transition_allowed is True:
            positive_score += float(self.rule_weights.get("allowed_transition", 0.18))
            positive_hits.append("ALLOWED_TRANSITION")
        elif transition_allowed is False:
            negative_score += float(self.negative_rule_weights.get("forbidden_transition", 0.45))
            negative_hits.append("FORBIDDEN_TRANSITION")

        if features.get("strong_trigger_hit"):
            positive_score += float(self.rule_weights.get("strong_trigger", 0.24))
            positive_hits.append("STRONG_TRIGGER")
        elif features.get("weak_trigger_hit"):
            positive_hits.append("WEAK_TRIGGER_ONLY")

        if features.get("shared_evidence"):
            positive_score += float(self.rule_weights.get("shared_evidence", 0.18))
            positive_hits.append("SHARED_EVIDENCE")

        if explicit_evidence_pointer or features.get("matched_evidence_unit_id"):
            positive_score += float(self.rule_weights.get("explicit_evidence_pointer", 0.06))
            positive_hits.append("EXPLICIT_EVIDENCE")

        if rel_prop_conf >= self.min_relation_prop_confidence:
            positive_score += float(self.rule_weights.get("relation_prop_conf", 0.10))
            positive_hits.append("REL_PROP_CONF")

        if features.get("negation_hit"):
            negative_score += float(self.negative_rule_weights.get("negated_trigger", 0.30))
            negative_hits.append("NEGATED_TRIGGER")

        if features.get("uncertainty_hit") and not features.get("strong_trigger_hit"):
            negative_score += float(self.negative_rule_weights.get("uncertainty_only", 0.15))
            negative_hits.append("UNCERTAINTY_ONLY")

        if overlap >= 0.95:
            # 两端几乎同名，降低正例置信
            positive_score -= 0.12
            negative_hits.append("NEAR_DUPLICATE_NODE_TEXT")

        if features.get("min_gap_chars") is not None and features["min_gap_chars"] > self.max_entity_gap_chars:
            positive_score -= 0.10
            negative_hits.append("ENTITY_GAP_TOO_LARGE")

        if chunk_distance is not None and chunk_distance > 4:
            positive_score -= 0.05

        positive_score = max(0.0, min(1.0, positive_score))
        negative_score = max(0.0, min(1.0, negative_score))

        positive_decision = (
            relation_whitelist_hit
            and transition_allowed is not False
            and features.get("shared_evidence")
            and features.get("strong_trigger_hit")
            and not features.get("negation_hit")
            and positive_score >= self.positive_min_confidence
        )

        fallback_positive_decision = (
            relation_whitelist_hit
            and transition_allowed is not False
            and (explicit_evidence_pointer or rel_prop_conf >= self.min_relation_prop_confidence)
            and features.get("shared_evidence")
            and not features.get("negation_hit")
            and positive_score >= self.positive_min_confidence
        )

        negative_decision = (
            transition_allowed is False and negative_score >= self.negative_min_confidence
        ) or (
            features.get("negation_hit")
            and features.get("shared_evidence")
            and negative_score >= self.negative_min_confidence
        )

        if positive_decision:
            primary_rule = "HP_REL_TRIGGER_EVIDENCE"
            return PseudoLabelDecision(
                label=1,
                confidence=positive_score,
                relation_type=relation_type_u,
                primary_rule=primary_rule,
                rule_hits=[primary_rule] + positive_hits,
                features=features,
                evidence_unit_id=features.get("matched_evidence_unit_id"),
                evidence_text=features.get("matched_evidence_text"),
            )

        if fallback_positive_decision:
            primary_rule = "HP_REL_CONF_EVIDENCE"
            return PseudoLabelDecision(
                label=1,
                confidence=positive_score,
                relation_type=relation_type_u,
                primary_rule=primary_rule,
                rule_hits=[primary_rule] + positive_hits,
                features=features,
                evidence_unit_id=features.get("matched_evidence_unit_id"),
                evidence_text=features.get("matched_evidence_text"),
            )

        if negative_decision:
            primary_rule = "HP_FORBIDDEN_OR_NEGATED"
            return PseudoLabelDecision(
                label=0,
                confidence=negative_score,
                relation_type=relation_type_u,
                primary_rule=primary_rule,
                rule_hits=[primary_rule] + negative_hits,
                features=features,
                evidence_unit_id=features.get("matched_evidence_unit_id"),
                evidence_text=features.get("matched_evidence_text"),
            )

        return PseudoLabelDecision(
            label=None,
            confidence=max(positive_score, negative_score),
            relation_type=relation_type_u,
            primary_rule=None,
            rule_hits=positive_hits + negative_hits,
            features=features,
            evidence_unit_id=features.get("matched_evidence_unit_id"),
            evidence_text=features.get("matched_evidence_text"),
        )

    def decide_multitask(
            self,
            *,
            source_node_id: str,
            source_text: str,
            source_layer: Optional[str],
            target_node_id: str,
            target_text: str,
            target_layer: Optional[str],
            relation_type: Optional[str],
            rel_props: Optional[Dict[str, Any]] = None,
            evidence_units: Optional[Sequence[Dict[str, Any]]] = None,
            explicit_evidence_pointer: bool = False,
            chunk_distance: Optional[int] = None,
    ) -> MultiTaskPseudoLabelDecision:
        rel_props = rel_props or {}
        evidence_units = list(evidence_units or [])
        relation_type_u = str(relation_type or "").strip().upper()
        evidence_feats, _ = self._evidence_features(source_text, target_text, evidence_units)

        from .pseudo_tasks import PseudoTaskFactory

        task_factory = PseudoTaskFactory(self.config)
        shared_text = " ".join(
            [
                source_text or "",
                relation_type_u,
                target_text or "",
                str(evidence_feats.get("matched_evidence_text") or ""),
            ]
        )
        edge_causal = task_factory.label_edge_causal(shared_text, relation_type_u, evidence_feats)
        edge_enable = task_factory.label_edge_enable(shared_text, relation_type_u, evidence_feats)
        dir_decision = task_factory.label_edge_direction(shared_text, evidence_feats)
        temporal_decision = task_factory.label_edge_temporal(shared_text, evidence_feats)
        src_first, dst_first = task_factory.label_node_first(
            source_text=source_text,
            target_text=target_text,
            source_layer=source_layer,
            target_layer=target_layer,
        )

        active_confs = [
            c
            for l, c in [
                (edge_causal.label, edge_causal.confidence),
                (edge_enable.label, edge_enable.confidence),
                (dir_decision.label, dir_decision.confidence),
                (temporal_decision.label, temporal_decision.confidence),
                (src_first.label, src_first.confidence),
                (dst_first.label, dst_first.confidence),
            ]
            if l != -1
        ]
        sample_weight = float(sum(active_confs) / max(1, len(active_confs)))

        return MultiTaskPseudoLabelDecision(
            silver_edge_causal=edge_causal.label,
            causal_conf=edge_causal.confidence,
            silver_edge_enable=edge_enable.label,
            enable_conf=edge_enable.confidence,
            silver_causal_dir=dir_decision.label,
            dir_conf=dir_decision.confidence,
            silver_temporal_before=temporal_decision.label,
            temporal_conf=temporal_decision.confidence,
            silver_node_first_src=src_first.label,
            src_first_conf=src_first.confidence,
            silver_node_first_dst=dst_first.label,
            dst_first_conf=dst_first.confidence,
            rule_hits={
                "edge_causal": edge_causal.rule_hits,
                "edge_enable": edge_enable.rule_hits,
                "causal_dir": dir_decision.rule_hits,
                "temporal_before": temporal_decision.rule_hits,
                "node_first_src": src_first.rule_hits,
                "node_first_dst": dst_first.rule_hits,
            },
            sample_weight=sample_weight,
            evidence_unit_id=evidence_feats.get("matched_evidence_unit_id"),
            evidence_text=evidence_feats.get("matched_evidence_text"),
            features={
                "source_node_id": source_node_id,
                "target_node_id": target_node_id,
                "relation_type": relation_type_u,
                "chunk_distance": chunk_distance,
                "explicit_evidence_pointer": explicit_evidence_pointer,
                **evidence_feats,
            },
        )