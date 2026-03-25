from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .pseudo_labeler import normalize_text

ABSTAIN = -1


def _contains_any(text: str, phrases: Sequence[str]) -> List[str]:
    hits: List[str] = []
    for phrase in phrases:
        p = normalize_text(str(phrase))
        if p and p in text:
            hits.append(p)
    return hits


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


@dataclass
class TaskDecision:
    label: int = ABSTAIN
    confidence: float = 0.0
    rule_hits: List[str] = field(default_factory=list)


@dataclass
class MultiTaskDecision:
    edge_causal: TaskDecision
    edge_enable: TaskDecision
    causal_dir: TaskDecision
    temporal_before: TaskDecision
    node_first_src: TaskDecision
    node_first_dst: TaskDecision
    evidence_unit_id: Optional[str] = None
    evidence_text: Optional[str] = None
    sample_weight: float = 1.0
    twin_group_id: Optional[str] = None
    review_status: str = "pending"


class PseudoTaskFactory:
    """Rule factory for five weak-supervision tasks.

    每个任务都允许 abstain（-1），并返回任务独立的置信度与规则命中。
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.thresholds = self.config.get("task_thresholds") or {}

        self.causal_kw = [normalize_text(x) for x in (self.config.get("strong_triggers") or []) if normalize_text(x)]
        self.enable_kw = [normalize_text(x) for x in (self.config.get("enable_triggers") or ["使得", "促成", "enable", "enables", "allow", "allows"]) if normalize_text(x)]
        self.dir_forward_kw = [normalize_text(x) for x in (self.config.get("direction_forward_cues") or ["导致", "引发", "causes", "leads to", "result in"]) if normalize_text(x)]
        self.dir_reverse_kw = [normalize_text(x) for x in (self.config.get("direction_reverse_cues") or ["由", "源于", "caused by", "resulted from", "due to"]) if normalize_text(x)]
        self.temp_before_kw = [normalize_text(x) for x in (self.config.get("temporal_before_cues") or ["先", "随后", "然后", "after", "then", "followed by"]) if normalize_text(x)]
        self.temp_reverse_kw = [normalize_text(x) for x in (self.config.get("temporal_reverse_cues") or ["此前", "之前", "before", "prior to"]) if normalize_text(x)]
        self.first_kw = [normalize_text(x) for x in (self.config.get("first_induced_cues") or ["诱因", "隐患", "根本原因", "先是", "起因", "root cause"]) if normalize_text(x)]

    def _task_threshold(self, key: str, default: Tuple[float, float]) -> Tuple[float, float]:
        cfg = self.thresholds.get(key) or {}
        pos = float(cfg.get("positive", default[0]))
        neg = float(cfg.get("negative", default[1]))
        return pos, neg

    def _discretize(self, score: float, key: str, default: Tuple[float, float]) -> int:
        pos_t, neg_t = self._task_threshold(key, default)
        if score >= pos_t:
            return 1
        if score <= neg_t:
            return 0
        return ABSTAIN

    def label_edge_causal(self, text: str, relation_type: str, evidence: Dict[str, Any]) -> TaskDecision:
        text_n = normalize_text(text)
        hits = _contains_any(text_n, self.causal_kw)
        score = 0.15 + 0.25 * bool(hits) + 0.20 * bool(evidence.get("shared_evidence"))
        if str(relation_type).upper() in {"CAUSES", "LEADS_TO", "RESULTS_IN", "TRIGGERS", "INDUCES"}:
            score += 0.25
            hits.append("rel=causal")
        if evidence.get("negation_hit"):
            score -= 0.30
            hits.append("negation")
        if evidence.get("uncertainty_hit"):
            score -= 0.20
            hits.append("uncertain")
        score = _clamp01(score)
        return TaskDecision(label=self._discretize(score, "edge_causal", (0.80, 0.20)), confidence=score, rule_hits=hits)

    def label_edge_enable(self, text: str, relation_type: str, evidence: Dict[str, Any]) -> TaskDecision:
        text_n = normalize_text(text)
        hits = _contains_any(text_n, self.enable_kw)
        score = 0.10 + 0.35 * bool(hits)
        if str(relation_type).upper() == "ENABLES":
            score += 0.35
            hits.append("rel=enables")
        if evidence.get("negation_hit"):
            score -= 0.25
        score = _clamp01(score)
        return TaskDecision(label=self._discretize(score, "edge_enable", (0.78, 0.25)), confidence=score, rule_hits=hits)

    def label_edge_direction(self, text: str, evidence: Dict[str, Any]) -> TaskDecision:
        text_n = normalize_text(text)
        forward_hits = _contains_any(text_n, self.dir_forward_kw)
        reverse_hits = _contains_any(text_n, self.dir_reverse_kw)
        score = 0.50
        if forward_hits:
            score += 0.30
        if reverse_hits:
            score -= 0.30
        if evidence.get("source_before_target") is True:
            score += 0.10
        elif evidence.get("source_before_target") is False:
            score -= 0.10
        score = _clamp01(score)
        hits = [f"fwd:{h}" for h in forward_hits] + [f"rev:{h}" for h in reverse_hits]
        return TaskDecision(label=self._discretize(score, "causal_dir", (0.72, 0.28)), confidence=score, rule_hits=hits)

    def label_edge_temporal(self, text: str, evidence: Dict[str, Any]) -> TaskDecision:
        text_n = normalize_text(text)
        before_hits = _contains_any(text_n, self.temp_before_kw)
        reverse_hits = _contains_any(text_n, self.temp_reverse_kw)
        score = 0.50
        score += 0.20 * bool(before_hits)
        score -= 0.20 * bool(reverse_hits)
        if evidence.get("source_before_target") is True:
            score += 0.15
        elif evidence.get("source_before_target") is False:
            score -= 0.15
        score = _clamp01(score)
        hits = [f"before:{h}" for h in before_hits] + [f"reverse:{h}" for h in reverse_hits]
        return TaskDecision(label=self._discretize(score, "temporal_before", (0.70, 0.30)), confidence=score, rule_hits=hits)

    def label_node_first(
        self,
        source_text: str,
        target_text: str,
        source_layer: Optional[str],
        target_layer: Optional[str],
    ) -> Tuple[TaskDecision, TaskDecision]:
        s = normalize_text(source_text)
        t = normalize_text(target_text)
        src_hits = _contains_any(s, self.first_kw)
        dst_hits = _contains_any(t, self.first_kw)
        src_score = 0.50 + 0.25 * bool(src_hits)
        dst_score = 0.50 + 0.25 * bool(dst_hits)

        if source_layer and target_layer:
            src = str(source_layer).upper()
            tgt = str(target_layer).upper()
            if src in {"ROOT", "CAUSE", "FACTOR", "SOURCESTATE", "SOURCEEVENT"}:
                src_score += 0.10
            if tgt in {"OUTCOME", "CONSEQUENCE", "HARMEVENT"}:
                dst_score -= 0.10

        src_score = _clamp01(src_score)
        dst_score = _clamp01(dst_score)
        src_d = TaskDecision(
            label=self._discretize(src_score, "node_first", (0.70, 0.30)),
            confidence=src_score,
            rule_hits=[f"src:{h}" for h in src_hits],
        )
        dst_d = TaskDecision(
            label=self._discretize(dst_score, "node_first", (0.70, 0.30)),
            confidence=dst_score,
            rule_hits=[f"dst:{h}" for h in dst_hits],
        )
        return src_d, dst_d