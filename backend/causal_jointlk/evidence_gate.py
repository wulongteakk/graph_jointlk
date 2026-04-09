
import math
import re
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .prior import CausalPrior, normalize_name


_WORD_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9_\-]+")


def _tokens(text: Optional[str]) -> List[str]:
    return _WORD_RE.findall((text or "").lower())


def lexical_overlap(a: Optional[str], b: Optional[str]) -> float:
    ta = set(_tokens(a))
    tb = set(_tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def find_best_span(text: Optional[str], phrase: Optional[str]) -> Optional[Tuple[int, int]]:
    text = text or ""
    phrase = phrase or ""
    if not text or not phrase:
        return None

    idx = text.lower().find(phrase.lower())
    if idx >= 0:
        return (idx, idx + len(phrase))

    # fallback: rough fuzzy span
    best_score = 0.0
    best_span = None
    window = max(len(phrase), 4)
    for start in range(0, max(len(text) - window + 1, 1)):
        segment = text[start : start + window]
        score = SequenceMatcher(None, segment.lower(), phrase.lower()).ratio()
        if score > best_score:
            best_score = score
            best_span = (start, start + window)
    return best_span if best_score >= 0.70 else None


def score_edge_support(
    prior: CausalPrior,
    source_text: str,
    target_text: str,
    relation: str,
    evidence_text: Optional[str],
    relation_text: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = prior.as_evidence_params()
    evidence_text = evidence_text or ""
    source_text = source_text or ""
    target_text = target_text or ""
    relation = relation or "UNK"

    source_span = find_best_span(evidence_text, source_text)
    target_span = find_best_span(evidence_text, target_text)
    source_hit = source_span is not None
    target_hit = target_span is not None

    trigger_hits = prior.find_trigger_words(evidence_text)
    rel_match = relation.upper() in (relation_text or "").upper() or relation.upper() in evidence_text.upper()
    overlap = max(
        lexical_overlap(f"{source_text} {target_text}", evidence_text),
        lexical_overlap(f"{source_text} {relation} {target_text}", evidence_text),
    )

    score = 0.0
    score += min(overlap, 0.35)
    if source_hit:
        score += 0.20
    if target_hit:
        score += 0.20
    if source_hit and target_hit:
        score += float(cfg.get("two_sided_entity_bonus", 0.20))
    if trigger_hits:
        score += float(cfg.get("trigger_bonus", 0.25))
    if rel_match:
        score += 0.10

    ordered_bonus = float(cfg.get("ordered_bonus", 0.10))
    if source_span and target_span and source_span[0] <= target_span[0]:
        score += ordered_bonus

    negative_cues = []
    for xs in (cfg.get("negative_cues") or {}).values():
        negative_cues.extend(xs)
    if any(cue.lower() in evidence_text.lower() for cue in negative_cues):
        score -= 0.25

    require_trigger_or_reltype = bool(cfg.get("require_trigger_or_reltype", True))
    min_overlap = float(cfg.get("min_lexical_overlap", 0.18))
    min_support = float(cfg.get("min_support_score", 0.55))

    supported = (
        overlap >= min_overlap
        and source_hit
        and target_hit
        and (not require_trigger_or_reltype or bool(trigger_hits or rel_match))
        and score >= min_support
    )

    temporal_score = score_temporal_support(prior, evidence_text, source_text, target_text)
    direction_score = score_direction_support(prior, evidence_text, source_text, target_text)
    first_induced_score = score_first_induced_support(prior, evidence_text, source_text)
    severity_signals = extract_severity_signals(prior, evidence_text)

    return {
        "supported": supported,
        "support_score": max(0.0, min(score, 1.0)),
        "temporal_score": temporal_score,
        "direction_score": direction_score,
        "first_induced_score": first_induced_score,
        "severity_signals": severity_signals,
        "trigger_hits": trigger_hits,
        "source_span": source_span,
        "target_span": target_span,
    }

def score_temporal_support(prior: CausalPrior, evidence_text: str, source_text: str, target_text: str) -> float:
    _ = prior
    text = (evidence_text or "").lower()
    temporal_words = ["先", "随后", "然后", "之后", "before", "after", "then", "followed"]
    source_hit = lexical_overlap(source_text, evidence_text) > 0.0
    target_hit = lexical_overlap(target_text, evidence_text) > 0.0
    cue = 1.0 if any(w in text for w in temporal_words) else 0.0
    return max(0.0, min(1.0, 0.4 * cue + 0.3 * float(source_hit) + 0.3 * float(target_hit)))


def score_direction_support(prior: CausalPrior, evidence_text: str, source_text: str, target_text: str) -> float:
    cfg = prior.config.get("direction_rules") or {}
    forward = [str(x).lower() for x in (cfg.get("forward_cues") or [])]
    reverse = [str(x).lower() for x in (cfg.get("reverse_cues") or [])]
    text = (evidence_text or "").lower()
    source_first = find_best_span(evidence_text, source_text)
    target_first = find_best_span(evidence_text, target_text)
    ordered = 1.0 if source_first and target_first and source_first[0] <= target_first[0] else 0.0
    forward_hit = 1.0 if any(c in text for c in forward) else 0.0
    reverse_hit = 1.0 if any(c in text for c in reverse) else 0.0
    value = 0.55 * ordered + 0.35 * forward_hit - 0.25 * reverse_hit
    return max(0.0, min(1.0, value))


def score_first_induced_support(prior: CausalPrior, evidence_text: str, source_text: str) -> float:
    cfg = prior.config.get("first_induced") or {}
    cues = [str(x).lower() for x in (cfg.get("cue_words_zh") or [])]
    rules_cues = [str(x).lower() for x in ((prior.config.get("first_induced_rules") or {}).get("cues") or [])]
    text = (evidence_text or "").lower()
    cue_hit = 1.0 if any(c in text for c in (cues + rules_cues)) else 0.0
    source_hit = 1.0 if lexical_overlap(source_text, evidence_text) > 0.0 else 0.0
    return max(0.0, min(1.0, 0.6 * cue_hit + 0.4 * source_hit))


def extract_severity_signals(prior: CausalPrior, evidence_text: str) -> Dict[str, float]:
    _ = prior
    text = evidence_text or ""
    numbers = re.findall(r"(死亡|重伤|轻伤)\s*(\d+)\s*人|(\d+)\s*人(死亡|重伤|轻伤)", text)
    death = serious = light = 0.0
    for g1, g2, g3, g4 in numbers:
        label = g1 or g4
        value = float(g2 or g3 or 0.0)
        if label == "死亡":
            death += value
        elif label == "重伤":
            serious += value
        elif label == "轻伤":
            light += value
    energy = 1.0 if any(k in text for k in ["爆炸", "爆燃", "火灾", "坍塌"]) else 0.0
    toxicity = 1.0 if any(k in text for k in ["中毒", "窒息", "有毒", "缺氧"]) else 0.0
    return {
        "death_count": death,
        "serious_injury_count": serious,
        "light_injury_count": light,
        "energy_level": energy,
        "toxicity_level": toxicity,
    }
def choose_best_evidence(
    prior: CausalPrior,
    source_text: str,
    target_text: str,
    relation: str,
    evidence_units: Sequence[Dict[str, Any]],
    relation_text: Optional[str] = None,
) -> Dict[str, Any]:
    best = {
        "supported": False,
        "support_score": 0.0,
        "temporal_score": 0.0,
        "direction_score": 0.0,
        "first_induced_score": 0.0,
        "severity_signals": {},
        "unit_id": None,
        "content": None,
        "source_span": None,
        "target_span": None,
        "trigger_hits": [],
    }
    for unit in evidence_units:
        current = score_edge_support(
            prior=prior,
            source_text=source_text,
            target_text=target_text,
            relation=relation,
            evidence_text=unit.get("content"),
            relation_text=relation_text,
        )
        if current["support_score"] > best["support_score"]:
            best = {
                **current,
                "unit_id": unit.get("unit_id"),
                "content": unit.get("content"),
            }
    return best
