from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml

from .gbt4754_lookup import GBT4754Lookup, GBT4754Entry
from .industry_rules import (
    collect_process_terms,
    collect_profile_spans,
    collect_title_spans,
    normalize_evidence_texts,
    rank_repeated_activity_terms,
)
from .schemas import CandidateBranch, IndustryCandidate, IndustryPrediction


class IndustryClassifier:
    def __init__(
        self,
        lookup: Optional[GBT4754Lookup] = None,
        priors_path: str = "configs/standards/gbt4754_priors.yaml",
    ) -> None:
        self.lookup = lookup or GBT4754Lookup()
        self.priors = yaml.safe_load(Path(priors_path).read_text(encoding="utf-8")) if Path(priors_path).exists() else {}

    def classify(
        self,
        doc_meta: Optional[Dict[str, Any]] = None,
        selected_branch: Optional[CandidateBranch] = None,
        all_branches: Optional[Sequence[CandidateBranch]] = None,
        evidence_units: Optional[Iterable[object]] = None,
        module_candidates: Optional[Sequence[str]] = None,
    ) -> IndustryPrediction:
        doc_meta = doc_meta or {}
        all_branches = list(all_branches or [])
        evidence_texts = normalize_evidence_texts(evidence_units or [])
        title = str(doc_meta.get("title") or doc_meta.get("doc_title") or doc_meta.get("doc_id") or "")
        unit_name = str(doc_meta.get("company_name") or doc_meta.get("org_name") or "")

        candidate_scores: Dict[str, float] = defaultdict(float)
        candidate_rules: Dict[str, List[str]] = defaultdict(list)
        candidate_evidence: Dict[str, List[str]] = defaultdict(list)

        # 1) 标题与单位概况权重高
        for text, weight, tag in collect_title_spans(title):
            self._accumulate_text(text, weight, tag, candidate_scores, candidate_rules, candidate_evidence)
        for text, weight, tag in collect_profile_spans([title, unit_name, *evidence_texts]):
            self._accumulate_text(text, weight, tag, candidate_scores, candidate_rules, candidate_evidence)

        # 2) 正文与证据命中
        for block in evidence_texts:
            self._accumulate_text(block, 1.0, "body_hit", candidate_scores, candidate_rules, candidate_evidence)

        # 3) branch 辅助线索
        if selected_branch:
            for cue in selected_branch.industry_cue_texts:
                self._accumulate_text(cue, 0.6, "branch_context_hit", candidate_scores, candidate_rules, candidate_evidence)
            for term in selected_branch.site_or_process_terms:
                self._accumulate_text(term, 0.5, "branch_term_hit", candidate_scores, candidate_rules, candidate_evidence)

        # 4) 重复活动词加分
        repeated = rank_repeated_activity_terms(evidence_texts + ([title] if title else []))
        for term, freq in repeated.items():
            if freq >= 2:
                candidates = self.lookup.lookup_candidates_from_text(term, top_k=3)
                for row in candidates:
                    entry: GBT4754Entry = row["entry"]
                    candidate_scores[entry.full_code] += 0.3 * (freq - 1)
                    candidate_rules[entry.full_code].append(f"repeated_term:{term}:{freq}")
                    candidate_evidence[entry.full_code].append(term)

        # 5) 模块软先验
        module_id = self._infer_module_id(module_candidates, selected_branch, all_branches)
        self._apply_priors(module_id, selected_branch, candidate_scores, candidate_rules)

        ranked = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        if not ranked:
            return IndustryPrediction(
                gbt4754_full_code=None,
                gbt4754_type_code=None,
                gbt4754_name=None,
                section_code=None,
                section_name=None,
                confidence=0.0,
                rule_hits=["industry_empty"],
                evidence_ids=list(dict.fromkeys((selected_branch.industry_evidence_ids if selected_branch else []))),
                evidence_texts=evidence_texts[:5],
                candidates=[],
            )

        candidates: List[IndustryCandidate] = []
        top_score = ranked[0][1]
        for code, score in ranked[:5]:
            entry = self.lookup.lookup_by_code(code)
            if not entry:
                continue
            candidates.append(
                IndustryCandidate(
                    gbt4754_full_code=entry.full_code,
                    gbt4754_type_code=entry.type_code,
                    gbt4754_name=entry.full_name,
                    section_code=entry.section_code,
                    section_name=entry.section_name,
                    score=round(score, 4),
                    evidence_ids=list(dict.fromkeys((selected_branch.industry_evidence_ids if selected_branch else []))),
                    evidence_texts=list(dict.fromkeys(candidate_evidence.get(code, [])))[:5],
                    rule_hits=list(dict.fromkeys(candidate_rules.get(code, []))),
                )
            )

        winner = candidates[0]
        confidence = 0.0
        if top_score > 0:
            second = candidates[1].score if len(candidates) > 1 else 0.0
            confidence = max(0.0, min(1.0, 0.5 + (winner.score - second) / (2 * max(top_score, 1e-6))))

        return IndustryPrediction(
            gbt4754_full_code=winner.gbt4754_full_code,
            gbt4754_type_code=winner.gbt4754_type_code,
            gbt4754_name=winner.gbt4754_name,
            section_code=winner.section_code,
            section_name=winner.section_name,
            confidence=round(confidence, 4),
            rule_hits=winner.rule_hits,
            evidence_ids=winner.evidence_ids,
            evidence_texts=winner.evidence_texts,
            candidates=candidates,
            meta={"module_id": module_id, "process_terms": collect_process_terms(evidence_texts)},
        )

    def _accumulate_text(
        self,
        text: str,
        weight: float,
        tag: str,
        candidate_scores: Dict[str, float],
        candidate_rules: Dict[str, List[str]],
        candidate_evidence: Dict[str, List[str]],
    ) -> None:
        for row in self.lookup.lookup_candidates_from_text(text, top_k=8):
            entry: GBT4754Entry = row["entry"]
            score = float(row["score"]) * weight
            candidate_scores[entry.full_code] += score
            candidate_rules[entry.full_code].append(tag)
            candidate_rules[entry.full_code].extend(row["rule_hits"])
            candidate_evidence[entry.full_code].append(text)

    def _infer_module_id(
        self,
        module_candidates: Optional[Sequence[str]],
        selected_branch: Optional[CandidateBranch],
        all_branches: Sequence[CandidateBranch],
    ) -> str:
        if selected_branch and selected_branch.meta.get("module_id"):
            return str(selected_branch.meta["module_id"])
        if module_candidates:
            for item in module_candidates:
                if "construction" in item or "施工" in item:
                    return "construction_safety"
        for branch in all_branches:
            if branch.meta.get("module_id"):
                return str(branch.meta["module_id"])
        return "production_safety"

    def _apply_priors(
        self,
        module_id: str,
        selected_branch: Optional[CandidateBranch],
        candidate_scores: Dict[str, float],
        candidate_rules: Dict[str, List[str]],
    ) -> None:
        module_priors = ((self.priors or {}).get("module_priors") or {}).get(module_id) or {}
        for section_code, bonus in (module_priors.get("default_section_bonus") or {}).items():
            for code in list(candidate_scores.keys()):
                entry = self.lookup.lookup_by_code(code)
                if entry and entry.section_code == section_code:
                    candidate_scores[code] += float(bonus)
                    candidate_rules[code].append(f"module_section_prior:{section_code}")
        scenario_bonus = module_priors.get("scenario_bonus") or {}
        for scenario in (selected_branch.meta.get("scenario_tags", []) if selected_branch else []):
            for code, bonus in (scenario_bonus.get(scenario) or {}).items():
                entry = self.lookup.lookup_by_code(code)
                if entry:
                    candidate_scores[entry.full_code] += float(bonus)
                    candidate_rules[entry.full_code].append(f"scenario_prior:{scenario}:{code}")
