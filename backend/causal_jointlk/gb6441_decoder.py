from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .schemas import DecodedAccidentResult, IndustryPrediction


class GB6441Decoder:
    def __init__(self, base_dir: str = "configs/standards") -> None:
        base = Path(base_dir)
        self.basic_types = json.loads((base / "gb6441_basic_types.json").read_text(encoding="utf-8")) if (base / "gb6441_basic_types.json").exists() else []
        self.injury = json.loads((base / "gb6441_injury_severity.json").read_text(encoding="utf-8")) if (base / "gb6441_injury_severity.json").exists() else []
        self.industry_map = json.loads((base / "gb6441_industry_map.json").read_text(encoding="utf-8"))
        self.codegen = json.loads((base / "gb6441_codegen_rules.json").read_text(encoding="utf-8")) if (base / "gb6441_codegen_rules.json").exists() else {}

    def decode(self, payload: Dict[str, Any]) -> DecodedAccidentResult:
        selected_branch = payload.get("selected_branch")
        branch_candidates = payload.get("module_candidates") or []
        if selected_branch is not None:
            branch_meta = getattr(selected_branch, "meta", {}) or {}
            scored = branch_meta.get("basic_type_scores") or {}
            if scored:
                branch_candidates = [k for k, _ in sorted(scored.items(), key=lambda x: x[1], reverse=True)]
        candidates = branch_candidates
        basic = self._decode_basic(candidates)
        sev = self._decode_severity(payload.get("severity_signals") or {})
        industry_pred: Optional[IndustryPrediction] = payload.get("industry_prediction")

        industry_type = None
        industry_code = None
        gbt_full_code = None
        gbt_type_code = None
        section_code = None
        section_name = None
        industry_rule_hits = []
        industry_evidence_ids = []
        industry_candidates = []
        confidence = 0.0

        if industry_pred and industry_pred.gbt4754_type_code:
            gbt_full_code = industry_pred.gbt4754_full_code
            gbt_type_code = industry_pred.gbt4754_type_code
            industry_type = industry_pred.gbt4754_name
            section_code = industry_pred.section_code
            section_name = industry_pred.section_name
            confidence = industry_pred.confidence
            industry_rule_hits = list(industry_pred.rule_hits)
            industry_evidence_ids = list(industry_pred.evidence_ids)
            industry_candidates = list(industry_pred.candidates)
            industry_code = self._gen_industry_code(gbt_type_code)

        branch_first = float(getattr(selected_branch, "p_branch_first", 0.0) or 0.0) if selected_branch else 0.0
        severity_score = float(getattr(selected_branch, "severity_score", 0.0) or 0.0) if selected_branch else 0.0
        severity_conf = min(1.0, max(0.0, severity_score / 10.0))
        conf_sources = {
            "branch_first_conf": round(branch_first, 4),
            "severity_conf": round(severity_conf, 4),
            "industry_conf": round(float(confidence or 0.0), 4),
        }
        decode_conf = round(
            0.5 * conf_sources["branch_first_conf"]
            + 0.2 * conf_sources["severity_conf"]
            + 0.3 * conf_sources["industry_conf"],
            4,
        )
        full_code = self._compose_full_code(
            basic.get("code"),
            sev.get("code"),
            industry_code,
        )

        decode_hits = ["basic_lookup", "severity_lookup"]
        if not candidates:
            decode_hits.append("basic_fallback_from_standard")
        if industry_code:
            decode_hits.append("industry_from_gbt4754")
        else:
            decode_hits.append("industry_missing_fallback")

        decode_hits.append(f"confidence_sources:{json.dumps(conf_sources, ensure_ascii=False)}")

        return DecodedAccidentResult(
            basic_type=basic.get("name"),
            injury_severity=sev.get("name"),
            industry_type=industry_type,
            basic_code=full_code.get("basic_code"),
            injury_code=sev.get("code"),
            industry_code=industry_code,
            gbt4754_full_code=gbt_full_code,
            gbt4754_type_code=gbt_type_code,
            industry_section_code=section_code,
            industry_section_name=section_name,
            industry_confidence=confidence,
            industry_rule_hits=industry_rule_hits,
            industry_evidence_ids=industry_evidence_ids,
            industry_candidates=industry_candidates,
            decode_rule_hits=decode_hits + [f"full_code:{full_code.get('full_code') or ''}"],
            decode_confidence=decode_conf
        )

    def _decode_basic(self, candidates: Any) -> Dict[str, Any]:
        if not self.basic_types:
            return {"name": None, "code": None, "confidence": 0.0}
        for item in self.basic_types:
            if item.get("name") in candidates or item.get("id") in candidates:
                item = dict(item)
                item["confidence"] = 0.85
                return item
        item = dict(self.basic_types[0])
        item["confidence"] = 0.65
        return item

    @staticmethod
    def _compose_full_code(
            basic_code: Optional[str],
            injury_code: Optional[str],
            industry_code: Optional[str],
    ) -> Dict[str, Optional[str]]:
        basic_code = str(basic_code or "").strip() or None
        injury_code = str(injury_code or "").strip() or None
        industry_code = str(industry_code or "").strip() or None
        if basic_code and injury_code and industry_code:
            full = f"{basic_code}-{injury_code}-{industry_code}"
        else:
            full = None
        return {
            "basic_code": basic_code,
            "injury_code": injury_code,
            "industry_code": industry_code,
            "full_code": full,
        }

    def _decode_severity(self, severity_signals: Dict[str, Any]) -> Dict[str, Any]:
        if not self.injury:
            return {"name": None, "code": None, "confidence": 0.0}
        death = float(severity_signals.get("death_count", 0) or 0)
        serious = float(severity_signals.get("serious_injury_count", 0) or 0)
        light = float(severity_signals.get("light_injury_count", 0) or 0)

        if death > 0:
            target = "死亡"
        elif serious > 0:
            target = "重伤"
        elif light > 0:
            target = "轻伤"
        else:
            return {**self.injury[0], "confidence": 0.55}

        for item in self.injury:
            if target in item.get("name", ""):
                return {**item, "confidence": 0.85}
        return {**self.injury[0], "confidence": 0.55}

    def _gen_industry_code(self, type_code: str) -> str:
        category_way_code = self.industry_map["category_way_code"]
        digits = "".join(ch for ch in str(type_code) if ch.isdigit())[-4:]
        digits = digits.rjust(4, "0")
        pattern = self.industry_map["industry_code_rule"]["output_pattern"]
        return pattern.format(category_way_code=category_way_code, type_code=digits)
