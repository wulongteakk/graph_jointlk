from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .schemas import DecodedAccidentResult


class GB6441Decoder:
    def __init__(self, base_dir: str = "configs/standards"):
        base = Path(base_dir)
        self.basic_types = json.loads((base / "gb6441_basic_types.json").read_text(encoding="utf-8"))
        self.injury = json.loads((base / "gb6441_injury_severity.json").read_text(encoding="utf-8"))
        self.industry = json.loads((base / "gb6441_industry_map.json").read_text(encoding="utf-8"))
        self.codegen = json.loads((base / "gb6441_codegen_rules.json").read_text(encoding="utf-8"))

    def decode(self, payload: Dict[str, Any]) -> DecodedAccidentResult:
        candidates = payload.get("module_candidates") or ["高处坠落"]
        basic = self.basic_types[0]
        for item in self.basic_types:
            if item["name"] in candidates or item["id"] in candidates:
                basic = item
                break
        sev = self.injury[0]
        ind = self.industry[0]
        return DecodedAccidentResult(
            basic_type=basic["name"],
            injury_severity=sev["name"],
            industry_type=ind["name"],
            basic_code=basic["code"],
            injury_code=sev["code"],
            industry_code=ind["code"],
            decode_rule_hits=["basic_lookup", "severity_default", "industry_default"],
            decode_confidence=0.75,
        )