from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.domain_packs.base import DomainPack
from src.ontology.core_ontology import DEFAULT_LAYER_BY_CORE_TYPE


class ConstructionPack(DomainPack):
    def __init__(self):
        base_dir = Path(__file__).resolve().parent
        super().__init__(
            pack_id="construction",
            manifest=self._load_yaml(base_dir / "manifest.yaml"),
            type_lexicon=self._load_yaml(base_dir / "type_lexicon.yaml"),
            relation_aliases=self._load_yaml(base_dir / "relation_aliases.yaml"),
            module_library=self._load_yaml(base_dir / "module_library.yaml"),
            exclusion_rules=self._load_yaml(base_dir / "exclusion_rules.yaml"),
            hfsca_priors=self._load_yaml(base_dir / "hfsca_priors.yaml"),
        )

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def map_to_hfsca(self, core_type: str, text: str, props: Dict[str, Any]) -> Dict[str, Any]:
        priors = self.hfsca_priors or {}
        core_defaults = priors.get("core_type_defaults") or {}
        result = dict(core_defaults.get(core_type) or {})

        lower_text = (text or "").lower()
        for rule in priors.get("keyword_rules") or []:
            keywords = [str(x).lower() for x in (rule.get("keywords") or [])]
            if keywords and any(keyword in lower_text for keyword in keywords):
                result.update({k: v for k, v in rule.items() if k != "keywords"})
                break

        result.setdefault("hfsca_layer", DEFAULT_LAYER_BY_CORE_TYPE.get(core_type))
        result.setdefault("hfsca_category", None)
        result.setdefault("confidence", 0.7)
        result.setdefault("reason", f"construction_pack:{core_type}")
        return result

    def match_module(self, text: str) -> Optional[str]:
        lower_text = (text or "").lower()
        for item in self.module_library.get("modules") or []:
            keywords = [str(x).lower() for x in (item.get("keywords") or [])]
            if keywords and any(keyword in lower_text for keyword in keywords):
                return item.get("module_id")
        return None