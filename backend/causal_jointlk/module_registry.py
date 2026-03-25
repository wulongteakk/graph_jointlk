from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


class ModuleRegistry:
    def __init__(self, root: str = "configs/domain_modules/production_safety"):
        self.root = Path(root)
        self.registry = self._load_yaml(self.root / "module_registry.yaml")

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def assign_module(self, text: str) -> Dict[str, Any]:
        text = text or ""
        modules = self.registry.get("modules") or []
        for module in modules:
            for scenario in module.get("scenarios", []):
                if any(keyword in text for keyword in scenario.get("keywords", [])):
                    return {
                        "domain_id": self.registry.get("domain_id", "production_safety"),
                        "module_id": module.get("module_id"),
                        "scenario_tags": [scenario.get("scenario_id")],
                    }
        default_module = modules[0].get("module_id") if modules else "construction_safety"
        return {
            "domain_id": self.registry.get("domain_id", "production_safety"),
            "module_id": default_module,
            "scenario_tags": [],
        }