from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class DomainPack:
    pack_id: str
    manifest: Dict[str, Any]
    type_lexicon: Dict[str, Any]
    relation_aliases: Dict[str, Any]
    module_library: Dict[str, Any]
    exclusion_rules: Dict[str, Any]
    hfsca_priors: Dict[str, Any]

    def allowed_core_node_types(self) -> List[str]:
        return self.manifest.get("allowed_core_node_types", [])

    def allowed_rel_types(self) -> List[str]:
        return self.manifest.get("allowed_rel_types", [])

    def map_to_hfsca(self, core_type: str, text: str, props: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def match_module(self, text: str) -> Optional[str]:
        raise NotImplementedError