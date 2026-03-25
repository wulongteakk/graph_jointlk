from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence

from .schemas import CandidateBranch, CandidateChain, CausalEdge


class BranchBuilder:
    HARM_LAYERS = {"EVENT", "OUTCOME", "CONSEQUENCE"}
    BASIC_TYPE_KEYWORDS = {
        "高处坠落": ["坠落", "高处", "脚手架", "平台", "坠下"],
        "坍塌": ["坍塌", "塌落", "垮塌", "倒塌"],
        "起重伤害": ["起重", "吊装", "塔吊", "吊物", "吊钩"],
        "中毒和窒息": ["中毒", "窒息", "有毒", "缺氧"],
        "触电": ["触电", "带电", "漏电"],
        "爆炸": ["爆炸", "爆燃"],
    }

    def build(self, chains: Sequence[CandidateChain]) -> List[CandidateBranch]:
        grouped: Dict[str, CandidateBranch] = {}
        for idx, chain in enumerate(chains):
            path_nodes = list(chain.nodes)
            path_edges = list(chain.edges)
            harm_node = self._infer_harm_node(path_edges, path_nodes)
            consequence_nodes = self._infer_consequence_nodes(path_edges, path_nodes)
            root_nodes = self._infer_root_nodes(path_nodes, path_edges)
            evidence_ids = list(dict.fromkeys([e.evidence_unit_id for e in path_edges if e.evidence_unit_id]))
            industry_cues, process_terms = self._infer_industry_cues(chain)

            cluster_key = f"{harm_node}|{','.join(sorted(consequence_nodes))}|{','.join(sorted(root_nodes))}"
            if cluster_key not in grouped:
                grouped[cluster_key] = CandidateBranch(
                    branch_id=f"branch::{len(grouped)}",
                    chain_ids=[chain.chain_id],
                    first_node_id=path_nodes[0] if path_nodes else None,
                    score=chain.score,
                    root_nodes=root_nodes,
                    path_nodes=path_nodes,
                    path_edges=path_edges,
                    harm_node=harm_node,
                    consequence_nodes=consequence_nodes,
                    evidence_unit_ids=evidence_ids,
                    basic_type_candidates=self._infer_basic_types(path_edges),
                    rule_hits=["chain_score", "branch_cluster"],
                    industry_cue_texts=industry_cues,
                    industry_evidence_ids=evidence_ids,
                    site_or_process_terms=process_terms,
                    meta={"module_id": self._infer_module_id(path_edges), "scenario_tags": self._infer_scenario_tags(path_edges)},
                )
            else:
                b = grouped[cluster_key]
                b.chain_ids.append(chain.chain_id)
                b.score = max(b.score, chain.score)
                b.evidence_unit_ids = list(dict.fromkeys(b.evidence_unit_ids + evidence_ids))
                b.basic_type_candidates = list(dict.fromkeys(b.basic_type_candidates + self._infer_basic_types(path_edges)))
                b.industry_cue_texts = list(dict.fromkeys(b.industry_cue_texts + industry_cues))
                b.site_or_process_terms = list(dict.fromkeys(b.site_or_process_terms + process_terms))

        branches = sorted(grouped.values(), key=lambda x: x.score, reverse=True)
        return branches

    def _infer_harm_node(self, edges: Sequence[CausalEdge], path_nodes: Sequence[str]) -> str | None:
        for edge in reversed(edges):
            if (edge.target_layer or "").upper() in self.HARM_LAYERS:
                return edge.target_id
        return path_nodes[-1] if path_nodes else None

    def _infer_consequence_nodes(self, edges: Sequence[CausalEdge], path_nodes: Sequence[str]) -> List[str]:
        rows = []
        for edge in edges:
            if (edge.target_layer or "").upper() in {"OUTCOME", "CONSEQUENCE"}:
                rows.append(edge.target_id)
        if not rows and path_nodes:
            rows.append(path_nodes[-1])
        return list(dict.fromkeys(rows))

    def _infer_root_nodes(self, path_nodes: Sequence[str], edges: Sequence[CausalEdge]) -> List[str]:
        incoming = defaultdict(int)
        for edge in edges:
            incoming[edge.target_id] += 1
        roots = [nid for nid in path_nodes if incoming[nid] == 0]
        return roots or ([path_nodes[0]] if path_nodes else [])

    def _infer_basic_types(self, edges: Sequence[CausalEdge]) -> List[str]:
        text = " ".join(filter(None, [e.source_text for e in edges] + [e.target_text for e in edges] + [e.evidence_text or "" for e in edges]))
        hits = []
        for basic_type, kws in self.BASIC_TYPE_KEYWORDS.items():
            if any(kw in text for kw in kws):
                hits.append(basic_type)
        return hits or ["高处坠落"]

    def _infer_industry_cues(self, chain: CandidateChain) -> tuple[list[str], list[str]]:
        cue_texts: List[str] = []
        process_terms: List[str] = []
        for edge in chain.edges:
            for text in [edge.source_text, edge.target_text, edge.evidence_text or ""]:
                if not text:
                    continue
                cue_texts.append(text)
                for term in ["施工", "吊装", "运输", "检修", "安装", "生产", "掘进"]:
                    if term in text:
                        process_terms.append(term)
        return list(dict.fromkeys(cue_texts))[:10], list(dict.fromkeys(process_terms))

    def _infer_module_id(self, edges: Sequence[CausalEdge]) -> str:
        text = " ".join(filter(None, [e.source_text for e in edges] + [e.target_text for e in edges]))
        if any(token in text for token in ["施工", "脚手架", "吊装", "塔吊", "坍塌"]):
            return "construction_safety"
        return "production_safety"

    def _infer_scenario_tags(self, edges: Sequence[CausalEdge]) -> List[str]:
        text = " ".join(filter(None, [e.source_text for e in edges] + [e.target_text for e in edges]))
        rows = []
        if any(k in text for k in ["坠落", "高处", "脚手架"]):
            rows.append("high_place_fall")
        if any(k in text for k in ["坍塌", "倒塌", "塌落"]):
            rows.append("collapse")
        if any(k in text for k in ["吊装", "起重", "塔吊"]):
            rows.append("lifting_injury")
        return rows
