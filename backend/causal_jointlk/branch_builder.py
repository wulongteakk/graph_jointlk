from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence

from .schemas import CandidateBranch, CandidateChain, CausalEdge,CausalNode


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

    def build(
        self,
        nodes: Sequence[CausalNode],
        edges: Sequence[CausalEdge],
        chains: Sequence[CandidateChain],
    ) -> List[CandidateBranch]:
        node_map = {n.node_id: n for n in nodes}
        graph_outgoing: Dict[str, List[CausalEdge]] = defaultdict(list)
        graph_incoming: Dict[str, List[CausalEdge]] = defaultdict(list)
        for edge in edges:
            graph_outgoing[edge.source_id].append(edge)
            graph_incoming[edge.target_id].append(edge)

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
                basic_candidates = self._infer_basic_types(path_edges)
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
                    basic_type_candidates=basic_candidates,
                    rule_hits=["chain_score", "branch_cluster"],
                    industry_cue_texts=industry_cues,
                    industry_evidence_ids=evidence_ids,
                    site_or_process_terms=process_terms,
                    meta={
                        "module_id": self._infer_module_id(path_edges),
                        "scenario_tags": self._infer_scenario_tags(path_edges),
                        "basic_type_scores": self._score_basic_types(path_edges),
                        "edge_ids": [e.edge_id for e in path_edges if e.edge_id],
                    },
                )
            else:
                b = grouped[cluster_key]
                b.chain_ids.append(chain.chain_id)
                b.score = max(b.score, chain.score)
                b.evidence_unit_ids = list(dict.fromkeys(b.evidence_unit_ids + evidence_ids))
                b.basic_type_candidates = list(
                    dict.fromkeys(b.basic_type_candidates + self._infer_basic_types(path_edges)))
                b.industry_cue_texts = list(dict.fromkeys(b.industry_cue_texts + industry_cues))
                b.site_or_process_terms = list(dict.fromkeys(b.site_or_process_terms + process_terms))
                merged_scores = dict(b.meta.get("basic_type_scores") or {})
                for key, val in self._score_basic_types(path_edges).items():
                    merged_scores[key] = max(float(merged_scores.get(key, 0.0)), float(val))
                b.meta["basic_type_scores"] = merged_scores
                edge_ids = list(dict.fromkeys((b.meta.get("edge_ids") or []) + [e.edge_id for e in path_edges if e.edge_id]))
                b.meta["edge_ids"] = edge_ids

        for branch in grouped.values():
            branch.path_edges = self._merge_graph_edges(branch.path_nodes, branch.path_edges, graph_outgoing)
            branch.root_nodes = self._infer_graph_roots(branch.path_nodes, graph_incoming)
            branch.evidence_unit_ids = list(
                dict.fromkeys(
                    branch.evidence_unit_ids
                    + [e.evidence_unit_id for e in branch.path_edges if e.evidence_unit_id]
                    + self._collect_node_evidence(branch.path_nodes, node_map)
                )
            )
            graph_type_scores = self._score_basic_types(branch.path_edges)
            branch.meta["basic_type_scores"] = self._merge_scores(
                branch.meta.get("basic_type_scores") or {},
                graph_type_scores,
            )
            branch.basic_type_candidates = [
                name for name, _ in sorted((branch.meta.get("basic_type_scores") or {}).items(), key=lambda x: x[1], reverse=True)
            ]
            branch.meta["graph_stats"] = {
                "num_nodes": len(branch.path_nodes),
                "num_edges": len(branch.path_edges),
                "num_root_nodes": len(branch.root_nodes),
                "num_consequence_nodes": len(branch.consequence_nodes),
            }

        branches = sorted(grouped.values(), key=lambda x: x.score, reverse=True)
        return branches

    @staticmethod
    def _collect_node_evidence(node_ids: Sequence[str], node_map: Dict[str, CausalNode]) -> List[str]:
        rows: List[str] = []
        for node_id in node_ids:
            node = node_map.get(node_id)
            if node is None:
                continue
            rows.extend(node.evidence_unit_ids or [])
        return rows

    @staticmethod
    def _merge_scores(base_scores: Dict[str, float], add_scores: Dict[str, float]) -> Dict[str, float]:
        merged = dict(base_scores)
        for key, val in add_scores.items():
            merged[key] = max(float(merged.get(key, 0.0)), float(val))
        return merged

    @staticmethod
    def _merge_graph_edges(path_nodes: Sequence[str], path_edges: Sequence[CausalEdge], graph_outgoing: Dict[str, List[CausalEdge]]) -> List[CausalEdge]:
        merged = {e.edge_id: e for e in path_edges if e.edge_id}
        for node_id in path_nodes:
            for edge in graph_outgoing.get(node_id, []):
                if edge.target_id in path_nodes and edge.edge_id:
                    merged[edge.edge_id] = edge
        return list(merged.values()) if merged else list(path_edges)

    @staticmethod
    def _infer_graph_roots(path_nodes: Sequence[str], graph_incoming: Dict[str, List[CausalEdge]]) -> List[str]:
        rows: List[str] = []
        path_set = set(path_nodes)
        for node_id in path_nodes:
            incoming = [e for e in graph_incoming.get(node_id, []) if e.source_id in path_set]
            if not incoming:
                rows.append(node_id)
        return rows or ([path_nodes[0]] if path_nodes else [])

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
        scores = self._score_basic_types(edges)
        return [name for name, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

    def _score_basic_types(self, edges: Sequence[CausalEdge]) -> Dict[str, float]:
        text = " ".join(filter(None,
                               [e.source_text for e in edges] + [e.target_text for e in edges] + [e.evidence_text or ""
                                                                                                  for e in edges]))
        scores: Dict[str, float] = {}
        for basic_type, kws in self.BASIC_TYPE_KEYWORDS.items():
            hit_count = sum(1 for kw in kws if kw in text)
            if hit_count > 0:
                scores[basic_type] = round(hit_count / max(len(kws), 1), 4)
        return scores

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
