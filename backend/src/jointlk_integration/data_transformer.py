import logging
import re
from typing import Any, Callable, Dict, List, Tuple,Optional

import torch

from utils.conceptnet import del_pos, merged_relations


class GraphDataTransformer:
    """将检索到的图和问题文本组装为 JointLK decoder 所需输入。"""

    def __init__(
        self,
        tokenizer: Callable,
        cpnet_vocab_path: str,
        max_seq_len: int = 128,
        max_node_num: int = 200,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_node_num = max_node_num

        self.concept2id, self.id2concept = self._load_concept_vocab(cpnet_vocab_path)
        self.relation2id = {r: i for i, r in enumerate(merged_relations)}

        # 与训练构图逻辑一致：0/1 预留给 context->question / context->answer
        self.base_relation_offset = 2
        self.half_n_rel = len(merged_relations) + self.base_relation_offset

    def _load_concept_vocab(self, cpnet_vocab_path: str) -> Tuple[Dict[str, int], List[str]]:
        with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
            id2concept = [w.strip() for w in fin]
        concept2id = {w: i for i, w in enumerate(id2concept)}
        return concept2id, id2concept

    def get_concept_id(self, node_text: str) -> int:
        processed_text = del_pos(node_text)
        processed_text = processed_text.split("/")[-1].lower().strip()

        cleaned = re.sub(r"[_-]", "", processed_text)
        if not cleaned.isalpha():
            return -1
        return self.concept2id.get(processed_text, -1)

    def _map_relation_type(self, relation_name: str) -> int:
        if not relation_name:
            return self.base_relation_offset
        normalized = relation_name.lower().strip()
        if normalized in self.relation2id:
            return self.relation2id[normalized] + self.base_relation_offset

        normalized = normalized.replace(" ", "_")
        return self.relation2id.get(normalized, 0) + self.base_relation_offset

    @staticmethod
    def _extract_node_uid(node: Dict[str, Any]) -> Optional[Any]:
        """兼容 graph_query / chunkid_entities 等返回格式的节点 ID。"""
        return (
                node.get("id")
                or node.get("element_id")
                or node.get("node_id")
                or node.get("uid")
        )

    @staticmethod
    def _extract_node_text(node: Dict[str, Any]) -> str:
        props = node.get("properties", {}) if isinstance(node.get("properties"), dict) else {}
        return str(
            props.get("name")
            or props.get("id")
            or props.get("description")
            or node.get("name")
            or node.get("text")
            or ""
        ).strip()

    @staticmethod
    def _extract_rel_ends(rel: Dict[str, Any]) -> Tuple[Optional[Any], Optional[Any]]:
        start_id = (
                rel.get("start_node_id")
                or rel.get("start_node_element_id")
                or rel.get("start")
                or rel.get("source")
        )
        end_id = (
                rel.get("end_node_id")
                or rel.get("end_node_element_id")
                or rel.get("end")
                or rel.get("target")
        )
        return start_id, end_id

    def format_for_jointlk(
        self,
        subgraph_nodes: List[Dict[str, Any]],
        subgraph_relationships: List[Dict[str, Any]],
        question: str,
    ) -> Dict[str, Any]:
        logging.info(
            "Transform graph to JointLK tensors. nodes=%d rels=%d",
            len(subgraph_nodes),
            len(subgraph_relationships),
        )

        concept_ids = torch.ones((1, self.max_node_num), dtype=torch.long)
        node_type_ids = torch.full((1, self.max_node_num), 2, dtype=torch.long)
        node_scores = torch.zeros((1, self.max_node_num, 1), dtype=torch.float)

        concept_ids[0, 0] = 0  # context node
        node_type_ids[0, 0] = 3

        neo4j_to_slot: Dict[Any, int] = {}
        context_texts: List[str] = []
        next_slot = 1
        dropped_nodes = 0

        for node in subgraph_nodes:
            if next_slot >= self.max_node_num:
                break
            neo4j_id = self._extract_node_uid(node)
            if neo4j_id in neo4j_to_slot:
                continue

            node_text = self._extract_node_text(node)
            if node_text:
                context_texts.append(node_text)

            cid = self.get_concept_id(node_text)
            if cid < 0:
                dropped_nodes += 1
                continue

            concept_ids[0, next_slot] = cid + 1  # 训练时为 context node 偏移了 +1
            neo4j_to_slot[neo4j_id] = next_slot
            next_slot += 1

        adj_len = max(1, next_slot)
        adj_lengths = torch.tensor([adj_len], dtype=torch.long)

        starts: List[int] = []
        ends: List[int] = []
        etypes: List[int] = []
        dropped_edges = 0
        for rel in subgraph_relationships:
            start_id, end_id = self._extract_rel_ends(rel)
            s = neo4j_to_slot.get(start_id)
            t = neo4j_to_slot.get(end_id)
            if s is None or t is None:
                dropped_edges += 1
                continue

            rel_id = self._map_relation_type(str(rel.get("type", "")))
            starts.append(s)
            ends.append(t)
            etypes.append(rel_id)

            # 补双向边，与训练装载逻辑一致
            starts.append(t)
            ends.append(s)
            etypes.append(rel_id + self.half_n_rel)

        if starts:
            edge_index = torch.tensor([starts, ends], dtype=torch.long)
            edge_type = torch.tensor(etypes, dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_type = torch.empty((0,), dtype=torch.long)

        context_string = " ".join(dict.fromkeys(context_texts))
        input_text = f"{question} [SEP] {context_string}" if context_string else question
        encoded = self.tokenizer(
            input_text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "concept_ids": concept_ids,
            "node_type_ids": node_type_ids,
            "node_scores": node_scores,
            "adj_lengths": adj_lengths,
            "adj": (edge_index, edge_type),
            "metadata": {
                "input_nodes": int(len(subgraph_nodes)),
                "input_edges": int(len(subgraph_relationships)),
                "processed_nodes": int(len(neo4j_to_slot)),
                "processed_edges": int(len(starts)),
                "dropped_nodes": int(dropped_nodes),
                "dropped_edges": int(dropped_edges),
                "max_node_num": int(self.max_node_num),
            },
        }