
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvidenceUnitRef:
    unit_id: str
    content: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalNode:
    node_id: str
    text: str
    layer: str = "UNK"
    doc_id: Optional[str] = None
    kg_scope: Optional[str] = None
    kg_id: Optional[str] = None
    labels: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalEdge:
    edge_id: str
    source_id: str
    target_id: str
    relation: str
    source_text: str
    target_text: str
    source_layer: str = "UNK"
    target_layer: str = "UNK"
    evidence_unit_id: Optional[str] = None
    evidence_text: Optional[str] = None
    doc_id: Optional[str] = None
    kg_scope: Optional[str] = None
    kg_id: Optional[str] = None
    score: float = 0.0
    supported: bool = False
    support_score: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateChain:
    chain_id: str
    nodes: List[str]
    edges: List[CausalEdge]
    score: float
    missing_layers: List[str] = field(default_factory=list)
    unsupported_edges: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "nodes": self.nodes,
            "score": self.score,
            "missing_layers": self.missing_layers,
            "unsupported_edges": self.unsupported_edges,
            "edges": [
                {
                    "edge_id": e.edge_id,
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "relation": e.relation,
                    "source_text": e.source_text,
                    "target_text": e.target_text,
                    "source_layer": e.source_layer,
                    "target_layer": e.target_layer,
                    "evidence_unit_id": e.evidence_unit_id,
                    "evidence_text": e.evidence_text,
                    "score": e.score,
                    "supported": e.supported,
                    "support_score": e.support_score,
                    "meta": e.meta,
                }
                for e in self.edges
            ],
        }


@dataclass
class ExtractionResult:
    mode: str
    query: Optional[str]
    doc_id: Optional[str]
    target_node_id: Optional[str]
    chains: List[CandidateChain]
    subgraph_nodes: List[CausalNode]
    subgraph_edges: List[CausalEdge]
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "query": self.query,
            "doc_id": self.doc_id,
            "target_node_id": self.target_node_id,
            "chains": [c.to_dict() for c in self.chains],
            "subgraph_nodes": [vars(n) for n in self.subgraph_nodes],
            "subgraph_edges": [vars(e) for e in self.subgraph_edges],
            "meta": self.meta,
        }
