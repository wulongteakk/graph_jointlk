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
    core_type: Optional[str] = None
    hfsca_layer: Optional[str] = None
    hfsca_category: Optional[str] = None
    module_id: Optional[str] = None
    event_stage: Optional[str] = None
    is_main_chain_candidate: Optional[bool] = None
    canonical_type: Optional[str] = None
    domain_id: str = "production_safety"
    scenario_tags: List[str] = field(default_factory=list)
    role_tags: List[str] = field(default_factory=list)
    event_time: Optional[str] = None
    temporal_rank: Optional[float] = None
    severity_signals: Dict[str, float] = field(default_factory=dict)
    evidence_unit_ids: List[str] = field(default_factory=list)
    p_node_first: Optional[float] = None
    silver_node_first: Optional[int] = None
    first_conf: Optional[float] = None


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
    p_causal: Optional[float] = None
    p_enable: Optional[float] = None
    p_dir: Optional[float] = None
    p_temporal_before: Optional[float] = None
    p_evidence: Optional[float] = None
    p_node_first: Optional[float] = None
    relation_family: Optional[str] = None
    semantic_role: Optional[str] = None
    domain_id: str = "production_safety"
    module_id: Optional[str] = None
    scenario_tags: List[str] = field(default_factory=list)
    direction_support: Optional[float] = None
    temporal_support: Optional[float] = None
    basic_type_supports: Dict[str, float] = field(default_factory=dict)
    silver_edge_causal: Optional[int] = None
    silver_edge_enable: Optional[int] = None
    silver_causal_dir: Optional[int] = None
    silver_temporal_before: Optional[int] = None
    sample_weight: float = 1.0
    twin_group_id: Optional[str] = None
    label_source: Optional[str] = None
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
            "edges": [vars(e) for e in self.edges],
        }


@dataclass
class CandidateBranch:
    branch_id: str
    chain_ids: List[str]
    first_node_id: Optional[str]
    score: float
    evidence_unit_ids: List[str] = field(default_factory=list)
    rule_hits: List[str] = field(default_factory=list)


@dataclass
class BranchDecision:
    selected_branch_id: Optional[str]
    decision_gap: float
    needs_severity_fallback: bool
    reason: str


@dataclass
class DecodedAccidentResult:
    basic_type: Optional[str]
    injury_severity: Optional[str]
    industry_type: Optional[str]
    basic_code: Optional[str]
    injury_code: Optional[str]
    industry_code: Optional[str]
    decode_rule_hits: List[str] = field(default_factory=list)
    decode_confidence: float = 0.0


@dataclass
class ExtractionResult:
    mode: str
    query: Optional[str]
    doc_id: Optional[str]
    target_node_id: Optional[str]
    chains: List[CandidateChain]
    subgraph_nodes: List[CausalNode]
    subgraph_edges: List[CausalEdge]
    branches: List[CandidateBranch] = field(default_factory=list)
    branch_decision: Optional[BranchDecision] = None
    selected_branch: Optional[CandidateBranch] = None
    rejected_branches: List[CandidateBranch] = field(default_factory=list)
    decoded_result: Optional[DecodedAccidentResult] = None
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
            "branches": [vars(b) for b in self.branches],
            "branch_decision": vars(self.branch_decision) if self.branch_decision else None,
            "selected_branch": vars(self.selected_branch) if self.selected_branch else None,
            "rejected_branches": [vars(b) for b in self.rejected_branches],
            "decoded_result": vars(self.decoded_result) if self.decoded_result else None,
            "meta": self.meta,
        }