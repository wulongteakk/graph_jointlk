from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional, Set

from pydantic import BaseModel, Field, field_validator


class RetrievalDoc(BaseModel):
    doc_id: str
    library_name: str = Field(description="standard_rule_library / precedent_library")
    dimension: str = Field(description="basic_accident_type / injury_severity / industry_type / coding / precedent")
    source_type: str
    standard_no: Optional[str] = None
    effective_from: Optional[str] = None
    clause_id: Optional[str] = None
    title: str
    text: str
    keywords: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def search_text(self) -> str:
        pieces = [self.title, self.text]
        if self.keywords:
            pieces.append(" ".join(self.keywords))
        if self.standard_no:
            pieces.append(self.standard_no)
        if self.clause_id:
            pieces.append(self.clause_id)
        return "\n".join(x for x in pieces if x)


class SearchFilters(BaseModel):
    dimensions: Optional[Set[str]] = None
    source_types: Optional[Set[str]] = None
    standard_nos: Optional[Set[str]] = None
    library_names: Optional[Set[str]] = None
    effective_on: Optional[date] = None
    metadata_contains: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("effective_on", mode="before")
    @classmethod
    def _parse_effective_on(cls, v: Any) -> Any:
        if v is None or isinstance(v, date):
            return v
        if isinstance(v, str):
            return datetime.fromisoformat(v).date()
        raise TypeError(f"Unsupported effective_on type: {type(v)!r}")


class SearchHit(BaseModel):
    doc_id: str
    title: str
    text: str
    score: float
    bm25_score: float = 0.0
    dense_score: float = 0.0
    rerank_score: float = 0.0
    dimension: str
    source_type: str
    standard_no: Optional[str] = None
    clause_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    task: Literal["type_retrieval", "severity_retrieval", "industry_retrieval", "precedent_retrieval"]
    query: str
    top_k: int = 10
    filters: Optional[SearchFilters] = None
    fact_basis: Dict[str, Any] = Field(default_factory=dict)
    candidate_types: List[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    task: str
    normalized_query: str
    hits: List[SearchHit] = Field(default_factory=list)
    debug: Dict[str, Any] = Field(default_factory=dict)


class EvidenceUnit(BaseModel):
    evidence_id: str
    report_id: Optional[str] = None
    text: str
    section_name: Optional[str] = None
    event_role: Literal["initial_abnormal", "intermediate_event", "harm_event", "injury_outcome", "other"] = "other"
    temporal_cues: List[str] = Field(default_factory=list)
    causal_cues: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
