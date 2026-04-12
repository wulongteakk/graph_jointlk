from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .hybrid_retriever import HybridRetriever
from .precedent_loader import PrecedentLoader
from .query_builder import QueryBuilder
from .rule_library_loader import RuleLibraryLoader
from .schemas import RetrievalDoc, SearchFilters, SearchHit, SearchRequest, SearchResponse
from .settings import RetrievalSettings


TASK_SOURCE_TYPES = {
    "type_retrieval": {"clause_rule", "basic_type_item", "coding_rule"},
    "severity_retrieval": {"severity_item", "gb15499_core_rule", "gb15499_appendix_row", "coding_rule", "clause_rule"},
    "industry_retrieval": {"industry_rule", "industry_item", "coding_rule", "clause_rule"},
    "precedent_retrieval": {"precedent_case"},
}

TASK_DIMENSIONS = {
    "type_retrieval": {"basic_accident_type", "coding"},
    "severity_retrieval": {"injury_severity", "injury_lost_workdays", "coding"},
    "industry_retrieval": {"industry_type", "coding"},
    "precedent_retrieval": {"precedent"},
}


class AccidentRetrievalService:
    def __init__(self, docs: List[RetrievalDoc], retrieval_profiles: Dict[str, Any], settings: RetrievalSettings):
        self.docs = docs
        self.settings = settings
        self.retrieval_profiles = retrieval_profiles
        self.standard_docs = [d for d in docs if d.library_name == "standard_rule_library"]
        self.precedent_docs = [d for d in docs if d.library_name == "precedent_library"]
        self.standard_retriever = HybridRetriever(
            self.standard_docs,
            dense_backend=settings.embedder_backend,
            dense_model_name_or_path=settings.embedder_model_name_or_path,
            reranker_backend=settings.reranker_backend,
            reranker_model_name_or_path=settings.reranker_model_name_or_path,
            use_faiss=settings.use_faiss,
        )
        self.precedent_retriever = None
        if self.precedent_docs:
            self.precedent_retriever = HybridRetriever(
                self.precedent_docs,
                dense_backend=settings.embedder_backend,
                dense_model_name_or_path=settings.embedder_model_name_or_path,
                reranker_backend=settings.reranker_backend,
                reranker_model_name_or_path=settings.reranker_model_name_or_path,
                use_faiss=settings.use_faiss,
            )

    @classmethod
    def from_settings(cls, settings: RetrievalSettings) -> "AccidentRetrievalService":
        loader = RuleLibraryLoader(settings.rule_library_path)
        docs, retrieval_profiles = loader.load_docs()
        if settings.has_precedent:
            docs.extend(PrecedentLoader(settings.precedent_path).load_docs())
        return cls(docs=docs, retrieval_profiles=retrieval_profiles, settings=settings)

    def search(self, request: SearchRequest) -> SearchResponse:
        merged_filters = self._merge_task_filters(request.task, request.filters)
        normalized_query = QueryBuilder.build_query(
            task=request.task,
            raw_query=request.query,
            retrieval_profiles=self.retrieval_profiles,
            fact_basis=request.fact_basis,
            candidate_types=request.candidate_types,
        )
        retriever = self.precedent_retriever if request.task == "precedent_retrieval" else self.standard_retriever
        if retriever is None:
            return SearchResponse(task=request.task, normalized_query=normalized_query, hits=[], debug={"warning": "precedent retriever not configured"})

        raw_hits = retriever.search(
            query=normalized_query,
            top_k=request.top_k,
            filters=merged_filters,
            lexical_top_k=self.settings.lexical_top_k,
            dense_top_k=self.settings.dense_top_k,
            rerank_top_k=self.settings.rerank_top_k,
        )

        boosted = self._apply_profile_boosts(request.task, raw_hits)
        boosted.sort(key=lambda x: x["score"], reverse=True)

        hits = [
            SearchHit(
                doc_id=item["doc_id"],
                title=item["doc"].title,
                text=item["doc"].text,
                score=float(item["score"]),
                bm25_score=float(item.get("bm25_score", 0.0)),
                dense_score=float(item.get("dense_score", 0.0)),
                rerank_score=float(item.get("rerank_score", 0.0)),
                dimension=item["doc"].dimension,
                source_type=item["doc"].source_type,
                standard_no=item["doc"].standard_no,
                clause_id=item["doc"].clause_id,
                metadata=item["doc"].metadata,
            )
            for item in boosted[: request.top_k]
            if item["score"] >= self.settings.min_score_threshold
        ]
        return SearchResponse(
            task=request.task,
            normalized_query=normalized_query,
            hits=hits,
            debug={
                "docs_total": len(self.docs),
                "task_dimensions": sorted(list(merged_filters.dimensions or [])),
                "task_source_types": sorted(list(merged_filters.source_types or [])),
            },
        )

    def multi_route_search(self, query: str, fact_basis: Dict[str, Any], candidate_types: List[str], top_k: int = 5) -> Dict[str, SearchResponse]:
        responses = {}
        for task in ["type_retrieval", "severity_retrieval", "industry_retrieval"]:
            responses[task] = self.search(
                SearchRequest(task=task, query=query, top_k=top_k, fact_basis=fact_basis, candidate_types=candidate_types)
            )
        if self.precedent_retriever is not None:
            responses["precedent_retrieval"] = self.search(
                SearchRequest(task="precedent_retrieval", query=query, top_k=top_k, fact_basis=fact_basis, candidate_types=candidate_types)
            )
        return responses

    def _merge_task_filters(self, task: str, user_filters: Optional[SearchFilters]) -> SearchFilters:
        task_filters = SearchFilters(
            dimensions=TASK_DIMENSIONS.get(task, set()),
            source_types=TASK_SOURCE_TYPES.get(task, set()),
            library_names={"precedent_library"} if task == "precedent_retrieval" else {"standard_rule_library"},
        )
        if self.settings.default_effective_on and not task_filters.effective_on:
            task_filters.effective_on = self.settings.default_effective_on  # pydantic 会在外部 request 中转；这里允许字符串原样传递
        if not user_filters:
            return task_filters
        merged = SearchFilters(
            dimensions=set(task_filters.dimensions or set()) | set(user_filters.dimensions or set()),
            source_types=set(task_filters.source_types or set()) | set(user_filters.source_types or set()),
            standard_nos=user_filters.standard_nos,
            library_names=set(task_filters.library_names or set()) | set(user_filters.library_names or set()),
            effective_on=user_filters.effective_on or task_filters.effective_on,
            metadata_contains={**task_filters.metadata_contains, **user_filters.metadata_contains},
        )
        return merged

    def _apply_profile_boosts(self, task: str, raw_hits: List[dict]) -> List[dict]:
        profile = self.retrieval_profiles.get(task, {})
        primary_rules = set(profile.get("primary_rules", []))
        for item in raw_hits:
            doc = item["doc"]
            primary_boost = 0.0
            if doc.clause_id and any(doc.clause_id in x for x in primary_rules):
                primary_boost += 0.15
            if doc.standard_no and any((doc.standard_no or "") in x for x in primary_rules):
                primary_boost += 0.10
            if doc.source_type.endswith("_item"):
                primary_boost += 0.03
            item["score"] += primary_boost
        return raw_hits
