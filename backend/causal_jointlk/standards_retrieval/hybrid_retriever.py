from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .schemas import RetrievalDoc, SearchFilters
from .tokenizer import keyword_overlap, tokenize


try:  # 可选依赖
    import faiss  # type: ignore
except Exception:  # pragma: no cover - 环境可能未安装
    faiss = None

try:  # 可选依赖
    from sentence_transformers import CrossEncoder, SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    CrossEncoder = None
    SentenceTransformer = None


@dataclass(slots=True)
class RawScore:
    doc: RetrievalDoc
    score: float


class BM25Retriever:
    def __init__(self, docs: Sequence[RetrievalDoc], k1: float = 1.5, b: float = 0.75):
        self.docs = list(docs)
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(doc.search_text) for doc in self.docs]
        self.doc_lens = np.array([len(x) for x in self.doc_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_lens.mean()) if len(self.doc_lens) else 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.term_freqs: List[Dict[str, int]] = []
        for tokens in self.doc_tokens:
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.term_freqs.append(tf)
            for t in tf.keys():
                self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1
        self.N = len(self.docs)
        self.idf = {
            t: math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))
            for t, df in self.doc_freqs.items()
        }

    def search(self, query: str, top_k: int, filters: Optional[SearchFilters] = None) -> List[RawScore]:
        if not self.docs:
            return []
        query_terms = tokenize(query)
        if not query_terms:
            return []
        scores = np.zeros(self.N, dtype=np.float32)
        for idx, tf in enumerate(self.term_freqs):
            if filters and not _match_filters(self.docs[idx], filters):
                continue
            dl = self.doc_lens[idx]
            denom_const = self.k1 * (1.0 - self.b + self.b * dl / (self.avgdl + 1e-9))
            s = 0.0
            for term in query_terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                idf = self.idf.get(term, 0.0)
                s += idf * (f * (self.k1 + 1.0)) / (f + denom_const)
            scores[idx] = s
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [RawScore(doc=self.docs[i], score=float(scores[i])) for i in top_indices if scores[i] > 0]


class DenseRetriever:
    def __init__(
        self,
        docs: Sequence[RetrievalDoc],
        backend: str = "tfidf",
        model_name_or_path: Optional[str] = None,
        use_faiss: bool = True,
    ):
        self.docs = list(docs)
        self.backend = backend
        self.model_name_or_path = model_name_or_path
        self.use_faiss = use_faiss and faiss is not None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.model = None
        self.corpus_matrix = None
        self.faiss_index = None
        self._build()

    def _build(self) -> None:
        texts = [doc.search_text for doc in self.docs]
        if self.backend == "sentence_transformers":
            if SentenceTransformer is None:
                raise ImportError(
                    "sentence-transformers 未安装，无法使用 sentence_transformers 后端。"
                )
            model_name = self.model_name_or_path or "./models/bge-m3"
            self.model = SentenceTransformer(model_name)
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).astype("float32")
            self.corpus_matrix = embeddings
            if self.use_faiss:
                dim = embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatIP(dim)
                self.faiss_index.add(embeddings)
        else:
            # 用 char_wb n-gram 对中文/代码/条款混合文本更稳
            self.vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 4),
                lowercase=False,
                sublinear_tf=True,
                min_df=1,
            )
            self.corpus_matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int, filters: Optional[SearchFilters] = None) -> List[RawScore]:
        if not self.docs:
            return []
        if self.backend == "sentence_transformers":
            q = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True).astype("float32")
            if self.faiss_index is not None:
                scores, indices = self.faiss_index.search(q, min(top_k * 3, len(self.docs)))
                candidates = [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]
            else:
                sims = np.dot(self.corpus_matrix, q[0])
                order = np.argsort(sims)[::-1][: min(top_k * 3, len(self.docs))]
                candidates = [(int(i), float(sims[i])) for i in order]
        else:
            q = self.vectorizer.transform([query])
            sims = (self.corpus_matrix @ q.T).toarray().ravel()
            order = np.argsort(sims)[::-1][: min(top_k * 3, len(self.docs))]
            candidates = [(int(i), float(sims[i])) for i in order]

        out: List[RawScore] = []
        for i, s in candidates:
            doc = self.docs[i]
            if filters and not _match_filters(doc, filters):
                continue
            if s <= 0:
                continue
            out.append(RawScore(doc=doc, score=s))
            if len(out) >= top_k:
                break
        return out


class Reranker:
    def __init__(self, backend: str = "heuristic", model_name_or_path: Optional[str] = None):
        self.backend = backend
        self.model_name_or_path = model_name_or_path
        self.model = None
        if backend == "cross_encoder":
            if CrossEncoder is None:
                raise ImportError("sentence-transformers 未安装，无法使用 CrossEncoder reranker。")
            model_name = model_name_or_path or "./models/bge-reranker-v2-m3"
            self.model = CrossEncoder(model_name)

    def score(self, query: str, docs: Sequence[RetrievalDoc]) -> List[float]:
        if not docs:
            return []
        if self.backend == "cross_encoder":
            pairs = [(query, doc.search_text) for doc in docs]
            scores = self.model.predict(pairs)
            return [float(s) for s in scores]
        return [self._heuristic_score(query, doc) for doc in docs]

    @staticmethod
    def _heuristic_score(query: str, doc: RetrievalDoc) -> float:
        overlap = keyword_overlap(query, doc.search_text)
        title_bonus = 0.2 if keyword_overlap(query, doc.title) > 0.5 else 0.0
        code_bonus = 0.15 if doc.metadata.get("full_code") and str(doc.metadata.get("full_code")) in query else 0.0
        clause_bonus = 0.1 if doc.clause_id and str(doc.clause_id) in query else 0.0
        return overlap + title_bonus + code_bonus + clause_bonus


class HybridRetriever:
    def __init__(self, docs: Sequence[RetrievalDoc], dense_backend: str, dense_model_name_or_path: Optional[str], reranker_backend: str, reranker_model_name_or_path: Optional[str], use_faiss: bool = True):
        self.docs = list(docs)
        self.doc_by_id = {doc.doc_id: doc for doc in self.docs}
        self.bm25 = BM25Retriever(self.docs)
        self.dense = DenseRetriever(self.docs, backend=dense_backend, model_name_or_path=dense_model_name_or_path, use_faiss=use_faiss)
        self.reranker = Reranker(backend=reranker_backend, model_name_or_path=reranker_model_name_or_path)

    def search(self, query: str, top_k: int, filters: Optional[SearchFilters], lexical_top_k: int, dense_top_k: int, rerank_top_k: int) -> List[dict]:
        bm25_hits = self.bm25.search(query, lexical_top_k, filters)
        dense_hits = self.dense.search(query, dense_top_k, filters)
        fused = reciprocal_rank_fusion(bm25_hits, dense_hits)
        candidates = sorted(fused.values(), key=lambda x: x["score"], reverse=True)[:rerank_top_k]
        candidate_docs = [self.doc_by_id[item["doc_id"]] for item in candidates]
        rerank_scores = self.reranker.score(query, candidate_docs)
        for item, rerank_score in zip(candidates, rerank_scores):
            item["rerank_score"] = float(rerank_score)
            item["score"] = item["score"] + float(rerank_score)
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]


def reciprocal_rank_fusion(*rank_lists: Sequence[RawScore], k: int = 60) -> Dict[str, dict]:
    fused: Dict[str, dict] = {}
    for list_name, rank_list in zip(["bm25", "dense"], rank_lists):
        for rank, hit in enumerate(rank_list, start=1):
            item = fused.setdefault(
                hit.doc.doc_id,
                {
                    "doc_id": hit.doc.doc_id,
                    "doc": hit.doc,
                    "score": 0.0,
                    "bm25_score": 0.0,
                    "dense_score": 0.0,
                    "rerank_score": 0.0,
                },
            )
            item["score"] += 1.0 / (k + rank)
            item[f"{list_name}_score"] = float(hit.score)
    return fused


def _match_filters(doc: RetrievalDoc, filters: SearchFilters) -> bool:
    if filters.dimensions and doc.dimension not in filters.dimensions:
        return False
    if filters.source_types and doc.source_type not in filters.source_types:
        return False
    if filters.standard_nos and (doc.standard_no not in filters.standard_nos):
        return False
    if filters.library_names and (doc.library_name not in filters.library_names):
        return False
    if filters.effective_on and doc.effective_from:
        try:
            doc_date = np.datetime64(doc.effective_from).astype("datetime64[D]").astype(object)
            if doc_date > filters.effective_on:
                return False
        except Exception:
            pass
    for k, v in filters.metadata_contains.items():
        if doc.metadata.get(k) != v:
            return False
    return True
