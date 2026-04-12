from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


@dataclass(slots=True)
class RetrievalSettings:
    """检索配置。

    默认值偏向“本地可跑 + 后续可替换为更强模型”：
    - embedder_backend='tfidf' 时无需额外大模型依赖；
    - embedder_backend='sentence_transformers' 时，可传本地或 Hugging Face 模型路径；
    - reranker_backend='heuristic' 时无需额外模型；
    - reranker_backend='cross_encoder' 时，可传 CrossEncoder 模型路径。
    """

    rule_library_path: Path = Path("./gb6441_2025_rule_library_v2.json")
    precedent_path: Optional[Path] = None

    embedder_backend: Literal["tfidf", "sentence_transformers"] = "tfidf"
    embedder_model_name_or_path: Optional[str] = None

    reranker_backend: Literal["heuristic", "cross_encoder"] = "heuristic"
    reranker_model_name_or_path: Optional[str] = None

    use_faiss: bool = True
    dense_top_k: int = 60
    lexical_top_k: int = 60
    rerank_top_k: int = 20
    final_top_k: int = 10

    min_score_threshold: float = 0.0
    prefer_effective_rules: bool = True
    default_effective_on: Optional[str] = None

    @property
    def has_precedent(self) -> bool:
        return self.precedent_path is not None and self.precedent_path.exists()
