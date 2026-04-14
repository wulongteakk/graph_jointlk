from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .schemas import RetrievalDoc


class PrecedentLoader:
    """加载先例库，支持 json / jsonl。"""

    def __init__(self, path: Path):
        self.path = Path(path)

    def load_docs(self) -> List[RetrievalDoc]:
        if not self.path.exists():
            return []
        suffix = self.path.suffix.lower()
        if suffix == ".jsonl":
            items = self._load_jsonl()
        else:
            items = self._load_json()
        return [self._to_doc(x, i) for i, x in enumerate(items)]

    def _load_json(self) -> List[Dict[str, Any]]:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict):
            if isinstance(data.get("items"), list):
                return [x for x in data["items"] if isinstance(x, dict)]
            return [data]
        return []

    def _load_jsonl(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
        return out

    @staticmethod
    def _to_doc(item: Dict[str, Any], idx: int) -> RetrievalDoc:
        title = str(item.get("title") or item.get("case_name") or f"precedent_{idx}")
        text = str(item.get("text") or item.get("content") or "")
        doc_id = str(item.get("doc_id") or item.get("id") or f"precedent::{idx}")
        keywords = item.get("keywords") or []
        if not isinstance(keywords, list):
            keywords = []
        return RetrievalDoc(
            doc_id=doc_id,
            library_name="precedent_library",
            dimension="precedent",
            source_type="precedent_case",
            standard_no=item.get("standard_no"),
            clause_id=item.get("clause_id"),
            title=title,
            text=text,
            keywords=[str(x) for x in keywords],
            metadata=item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
        )