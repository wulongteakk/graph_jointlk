import json
import os
import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence


_DEFAULT_DB_PATH = os.getenv("EVIDENCE_DB_PATH", "./data/evidence_store.sqlite3")


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS evidence_chunks (
  evidence_id TEXT PRIMARY KEY,
  file_name TEXT NOT NULL,
  position INTEGER DEFAULT 0,
  content TEXT NOT NULL,
  meta_json TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_evidence_chunks_file_name ON evidence_chunks(file_name);
CREATE INDEX IF NOT EXISTS idx_evidence_chunks_position ON evidence_chunks(file_name, position);

-- 更细粒度证据单元（句子/段落等），正文仍存 SQLite，不进入 KG
CREATE TABLE IF NOT EXISTS evidence_units (
  unit_id TEXT PRIMARY KEY,
  parent_evidence_id TEXT NOT NULL,
  unit_kind TEXT NOT NULL,             -- sentence/paragraph/custom
  start_char INTEGER DEFAULT NULL,
  end_char INTEGER DEFAULT NULL,
  content TEXT NOT NULL,
  meta_json TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_evidence_units_parent ON evidence_units(parent_evidence_id);
CREATE INDEX IF NOT EXISTS idx_evidence_units_kind ON evidence_units(unit_kind);
"""


@dataclass(frozen=True)
class EvidenceChunk:
    evidence_id: str
    file_name: str
    position: int
    content: str
    meta: Dict[str, Any]
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class EvidenceUnit:
    unit_id: str
    parent_evidence_id: str
    unit_kind: str
    start_char: Optional[int]
    end_char: Optional[int]
    content: str
    meta: Dict[str, Any]
    created_at: str
    updated_at: str


class EvidenceStore:
    """SQLite Evidence Store。

    设计目标：
    - Chunk（全文/窗口）与 EvidenceUnit（句子/段落）都存储在证据库中
    - Neo4j 里只保存指针（evidence_id / unit_id）和少量结构化元数据

    线程安全：内部 lock + sqlite check_same_thread=False。
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or _DEFAULT_DB_PATH
        self._lock = threading.RLock()
        self._ensure_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(_SCHEMA_SQL)
                conn.commit()
            finally:
                conn.close()

    # ---------------------------------------------------------------------
    # Chunk API
    # ---------------------------------------------------------------------
    def upsert_chunk(
        self,
        evidence_id: str,
        file_name: str,
        content: str,
        position: int = 0,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = datetime.utcnow().isoformat()
        meta_json = json.dumps(meta or {}, ensure_ascii=False)

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO evidence_chunks(evidence_id, file_name, position, content, meta_json, created_at, updated_at)
                    VALUES(?,?,?,?,?,?,?)
                    ON CONFLICT(evidence_id) DO UPDATE SET
                      file_name=excluded.file_name,
                      position=excluded.position,
                      content=excluded.content,
                      meta_json=excluded.meta_json,
                      updated_at=excluded.updated_at
                    """,
                    (evidence_id, file_name, int(position), content, meta_json, now, now),
                )
                conn.commit()
            finally:
                conn.close()

    def get_content(self, evidence_id: str) -> Optional[str]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT content FROM evidence_chunks WHERE evidence_id = ?",
                    (evidence_id,),
                ).fetchone()
                return row["content"] if row else None
            finally:
                conn.close()

    def bulk_get_contents(self, evidence_ids: Sequence[str]) -> Dict[str, str]:
        ids = [x for x in evidence_ids if x]
        if not ids:
            return {}
        placeholders = ",".join(["?"] * len(ids))
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    f"SELECT evidence_id, content FROM evidence_chunks WHERE evidence_id IN ({placeholders})",
                    tuple(ids),
                ).fetchall()
                return {r["evidence_id"]: r["content"] for r in rows}
            finally:
                conn.close()

    def get_chunk(self, evidence_id: str) -> Optional[EvidenceChunk]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT evidence_id, file_name, position, content, meta_json, created_at, updated_at
                    FROM evidence_chunks WHERE evidence_id = ?
                    """,
                    (evidence_id,),
                ).fetchone()
                if not row:
                    return None
                meta = _safe_json_loads(row["meta_json"])
                return EvidenceChunk(
                    evidence_id=row["evidence_id"],
                    file_name=row["file_name"],
                    position=int(row["position"] or 0),
                    content=row["content"],
                    meta=meta,
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
            finally:
                conn.close()

    def list_chunks_for_file(self, file_name: str) -> List[EvidenceChunk]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT evidence_id, file_name, position, content, meta_json, created_at, updated_at
                    FROM evidence_chunks
                    WHERE file_name = ?
                    ORDER BY position ASC
                    """,
                    (file_name,),
                ).fetchall()
                out: List[EvidenceChunk] = []
                for row in rows:
                    out.append(
                        EvidenceChunk(
                            evidence_id=row["evidence_id"],
                            file_name=row["file_name"],
                            position=int(row["position"] or 0),
                            content=row["content"],
                            meta=_safe_json_loads(row["meta_json"]),
                            created_at=row["created_at"],
                            updated_at=row["updated_at"],
                        )
                    )
                return out
            finally:
                conn.close()

    # ---------------------------------------------------------------------
    # Evidence Unit API
    # ---------------------------------------------------------------------
    def upsert_unit(
        self,
        unit_id: str,
        parent_evidence_id: str,
        unit_kind: str,
        content: str,
        start_char: Optional[int] = None,
        end_char: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """插入或更新证据单元（句子/段落）。"""
        now = datetime.utcnow().isoformat()
        meta_json = json.dumps(meta or {}, ensure_ascii=False)

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO evidence_units(unit_id, parent_evidence_id, unit_kind, start_char, end_char, content, meta_json, created_at, updated_at)
                    VALUES(?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(unit_id) DO UPDATE SET
                      parent_evidence_id=excluded.parent_evidence_id,
                      unit_kind=excluded.unit_kind,
                      start_char=excluded.start_char,
                      end_char=excluded.end_char,
                      content=excluded.content,
                      meta_json=excluded.meta_json,
                      updated_at=excluded.updated_at
                    """,
                    (
                        unit_id,
                        parent_evidence_id,
                        unit_kind,
                        None if start_char is None else int(start_char),
                        None if end_char is None else int(end_char),
                        content,
                        meta_json,
                        now,
                        now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def get_unit(self, unit_id: str) -> Optional[EvidenceUnit]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT unit_id, parent_evidence_id, unit_kind, start_char, end_char, content, meta_json, created_at, updated_at
                    FROM evidence_units WHERE unit_id = ?
                    """,
                    (unit_id,),
                ).fetchone()
                if not row:
                    return None
                meta = _safe_json_loads(row["meta_json"])
                return EvidenceUnit(
                    unit_id=row["unit_id"],
                    parent_evidence_id=row["parent_evidence_id"],
                    unit_kind=row["unit_kind"],
                    start_char=row["start_char"],
                    end_char=row["end_char"],
                    content=row["content"],
                    meta=meta,
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
            finally:
                conn.close()

    def bulk_get_unit_contents(self, unit_ids: Sequence[str]) -> Dict[str, str]:
        ids = [x for x in unit_ids if x]
        if not ids:
            return {}
        placeholders = ",".join(["?"] * len(ids))
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    f"SELECT unit_id, content FROM evidence_units WHERE unit_id IN ({placeholders})",
                    tuple(ids),
                ).fetchall()
                return {r["unit_id"]: r["content"] for r in rows}
            finally:
                conn.close()

    def list_units_for_parent(self, parent_evidence_id: str, unit_kind: Optional[str] = None) -> List[EvidenceUnit]:
        with self._lock:
            conn = self._connect()
            try:
                if unit_kind:
                    rows = conn.execute(
                        """
                        SELECT unit_id, parent_evidence_id, unit_kind, start_char, end_char, content, meta_json, created_at, updated_at
                        FROM evidence_units
                        WHERE parent_evidence_id = ? AND unit_kind = ?
                        ORDER BY start_char ASC
                        """,
                        (parent_evidence_id, unit_kind),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT unit_id, parent_evidence_id, unit_kind, start_char, end_char, content, meta_json, created_at, updated_at
                        FROM evidence_units
                        WHERE parent_evidence_id = ?
                        ORDER BY start_char ASC
                        """,
                        (parent_evidence_id,),
                    ).fetchall()

                return [
                    EvidenceUnit(
                        unit_id=row["unit_id"],
                        parent_evidence_id=row["parent_evidence_id"],
                        unit_kind=row["unit_kind"],
                        start_char=row["start_char"],
                        end_char=row["end_char"],
                        content=row["content"],
                        meta=_safe_json_loads(row["meta_json"]),
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                    )
                    for row in rows
                ]
            finally:
                conn.close()


# -------------------------------------------------------------------------
# internal helpers
# -------------------------------------------------------------------------

def _safe_json_loads(s: Optional[str]) -> Dict[str, Any]:
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}


def normalize_for_search(text: str) -> str:
    """用于轻量搜索的归一化（不保证可逆）。"""
    text = text or ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()
