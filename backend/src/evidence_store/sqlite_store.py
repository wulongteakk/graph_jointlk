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

-- 因果链主表：按报告/chunk 持久化生成结果，便于后续分析与复用
CREATE TABLE IF NOT EXISTS causal_chains (
  chain_id TEXT PRIMARY KEY,
  file_name TEXT,
  parent_evidence_id TEXT,
  chain_text TEXT,
  chain_json TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_causal_chains_file ON causal_chains(file_name);
CREATE INDEX IF NOT EXISTS idx_causal_chains_parent ON causal_chains(parent_evidence_id);

-- 因果链边表：保留每一条边及顺序，并挂接证据定位
CREATE TABLE IF NOT EXISTS causal_chain_edges (
  edge_id TEXT PRIMARY KEY,
  chain_id TEXT NOT NULL,
  seq INTEGER NOT NULL,
  source_node_id TEXT NOT NULL,
  source_layer TEXT,
  target_node_id TEXT NOT NULL,
  target_layer TEXT,
  evidence_unit_id TEXT,
  evidence_start INTEGER,
  evidence_end INTEGER,
  meta_json TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_causal_chain_edges_chain ON causal_chain_edges(chain_id, seq);
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
    # Causal Chain API
    # ---------------------------------------------------------------------
    def upsert_causal_chain(
        self,
        chain_id: str,
        file_name: Optional[str],
        parent_evidence_id: Optional[str],
        chain_text: str,
        chain_json: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = datetime.utcnow().isoformat()
        chain_json_str = json.dumps(chain_json or {}, ensure_ascii=False)

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO causal_chains(chain_id, file_name, parent_evidence_id, chain_text, chain_json, created_at, updated_at)
                    VALUES(?,?,?,?,?,?,?)
                    ON CONFLICT(chain_id) DO UPDATE SET
                      file_name=excluded.file_name,
                      parent_evidence_id=excluded.parent_evidence_id,
                      chain_text=excluded.chain_text,
                      chain_json=excluded.chain_json,
                      updated_at=excluded.updated_at
                    """,
                    (chain_id, file_name, parent_evidence_id, chain_text, chain_json_str, now, now),
                )
                conn.commit()
            finally:
                conn.close()

    def upsert_causal_chain_edges(self, chain_id: str, edges: Sequence[Dict[str, Any]]) -> None:
        if not edges:
            return
        now = datetime.utcnow().isoformat()

        with self._lock:
            conn = self._connect()
            try:
                for edge in edges:
                    edge_id = edge.get("edge_id")
                    if not edge_id:
                        continue
                    meta_json = json.dumps(edge.get("meta") or {}, ensure_ascii=False)
                    conn.execute(
                        """
                        INSERT INTO causal_chain_edges(
                          edge_id, chain_id, seq, source_node_id, source_layer,
                          target_node_id, target_layer, evidence_unit_id,
                          evidence_start, evidence_end, meta_json, created_at, updated_at
                        )
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                        ON CONFLICT(edge_id) DO UPDATE SET
                          chain_id=excluded.chain_id,
                          seq=excluded.seq,
                          source_node_id=excluded.source_node_id,
                          source_layer=excluded.source_layer,
                          target_node_id=excluded.target_node_id,
                          target_layer=excluded.target_layer,
                          evidence_unit_id=excluded.evidence_unit_id,
                          evidence_start=excluded.evidence_start,
                          evidence_end=excluded.evidence_end,
                          meta_json=excluded.meta_json,
                          updated_at=excluded.updated_at
                        """,
                        (
                            edge_id,
                            chain_id,
                            int(edge.get("seq") or 0),
                            edge.get("source_node_id"),
                            edge.get("source_layer"),
                            edge.get("target_node_id"),
                            edge.get("target_layer"),
                            edge.get("evidence_unit_id"),
                            edge.get("evidence_start"),
                            edge.get("evidence_end"),
                            meta_json,
                            now,
                            now,
                        ),
                    )
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
