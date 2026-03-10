from langchain_community.graphs import Neo4jGraph
from langchain.docstore.document import Document

from src.shared.common_fn import load_embedding_model
from src.evidence_store.sqlite_store import EvidenceStore
from src.evidence_store.text_span import (
    split_paragraphs_with_offsets,
    split_sentences_with_offsets,
    find_best_span,
    pick_span_paragraph,
    pick_span_sentence,
)

import logging
from typing import List, Dict, Any, Optional, Tuple
import os
import hashlib


logging.basicConfig(format="%(asctime)s - %(message)s", level="INFO")


def _node_props(node) -> Dict[str, Any]:
    props = getattr(node, "properties", None)
    return props if isinstance(props, dict) else {}


def _pick_evidence_snippet(node) -> str:
    """从 GraphDocument Node 中挑一个用于证据定位的短文本。"""
    props = _node_props(node)
    for k in [
        "evidence",
        "evidence_text",
        "quote",
        "excerpt",
        "supporting_text",
        "support",
        "reason_evidence",
    ]:
        v = props.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()[:120]

    name = props.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()[:120]

    return str(getattr(node, "id", "")).strip()[:120]


def _parse_scope_from_doc_id(doc_id: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """doc_id expected: '<kg_scope>|<kg_id>|<file_name_clean>'"""
    if not doc_id or "|" not in doc_id:
        return None, None
    parts = doc_id.split("|", 2)
    if len(parts) < 2:
        return None, None
    return parts[0], parts[1]


def _infer_doc_fields(chunk_id: str, md: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Infer doc_id/kg_scope/kg_id from metadata, with chunk_id fallback.

    - Preferred: metadata['doc_id'], metadata['kg_scope'], metadata['kg_id']
    - Fallback: if chunk_id == '<doc_id>|<hash>', then doc_id = chunk_id.rsplit('|',1)[0]
    - Fallback parse: kg_scope, kg_id from doc_id
    """

    doc_id = md.get("doc_id") or md.get("docId")
    kg_scope = md.get("kg_scope")
    kg_id = md.get("kg_id")

    # chunk_id fallback
    if not doc_id and isinstance(chunk_id, str) and "|" in chunk_id:
        # chunk_id = '<doc_id>|<sha1>'
        doc_id = chunk_id.rsplit("|", 1)[0]

    # parse from doc_id
    if (not kg_scope or not kg_id) and doc_id:
        s, kid = _parse_scope_from_doc_id(doc_id)
        kg_scope = kg_scope or s
        kg_id = kg_id or kid

    return {"doc_id": doc_id, "kg_scope": kg_scope, "kg_id": kg_id}


def merge_relationship_between_chunk_and_entites(graph: Neo4jGraph, graph_documents_chunk_chunk_Id: list):
    """Create HAS_ENTITY relationship between chunks and entities.

    本版本：段落级证据优先（unit_kind='paragraph'），句子级作为次级补充。

    新增 / 更新：
    - 证据跨度定位（evidence_offset）：在 HAS_ENTITY / SUPPORTED_BY 上写入 start/end
    - EvidenceUnit（段落/句子）：
        * EvidenceStore(SQLite) 存正文
        * Neo4j 只存 unit_id 指针、span、unit_kind
    - “优先段落，再落句子”：
        * (Entity)-[:SUPPORTED_BY {primary:true}]->(paragraph EvidenceUnit)
        * (Entity)-[:SUPPORTED_BY {primary:false}]->(sentence EvidenceUnit)

    目标：让 Stage1 抽取出来的实体具备“可追溯到段落级”的证据定位。

    补齐：EvidenceUnit / HAS_ENTITY / SUPPORTED_BY 写入 doc_id/kg_scope/kg_id 属性（非 MERGE key）。
    """

    logging.info("Create HAS_ENTITY relationship between chunks and entities (paragraph-first evidence units)")

    evidence_store = EvidenceStore()

    # 整理输入为 (chunk_id, graph_doc)
    pairs: List[Dict[str, Any]] = []
    for item in graph_documents_chunk_chunk_Id:
        chunk_ids: List[str] = []
        graph_doc = None

        if isinstance(item, dict):
            graph_doc = item.get("graph_doc")
            chunk_id = item.get("chunk_id")
            if chunk_id:
                chunk_ids.append(chunk_id)
        elif isinstance(item, tuple) and len(item) == 2:
            chunk_ids, graph_doc = item
            if not isinstance(chunk_ids, list):
                chunk_ids = [chunk_ids]

        if not graph_doc:
            logging.warning("graph_doc is missing from entry %s; skipping", item)
            continue

        if not chunk_ids and graph_doc.source and graph_doc.source.metadata:
            metadata_chunk_ids = graph_doc.source.metadata.get("combined_chunk_ids")
            if metadata_chunk_ids:
                chunk_ids = metadata_chunk_ids if isinstance(metadata_chunk_ids, list) else [metadata_chunk_ids]

        for cid in chunk_ids:
            pairs.append({"chunk_id": cid, "graph_doc": graph_doc})

    # 预取正文并分段/分句（每个 chunk 一次）
    cache: Dict[str, Dict[str, Any]] = {}

    para_max_len = int(os.getenv("EVIDENCE_PARA_MAX_LEN", "1200"))
    sent_max_len = int(os.getenv("EVIDENCE_SENT_MAX_LEN", os.getenv("EVIDENCE_UNIT_MAX_LEN", "400")))

    def _get_chunk_cache(chunk_id: str) -> Dict[str, Any]:
        if chunk_id in cache:
            return cache[chunk_id]

        text = evidence_store.get_content(chunk_id) or ""
        paras = split_paragraphs_with_offsets(text, max_len=para_max_len)
        sents = split_sentences_with_offsets(text, max_len=sent_max_len)
        cache[chunk_id] = {"text": text, "paras": paras, "sents": sents}
        return cache[chunk_id]

    primary_batch: List[Dict[str, Any]] = []
    secondary_batch: List[Dict[str, Any]] = []

    for p in pairs:
        chunk_id = p["chunk_id"]
        graph_doc = p["graph_doc"]

        md = graph_doc.source.metadata if (graph_doc.source and graph_doc.source.metadata) else {}
        file_name = md.get("fileName")

        scope_fields = _infer_doc_fields(chunk_id, md)
        doc_id = scope_fields["doc_id"]
        kg_scope = scope_fields["kg_scope"]
        kg_id = scope_fields["kg_id"]

        chunk_info = _get_chunk_cache(chunk_id)
        chunk_text = chunk_info["text"]
        paras = chunk_info["paras"]
        sents = chunk_info["sents"]

        for node in graph_doc.nodes:
            node_id = node.id
            node_type = node.type
            snippet = _pick_evidence_snippet(node)

            start, end = find_best_span(chunk_text, snippet)
            if start is None:
                start, end = find_best_span(chunk_text, str(node_id))

            # 段落
            para_idx = pick_span_paragraph(paras, start, end) if paras else 0
            para = paras[para_idx] if paras else None
            para_unit_id = f"{chunk_id}::para::{para_idx}" if paras else None

            # 句子（次级）
            sent_idx = pick_span_sentence(sents, start, end) if sents else 0
            sent = sents[sent_idx] if sents else None
            sent_unit_id = f"{chunk_id}::sent::{sent_idx}" if sents else None

            # primary: 优先段落，其次句子；都不可用时回落到 chunk 级别，避免 Neo4j MERGE 空 id
            if para and para.text and para.text.strip():
                primary_unit_id = para_unit_id
                primary_kind = "paragraph"
                primary_span = para
            elif sent and sent.text and sent.text.strip():
                primary_unit_id = sent_unit_id
                primary_kind = "sentence"
                primary_span = sent
            else:
                primary_unit_id = f"{chunk_id}::chunk"
                primary_kind = "chunk"
                primary_span = None

            # secondary: 如果 primary 是段落，则补一个句子
            secondary_unit_id = None
            secondary_kind = None
            secondary_span = None
            if primary_kind == "paragraph" and sent and sent.text and sent.text.strip():
                secondary_unit_id = sent_unit_id
                secondary_kind = "sentence"
                secondary_span = sent

            # 写入 EvidenceStore（正文不进入 KG）
            try:
                # 段落单位
                if para_unit_id and para and para.text and para.text.strip():
                    evidence_store.upsert_unit(
                        unit_id=para_unit_id,
                        parent_evidence_id=chunk_id,
                        unit_kind="paragraph",
                        content=para.text,
                        start_char=para.start,
                        end_char=para.end,
                        meta={
                            "file_name": file_name,
                            "chunk_id": chunk_id,
                            "paragraph_index": para_idx,
                            "doc_id": doc_id,
                            "kg_scope": kg_scope,
                            "kg_id": kg_id,
                        },
                    )
                # 句子单位（用于 fallback/次级解释）
                if sent_unit_id and sent and sent.text and sent.text.strip():
                    evidence_store.upsert_unit(
                        unit_id=sent_unit_id,
                        parent_evidence_id=chunk_id,
                        unit_kind="sentence",
                        content=sent.text,
                        start_char=sent.start,
                        end_char=sent.end,
                        meta={
                            "file_name": file_name,
                            "chunk_id": chunk_id,
                            "sentence_index": sent_idx,
                            "doc_id": doc_id,
                            "kg_scope": kg_scope,
                            "kg_id": kg_id,
                        },
                    )
            except Exception as e:
                logging.warning(f"EvidenceStore.upsert_unit failed: {e}")

            # primary batch (HAS_ENTITY + primary SUPPORTED_BY)
            primary_batch.append(
                {
                    "chunk_id": chunk_id,
                    "node_type": node_type,
                    "node_id": node_id,
                    "file_name": file_name,
                    "doc_id": doc_id,
                    "kg_scope": kg_scope,
                    "kg_id": kg_id,
                    "evidence_text": snippet,
                    "evidence_start": start,
                    "evidence_end": end,
                    "primary_unit_id": primary_unit_id,
                    "primary_unit_kind": primary_kind,
                    "primary_unit_start": (primary_span.start if primary_span else start),
                    "primary_unit_end": (primary_span.end if primary_span else end),
                    "paragraph_unit_id": para_unit_id,
                    "paragraph_start": (para.start if para else None),
                    "paragraph_end": (para.end if para else None),
                    "sentence_unit_id": sent_unit_id,
                    "sentence_start": (sent.start if sent else None),
                    "sentence_end": (sent.end if sent else None),
                    "parent_evidence_id": chunk_id,
                }
            )

            # secondary batch (secondary SUPPORTED_BY)
            if secondary_unit_id and secondary_unit_id != primary_unit_id:
                secondary_batch.append(
                    {
                        "chunk_id": chunk_id,
                        "node_type": node_type,
                        "node_id": node_id,
                        "file_name": file_name,
                        "doc_id": doc_id,
                        "kg_scope": kg_scope,
                        "kg_id": kg_id,
                        "evidence_text": snippet,
                        "evidence_start": start,
                        "evidence_end": end,
                        "unit_id": secondary_unit_id,
                        "unit_kind": secondary_kind,
                        "unit_start": (secondary_span.start if secondary_span else None),
                        "unit_end": (secondary_span.end if secondary_span else None),
                        "parent_evidence_id": chunk_id,
                    }
                )

    if not primary_batch:
        logging.warning("No primary_batch for HAS_ENTITY/SUPPORTED_BY creation")
        return

    # --------------------------
    # 1) primary: HAS_ENTITY + primary EvidenceUnit + primary SUPPORTED_BY
    # --------------------------

    primary_query = """
    UNWIND $batch AS data
    MERGE (c:Chunk {id: data.chunk_id})
    SET c.fileName = coalesce(data.file_name, c.fileName),
        c.doc_id = coalesce(data.doc_id, c.doc_id),
        c.kg_scope = coalesce(data.kg_scope, c.kg_scope),
        c.kg_id = coalesce(data.kg_id, c.kg_id)
    WITH c, data

    CALL apoc.merge.node([data.node_type], {id: data.node_id}) YIELD node AS n

    MERGE (c)-[he:HAS_ENTITY]->(n)
    SET he.evidence_text = data.evidence_text,
        he.evidence_start = data.evidence_start,
        he.evidence_end = data.evidence_end,
        he.primary_unit_id = data.primary_unit_id,
        he.primary_unit_kind = data.primary_unit_kind,
        he.primary_unit_start = data.primary_unit_start,
        he.primary_unit_end = data.primary_unit_end,
        he.paragraph_unit_id = data.paragraph_unit_id,
        he.paragraph_start = data.paragraph_start,
        he.paragraph_end = data.paragraph_end,
        he.sentence_unit_id = data.sentence_unit_id,
        he.sentence_start = data.sentence_start,
        he.sentence_end = data.sentence_end,
        he.doc_id = coalesce(data.doc_id, he.doc_id),
        he.kg_scope = coalesce(data.kg_scope, he.kg_scope),
        he.kg_id = coalesce(data.kg_id, he.kg_id)

    WITH c, n, data

    MERGE (eu:EvidenceUnit {id: data.primary_unit_id})
    SET eu.parent_evidence_id = data.parent_evidence_id,
        eu.unit_kind = data.primary_unit_kind,
        eu.start_char = data.primary_unit_start,
        eu.end_char = data.primary_unit_end,
        eu.chunk_id = data.chunk_id,
        eu.fileName = coalesce(data.file_name, eu.fileName),
        eu.doc_id = coalesce(data.doc_id, eu.doc_id),
        eu.kg_scope = coalesce(data.kg_scope, eu.kg_scope),
        eu.kg_id = coalesce(data.kg_id, eu.kg_id)

    MERGE (eu)-[:PART_OF]->(c)

    MERGE (n)-[sb:SUPPORTED_BY]->(eu)
    SET sb.primary = true,
        sb.parent_evidence_id = data.parent_evidence_id,
        sb.unit_kind = data.primary_unit_kind,
        sb.evidence_unit_id = data.primary_unit_id,
        sb.evidence_start = data.evidence_start,
        sb.evidence_end = data.evidence_end,
        sb.evidence_text = data.evidence_text,
        sb.doc_id = coalesce(data.doc_id, sb.doc_id),
        sb.kg_scope = coalesce(data.kg_scope, sb.kg_scope),
        sb.kg_id = coalesce(data.kg_id, sb.kg_id)

    WITH c, data

    CALL apoc.do.when(
        data.doc_id IS NULL,
        'WITH $c AS c MATCH (d:Document {fileName: $file_name}) MERGE (c)-[:PART_OF]->(d) RETURN 1 as _',
        'WITH $c AS c MATCH (d:Document {doc_id: $doc_id}) MERGE (c)-[:PART_OF]->(d) RETURN 1 as _',
        {c: c, file_name: data.file_name, doc_id: data.doc_id}
    ) YIELD value

    RETURN count(*)
    """

    graph.query(primary_query, params={"batch": primary_batch})

    # --------------------------
    # 2) secondary: secondary EvidenceUnit + secondary SUPPORTED_BY (primary=false)
    # --------------------------

    if secondary_batch:
        secondary_query = """
        UNWIND $batch AS data
        MERGE (c:Chunk {id: data.chunk_id})
        SET c.fileName = coalesce(data.file_name, c.fileName),
            c.doc_id = coalesce(data.doc_id, c.doc_id),
            c.kg_scope = coalesce(data.kg_scope, c.kg_scope),
            c.kg_id = coalesce(data.kg_id, c.kg_id)
        WITH c, data

        CALL apoc.merge.node([data.node_type], {id: data.node_id}) YIELD node AS n

        MERGE (eu:EvidenceUnit {id: data.unit_id})
        SET eu.parent_evidence_id = data.parent_evidence_id,
            eu.unit_kind = data.unit_kind,
            eu.start_char = data.unit_start,
            eu.end_char = data.unit_end,
            eu.chunk_id = data.chunk_id,
            eu.fileName = coalesce(data.file_name, eu.fileName),
            eu.doc_id = coalesce(data.doc_id, eu.doc_id),
            eu.kg_scope = coalesce(data.kg_scope, eu.kg_scope),
            eu.kg_id = coalesce(data.kg_id, eu.kg_id)

        MERGE (eu)-[:PART_OF]->(c)

        MERGE (n)-[sb:SUPPORTED_BY]->(eu)
        SET sb.primary = false,
            sb.parent_evidence_id = data.parent_evidence_id,
            sb.unit_kind = data.unit_kind,
            sb.evidence_unit_id = data.unit_id,
            sb.evidence_start = data.evidence_start,
            sb.evidence_end = data.evidence_end,
            sb.evidence_text = data.evidence_text,
            sb.doc_id = coalesce(data.doc_id, sb.doc_id),
            sb.kg_scope = coalesce(data.kg_scope, sb.kg_scope),
            sb.kg_id = coalesce(data.kg_id, sb.kg_id)

        RETURN count(*)
        """
        graph.query(secondary_query, params={"batch": secondary_batch})


def update_embedding_create_vector_index(graph, chunkId_chunkDoc_list, file_name, doc_id: Optional[str] = None):
    # create embedding
    isEmbedding = os.getenv("IS_EMBEDDING")
    embedding_model = os.getenv("EMBEDDING_MODEL")

    embeddings, dimension = load_embedding_model(embedding_model)
    logging.info(f"embedding model:{embeddings} and dimesion:{dimension}")
    data_for_query = []
    logging.info(f"update embedding and vector index for chunks")

    for row in chunkId_chunkDoc_list:
        if isEmbedding and isEmbedding.upper() == "TRUE":
            embeddings_arr = embeddings.embed_query(row["chunk_doc"].page_content)
            data_for_query.append({"chunkId": row["chunk_id"], "embeddings": embeddings_arr})

            graph.query(
                """CREATE VECTOR INDEX `vector` if not exists for (c:Chunk) on (c.embedding)
                            OPTIONS {indexConfig: {
                            `vector.dimensions`: $dimensions,
                            `vector.similarity_function`: 'cosine'
                            }}
                        """,
                {"dimensions": dimension},
            )

    if not data_for_query:
        return

    if doc_id:
        query_to_create_embedding = """
            UNWIND $data AS row
            MATCH (d:Document {doc_id: $doc_id})
            MERGE (c:Chunk {id: row.chunkId})
            SET c.embedding = row.embeddings
            MERGE (c)-[:PART_OF]->(d)
        """
        graph.query(query_to_create_embedding, params={"doc_id": doc_id, "data": data_for_query})
    else:
        query_to_create_embedding = """
            UNWIND $data AS row
            MATCH (d:Document {fileName: $fileName})
            MERGE (c:Chunk {id: row.chunkId})
            SET c.embedding = row.embeddings
            MERGE (c)-[:PART_OF]->(d)
        """
        graph.query(query_to_create_embedding, params={"fileName": file_name, "data": data_for_query})


def create_relation_between_chunks(
    graph,
    file_name: str,
    chunks: List[Document],
    doc_id: Optional[str] = None,
    kg_scope: Optional[str] = None,
    kg_id: Optional[str] = None,
):
    """将文档所有 chunks 合并为一个 KG Chunk 节点，但把正文存到证据库（SQLite）。

    - Neo4j: 只存 Chunk.id / evidence_id / position / length / offsets / embedding 等
    - EvidenceStore(SQLite): 存 evidence_id -> content（全文或合并片段）

    这样可以避免 KG 过度膨胀，同时为“可追溯证据”提供独立存储。

    重要：Chunk.id / evidence_id 使用 doc_id 前缀，避免 BG/Instance 或多文档冲突。
    """

    logging.info("Merging all chunks into one for Knowledge Graph (KG<->Evidence split enabled)")

    combined_content = " ".join([chunk.page_content for chunk in chunks])
    base_hash = hashlib.sha1(combined_content.encode()).hexdigest()
    current_chunk_id = f"{doc_id}|{base_hash}" if doc_id else base_hash

    position = 1
    length = len(combined_content)
    offset = 0

    metadata = {
        "position": position,
        "length": length,
        "content_offset": offset,
        "fileName": file_name,
        "combined_chunk_ids": [current_chunk_id],
        "doc_id": doc_id,
        "kg_scope": kg_scope,
        "kg_id": kg_id,
    }
    chunk_document = Document(page_content=combined_content, metadata=metadata)

    # 1) 写入证据库（SQLite）
    evidence_store = EvidenceStore()
    evidence_store.upsert_chunk(
        evidence_id=current_chunk_id,
        file_name=file_name,
        content=combined_content,
        position=position,
        meta={
            "mode": "merged_all_chunks",
            "length": length,
            "content_offset": offset,
            "doc_id": doc_id,
            "kg_scope": kg_scope,
            "kg_id": kg_id,
        },
    )

    # 2) Neo4j 只存指针，不存全文
    batch_data = [
        {
            "id": current_chunk_id,
            "evidence_id": current_chunk_id,
            "position": position,
            "length": length,
            "f_name": file_name,
            "content_offset": offset,
            "doc_id": doc_id,
            "kg_scope": kg_scope,
            "kg_id": kg_id,
        }
    ]

    query = """
    UNWIND $batch_data AS data
    MERGE (c:Chunk {id: data.id})
    SET c.evidence_id = data.evidence_id,
        c.position = data.position,
        c.length = data.length,
        c.fileName = data.f_name,
        c.content_offset = data.content_offset,
        c.doc_id = coalesce(data.doc_id, c.doc_id),
        c.kg_scope = coalesce(data.kg_scope, c.kg_scope),
        c.kg_id = coalesce(data.kg_id, c.kg_id)
    WITH c, data

    CALL apoc.do.when(
        data.doc_id IS NULL,
        'WITH $c AS c MATCH (d:Document {fileName: $file_name}) MERGE (c)-[:PART_OF]->(d) RETURN 1 as _',
        'WITH $c AS c MATCH (d:Document {doc_id: $doc_id}) MERGE (c)-[:PART_OF]->(d) RETURN 1 as _',
        {c: c, file_name: data.f_name, doc_id: data.doc_id}
    ) YIELD value

    RETURN count(*)
    """
    graph.query(query, params={"batch_data": batch_data})

    # FIRST_CHUNK
    if doc_id:
        graph.query(
            """
            MATCH (d:Document {doc_id: $doc_id})
            MATCH (c:Chunk {id: $chunk_id})
            MERGE (d)-[:FIRST_CHUNK]->(c)
            """,
            params={"doc_id": doc_id, "chunk_id": current_chunk_id},
        )
    else:
        graph.query(
            """
            MATCH (d:Document {fileName: $f_name})
            MATCH (c:Chunk {id: $chunk_id})
            MERGE (d)-[:FIRST_CHUNK]->(c)
            """,
            params={"f_name": file_name, "chunk_id": current_chunk_id},
        )

    return [{"chunk_id": current_chunk_id, "chunk_doc": chunk_document}]

