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


def persist_evidence_units(
    graph: Neo4jGraph,
    evidence_store: Optional[EvidenceStore],
    evidence_units: List[Any],
    file_name: str,
    doc_id: Optional[str],
    kg_scope: Optional[str],
    kg_id: Optional[str],
):
    """Persist evidence-unit content to SQLite and create lightweight Neo4j nodes."""
    if not evidence_units:
        return
    store = evidence_store or EvidenceStore()
    store.upsert_evidence_units(evidence_units)

    batch = []
    for unit in evidence_units:
        payload = unit.__dict__ if hasattr(unit, "__dict__") else dict(unit)
        page_range = payload.get("page_range")
        batch.append({
            "unit_id": payload.get("unit_id"),
            "parent_chunk_id": payload.get("parent_chunk_id") or payload.get("parent_evidence_id"),
            "unit_kind": payload.get("unit_kind"),
            "start_char": payload.get("start_char"),
            "end_char": payload.get("end_char"),
            "file_name": payload.get("file_name") or file_name,
            "doc_id": payload.get("report_id") or doc_id,
            "kg_scope": (payload.get("meta") or {}).get("kg_scope") or kg_scope,
            "kg_id": (payload.get("meta") or {}).get("kg_id") or kg_id,
            "page_start": page_range[0] if page_range else None,
            "page_end": page_range[1] if page_range else None,
            "section_name": payload.get("section_name"),
            "trigger_words": payload.get("trigger_words") or [],
            "temporal_cues": payload.get("temporal_cues") or [],
            "actors": payload.get("actors") or [],
            "prev_unit_id": payload.get("prev_unit_id"),
            "next_unit_id": payload.get("next_unit_id"),
        })

    graph.query("""
    UNWIND $batch AS data
    MERGE (eu:EvidenceUnit {id: data.unit_id})
    SET eu.parent_evidence_id = data.parent_chunk_id,
        eu.unit_kind = data.unit_kind,
        eu.start_char = data.start_char,
        eu.end_char = data.end_char,
        eu.fileName = data.file_name,
        eu.doc_id = data.doc_id,
        eu.kg_scope = data.kg_scope,
        eu.kg_id = data.kg_id,
        eu.page_start = data.page_start,
        eu.page_end = data.page_end,
        eu.section_name = data.section_name,
        eu.trigger_words = data.trigger_words,
        eu.temporal_cues = data.temporal_cues,
        eu.actors = data.actors
    RETURN count(*)
    """, params={"batch": batch})

    link_chunk_to_evidence_units(graph, evidence_units)


def link_chunk_to_evidence_units(graph: Neo4jGraph, evidence_units: List[Any]):
    if not evidence_units:
        return
    batch = []
    next_edges = []
    for unit in evidence_units:
        payload = unit.__dict__ if hasattr(unit, "__dict__") else dict(unit)
        batch.append({"chunk_id": payload.get("parent_chunk_id") or payload.get("parent_evidence_id"), "unit_id": payload.get("unit_id")})
        if payload.get("next_unit_id"):
            next_edges.append({"source_id": payload.get("unit_id"), "target_id": payload.get("next_unit_id")})
    graph.query("""
    UNWIND $batch AS data
    MATCH (c:Chunk {id: data.chunk_id})
    MATCH (eu:EvidenceUnit {id: data.unit_id})
    MERGE (c)-[:HAS_EVIDENCE_UNIT]->(eu)
    RETURN count(*)
    """, params={"batch": batch})
    if next_edges:
        graph.query("""
        UNWIND $batch AS data
        MATCH (a:EvidenceUnit {id: data.source_id})
        MATCH (b:EvidenceUnit {id: data.target_id})
        MERGE (a)-[:NEXT_EVIDENCE_UNIT]->(b)
        RETURN count(*)
        """, params={"batch": next_edges})


def link_entities_to_evidence_units(
    graph: Neo4jGraph,
    graph_documents_chunk_chunk_Id: list,
    evidence_units_by_chunk: Optional[Dict[str, List[Any]]] = None,
):
    logging.info("Create HAS_ENTITY / SUPPORTED_BY relationships using evidence units")
    evidence_store = EvidenceStore()
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
            continue
        if not chunk_ids and graph_doc.source and graph_doc.source.metadata:
            metadata_chunk_ids = graph_doc.source.metadata.get("combined_chunk_ids")
            if metadata_chunk_ids:
                chunk_ids = metadata_chunk_ids if isinstance(metadata_chunk_ids, list) else [metadata_chunk_ids]
        for cid in chunk_ids:
            pairs.append({"chunk_id": cid, "graph_doc": graph_doc})

    primary_batch: List[Dict[str, Any]] = []
    for pair in pairs:
        chunk_id = pair["chunk_id"]
        graph_doc = pair["graph_doc"]
        md = graph_doc.source.metadata if (graph_doc.source and graph_doc.source.metadata) else {}
        file_name = md.get("fileName")
        scope_fields = _infer_doc_fields(chunk_id, md)
        chunk_text = evidence_store.get_content(chunk_id) or ""
        units = evidence_units_by_chunk.get(chunk_id, []) if evidence_units_by_chunk else []

        for node in graph_doc.nodes:
            snippet = _pick_evidence_snippet(node)
            start, end = find_best_span(chunk_text, snippet)
            if start is None:
                start, end = find_best_span(chunk_text, str(node.id))

            best_unit = None
            for unit in units:
                payload = unit.__dict__ if hasattr(unit, "__dict__") else dict(unit)
                s = payload.get("start_char")
                e = payload.get("end_char")
                if s is None or e is None:
                    continue
                overlap = min(e, end or e) - max(s, start or s)
                if overlap >= 0:
                    best_unit = payload
                    break
            if best_unit is None and units:
                sample = units[0]
                best_unit = sample.__dict__ if hasattr(sample, "__dict__") else dict(sample)

            primary_batch.append({
                "chunk_id": chunk_id,
                "node_type": node.type,
                "node_id": node.id,
                "file_name": file_name,
                "doc_id": scope_fields["doc_id"],
                "kg_scope": scope_fields["kg_scope"],
                "kg_id": scope_fields["kg_id"],
                "evidence_text": snippet,
                "evidence_start": start,
                "evidence_end": end,
                "unit_id": best_unit.get("unit_id") if best_unit else f"{chunk_id}::chunk",
                "unit_kind": best_unit.get("unit_kind") if best_unit else "chunk",
                "unit_start": best_unit.get("start_char") if best_unit else start,
                "unit_end": best_unit.get("end_char") if best_unit else end,
                "parent_evidence_id": chunk_id,
            })

    if not primary_batch:
        logging.warning("No evidence-linked entities were generated")
        return

    graph.query("""
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
        he.primary_unit_id = data.unit_id,
        he.primary_unit_kind = data.unit_kind,
        he.doc_id = coalesce(data.doc_id, he.doc_id),
        he.kg_scope = coalesce(data.kg_scope, he.kg_scope),
        he.kg_id = coalesce(data.kg_id, he.kg_id)
    WITH n, data
    MATCH (eu:EvidenceUnit {id: data.unit_id})
    MERGE (n)-[sb:SUPPORTED_BY]->(eu)
    SET sb.primary = true,
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
    """, params={"batch": primary_batch})


def merge_relationship_between_chunk_and_entites(
    graph: Neo4jGraph,
    graph_documents_chunk_chunk_Id: list,
    evidence_units_by_chunk: Optional[Dict[str, List[Any]]] = None,
):
    link_entities_to_evidence_units(graph, graph_documents_chunk_chunk_Id, evidence_units_by_chunk)


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
    """Create fixed chunks for retrieval/vector indexing/raw storage only.

    Evidence units are built later from these chunk records.
    """

    logging.info("Creating fixed chunk nodes and storing chunk text in SQLite evidence store")

    evidence_store = EvidenceStore()
    batch_data = []
    chunk_records: List[Dict[str, Any]] = []
    offset = 0

    for index, chunk in enumerate(chunks):
        content = chunk.page_content or ""
        base_hash = hashlib.sha1(f"{index}:{content}".encode()).hexdigest()
        current_chunk_id = f"{doc_id}|{base_hash}" if doc_id else base_hash
        metadata = dict(chunk.metadata or {})
        metadata.update(
            {
                "id": current_chunk_id,
                "position": index + 1,
                "length": len(content),
                "content_offset": offset,
                "fileName": file_name,
                "file_name": file_name,
                "doc_id": doc_id,
                "kg_scope": kg_scope,
                "kg_id": kg_id,
                "combined_chunk_ids": [current_chunk_id],
            }
        )
        chunk_document = Document(page_content=content, metadata=metadata)
        evidence_store.upsert_chunk(
            evidence_id=current_chunk_id,
            file_name=file_name,
            content=content,
            position=index + 1,
            meta={
                "mode": "fixed_chunk",
                "length": len(content),
                "content_offset": offset,
                "chunk_index": metadata.get("chunk_index", index),
                "page": metadata.get("page") or metadata.get("page_number"),
                "doc_id": doc_id,
                "kg_scope": kg_scope,
                "kg_id": kg_id,
            },
        )
        batch_data.append(
            {
                "id": current_chunk_id,
                "evidence_id": current_chunk_id,
                "position": index + 1,
                "length": len(content),
                "f_name": file_name,
                "content_offset": offset,
                "doc_id": doc_id,
                "kg_scope": kg_scope,
                "kg_id": kg_id,
            }
        )
        chunk_records.append({"chunk_id": current_chunk_id, "chunk_doc": chunk_document})
        offset += len(content)

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
    first_chunk_id = chunk_records[0]["chunk_id"] if chunk_records else None
    if first_chunk_id and doc_id:
        graph.query(
            """
            MATCH (d:Document {doc_id: $doc_id})
            MATCH (c:Chunk {id: $chunk_id})
            MERGE (d)-[:FIRST_CHUNK]->(c)
            """,
            params={"doc_id": doc_id, "chunk_id": first_chunk_id},
        )
    elif first_chunk_id:
        graph.query(
            """
            MATCH (d:Document {fileName: $f_name})
            MATCH (c:Chunk {id: $chunk_id})
            MERGE (d)-[:FIRST_CHUNK]->(c)
            """,
            params={"f_name": file_name, "chunk_id": first_chunk_id},
        )

    if len(chunk_records) > 1:
        graph.query(
            """
            UNWIND range(0, size($chunk_ids) - 2) AS idx
            MATCH (a:Chunk {id: $chunk_ids[idx]})
            MATCH (b:Chunk {id: $chunk_ids[idx + 1]})
            MERGE (a)-[:NEXT_CHUNK]->(b)
            RETURN count(*)
            """,
            params={"chunk_ids": [item["chunk_id"] for item in chunk_records]},
        )

    return chunk_records

