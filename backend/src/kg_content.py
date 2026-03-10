"""KG scoping utilities (BG-KG / Instance-KG minimal separation).

Goal
----
Make it possible to build multiple logical KGs in ONE Neo4j database without
node/edge collisions, by introducing a (kg_scope, kg_id, doc_id) context.

- kg_scope: 'bg' (background) or 'inst' (instance / per-report)
- kg_id:    a dataset/project identifier inside the scope (e.g. 'safety_std', 'default')
- doc_id:   a document identifier inside (kg_scope, kg_id)

Recommended identifiers
-----------------------
- doc_id  = f"{kg_scope}|{kg_id}|{file_name_clean}"
- chunk_id = f"{doc_id}|{sha1(text)}"  (keeps Chunk/Evidence ids unique per doc)

Entity uid strategy (important!)
--------------------------------
To avoid Instance-KG entities from merging across documents, we scope entity IDs
by doc_id when kg_scope == 'inst'. For BG-KG, we scope by (kg_scope, kg_id)
so background entities can merge across background documents.

This is intentionally a *minimal* strategy: it changes only IDs/properties,
while keeping the existing upload/extract flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


def _clean_part(x: str) -> str:
    """Prevent delimiter collisions in ids."""
    if x is None:
        return ""
    return str(x).replace("|", "_").strip()


@dataclass(frozen=True)
class KGContext:
    kg_scope: str
    kg_id: str
    file_name: str

    @property
    def file_name_clean(self) -> str:
        return _clean_part(self.file_name)

    @property
    def doc_id(self) -> str:
        return f"{self.kg_scope}|{self.kg_id}|{self.file_name_clean}"

    def entity_uid(self, local_id: str) -> str:
        """Return the Neo4j MERGE key used by entity nodes.

        - Instance KG: doc-scoped (avoid cross-report merge)
        - BG KG: scope+kg_id-scoped (allow merge across BG documents)
        """
        local = _clean_part(local_id)
        if self.kg_scope == "inst":
            return f"{self.doc_id}|{local}"
        return f"{self.kg_scope}|{self.kg_id}|{local}"

    def chunk_uid(self, chunk_hash: str) -> str:
        return f"{self.doc_id}|{_clean_part(chunk_hash)}"


def build_kg_context(
    kg_scope: Optional[str],
    kg_id: Optional[str],
    file_name: str,
    default_scope: str = "inst",
    default_kg_id: str = "default",
) -> KGContext:
    scope = (kg_scope or default_scope).strip().lower()
    kid = (kg_id or default_kg_id).strip()
    return KGContext(kg_scope=scope, kg_id=kid, file_name=file_name)


def scope_graph_documents(graph_documents: List[Any], ctx: KGContext) -> List[Any]:
    """In-place scope GraphDocument nodes/edges ids, and attach kg properties.

    This should be called **after** LLM extraction (GraphDocuments created),
    and **before** save_graphDocuments_in_neo4j() to avoid collisions.

    We do NOT change node labels/types, only node.id (MERGE key) and
    add properties for traceability.
    """

    if not graph_documents:
        return graph_documents

    # Build mapping from original node id -> new uid for each graph_doc
    for gd in graph_documents:
        nodes = getattr(gd, "nodes", None) or []
        rels = getattr(gd, "relationships", None) or []

        id_map: Dict[str, str] = {}
        for n in nodes:
            old_id = getattr(n, "id", None)
            if old_id is None:
                continue
            new_id = ctx.entity_uid(str(old_id))
            id_map[str(old_id)] = new_id

            # Attach properties
            props = getattr(n, "properties", None)
            if not isinstance(props, dict):
                props = {}
            props.setdefault("orig_id", str(old_id))
            props["kg_scope"] = ctx.kg_scope
            props["kg_id"] = ctx.kg_id
            props["doc_id"] = ctx.doc_id
            setattr(n, "properties", props)

            # Overwrite MERGE id
            setattr(n, "id", new_id)

        # Update relationship endpoints
        for r in rels:
            s = getattr(r, "source", None)
            t = getattr(r, "target", None)
            if s is not None and getattr(s, "id", None) in id_map:
                setattr(s, "id", id_map[getattr(s, "id")])
            if t is not None and getattr(t, "id", None) in id_map:
                setattr(t, "id", id_map[getattr(t, "id")])

            # Attach rel properties too (optional but useful)
            rprops = getattr(r, "properties", None)
            if not isinstance(rprops, dict):
                rprops = {}
            rprops["kg_scope"] = ctx.kg_scope
            rprops["kg_id"] = ctx.kg_id
            rprops["doc_id"] = ctx.doc_id
            setattr(r, "properties", rprops)

        # Attach ctx into source metadata for later evidence writing
        src = getattr(gd, "source", None)
        if src is not None:
            md = getattr(src, "metadata", None)
            if not isinstance(md, dict):
                md = {}
            md["kg_scope"] = ctx.kg_scope
            md["kg_id"] = ctx.kg_id
            md["doc_id"] = ctx.doc_id
            setattr(src, "metadata", md)

    return graph_documents
