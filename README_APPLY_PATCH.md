# graph_jointlk BG-KG / Instance-KG scoped patch (minimal)

This bundle contains **only the modified /新增** files needed to:

- Separate BG-KG vs Instance-KG in ONE Neo4j DB using `(kg_scope, kg_id, doc_id)`.
- Scope **Chunk/Evidence** ids by `doc_id` to avoid collisions.
- Scope **entity node ids** (GraphDocuments) via `scope_graph_documents()` to avoid collisions.
- Add `doc_id/kg_scope/kg_id` **properties** onto `EvidenceUnit`, `HAS_ENTITY`, `SUPPORTED_BY` (NOT merge keys).

## Apply

From your repo root:

```bash
unzip -o graph_jointlk_bg_inst_scoped_patch_v2.zip -d .
```

(or copy files manually).

## Neo4j constraints (recommended)

Run `cypher/constraints_kg_scope.cypher` in Neo4j Browser.

## API (minimal compatible changes)

### /upload
Add optional form fields:
- `kg_scope`: `inst` (default) or `bg`
- `kg_id`: default `default` (e.g. `safety_bg`)

### /extract
Add same optional form fields: `kg_scope`, `kg_id`.

If frontend doesn't send them, everything stays in `inst|default|<fileName>`.

## Notes

- Document MERGE key becomes `doc_id` when using the scoped create/update methods.
- For legacy UI/status endpoints that still query by `fileName`, the Document node still keeps `fileName` for display.
