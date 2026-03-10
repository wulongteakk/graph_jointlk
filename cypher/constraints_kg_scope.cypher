// Optional but recommended constraints for scoped KGs
// Run in Neo4j Browser. They are safe (IF NOT EXISTS).

// Documents should be unique by doc_id (scoped)
CREATE CONSTRAINT document_doc_id_unique IF NOT EXISTS
FOR (d:Document)
REQUIRE d.doc_id IS UNIQUE;

// Chunks should be unique by id (we make it doc_id|hash)
CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
FOR (c:Chunk)
REQUIRE c.id IS UNIQUE;

// Evidence units are scoped by chunk_id prefix, so should be unique too
CREATE CONSTRAINT evidence_unit_id_unique IF NOT EXISTS
FOR (eu:EvidenceUnit)
REQUIRE eu.id IS UNIQUE;
