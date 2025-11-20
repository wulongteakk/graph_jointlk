from langchain_community.graphs import Neo4jGraph
from langchain.docstore.document import Document
from src.shared.common_fn import load_embedding_model
import logging
from typing import List
import os
import hashlib
import time


logging.basicConfig(format='%(asctime)s - %(message)s',level='INFO')

def merge_relationship_between_chunk_and_entites(graph: Neo4jGraph, graph_documents_chunk_chunk_Id : list):
    batch_data = []
    logging.info("Create HAS_ENTITY relationship between chunks and entities")

    for item in graph_documents_chunk_chunk_Id:
        chunk_ids = []
        graph_doc = None

        # 支持当前的 dict 结构和历史的 (chunk_ids, graph_doc) 元组结构
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

        # 回退：如果未显式传入 chunk_id，则从 GraphDocument 的 metadata 中读取
        if not chunk_ids and graph_doc.source and graph_doc.source.metadata:
            metadata_chunk_ids = graph_doc.source.metadata.get("combined_chunk_ids")
            if metadata_chunk_ids:
                chunk_ids = metadata_chunk_ids if isinstance(metadata_chunk_ids, list) else [metadata_chunk_ids]

        file_name = None
        if graph_doc.source and graph_doc.source.metadata:
            file_name = graph_doc.source.metadata.get("fileName")

        for chunk_id in chunk_ids:
            for node in graph_doc.nodes:
                batch_data.append({
                    "chunk_id": chunk_id,
                    "node_type": node.type,
                    "node_id": node.id,
                    "file_name": file_name
                })

    if batch_data:
        unwind_query = """
                    UNWIND $batch_data AS data
                    MERGE (c:Chunk {id: data.chunk_id})
                    SET c.fileName = coalesce(data.file_name, c.fileName)
                    WITH c, data
                    CALL apoc.merge.node([data.node_type], {id: data.node_id}) YIELD node AS n
                    MERGE (c)-[:HAS_ENTITY]->(n)
                    WITH c, data
                    CALL apoc.do.when(
                        data.file_name IS NULL,
                        'RETURN 0 as _',
                        'MATCH (d:Document {fileName: $file_name}) MERGE (c)-[:PART_OF]->(d) RETURN 1 as _',
                        {c: c, file_name: data.file_name}
                    ) YIELD value
                    RETURN count(*)
                """
        graph.query(unwind_query, params={"batch_data": batch_data})



def update_embedding_create_vector_index(graph, chunkId_chunkDoc_list, file_name):
    #create embedding
    isEmbedding = os.getenv('IS_EMBEDDING')
    embedding_model = os.getenv('EMBEDDING_MODEL')

    embeddings, dimension = load_embedding_model(embedding_model)
    logging.info(f'embedding model:{embeddings} and dimesion:{dimension}')
    data_for_query = []
    logging.info(f"update embedding and vector index for chunks")
    logging.info(f"看看chunkId_chunkDoc_list{chunkId_chunkDoc_list}")
    for row in chunkId_chunkDoc_list:
        # for graph_document in row['graph_doc']:
        if isEmbedding.upper() == "TRUE":
            embeddings_arr = embeddings.embed_query(row['chunk_doc'].page_content)
            # logging.info(f'Embedding list {embeddings_arr}')

            data_for_query.append({
                "chunkId": row['chunk_id'],
                "embeddings": embeddings_arr
            })
            # graph.query("""MATCH (d:Document {fileName : $fileName})
            #                MERGE (c:Chunk {id:$chunkId}) SET c.embedding = $embeddings
            #                MERGE (c)-[:PART_OF]->(d)
            #             """,
            #             {
            #                 "fileName" : file_name,
            #                 "chunkId": row['chunk_id'],
            #                 "embeddings" : embeddings_arr
            #             }
            #             )
            # logging.info('create vector index on chunk embedding')

            graph.query("""CREATE VECTOR INDEX `vector` if not exists for (c:Chunk) on (c.embedding)
                            OPTIONS {indexConfig: {
                            `vector.dimensions`: $dimensions,
                            `vector.similarity_function`: 'cosine'
                            }}
                        """,
                        {
                            "dimensions" : dimension
                        }
                        )

    query_to_create_embedding = """
        UNWIND $data AS row
        MATCH (d:Document {fileName: $fileName})
        MERGE (c:Chunk {id: row.chunkId})
        SET c.embedding = row.embeddings
        MERGE (c)-[:PART_OF]->(d)
    """
    graph.query(query_to_create_embedding, params={"fileName":file_name, "data":data_for_query})

# def create_relation_between_chunks(graph, file_name, chunks: List[Document])->list:
#     logging.info("creating FIRST_CHUNK and NEXT_CHUNK relationships between chunks")
#     current_chunk_id = ""
#     lst_chunks_including_hash = []
#     batch_data = []
#     relationships = []
#     offset=0
#     for i, chunk in enumerate(chunks):
#         page_content_sha1 = hashlib.sha1(chunk.page_content.encode())
#         previous_chunk_id = current_chunk_id
#         current_chunk_id = page_content_sha1.hexdigest()
#         position = i + 1
#         if i>0:
#             #offset += len(tiktoken.encoding_for_model("gpt2").encode(chunk.page_content))
#             offset += len(chunks[i-1].page_content)
#         if i == 0:
#             firstChunk = True
#         else:
#             firstChunk = False
#         metadata = {"position": position,"length": len(chunk.page_content), "content_offset":offset}
#         chunk_document = Document(
#             page_content=chunk.page_content, metadata=metadata
#         )
#
#         chunk_data = {
#             "id": current_chunk_id,
#             "pg_content": chunk_document.page_content,
#             "position": position,
#             "length": chunk_document.metadata["length"],
#             "f_name": file_name,
#             "previous_id" : previous_chunk_id,
#             "content_offset" : offset
#         }
#
#         if 'page_number' in chunk.metadata:
#             chunk_data['page_number'] = chunk.metadata['page_number']
#
#         if 'start_time' in chunk.metadata and 'end_time' in chunk.metadata:
#             chunk_data['start_time'] = chunk.metadata['start_time']
#             chunk_data['end_time'] = chunk.metadata['end_time']
#
#         batch_data.append(chunk_data)
#
#         lst_chunks_including_hash.append({'chunk_id': current_chunk_id, 'chunk_doc': chunk})
#
#         # create relationships between chunks
#         if firstChunk:
#             relationships.append({"type": "FIRST_CHUNK", "chunk_id": current_chunk_id})
#         else:
#             relationships.append({
#                 "type": "NEXT_CHUNK",
#                 "previous_chunk_id": previous_chunk_id,  # ID of previous chunk
#                 "current_chunk_id": current_chunk_id
#             })
#
#     query_to_create_chunk_and_PART_OF_relation = """
#         UNWIND $batch_data AS data
#         MERGE (c:Chunk {id: data.id})
#         SET c.text = data.pg_content, c.position = data.position, c.length = data.length, c.fileName=data.f_name, c.content_offset=data.content_offset
#         WITH data, c
#         SET c.page_number = CASE WHEN data.page_number IS NOT NULL THEN data.page_number END,
#             c.start_time = CASE WHEN data.start_time IS NOT NULL THEN data.start_time END,
#             c.end_time = CASE WHEN data.end_time IS NOT NULL THEN data.end_time END
#         WITH data, c
#         MATCH (d:Document {fileName: data.f_name})
#         MERGE (c)-[:PART_OF]->(d)
#     """
#     graph.query(query_to_create_chunk_and_PART_OF_relation, params={"batch_data": batch_data})
#
#     query_to_create_FIRST_relation = """
#         UNWIND $relationships AS relationship
#         MATCH (d:Document {fileName: $f_name})
#         MATCH (c:Chunk {id: relationship.chunk_id})
#         FOREACH(r IN CASE WHEN relationship.type = 'FIRST_CHUNK' THEN [1] ELSE [] END |
#                 MERGE (d)-[:FIRST_CHUNK]->(c))
#         """
#     graph.query(query_to_create_FIRST_relation, params={"f_name": file_name, "relationships": relationships})
#
#     query_to_create_NEXT_CHUNK_relation = """
#         UNWIND $relationships AS relationship
#         MATCH (c:Chunk {id: relationship.current_chunk_id})
#         WITH c, relationship
#         MATCH (pc:Chunk {id: relationship.previous_chunk_id})
#         FOREACH(r IN CASE WHEN relationship.type = 'NEXT_CHUNK' THEN [1] ELSE [] END |
#                 MERGE (c)<-[:NEXT_CHUNK]-(pc))
#         """
#     graph.query(query_to_create_NEXT_CHUNK_relation, params={"relationships": relationships})
#
#     return lst_chunks_including_hash

def create_relation_between_chunks(graph, file_name, chunks):
    logging.info("Merging all chunks into one for Knowledge Graph")
    # 合并所有chunks的内容
    combined_content = " ".join([chunk.page_content for chunk in chunks])
    page_content_sha1 = hashlib.sha1(combined_content.encode())
    current_chunk_id = page_content_sha1.hexdigest()
    original_chunk_ids = [hashlib.sha1(chunk.page_content.encode()).hexdigest() for chunk in chunks]

    position = 1
    length = len(combined_content)
    offset = 0
    metadata = {
        "position": position,
        "length": length,
        "content_offset": offset,
        "fileName": file_name,
        "combined_chunk_ids": original_chunk_ids + [current_chunk_id],
    }
    chunk_document = Document(
        page_content=combined_content, metadata=metadata
    )

    chunk_data = {
        "id": current_chunk_id,
        "pg_content": chunk_document.page_content,
        "position": position,
        "length": chunk_document.metadata["length"],
        "f_name": file_name,
        "previous_id": "",
        "content_offset": offset
    }

    if 'page_number' in chunks[0].metadata:
        chunk_data['page_number'] = chunks[0].metadata['page_number']
    if 'start_time' in chunks[0].metadata and 'end_time' in chunks[0].metadata:
        chunk_data['start_time'] = chunks[0].metadata['start_time']
        chunk_data['end_time'] = chunks[0].metadata['end_time']

    batch_data = [chunk_data]

    lst_chunks_including_hash = [{'chunk_id': current_chunk_id, 'chunk_doc': chunk_document}]

    # 创建合并后chunk与Document的PART_OF关系
    query_to_create_chunk_and_PART_OF_relation = """
        UNWIND $batch_data AS data
        MERGE (c:Chunk {id: data.id})
        SET c.text = data.pg_content, c.position = data.position, c.length = data.length, c.fileName=data.f_name, c.content_offset=data.content_offset
        WITH data, c
        SET c.page_number = CASE WHEN data.page_number IS NOT NULL THEN data.page_number END,
            c.start_time = CASE WHEN data.start_time IS NOT NULL THEN data.start_time END,
            c.end_time = CASE WHEN data.end_time IS NOT NULL THEN data.end_time END
        WITH data, c
        MATCH (d:Document {fileName: data.f_name})
        MERGE (c)-[:PART_OF]->(d)
    """
    graph.query(query_to_create_chunk_and_PART_OF_relation, params={"batch_data": batch_data})

    # 创建FIRST_CHUNK关系
    query_to_create_FIRST_relation = """ 
        UNWIND $relationships AS relationship
        MATCH (d:Document {fileName: $f_name})
        MATCH (c:Chunk {id: relationship.chunk_id})
        FOREACH(r IN CASE WHEN relationship.type = 'FIRST_CHUNK' THEN [1] ELSE [] END |
                MERGE (d)-[:FIRST_CHUNK]->(c))
        """
    relationships = [{"type": "FIRST_CHUNK", "chunk_id": current_chunk_id}]
    graph.query(query_to_create_FIRST_relation, params={"f_name": file_name, "relationships": relationships})


    original_chunk_ids = [hashlib.sha1(chunk.page_content.encode()).hexdigest() for chunk in chunks]

    # 将原始chunks的实体和关系迁移到合并后的chunk上
    query_to_migrate_entities = """
        UNWIND $original_chunk_ids AS original_chunk_id
        MATCH (oc:Chunk {id: original_chunk_id})-[r:HAS_ENTITY]->(e)
        MERGE (nc:Chunk {id: $new_chunk_id})-[:HAS_ENTITY]->(e)
        DELETE r
    """
    graph.query(query_to_migrate_entities, params={"original_chunk_ids": original_chunk_ids, "new_chunk_id": current_chunk_id})

    query_to_migrate_part_of = """
        UNWIND $original_chunk_ids AS original_chunk_id
        MATCH (oc:Chunk {id: original_chunk_id})-[:PART_OF]->(d:Document)
        MERGE (nc:Chunk {id: $new_chunk_id})-[:PART_OF]->(d)
    """
    graph.query(query_to_migrate_part_of, params={"original_chunk_ids": original_chunk_ids, "new_chunk_id": current_chunk_id})

    # 迁移完成后再安全删除旧的 Chunk 节点（避免删除当前合并的 Chunk）
    chunk_ids_to_delete = [cid for cid in original_chunk_ids if cid != current_chunk_id]
    if chunk_ids_to_delete:
        query_to_delete_original_chunks = """
            UNWIND $chunk_ids AS chunk_id
            MATCH (c:Chunk {id: chunk_id})
            DETACH DELETE c
        """
        graph.query(query_to_delete_original_chunks, params={"chunk_ids": chunk_ids_to_delete})

    return lst_chunks_including_hash