from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv
import os
import logging
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import List
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from src.llm import get_llm
from src.chunk_utils import get_combined_chunks

load_dotenv()
logging.basicConfig(format='%(asctime)s - %(message)s',level='INFO')

def get_graph_from_Groq_Llama3(model_version,
                               graph: Neo4jGraph,
                               chunkId_chunkDoc_list: List,
                               allowedNodes,
                               allowedRelationship):
    """Extract graph documents from Groq Llama3 model."""
    from src.llm import get_llm
    logging.info(f"Get graphDocuments from {model_version}")

    graph_document_list = []
    combined_chunk_document_list = get_combined_chunks(chunkId_chunkDoc_list)
    #api_key = os.environ.get('GROQ_API_KEY') 
    llm, model_name = get_llm(model_version)
    llm_transformer = LLMGraphTransformer(
        llm=llm,
        node_properties=["description"],
        allowed_nodes=allowedNodes,
        allowed_relationships=allowedRelationship,
    )
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for chunk in combined_chunk_document_list:
            chunk_doc = Document(page_content=chunk.page_content.encode("utf-8"), metadata=chunk.metadata)
            futures.append(executor.submit(llm_transformer.convert_to_graph_documents, [chunk_doc]))

        for future in concurrent.futures.as_completed(futures):
            graph_document = future.result()
            graph_document_list.append(graph_document[0])
           
    return  graph_document_list