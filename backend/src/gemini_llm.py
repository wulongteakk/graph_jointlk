from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv

import logging

import google.auth 
from typing import List
from langchain_core.documents import Document
import vertexai

from src.chunk_utils import get_combined_chunks

load_dotenv()
logging.basicConfig(format='%(asctime)s - %(message)s', level='DEBUG')


def get_graph_from_Gemini(model_version,
                          graph: Neo4jGraph,
                          chunkId_chunkDoc_list: List,
                          allowedNodes,
                          allowedRelationship):
    """Extract graph documents from Gemini model."""
    from src.llm import get_graph_document_list, get_llm

    logging.info(f"Get graphDocuments from {model_version}")
    location = "us-central1"
    #project_id = "llm-experiments-387609"                            
    credentials, project_id = google.auth.default()
    if hasattr(credentials, "service_account_email"):
      logging.info(credentials.service_account_email)
    else:
        logging.info("WARNING: no service account credential. User account credential?")                           
    vertexai.init(project=project_id, location=location)
    
    combined_chunk_document_list = get_combined_chunks(chunkId_chunkDoc_list)
    llm, model_name = get_llm(model_version)
    return get_graph_document_list(llm, combined_chunk_document_list, allowedNodes, allowedRelationship)
           
       
