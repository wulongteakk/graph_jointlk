
from typing import List

import logging
from src.chunk_utils import get_combined_chunks


logging.basicConfig(format='%(asctime)s - %(message)s', level='INFO')


def get_graph_from_diffbot(graph, chunkId_chunkDoc_list: List):
    # 延迟导入，避免 src.llm 初始化期间与本模块产生循环依赖
    from src.llm import get_llm
    logging.basicConfig(format='%(asctime)s - %(message)s', level='INFO')
    combined_chunk_document_list = get_combined_chunks(chunkId_chunkDoc_list)
    llm, model_name = get_llm('diffbot')
    graph_documents = llm.convert_to_graph_documents(combined_chunk_document_list)
    return graph_documents

    