import logging
import os
from typing import List

from langchain.docstore.document import Document


logging.basicConfig(format='%(asctime)s - %(message)s', level='INFO')


def get_combined_chunks(chunkId_chunkDoc_list: List):
    """按配置将多个 chunk 合并为一个 Document，统一承载 combined_chunk_ids 元数据。"""
    chunks_to_combine = int(os.environ.get("NUMBER_OF_CHUNKS_TO_COMBINE"))
    logging.info(f"Combining {chunks_to_combine} chunks before sending request to LLM")
    combined_chunk_document_list = []
    combined_chunks_page_content = [
        "".join(
            document["chunk_doc"].page_content
            for document in chunkId_chunkDoc_list[i: i + chunks_to_combine]
        )
        for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
    ]
    combined_chunks_ids = [
        [
            document["chunk_id"]
            for document in chunkId_chunkDoc_list[i: i + chunks_to_combine]
        ]
        for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
    ]

    combined_file_names = [
        next(
            (
                document["chunk_doc"].metadata.get("fileName")
                for document in chunkId_chunkDoc_list[i: i + chunks_to_combine]
                if document.get("chunk_doc") and document["chunk_doc"].metadata
            ),
            None,
        )
        for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
    ]

    for i in range(len(combined_chunks_page_content)):
        combined_chunk_document_list.append(
            Document(
                page_content=combined_chunks_page_content[i],
                metadata={
                    "combined_chunk_ids": combined_chunks_ids[i],
                    "fileName": combined_file_names[i],
                },
            )
        )
    return combined_chunk_document_list