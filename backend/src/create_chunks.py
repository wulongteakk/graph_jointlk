from langchain_text_splitters import TokenTextSplitter
from langchain.docstore.document import Document
from langchain_community.graphs import Neo4jGraph
import logging
import os
from src.document_sources.youtube import get_chunks_with_timestamps

logging.basicConfig(format="%(asctime)s - %(message)s", level="INFO")


class CreateChunksofDocument:
    def __init__(self, pages: list[Document], graph: Neo4jGraph):
        self.pages = pages
        self.graph = graph

    def split_file_into_chunks(self):
        """
        Split a list of documents(file pages) into chunks of fixed size.

        Args:
            pages: A list of pages to split. Each page is a list of text strings.

        Returns:
            A list of chunks each of which is a langchain Document.
        """
        logging.info("Split file into smaller chunks for retrieval/vector index/raw storage only; evidence units are built separately")

        chunk_size = int(os.getenv("KG_CHUNK_SIZE", "200"))
        chunk_overlap = int(os.getenv("KG_CHUNK_OVERLAP", "20"))
        text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if 'page' in self.pages[0].metadata:
            chunks = []
            for i, document in enumerate(self.pages):
                page_number = i + 1
                for chunk in text_splitter.split_documents([document]):
                    chunks.append(Document(page_content=chunk.page_content, metadata={
                        'page_number': page_number,
                        'page': page_number,
                        'chunk_index': len(chunks),
                        'chunk_source': 'fixed_chunk',
                        'fileName': document.metadata.get('fileName') or document.metadata.get('source') or '',
                        'file_name': document.metadata.get('fileName') or document.metadata.get('source') or '',
                        'doc_id': document.metadata.get('doc_id'),
                        'kg_scope': document.metadata.get('kg_scope'),
                        'kg_id': document.metadata.get('kg_id'),
                    }))

        elif 'length' in self.pages[0].metadata:
            chunks_without_timestamps = text_splitter.split_documents(self.pages)
            chunks = get_chunks_with_timestamps(chunks_without_timestamps, self.pages[0].metadata['source'])
            for idx, chunk in enumerate(chunks):
                chunk.metadata.setdefault('chunk_index', idx)
                chunk.metadata.setdefault('chunk_source', 'fixed_chunk')
                chunk.metadata.setdefault('fileName', self.pages[0].metadata.get('source', ''))
                chunk.metadata.setdefault('file_name', self.pages[0].metadata.get('source', ''))
                chunk.metadata.setdefault('doc_id', self.pages[0].metadata.get('doc_id'))
                chunk.metadata.setdefault('kg_scope', self.pages[0].metadata.get('kg_scope'))
                chunk.metadata.setdefault('kg_id', self.pages[0].metadata.get('kg_id'))
        else:
            chunks = text_splitter.split_documents(self.pages)
            for idx, chunk in enumerate(chunks):
                chunk.metadata.setdefault('chunk_index', idx)
                chunk.metadata.setdefault('chunk_source', 'fixed_chunk')
                chunk.metadata.setdefault('fileName', self.pages[0].metadata.get('fileName') or self.pages[0].metadata.get('source') or '')
                chunk.metadata.setdefault('file_name', self.pages[0].metadata.get('fileName') or self.pages[0].metadata.get('source') or '')
                chunk.metadata.setdefault('doc_id', self.pages[0].metadata.get('doc_id'))
                chunk.metadata.setdefault('kg_scope', self.pages[0].metadata.get('kg_scope'))
                chunk.metadata.setdefault('kg_id', self.pages[0].metadata.get('kg_id'))
        return chunks