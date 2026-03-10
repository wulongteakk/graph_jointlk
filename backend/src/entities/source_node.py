from datetime import datetime


class sourceNode:
    """Lightweight container used by graphDBdataAccess.* to upsert Document nodes.

    NOTE: This project historically used fileName as the Document MERGE key.
    For BG-KG / Instance-KG separation we introduce:

    - doc_id:   scoped document id (recommended MERGE key)
    - kg_scope: 'bg' or 'inst'
    - kg_id:    dataset/project id inside the scope

    These fields are optional so legacy flows keep working.
    """

    file_name: str = None
    file_size: int = None
    file_type: str = None
    file_source: str = None
    status: str = None
    url: str = None

    # New scoping fields
    doc_id: str = None
    kg_scope: str = None
    kg_id: str = None

    gcsBucket: str = None
    gcsBucketFolder: str = None
    gcsProjectId: str = None
    awsAccessKeyId: str = None

    node_count: int = None
    relationship_count: str = None
    model: str = None

    created_at: datetime = None
    updated_at: datetime = None
    processing_time: float = None
    error_message: str = None

    total_pages: int = None
    total_chunks: int = None
    language: str = None
    is_cancelled: bool = None
    processed_chunk: int = None

    access_token: str = None
