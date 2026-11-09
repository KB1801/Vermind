"""
Background Ingestion Workers & Async Pipeline Handlers.
Manages long-running OCR and web scraping tasks asynchronously.
"""
import uuid
from typing import Dict, Any, List
from backend.app.ingestion.pdf_loader import pdf_loader
from backend.app.ingestion.web_loader import web_loader
from backend.app.ingestion.chunker import document_chunker
from backend.app.chunking.semantic import semantic_chunker
from backend.app.chunking.sentence_window import sentence_window_chunker
from backend.app.chunking.parent_child import parent_child_chunker
from backend.app.core.logger import logger

async def process_document_async(
    content: str,
    title: str,
    strategy: str = "semantic",
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Background worker function to chunk and prepare documents for vector indexing.
    """
    metadata = metadata or {}
    doc_id = f"doc_{uuid.uuid4().hex[:8]}"
    metadata["title"] = title

    logger.info(f"Processing document '{title}' (ID: {doc_id}) with strategy '{strategy}'")

    if strategy == "semantic":
        chunks = semantic_chunker.split_text(content, doc_id, metadata)
    elif strategy == "sentence_window":
        chunks = sentence_window_chunker.split_text(content, doc_id, metadata)
    elif strategy == "parent_child":
        chunks = parent_child_chunker.split_text(content, doc_id, metadata)
    else:
        chunks = document_chunker.split_text(content, doc_id, metadata)

    return {
        "doc_id": doc_id,
        "chunks_count": len(chunks),
        "chunks": chunks
    }