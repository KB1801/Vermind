"""
FastAPI Standard Search Query Router.
Handles simple vector retrieval without full synthesis generation.
"""
from fastapi import APIRouter
from typing import List
from backend.app.schemas import QueryRequest, VectorSearchResult
from backend.app.retrieval.qdrant_store import qdrant_store

router = APIRouter(prefix="/search", tags=["Vector Search"])

@router.post("/chunks", response_model=List[VectorSearchResult])
async def search_chunks_only(request: QueryRequest):
    candidates = qdrant_store.hybrid_search(
        query=request.query,
        top_k=request.top_k,
        tenant_id=request.tenant_id
    )
    
    return [
        VectorSearchResult(
            chunk_id=c.get("chunk_id", "c_id"),
            doc_id=c.get("doc_id", "doc_id"),
            text=c.get("text", ""),
            score=c.get("score", 0.0),
            metadata=c.get("metadata", {})
        )
        for c in candidates
    ]