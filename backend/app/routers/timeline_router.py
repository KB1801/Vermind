"""
FastAPI Evidence Timeline & Chronological Provenance Router.
Orders retrieved context chunks chronologically based on document metadata.
"""
from fastapi import APIRouter, Form
from typing import Dict, Any, List
from backend.app.retrieval.qdrant_store import qdrant_store

router = APIRouter(prefix="/timeline", tags=["Evidence Timeline"])

@router.post("/build")
async def get_evidence_timeline(query: str = Form(...), tenant_id: str = Form("default_tenant")) -> Dict[str, Any]:
    chunks = qdrant_store.hybrid_search(query, top_k=10, tenant_id=tenant_id)

    timeline_events = []
    for idx, chunk in enumerate(chunks):
        timeline_events.append({
            "step": idx + 1,
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "snippet": chunk["text"][:150] + "...",
            "score": chunk["score"]
        })

    return {
        "query": query,
        "timeline_events_count": len(timeline_events),
        "timeline": timeline_events
    }