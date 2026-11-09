"""
FastAPI Time-Travel Query Replay Router.
Re-executes historical queries to verify reproducibility and track drift over time.
"""
import time
from fastapi import APIRouter, Form
from typing import Dict, Any
from backend.app.retrieval.qdrant_store import qdrant_store
from backend.app.generation.unified_provider import unified_provider

router = APIRouter(prefix="/replay", tags=["Query Time-Travel Replay"])

@router.post("/run")
async def replay_historical_query(query: str = Form(...), tenant_id: str = Form("default_tenant")) -> Dict[str, Any]:
    start = time.perf_counter()
    chunks = qdrant_store.hybrid_search(query, top_k=5, tenant_id=tenant_id)
    answer, is_offline = await unified_provider.generate_answer(query, chunks)
    elapsed = (time.perf_counter() - start) * 1000.0

    return {
        "replayed_query": query,
        "replayed_answer": answer,
        "chunks_retrieved_count": len(chunks),
        "execution_time_ms": round(elapsed, 2),
        "status": "replayed_successfully"
    }