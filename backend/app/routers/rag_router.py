"""
FastAPI RAG Router.
Exposes endpoints for executing single-step and agentic retrieval queries.
"""
import time
from fastapi import APIRouter, HTTPException, Depends
from backend.app.schemas import QueryRequest, QueryResponse
from backend.app.retrieval.qdrant_store import qdrant_store
from backend.app.retrieval.reranker import reranker
from backend.app.evaluation.jepa_style import jepa_adapter
from backend.app.generation.answer import answer_generator
from backend.app.core.queryguard import query_guard

router = APIRouter(prefix="/rag", tags=["RAG Execution"])

@router.post("/query", response_model=QueryResponse)
async def process_rag_query(request: QueryRequest):
    start_time = time.perf_counter()
    
    # 1. Sanitize input query
    clean_query, _ = query_guard.sanitize(request.query)

    # 2. Hybrid Retrieval (Qdrant BM25 + Dense BGE-small)
    candidates = qdrant_store.hybrid_search(
        query=clean_query,
        top_k=request.top_k * 2,
        tenant_id=request.tenant_id
    )

    # 3. Cross-Encoder Re-Ranking
    reranked = reranker.rerank(query=clean_query, candidates=candidates, top_k=request.top_k)

    # 4. I-JEPA Latent World Model Calibration
    calibrated = jepa_adapter.calibrate_scores(query_vec=[], candidate_chunks=reranked)

    # 5. Answer Synthesis
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    
    return await answer_generator.synthesize_response(
        query=clean_query,
        retrieved_chunks=calibrated,
        pattern=request.pattern,
        execution_time_ms=elapsed_ms,
        tenant_id=request.tenant_id
    )