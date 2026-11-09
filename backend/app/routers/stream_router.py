"""
FastAPI Server-Sent Events (SSE) Streaming Router.
Streams generated tokens and live citation events in real-time to Streamlit / React UI.
"""
import asyncio
import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from backend.app.schemas import QueryRequest
from backend.app.retrieval.qdrant_store import qdrant_store
from backend.app.generation.unified_provider import unified_provider

router = APIRouter(prefix="/stream", tags=["SSE Real-Time Streaming"])

@router.post("/query")
async def stream_rag_response(request: QueryRequest):
    """Streams RAG generation tokens token-by-token via SSE."""
    
    async def sse_event_generator():
        # Step 1: Retrieval event
        yield f"event: status\ndata: {json.dumps({'message': 'Retrieving relevant documents...'})}\n\n"
        await asyncio.sleep(0.1)

        chunks = qdrant_store.hybrid_search(request.query, top_k=request.top_k, tenant_id=request.tenant_id)
        
        # Stream retrieved source metadata
        sources_payload = [{"chunk_id": c["chunk_id"], "score": c["score"]} for c in chunks]
        yield f"event: sources\ndata: {json.dumps({'sources': sources_payload})}\n\n"

        # Step 2: Answer token streaming simulation
        answer, _ = await unified_provider.generate_answer(request.query, chunks)
        words = answer.split()

        for word in words:
            yield f"event: token\ndata: {json.dumps({'token': word + ' '})}\n\n"
            await asyncio.sleep(0.02)  # Simulate live streaming pacing

        yield f"event: complete\ndata: {json.dumps({'status': 'finished'})}\n\n"

    return StreamingResponse(sse_event_generator(), media_type="text/event-stream")