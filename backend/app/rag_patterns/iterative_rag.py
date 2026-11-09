"""
Iterative RAG Pattern.
Performs multi-step retrieval loops to iteratively refine context.
"""
import time
from backend.app.rag_patterns.base import BaseRAGPattern
from backend.app.schemas import QueryResponse
from backend.app.retrieval.qdrant_store import qdrant_store
from backend.app.generation.answer import answer_generator

class IterativeRAG(BaseRAGPattern):
    def __init__(self):
        super().__init__("iterative")

    async def execute(self, query: str, tenant_id: str = "default_tenant", top_k: int = 5) -> QueryResponse:
        start = time.perf_counter()
        
        # Step 1: Initial retrieval
        step1_chunks = qdrant_store.hybrid_search(query, top_k=top_k, tenant_id=tenant_id)
        
        # Step 2: Refined search based on step 1 terms
        refined_term = step1_chunks[0]["text"][:50] if step1_chunks else query
        step2_chunks = qdrant_store.hybrid_search(refined_term, top_k=2, tenant_id=tenant_id)

        all_chunks = step1_chunks + step2_chunks
        elapsed = (time.perf_counter() - start) * 1000.0

        return await answer_generator.synthesize_response(
            query=query,
            retrieved_chunks=all_chunks[:top_k],
            pattern=self.name,
            execution_time_ms=elapsed,
            tenant_id=tenant_id
        )

iterative_rag = IterativeRAG()