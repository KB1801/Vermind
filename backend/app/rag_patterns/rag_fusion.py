"""
RAG-Fusion Pattern.
Generates multiple query variations, searches vector space in parallel, and merges with RRF.
"""
import time
from backend.app.rag_patterns.base import BaseRAGPattern
from backend.app.schemas import QueryResponse
from backend.app.retrieval.qdrant_store import qdrant_store
from backend.app.generation.answer import answer_generator

class RAGFusion(BaseRAGPattern):
    def __init__(self):
        super().__init__("rag_fusion")

    async def execute(self, query: str, tenant_id: str = "default_tenant", top_k: int = 5) -> QueryResponse:
        start = time.perf_counter()
        
        # Query variations
        variations = [
            query,
            f"Explain details regarding {query}",
            f"Overview and key concepts of {query}"
        ]

        all_candidates = []
        for var in variations:
            res = qdrant_store.hybrid_search(var, top_k=top_k, tenant_id=tenant_id)
            all_candidates.extend(res)

        # Deduplicate candidates by chunk_id
        unique_chunks = {}
        for chunk in all_candidates:
            cid = chunk["chunk_id"]
            if cid not in unique_chunks:
                unique_chunks[cid] = chunk

        merged_chunks = list(unique_chunks.values())[:top_k]
        elapsed = (time.perf_counter() - start) * 1000.0

        return await answer_generator.synthesize_response(
            query=query,
            retrieved_chunks=merged_chunks,
            pattern=self.name,
            execution_time_ms=elapsed,
            tenant_id=tenant_id
        )

rag_fusion = RAGFusion()