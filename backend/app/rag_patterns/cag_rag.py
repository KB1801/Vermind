"""
Cache-Augmented Generation (CAG) Pattern.
Bypasses retrieval when queries match high-confidence semantic cache entries.
"""
import time
from backend.app.rag_patterns.base import BaseRAGPattern
from backend.app.schemas import QueryResponse
from backend.app.core.cache import cache_manager
from backend.app.retrieval.qdrant_store import qdrant_store
from backend.app.generation.answer import answer_generator

class CacheAugmentedGeneration(BaseRAGPattern):
    def __init__(self):
        super().__init__("cag")

    async def execute(self, query: str, tenant_id: str = "default_tenant", top_k: int = 5) -> QueryResponse:
        start = time.perf_counter()
        
        # Check cache
        cached = cache_manager.get("cag_query", query)
        if cached:
            elapsed = (time.perf_counter() - start) * 1000.0
            return QueryResponse(
                query=query,
                answer=cached["answer"],
                pattern_used="cag_cache_hit",
                retrieved_chunks=[],
                claims=[],
                evidence_integrity_index=1.0,
                execution_time_ms=elapsed,
                tenant_id=tenant_id
            )

        chunks = qdrant_store.hybrid_search(query, top_k=top_k, tenant_id=tenant_id)
        response = await answer_generator.synthesize_response(
            query=query,
            retrieved_chunks=chunks,
            pattern=self.name,
            execution_time_ms=(time.perf_counter() - start) * 1000.0,
            tenant_id=tenant_id
        )

        # Cache answer
        cache_manager.set("cag_query", query, {"answer": response.answer})
        return response

cag_pattern = CacheAugmentedGeneration()