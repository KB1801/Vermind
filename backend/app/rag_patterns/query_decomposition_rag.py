"""
Query Decomposition RAG Pattern.
Decomposes complex queries into sub-questions and retrieves evidence sequentially.
"""
import time
from backend.app.rag_patterns.base import BaseRAGPattern
from backend.app.schemas import QueryResponse
from backend.app.retrieval.qdrant_store import qdrant_store
from backend.app.generation.answer import answer_generator

class QueryDecompositionRAG(BaseRAGPattern):
    def __init__(self):
        super().__init__("query_decomposition")

    async def execute(self, query: str, tenant_id: str = "default_tenant", top_k: int = 5) -> QueryResponse:
        start = time.perf_counter()
        
        # Split multi-part query by keywords
        sub_queries = [q.strip() for q in query.replace(" and ", " | ").split("|") if q.strip()]
        if not sub_queries:
            sub_queries = [query]

        collected_chunks = []
        for sq in sub_queries:
            chunks = qdrant_store.hybrid_search(sq, top_k=max(2, top_k // len(sub_queries)), tenant_id=tenant_id)
            collected_chunks.extend(chunks)

        elapsed = (time.perf_counter() - start) * 1000.0
        return await answer_generator.synthesize_response(
            query=query,
            retrieved_chunks=collected_chunks[:top_k],
            pattern=self.name,
            execution_time_ms=elapsed,
            tenant_id=tenant_id
        )

query_decomposition_rag = QueryDecompositionRAG()