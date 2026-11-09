"""
Naive Baseline RAG Pattern.
Direct dense vector search without reranking, filtering, or claim verification.
"""
import time
from backend.app.rag_patterns.base import BaseRAGPattern
from backend.app.schemas import QueryResponse
from backend.app.retrieval.qdrant_store import qdrant_store
from backend.app.generation.answer import answer_generator

class NaiveRAG(BaseRAGPattern):
    def __init__(self):
        super().__init__("naive")

    async def execute(self, query: str, tenant_id: str = "default_tenant", top_k: int = 5) -> QueryResponse:
        start = time.perf_counter()
        chunks = qdrant_store.hybrid_search(query, top_k=top_k, tenant_id=tenant_id)
        elapsed = (time.perf_counter() - start) * 1000.0
        return await answer_generator.synthesize_response(
            query=query,
            retrieved_chunks=chunks,
            pattern=self.name,
            execution_time_ms=elapsed,
            tenant_id=tenant_id
        )

naive_rag = NaiveRAG()