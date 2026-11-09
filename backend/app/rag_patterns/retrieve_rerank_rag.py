"""
Retrieve-and-Rerank Pattern.
Standard high-precision strategy: Dense+Sparse retrieval followed by Cross-Encoder re-ranking.
"""
import time
from backend.app.rag_patterns.base import BaseRAGPattern
from backend.app.schemas import QueryResponse
from backend.app.retrieval.qdrant_store import qdrant_store
from backend.app.retrieval.reranker import reranker
from backend.app.generation.answer import answer_generator

class RetrieveRerankRAG(BaseRAGPattern):
    def __init__(self):
        super().__init__("retrieve_rerank")

    async def execute(self, query: str, tenant_id: str = "default_tenant", top_k: int = 5) -> QueryResponse:
        start = time.perf_counter()
        candidates = qdrant_store.hybrid_search(query, top_k=top_k * 2, tenant_id=tenant_id)
        reranked = reranker.rerank(query, candidates, top_k=top_k)
        elapsed = (time.perf_counter() - start) * 1000.0
        return await answer_generator.synthesize_response(
            query=query,
            retrieved_chunks=reranked,
            pattern=self.name,
            execution_time_ms=elapsed,
            tenant_id=tenant_id
        )

retrieve_rerank_rag = RetrieveRerankRAG()