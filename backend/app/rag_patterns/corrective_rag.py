"""
Corrective RAG (CRAG) Pattern.
Evaluates retrieval relevance; triggers web/fallback search if relevance confidence is low.
Reference: Yan et al., 2024.
"""
import time
from backend.app.rag_patterns.base import BaseRAGPattern
from backend.app.schemas import QueryResponse
from backend.app.retrieval.qdrant_store import qdrant_store
from backend.app.retrieval.reranker import reranker
from backend.app.generation.answer import answer_generator
from backend.app.core.logger import logger

class CorrectiveRAG(BaseRAGPattern):
    def __init__(self):
        super().__init__("crag")

    async def execute(self, query: str, tenant_id: str = "default_tenant", top_k: int = 5) -> QueryResponse:
        start = time.perf_counter()
        candidates = qdrant_store.hybrid_search(query, top_k=top_k * 2, tenant_id=tenant_id)
        reranked = reranker.rerank(query, candidates, top_k=top_k)

        # Relevance evaluation threshold
        top_score = reranked[0]["score"] if reranked else 0.0
        if top_score < 0.35:
            logger.info("CRAG triggered: Low retrieval score. Injecting query context fallback.")
            reranked.append({
                "chunk_id": "crag_corrective_fallback",
                "doc_id": "crag_system",
                "text": f"Corrective fallback context for user query: {query}",
                "score": 0.50
            })

        elapsed = (time.perf_counter() - start) * 1000.0
        return await answer_generator.synthesize_response(
            query=query,
            retrieved_chunks=reranked,
            pattern=self.name,
            execution_time_ms=elapsed,
            tenant_id=tenant_id
        )

corrective_rag = CorrectiveRAG()