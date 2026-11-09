"""
Adaptive RAG Pattern.
Dynamically routes queries based on intent classification and complexity score.
Reference: Jeong et al., 2024.
"""
import time
from backend.app.rag_patterns.base import BaseRAGPattern
from backend.app.schemas import QueryResponse
from backend.app.ml.intent_classifier import intent_classifier
from backend.app.retrieval.qdrant_store import qdrant_store
from backend.app.retrieval.reranker import reranker
from backend.app.generation.answer import answer_generator

class AdaptiveRAG(BaseRAGPattern):
    def __init__(self):
        super().__init__("adaptive_rag")

    async def execute(self, query: str, tenant_id: str = "default_tenant", top_k: int = 5) -> QueryResponse:
        start = time.perf_counter()
        intent, confidence = intent_classifier.predict(query)

        # Dynamically adjust top_k retrieval budget based on predicted intent
        fetch_k = top_k * 3 if intent in ["summary", "comparison"] else top_k

        candidates = qdrant_store.hybrid_search(query, top_k=fetch_k, tenant_id=tenant_id)
        reranked = reranker.rerank(query, candidates, top_k=top_k)

        elapsed = (time.perf_counter() - start) * 1000.0
        return await answer_generator.synthesize_response(
            query=query,
            retrieved_chunks=reranked,
            pattern=f"adaptive_{intent}",
            execution_time_ms=elapsed,
            tenant_id=tenant_id
        )

adaptive_rag = AdaptiveRAG()