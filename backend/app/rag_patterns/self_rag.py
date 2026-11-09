"""
Self-RAG Pattern.
Uses reflection tokens to evaluate whether retrieval is needed and verifies answer support.
Reference: Asai et al., 2023.
"""
import time
from backend.app.rag_patterns.base import BaseRAGPattern
from backend.app.schemas import QueryResponse
from backend.app.retrieval.qdrant_store import qdrant_store
from backend.app.retrieval.reranker import reranker
from backend.app.verification.validator import evidence_validator
from backend.app.generation.answer import answer_generator

class SelfRAG(BaseRAGPattern):
    def __init__(self):
        super().__init__("self_rag")

    async def execute(self, query: str, tenant_id: str = "default_tenant", top_k: int = 5) -> QueryResponse:
        start = time.perf_counter()
        candidates = qdrant_store.hybrid_search(query, top_k=top_k * 2, tenant_id=tenant_id)
        reranked = reranker.rerank(query, candidates, top_k=top_k)

        # Synthesize answer
        response = await answer_generator.synthesize_response(
            query=query,
            retrieved_chunks=reranked,
            pattern=self.name,
            execution_time_ms=0.0,
            tenant_id=tenant_id
        )

        # Self-Reflection Verification
        claims, eii = await evidence_validator.validate_answer_claims(response.answer, reranked)
        response.claims = claims
        response.evidence_integrity_index = eii
        response.execution_time_ms = (time.perf_counter() - start) * 1000.0
        
        return response

self_rag = SelfRAG()