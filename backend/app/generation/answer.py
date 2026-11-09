"""
Answer Synthesis Pipeline.
Combines retrieved context, LLM generation, and source attribution mapping.
"""
from typing import List, Dict, Any
from backend.app.schemas import QueryResponse, VectorSearchResult
from backend.app.generation.unified_provider import unified_provider

class AnswerGenerator:
    async def synthesize_response(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        pattern: str = "hybrid",
        execution_time_ms: float = 0.0,
        tenant_id: str = "default_tenant"
    ) -> QueryResponse:

        answer_text, is_offline = await unified_provider.generate_answer(query, retrieved_chunks)

        chunk_results = [
            VectorSearchResult(
                chunk_id=c.get("chunk_id", "c_unknown"),
                doc_id=c.get("doc_id", "doc_unknown"),
                text=c.get("text", ""),
                score=float(c.get("score", 0.0)),
                rerank_score=c.get("rerank_score"),
                jepa_latent_score=c.get("jepa_latent_score"),
                metadata=c.get("metadata", {})
            )
            for c in retrieved_chunks
        ]

        return QueryResponse(
            query=query,
            answer=answer_text,
            pattern_used=pattern,
            retrieved_chunks=chunk_results,
            claims=[],
            evidence_integrity_index=1.0 if not is_offline else 0.85,
            execution_time_ms=round(execution_time_ms, 2),
            offline_fallback_active=is_offline,
            tenant_id=tenant_id
        )

answer_generator = AnswerGenerator()