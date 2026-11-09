"""
Hypothetical Document Embeddings (HyDE) Pattern.
Generates a hypothetical answer document to embed and search vector space.
Reference: Gao et al., 2022.
"""
import time
from backend.app.rag_patterns.base import BaseRAGPattern
from backend.app.schemas import QueryResponse
from backend.app.generation.unified_provider import cloud_provider
from backend.app.retrieval.qdrant_store import qdrant_store
from backend.app.retrieval.reranker import reranker
from backend.app.generation.answer import answer_generator

class HyDERAG(BaseRAGPattern):
    def __init__(self):
        super().__init__("hyde")

    async def execute(self, query: str, tenant_id: str = "default_tenant", top_k: int = 5) -> QueryResponse:
        start = time.perf_counter()
        
        # Step 1: Generate hypothetical document
        hyde_prompt = f"Write a brief, factual mock passage that directly answers the question: '{query}'"
        hypothetical_doc = await cloud_provider.generate_response(hyde_prompt)
        
        search_query = hypothetical_doc if len(hypothetical_doc) > 20 else query

        # Step 2: Vector search using hypothetical text
        candidates = qdrant_store.hybrid_search(search_query, top_k=top_k * 2, tenant_id=tenant_id)
        reranked = reranker.rerank(query, candidates, top_k=top_k)

        elapsed = (time.perf_counter() - start) * 1000.0
        return await answer_generator.synthesize_response(
            query=query,
            retrieved_chunks=reranked,
            pattern=self.name,
            execution_time_ms=elapsed,
            tenant_id=tenant_id
        )

hyde_rag = HyDERAG()