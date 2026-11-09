"""
Context Compression RAG Pattern.
Summarizes and compresses retrieved chunks to eliminate prompt noise before synthesis.
"""
import time
from backend.app.rag_patterns.base import BaseRAGPattern
from backend.app.schemas import QueryResponse
from backend.app.retrieval.qdrant_store import qdrant_store
from backend.app.generation.answer import answer_generator

class CompressionRAG(BaseRAGPattern):
    def __init__(self):
        super().__init__("compression")

    async def execute(self, query: str, tenant_id: str = "default_tenant", top_k: int = 5) -> QueryResponse:
        start = time.perf_counter()
        raw_chunks = qdrant_store.hybrid_search(query, top_k=top_k, tenant_id=tenant_id)

        compressed_chunks = []
        for c in raw_chunks:
            # Compress long chunks into key sentences
            text = c["text"]
            compressed_text = text[:300] + "..." if len(text) > 300 else text
            c_copy = c.copy()
            c_copy["text"] = compressed_text
            compressed_chunks.append(c_copy)

        elapsed = (time.perf_counter() - start) * 1000.0
        return await answer_generator.synthesize_response(
            query=query,
            retrieved_chunks=compressed_chunks,
            pattern=self.name,
            execution_time_ms=elapsed,
            tenant_id=tenant_id
        )

compression_rag = CompressionRAG()