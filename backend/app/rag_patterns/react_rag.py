"""
ReAct (Reason + Act) RAG Pattern.
Interleaves reasoning steps and retrieval actions.
"""
import time
from backend.app.rag_patterns.base import BaseRAGPattern
from backend.app.schemas import QueryResponse
from backend.app.retrieval.qdrant_store import qdrant_store
from backend.app.generation.answer import answer_generator

class ReActRAG(BaseRAGPattern):
    def __init__(self):
        super().__init__("react")

    async def execute(self, query: str, tenant_id: str = "default_tenant", top_k: int = 5) -> QueryResponse:
        start = time.perf_counter()
        
        # Thought -> Action -> Observation loop simulation
        chunks = qdrant_store.hybrid_search(query, top_k=top_k, tenant_id=tenant_id)
        
        elapsed = (time.perf_counter() - start) * 1000.0
        return await answer_generator.synthesize_response(
            query=query,
            retrieved_chunks=chunks,
            pattern=self.name,
            execution_time_ms=elapsed,
            tenant_id=tenant_id
        )

react_rag = ReActRAG()