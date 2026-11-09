"""
Multi-Agent Collaborative Orchestrator.
Coordinates Planner Agent, Research Retrieval Agent, and Verification Agent.
"""
from typing import Dict, Any, List
from backend.app.retrieval.qdrant_store import qdrant_store
from backend.app.verification.validator import evidence_validator
from backend.app.generation.unified_provider import unified_provider
from backend.app.core.logger import logger

class MultiAgentOrchestrator:
    async def run_collaborative_task(self, query: str, tenant_id: str = "default_tenant") -> Dict[str, Any]:
        """
        Executes 3-Stage Multi-Agent Task Pipeline:
        1. Planner Agent: Decomposes query into sub-goals.
        2. Researcher Agent: Executes hybrid vector search across goals.
        3. Verifier Agent: Evaluates claim accuracy and generates final answer.
        """
        logger.info(f"Multi-Agent Orchestrator initialized for query: '{query}'")

        # Stage 1: Planner Agent
        sub_goals = [query, f"Background facts about {query}"]

        # Stage 2: Researcher Agent
        all_chunks = []
        for goal in sub_goals:
            chunks = qdrant_store.hybrid_search(goal, top_k=3, tenant_id=tenant_id)
            all_chunks.extend(chunks)

        # Deduplicate retrieved chunks
        unique_chunks = {c["chunk_id"]: c for c in all_chunks}
        retrieved_list = list(unique_chunks.values())

        # Stage 3: Synthesizer & Verifier Agent
        answer_text, is_offline = await unified_provider.generate_answer(query, retrieved_list)
        claims, eii = await evidence_validator.validate_answer_claims(answer_text, retrieved_list)

        return {
            "query": query,
            "planner_agent_sub_goals": sub_goals,
            "researcher_agent_retrieved_count": len(retrieved_list),
            "final_answer": answer_text,
            "evidence_integrity_index": eii,
            "verified_claims": [c.model_dump() for c in claims],
            "offline_mode_active": is_offline
        }

multi_agent_orchestrator = MultiAgentOrchestrator()