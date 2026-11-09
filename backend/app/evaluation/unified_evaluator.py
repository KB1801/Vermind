"""
Unified Quality & Reliability Evaluator.
Aggregates RAGAS metrics, EII verification scores, and execution latency into a single report.
"""
from typing import List, Dict, Any
from backend.app.evaluation.ragas_adapter import ragas_adapter
from backend.app.verification.validator import evidence_validator

class UnifiedEvaluator:
    async def evaluate_rag_execution(
        self,
        query: str,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]],
        execution_time_ms: float
    ) -> Dict[str, Any]:

        faithfulness = await ragas_adapter.evaluate_faithfulness(query, answer, retrieved_chunks)
        precision = await ragas_adapter.evaluate_context_precision(query, retrieved_chunks)
        claims, eii_score = await evidence_validator.validate_answer_claims(answer, retrieved_chunks)

        return {
            "query": query,
            "faithfulness": faithfulness,
            "context_precision": precision,
            "evidence_integrity_index": eii_score,
            "claims_count": len(claims),
            "execution_time_ms": round(execution_time_ms, 2)
        }

unified_evaluator = UnifiedEvaluator()