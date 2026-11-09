"""
DeepEval Test Suite Adapter.
Provides programmatic unit test assertions for retrieval precision and faithfulness thresholds.
"""
from typing import Dict, Any
from backend.app.core.logger import logger

class DeepEvalAdapter:
    def assert_relevance_threshold(self, score: float, threshold: float = 0.80) -> bool:
        if score < threshold:
            logger.error(f"DeepEval Threshold Violation: Score {score:.2f} < Threshold {threshold:.2f}")
            return False
        return True

deepeval_adapter = DeepEvalAdapter()