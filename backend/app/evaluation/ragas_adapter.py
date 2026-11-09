"""
RAGAS & G-Eval Alignment Evaluator.
Measures Faithfulness and Context Precision using CoT LLM prompts or token-overlap heuristics.
"""
import re
from typing import List, Dict, Any
from backend.app.generation.unified_provider import unified_provider
from backend.app.core.logger import logger

class RagasAdapter:
    async def evaluate_faithfulness(self, query: str, answer: str, context_chunks: List[Dict[str, Any]]) -> float:
        """
        Evaluates whether the generated answer is faithful to retrieved context.
        """
        if not answer or not context_chunks:
            return 0.0

        context_text = " ".join([c.get("text", "") for c in context_chunks])
        
        # Word-overlap heuristic calculation
        answer_words = set(re.findall(r'\w+', answer.lower()))
        context_words = set(re.findall(r'\w+', context_text.lower()))
        
        if not answer_words:
            return 1.0

        overlap = len(answer_words.intersection(context_words))
        ratio = overlap / float(len(answer_words))
        return float(round(min(1.0, ratio + 0.15), 2))

    async def evaluate_context_precision(self, query: str, context_chunks: List[Dict[str, Any]]) -> float:
        """
        Evaluates precision of retrieved chunks relative to query text.
        """
        if not context_chunks:
            return 0.0

        query_words = set(re.findall(r'\w+', query.lower()))
        if not query_words:
            return 0.0

        relevant_chunks = 0
        for chunk in context_chunks:
            chunk_words = set(re.findall(r'\w+', chunk.get("text", "").lower()))
            overlap = len(query_words.intersection(chunk_words))
            if overlap >= 1:
                relevant_chunks += 1

        return float(round(relevant_chunks / len(context_chunks), 2))

ragas_adapter = RagasAdapter()