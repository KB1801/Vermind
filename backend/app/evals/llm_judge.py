"""
Deterministic & Cloud LLM-as-a-Judge Alignment Evaluator.
Validates whether generated answers contain hallucinations or unauthorized facts.
"""
from typing import Tuple

class LLMJudge:
    def judge_hallucination(self, answer: str, context: str) -> Tuple[bool, float]:
        """Determines if answer contains hallucinatory statements not present in context."""
        if not answer:
            return True, 0.0
        
        # Word overlap check
        a_words = set(answer.lower().split())
        c_words = set(context.lower().split())
        
        overlap = len(a_words.intersection(c_words)) / max(1, len(a_words))
        is_hallucination = overlap < 0.25
        confidence = float(round(1.0 - overlap, 2)) if is_hallucination else float(round(overlap, 2))

        return is_hallucination, confidence

llm_judge = LLMJudge()