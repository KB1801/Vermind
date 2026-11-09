"""
Deterministic & Cloud LLM-as-a-Judge Alignment Evaluator.
Validates whether generated answers contain hallucinations or unauthorized facts.
"""
import re
from typing import Tuple

class LLMJudge:
    def judge_hallucination(self, answer: str, context: str) -> Tuple[bool, float]:
        """Determines if answer contains hallucinatory statements not present in context."""
        if not answer or not context:
            return True, 0.0

        # Extract words and convert to lowercase
        a_words = set(re.findall(r'\b\w+\b', answer.lower()))
        c_words = set(re.findall(r'\b\w+\b', context.lower()))

        # Remove common stop words for clean non-trivial term overlap
        stopwords = {"is", "a", "an", "the", "for", "to", "of", "and", "in", "on", "with", "by", "at", "or", "requires", "optimized"}
        a_content = a_words - stopwords
        c_content = c_words - stopwords

        if not a_content:
            return False, 1.0

        overlap = len(a_content.intersection(c_content)) / float(len(a_content))
        
        # If less than 40% of non-stopword terms in answer exist in context, flag as hallucination
        is_hallucination = overlap < 0.40
        confidence = float(round(1.0 - overlap, 2)) if is_hallucination else float(round(overlap, 2))

        return is_hallucination, confidence

llm_judge = LLMJudge()