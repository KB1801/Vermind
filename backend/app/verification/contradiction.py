"""
Contradiction & Evidence Alignment Detector (Natural Language Inference).
Checks atomic claims against retrieved evidence chunks to detect factual contradictions.
"""
import re
from typing import List, Dict, Any, Tuple
from backend.app.core.logger import logger

class ContradictionDetector:
    def check_claim_against_evidence(
        self,
        claim: str,
        evidence_chunks: List[Dict[str, Any]]
    ) -> Tuple[str, float, List[str], str]:
        """
        Verifies an atomic claim against retrieved evidence.
        Returns: (verdict, confidence, supporting_chunk_ids, reasoning)
        Verdict values: SUPPORTED | CONTRADICTED | NEUTRAL
        """
        if not evidence_chunks:
            return "NEUTRAL", 0.5, [], "No evidence chunks available to verify claim."

        claim_words = set(re.findall(r'\w+', claim.lower()))
        best_support_chunks = []
        max_overlap_ratio = 0.0

        for chunk in evidence_chunks:
            text = chunk.get("text", "").lower()
            chunk_words = set(re.findall(r'\w+', text))
            
            if not claim_words:
                continue

            overlap = len(claim_words.intersection(chunk_words))
            ratio = overlap / float(len(claim_words))

            if ratio > max_overlap_ratio:
                max_overlap_ratio = ratio

            if ratio >= 0.4:
                best_support_chunks.append(chunk.get("chunk_id", "chunk_ref"))

        # Decision threshold boundary mapping
        if max_overlap_ratio >= 0.55:
            return "SUPPORTED", round(min(1.0, max_overlap_ratio + 0.2), 2), best_support_chunks, "Claim strongly aligned with retrieved evidence."
        
        # Check explicit negation keywords for contradiction
        negation_terms = {"not", "never", "no", "false", "neither", "unable", "denies"}
        claim_has_negation = bool(claim_words.intersection(negation_terms))
        
        if max_overlap_ratio >= 0.35 and claim_has_negation:
            return "CONTRADICTED", 0.75, best_support_chunks, "Claim contains negation that conflicts with evidence terms."

        if max_overlap_ratio >= 0.25:
            return "SUPPORTED", round(max_overlap_ratio, 2), best_support_chunks, "Claim partially supported by evidence context."

        return "NEUTRAL", 0.40, [], "Insufficient evidence in dataset to verify claim."

contradiction_detector = ContradictionDetector()