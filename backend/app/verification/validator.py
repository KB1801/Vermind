"""
Evidence Integrity Index (EII) Computation Engine.
Calculates system-wide evidence trustworthiness scores across all decomposed answer claims.
"""
from typing import List, Dict, Any
from backend.app.schemas import ClaimVerificationResult
from backend.app.verification.claim_decomposer import claim_decomposer
from backend.app.verification.contradiction import contradiction_detector
from backend.app.core.logger import logger

class EvidenceIntegrityValidator:
    async def validate_answer_claims(
        self,
        answer_text: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Tuple[List[ClaimVerificationResult], float]:
        """
        Decomposes answer text, evaluates each claim, and computes overall Evidence Integrity Index (EII).
        """
        claims_text = await claim_decomposer.decompose_text(answer_text)
        if not claims_text:
            return [], 1.0  # Default perfect integrity for empty claim sets

        results = []
        supported_count = 0
        contradicted_count = 0

        for claim in claims_text:
            verdict, confidence, chunk_ids, reasoning = contradiction_detector.check_claim_against_evidence(
                claim=claim,
                evidence_chunks=retrieved_chunks
            )

            if verdict == "SUPPORTED":
                supported_count += 1
            elif verdict == "CONTRADICTED":
                contradicted_count += 1

            results.append(ClaimVerificationResult(
                claim=claim,
                verdict=verdict,
                confidence=confidence,
                supporting_chunk_ids=chunk_ids,
                reasoning=reasoning
            ))

        total_claims = len(claims_text)
        # EII Mathematical Formula: Supported Weight - Contradicted Penalty
        eii = (supported_count - (1.5 * contradicted_count)) / float(total_claims)
        eii_bounded = max(0.0, min(1.0, round(eii, 2)))

        logger.info(f"Verified {total_claims} claims: {supported_count} Supported, {contradicted_count} Contradicted. EII Index: {eii_bounded}")
        return results, eii_bounded

evidence_validator = EvidenceIntegrityValidator()