"""
Atomic Claim Decomposition Engine.
Deconstructs complex LLM answer text into standalone factual claims for individual verification.
"""
import re
from typing import List
from backend.app.generation.unified_provider import unified_provider
from backend.app.core.logger import logger

class ClaimDecomposer:
    def __init__(self):
        pass

    async def decompose_text(self, text: str) -> List[str]:
        """
        Decomposes complex paragraphs into standalone atomic factual claims.
        """
        if not text or len(text.strip()) < 10:
            return []

        prompt = (
            "Break down the following text into individual, standalone atomic factual claims. "
            "Each claim must express exactly one fact that can be independently verified. "
            "Return ONLY a raw JSON array of strings, e.g. [\"Claim 1\", \"Claim 2\"].\n\n"
            f"Text:\n{text}"
        )

        try:
            from backend.app.generation.llm_provider import cloud_provider
            raw_response = await cloud_provider.generate_response(prompt, "You are a factual claim decomposer. Respond ONLY in valid JSON.")
            parsed = unified_provider.parse_json_safely(raw_response)
            
            if isinstance(parsed, list) and len(parsed) > 0:
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception as e:
            logger.warning(f"LLM claim decomposition failed ({str(e)}). Falling back to rule-based sentence splitting.")

        # Rule-based sentence boundary fallback
        raw_sentences = re.split(r'(?<=[.!?])\s+', text)
        clean_claims = []
        for s in raw_sentences:
            cleaned = re.sub(r'\[.*?\]', '', s).strip()  # Strip citations like [c1]
            if len(cleaned) > 12:
                clean_claims.append(cleaned)

        return clean_claims

claim_decomposer = ClaimDecomposer()