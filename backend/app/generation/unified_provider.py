"""
Unified LLM & Offline Synthesis Orchestrator.
Manages automatic fallback cascade: Cloud Groq -> OpenAI -> Ollama -> Extractive Engine.
Includes safe regex JSON extraction for non-OpenAI models.
"""
import re
import json
from typing import List, Dict, Any, Tuple
from backend.app.config import settings
from backend.app.generation.llm_provider import cloud_provider
from backend.app.generation.offline_bot import offline_bot
from backend.app.core.logger import logger

class UnifiedLLMProvider:
    def __init__(self):
        self.offline_mode = settings.VERIMIND_OFFLINE_MODE

    async def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> Tuple[str, bool]:
        """
        Generates response using active LLM provider or offline fallback.
        Returns: (answer_string, is_offline_fallback)
        """
        if self.offline_mode:
            logger.info("VeriMind forced offline mode active. Using Extractive Engine.")
            return offline_bot.synthesize_answer(query, context_chunks), True

        context_str = "\n\n".join([f"Source [{c.get('chunk_id', 'ref')}]: {c['text']}" for c in context_chunks])
        
        system_prompt = (
            "You are VeriMind, a high-precision factual AI assistant. "
            "Answer the user query using ONLY the provided context sources. "
            "Include inline source citations like [chunk_id]."
        )
        prompt = f"Context:\n{context_str}\n\nQuestion: {query}"

        # Step 1: Attempt Cloud / Ollama LLM
        response = await cloud_provider.generate_response(prompt, system_prompt)
        
        if response and len(response.strip()) > 10:
            return response.strip(), False

        # Step 2: Fallback to Extractive Offline Bot
        logger.warning("LLM providers unreachable or empty. Triggering Extractive Engine fallback.")
        return offline_bot.synthesize_answer(query, context_chunks), True

    def parse_json_safely(self, raw_text: str) -> Dict[str, Any]:
        """
        Regex-based JSON parser to clean markdown code blocks from model output.
        """
        try:
            return json.loads(raw_text)
        except Exception:
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass
        return {}

unified_provider = UnifiedLLMProvider()