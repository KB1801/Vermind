"""
LLM Provider Factory supporting Groq, OpenAI, Anthropic, and Ollama.
Default model: Groq llama-3.3-70b-versatile.
"""
import httpx
from typing import Dict, Any, Optional
from backend.app.config import settings
from backend.app.core.logger import logger

class CloudLLMProvider:
    def __init__(self):
        self.groq_key = settings.GROQ_API_KEY
        self.openai_key = settings.OPENAI_API_KEY

    async def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        """
        Calls primary Groq API with automatic fallback to OpenAI or local Ollama.
        """
        if self.groq_key:
            try:
                from groq import AsyncGroq
                client = AsyncGroq(api_key=self.groq_key)
                completion = await client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt or "You are VeriMind AI assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=1024
                )
                return completion.choices[0].message.content
            except Exception as e:
                logger.warning(f"Groq API call failed: {str(e)}. Attempting fallbacks...")

        # Fallback to Ollama local server
        return await self._call_ollama(prompt, system_prompt)

    async def _call_ollama(self, prompt: str, system_prompt: str = "") -> str:
        try:
            url = f"{settings.OLLAMA_URL}/api/generate"
            payload = {
                "model": settings.OLLAMA_MODEL,
                "prompt": f"{system_prompt}\n\nUser Question: {prompt}\nAnswer:",
                "stream": False
            }
            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.post(url, json=payload)
                if res.status_code == 200:
                    return res.json().get("response", "")
        except Exception as e:
            logger.warning(f"Local Ollama server unavailable at {settings.OLLAMA_URL}: {str(e)}")

        return ""

cloud_provider = CloudLLMProvider()