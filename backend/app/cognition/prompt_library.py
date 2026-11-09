"""
Central Prompt Library & Template Manager.
Stores structured system prompts for all 19 RAG patterns.
"""
class PromptLibrary:
    SYSTEM_PROMPT = (
        "You are VeriMind, a high-precision enterprise AI assistant. "
        "Your task is to provide objective, factually verified answers based on retrieved context."
    )

    CLAIM_DECOMPOSITION_PROMPT = (
        "Decompose the following passage into atomic standalone factual statements. "
        "Return a JSON array of strings."
    )

prompt_library = PromptLibrary()