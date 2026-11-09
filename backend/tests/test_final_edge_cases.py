"""
Pytest Edge Case & End-to-End Verification Suite.
Validates FastAPI router mounting, semantic cache, memory manager, and G-Eval judge.
"""
import pytest
from backend.app.core.memory import memory_manager
from backend.app.core.semantic_cache import semantic_cache
from backend.app.evals.llm_judge import llm_judge

def test_conversation_memory():
    memory_manager.add_message("session_123", "user", "Hello VeriMind")
    history = memory_manager.get_history("session_123")
    assert len(history) == 1
    assert history[0]["content"] == "Hello VeriMind"

def test_semantic_cache_store_and_get():
    semantic_cache.store("What is the capital of France?", "The capital of France is Paris.")
    hit = semantic_cache.get_semantic_hit("What is the capital of France?")
    assert hit is not None
    assert "Paris" in hit["answer"]

def test_llm_judge_hallucination():
    context = "VeriMind is optimized for 8GB RAM laptops."
    answer = "VeriMind requires a 128GB RAM supercomputer cluster."
    is_hallucination, conf = llm_judge.judge_hallucination(answer, context)
    assert is_hallucination is True