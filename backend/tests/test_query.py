"""
Pytest Verification Suite for Hybrid Search & Dual Generation.
"""
import pytest
from backend.app.retrieval.qdrant_store import qdrant_store
from backend.app.retrieval.reranker import reranker
from backend.app.generation.offline_bot import offline_bot

def test_upsert_and_hybrid_search():
    sample_chunks = [
        {"chunk_id": "c1", "doc_id": "d1", "text": "VeriMind provides 19 RAG patterns with atomic claim verification."},
        {"chunk_id": "c2", "doc_id": "d1", "text": "PyTorch I-JEPA latent world model scores context alignment without tokens."}
    ]
    
    count = qdrant_store.upsert_chunks(sample_chunks, tenant_id="tenant_test")
    assert count == 2

    results = qdrant_store.hybrid_search("What is VeriMind?", top_k=2, tenant_id="tenant_test")
    assert len(results) > 0

def test_cross_encoder_rerank():
    candidates = [
        {"chunk_id": "c1", "text": "Unrelated apples and oranges text.", "score": 0.8},
        {"chunk_id": "c2", "text": "VeriMind features custom neural reranking engines.", "score": 0.5}
    ]
    reranked = reranker.rerank("VeriMind neural engines", candidates, top_k=2)
    assert len(reranked) > 0

def test_offline_extractive_bot():
    chunks = [{"chunk_id": "c1", "text": "VeriMind is engineered for 8GB RAM laptops."}]
    answer = offline_bot.synthesize_answer("What laptop does VeriMind support?", chunks)
    assert "8GB RAM" in answer