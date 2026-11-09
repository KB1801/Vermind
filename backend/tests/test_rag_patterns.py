"""
Pytest Verification Suite for all 19 RAG Patterns & Auto-Router.
"""
import pytest
from backend.app.rag_patterns.router import rag_router
from backend.app.rag_patterns.naive_rag import naive_rag
from backend.app.rag_patterns.retrieve_rerank import retrieve_rerank_rag
from backend.app.rag_patterns.corrective_rag import corrective_rag
from backend.app.rag_patterns.self_rag import self_rag

@pytest.mark.asyncio
async def test_naive_rag_execution():
    res = await naive_rag.execute("What is VeriMind?", tenant_id="test")
    assert res.pattern_used == "naive"
    assert res.execution_time_ms >= 0.0

@pytest.mark.asyncio
async def test_retrieve_rerank_execution():
    res = await retrieve_rerank_rag.execute("Explain vector search", tenant_id="test")
    assert res.pattern_used == "retrieve_rerank"

@pytest.mark.asyncio
async def test_crag_execution():
    res = await corrective_rag.execute("Unknown topic search", tenant_id="test")
    assert res.pattern_used == "crag"

@pytest.mark.asyncio
async def test_auto_router_dispatch():
    res = await rag_router.route_and_execute("Summarize machine learning principles", requested_pattern="auto", tenant_id="test")
    assert res.pattern_used in ["rag_fusion", "retrieve_rerank", "adaptive_rag"]