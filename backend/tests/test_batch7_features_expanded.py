"""
Pytest Verification Suite for MCP, Multi-Agent Orchestrator, SSE Router, and Exporters.
"""
import pytest
from backend.app.mcp.server import mcp_server
from backend.app.agents.multi_agent_orchestrator import multi_agent_orchestrator
from backend.app.agents.coding_agent import coding_agent

@pytest.mark.asyncio
async def test_mcp_list_tools():
    raw_req = '{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}'
    res_str = await mcp_server.handle_json_rpc(raw_req)
    assert "search_verimind" in res_str
    assert "verify_claim" in res_str

@pytest.mark.asyncio
async def test_multi_agent_orchestrator():
    res = await multi_agent_orchestrator.run_collaborative_task("What is VeriMind?", tenant_id="test")
    assert "final_answer" in res
    assert len(res["planner_agent_sub_goals"]) > 0

@pytest.mark.asyncio
async def test_coding_agent():
    res = await coding_agent.generate_code_solution("Write a function to calculate dot product", language="python")
    assert "def solution():" in res["generated_code"]
    assert res["status"] == "ready"
    