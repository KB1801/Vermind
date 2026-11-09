"""
FastAPI Multi-Agent Execution Router.
Exposes endpoints for triggering collaborative multi-agent tasks.
"""
from fastapi import APIRouter, Form
from typing import Dict, Any
from backend.app.agents.multi_agent_orchestrator import multi_agent_orchestrator

router = APIRouter(prefix="/agent", tags=["Multi-Agent Orchestration"])

@router.post("/run-task")
async def execute_agent_task(query: str = Form(...), tenant_id: str = Form("default_tenant")) -> Dict[str, Any]:
    return await multi_agent_orchestrator.run_collaborative_task(query=query, tenant_id=tenant_id)