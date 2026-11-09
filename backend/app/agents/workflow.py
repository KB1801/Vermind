"""
Agent Workflow Directed Acyclic Graph (DAG) Execution Engine.
Sequences multi-agent steps with condition evaluation and state passing.
"""
from typing import Dict, Any, List, Callable
from backend.app.core.logger import logger

class AgentWorkflow:
    def __init__(self, workflow_name: str):
        self.workflow_name = workflow_name
        self.steps: List[Dict[str, Any]] = []

    def add_step(self, step_name: str, action_fn: Callable):
        self.steps.append({"name": step_name, "action": action_fn})

    async def execute(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        state = initial_state.copy()
        logger.info(f"Executing Agent Workflow [{self.workflow_name}] with {len(self.steps)} steps...")

        for step in self.steps:
            step_name = step["name"]
            action = step["action"]
            logger.debug(f"Workflow [{self.workflow_name}] running step: {step_name}")
            try:
                state = await action(state)
            except Exception as e:
                logger.error(f"Workflow step [{step_name}] failed: {str(e)}")
                state["error"] = str(e)
                break

        return state