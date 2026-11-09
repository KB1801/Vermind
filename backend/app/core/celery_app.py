"""
Background Async Task Queue Orchestrator Fallback.
Provides async task scheduling interfaces with graceful local fallback execution.
"""
import asyncio
from typing import Callable, Any
from backend.app.core.logger import logger

class TaskQueueManager:
    async def run_in_background(self, task_fn: Callable, *args, **kwargs) -> Any:
        """Executes a long-running heavy function asynchronously without blocking main event loop."""
        logger.info(f"Dispatching background task: {task_fn.__name__}")
        return await asyncio.to_thread(task_fn, *args, **kwargs)

task_queue = TaskQueueManager()