"""
VeriMind Latency & Metrics Observability Tracker.
Monitors execution performance across RAG stages without blocking async loops.
"""
import time
from functools import wraps
from typing import Callable, Any
from backend.app.core.logger import logger

class ObservabilityTracker:
    def __init__(self):
        self.metrics_store = {
            "total_queries": 0,
            "failed_queries": 0,
            "latency_history_ms": []
        }

    def record_query_execution(self, latency_ms: float, success: bool = True):
        self.metrics_store["total_queries"] += 1
        if not success:
            self.metrics_store["failed_queries"] += 1
        self.metrics_store["latency_history_ms"].append(latency_ms)
        # Keep last 1000 records
        if len(self.metrics_store["latency_history_ms"]) > 1000:
            self.metrics_store["latency_history_ms"].pop(0)

    def get_summary(self) -> dict:
        history = self.metrics_store["latency_history_ms"]
        avg_latency = sum(history) / len(history) if history else 0.0
        return {
            "total_queries": self.metrics_store["total_queries"],
            "failed_queries": self.metrics_store["failed_queries"],
            "avg_latency_ms": round(avg_latency, 2)
        }

tracker = ObservabilityTracker()

def time_it(stage_name: str):
    """Decorator to measure execution time of any sync or async function."""
    def decorator(func: Callable):
        if hasattr(func, "__name__"):
            func_name = func.__name__
        else:
            func_name = stage_name

        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                elapsed = (time.perf_counter() - start_time) * 1000
                logger.debug(f"Stage [{stage_name}] completed in {elapsed:.2f}ms")
                return result
            except Exception as e:
                elapsed = (time.perf_counter() - start_time) * 1000
                logger.error(f"Stage [{stage_name}] failed after {elapsed:.2f}ms: {str(e)}")
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = (time.perf_counter() - start_time) * 1000
                logger.debug(f"Stage [{stage_name}] completed in {elapsed:.2f}ms")
                return result
            except Exception as e:
                elapsed = (time.perf_counter() - start_time) * 1000
                logger.error(f"Stage [{stage_name}] failed after {elapsed:.2f}ms: {str(e)}")
                raise

        import asyncio
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator