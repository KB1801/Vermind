"""
FastAPI System Telemetry & Resource Monitor Router.
Monitors memory usage, CPU load, and active models on 8GB RAM laptops.
"""
import psutil
from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter(prefix="/system", tags=["System Telemetry"])

@router.get("/metrics")
async def get_system_telemetry() -> Dict[str, Any]:
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=0.1)

    return {
        "ram_total_gb": round(mem.total / (1024**3), 2),
        "ram_available_gb": round(mem.available / (1024**3), 2),
        "ram_used_percent": mem.percent,
        "cpu_usage_percent": cpu,
        "status": "healthy" if mem.percent < 90.0 else "high_memory_load"
    }