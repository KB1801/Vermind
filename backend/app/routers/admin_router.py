"""
FastAPI System Status & Admin Management Router.
"""
from fastapi import APIRouter
from backend.app.config import settings
from backend.app.schemas import SystemStatusResponse
from backend.app.retrieval.qdrant_store import qdrant_store
from backend.app.core.cache import cache_manager

router = APIRouter(prefix="/admin", tags=["Admin & System Status"])

@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    qdrant_online = qdrant_store.client is not None
    redis_online = cache_manager.using_redis

    return SystemStatusResponse(
        status="healthy",
        version=settings.VERSION,
        qdrant_connected=qdrant_online,
        redis_connected=redis_online,
        offline_mode=settings.VERIMIND_OFFLINE_MODE,
        active_llm_provider="Groq (llama-3.3-70b-versatile)" if settings.GROQ_API_KEY else "Offline Extractive Engine",
        models_loaded={
            "intent_classifier": True,
            "ijepa_world_model": settings.ENABLE_JEPA_WORLD_MODEL,
            "cross_encoder": settings.ENABLE_CROSS_ENCODER
        }
    )