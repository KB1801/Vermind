"""
VeriMind Master FastAPI Web Application Entry Point.
Mounts all 15 API Routers, CORS Middleware, Lifecycle Event Handlers, and Exception Handlers.
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

from backend.app.config import settings
from backend.app.core.logger import logger
from backend.app.core.exceptions import VeriMindException

# Import All Routers
from backend.app.routers import (
    ingestion_router,
    query_router,
    rag_router,
    admin_router,
    ml_router,
    feedback_router,
    stream_router,
    timeline_router,
    graph_router,
    replay_router,
    export_router,
    agent_router,
    system_status_router,
    visualizer_3d_router
)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Enterprise Dual-Engine Hybrid RAG & Agentic AI Platform with Atomic Verification",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Latency Middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Process-Time-MS"] = f"{process_time:.2f}"
    return response

# Custom Exception Handler
@app.exception_handler(VeriMindException)
async def verimind_exception_handler(request: Request, exc: VeriMindException):
    logger.error(f"Domain Error on {request.url.path}: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message, "details": exc.details}
    )

# Include All API Routers
app.include_router(ingestion_router.router, prefix=settings.API_V1_STR)
app.include_router(query_router.router, prefix=settings.API_V1_STR)
app.include_router(rag_router.router, prefix=settings.API_V1_STR)
app.include_router(admin_router.router, prefix=settings.API_V1_STR)
app.include_router(ml_router.router, prefix=settings.API_V1_STR)
app.include_router(feedback_router.router, prefix=settings.API_V1_STR)
app.include_router(stream_router.router, prefix=settings.API_V1_STR)
app.include_router(timeline_router.router, prefix=settings.API_V1_STR)
app.include_router(graph_router.router, prefix=settings.API_V1_STR)
app.include_router(replay_router.router, prefix=settings.API_V1_STR)
app.include_router(export_router.router, prefix=settings.API_V1_STR)
app.include_router(agent_router.router, prefix=settings.API_V1_STR)
app.include_router(system_status_router.router, prefix=settings.API_V1_STR)
app.include_router(visualizer_3d_router.router, prefix=settings.API_V1_STR)

@app.on_event("startup")
async def startup_event():
    logger.info("==========================================================")
    logger.info(f"✓ VeriMind Engine v{settings.VERSION} Starting Up...")
    logger.info(f"✓ Active Mode: {'OFFLINE (Local Extractive)' if settings.VERIMIND_OFFLINE_MODE else 'ONLINE (Groq llama-3.3-70b)'}")
    logger.info("==========================================================")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("✓ VeriMind Engine Server Gracefully Shutting Down...")

@app.get("/")
async def root():
    return {
        "platform": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "documentation": "/docs"
    }