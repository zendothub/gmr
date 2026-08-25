"""Retail AI Platform - FastAPI application entry point.

Single modular FastAPI application for on-prem AI CCTV retail analytics.
All modules live inside this app with clean boundaries so they can be
separated into services later if needed.
"""

import sys

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.config import get_settings
from app.lifecycle import lifespan

# Module routers
from app.modules.auth.router import router as auth_router
from app.modules.users.router import router as users_router
from app.modules.cameras.router import router as cameras_router
from app.modules.cameras.v2_router import v2_router as cameras_v2_router
from app.modules.zones.router import router as zones_router
from app.modules.streaming.router import router as streaming_router

from app.modules.rules.router import router as rules_router
from app.modules.ai_runtime.router import router as runtime_router
from app.modules.events.router import router as events_router
from app.modules.billing.router import router as billing_router
from app.modules.analytics.router import router as analytics_router, v2_router as analytics_v2_router
from app.modules.storage.router import router as storage_router
from app.modules.feature_requests.router import router as feature_requests_router
from app.modules.stores.router import router as stores_router
from app.modules.debug.router import router as debug_router
from app.modules.recording.router import router as recording_router

settings = get_settings()

# Configure logging
import os
os.makedirs("logs", exist_ok=True)

logger.remove()
logger.add(
    sys.stderr,
    level=settings.LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
)
logger.add(
    "logs/ai_processing.log",
    rotation="50 MB",
    retention="7 days",
    level=settings.LOG_LEVEL,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "On-prem AI CCTV retail analytics platform for pharmacies. "
        "Cameras, zones, rules, AI runtime (YOLO + ByteTrack + OSNet ReID "
        "with pgvector), events, billing interactions and analytics."

    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS for dashboard integration (origins configured via CORS_ORIGINS env var)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------
# Global error handler
# ----------------------------------------------------------------------

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all error handler with logging."""
    logger.exception(f"Unhandled error on {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ----------------------------------------------------------------------
# Routers
# ----------------------------------------------------------------------

app.include_router(auth_router)
app.include_router(users_router)
app.include_router(cameras_router)
app.include_router(cameras_v2_router)

app.include_router(zones_router)
app.include_router(streaming_router)
app.include_router(rules_router)

app.include_router(runtime_router)
app.include_router(events_router)
app.include_router(billing_router)
app.include_router(analytics_router)
app.include_router(analytics_v2_router)
app.include_router(storage_router)
app.include_router(feature_requests_router)
app.include_router(stores_router)
app.include_router(debug_router)
app.include_router(recording_router)


# ----------------------------------------------------------------------
# Health
# ----------------------------------------------------------------------

@app.get("/", tags=["Health"])
async def root():
    """Service identification."""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
    }


@app.get("/health", tags=["Health"])
async def health():
    """Liveness/readiness probe."""
    return {"status": "ok"}