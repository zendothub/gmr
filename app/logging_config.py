"""Shared loguru logging configuration.

Importing :func:`setup_logging` configures the same sinks (stderr + rotating
``logs/ai_processing.log``) used by the FastAPI API server, so background
processes (e.g. ``app.worker``) show up in the same log file instead of only in
the systemd journal.

Safe to call multiple times — removes existing sinks first.
"""

from __future__ import annotations

import os
import sys

from loguru import logger

from app.config import get_settings


def setup_logging() -> None:
    """Configure loguru sinks for a process (API server or background worker)."""
    settings = get_settings()

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
