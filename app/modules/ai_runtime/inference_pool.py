"""Dedicated, capped thread pool for AI inference.

All camera workers route detection / embedding extraction through this
single small executor so the CPU/GPU is never oversubscribed when many
cameras submit work at the same time (the default asyncio.to_thread pool
is unbounded for practical purposes).
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Optional

from loguru import logger

from app.config import get_settings

_executor: Optional[ThreadPoolExecutor] = None


def _get_max_threads() -> int:
    """Get max inference threads from settings (default 10)."""
    return get_settings().MAX_WORKERS


def get_inference_executor() -> ThreadPoolExecutor:
    """Get (or lazily create) the shared inference executor."""
    global _executor
    if _executor is None:
        max_threads = _get_max_threads()
        _executor = ThreadPoolExecutor(
            max_workers=max_threads,
            thread_name_prefix="inference",
        )
        logger.info(f"Inference executor started (max_workers={max_threads})")
    return _executor


async def run_inference(func, *args, **kwargs):
    """Run a blocking inference call on the capped executor.

    Supports keyword args (e.g. YOLO.predict(..., conf=0.3, verbose=False))
    via functools.partial. Positional-only calls stay on the fast path.

    BUGFIX 2026-07-20: previous signature ignored kwargs, so
    `run_inference(yolo_pose.predict, crop, verbose=False, conf=0.3)` raised
    TypeError every frame. Pose was silently skipped (DEBUG-only catch) and
    track_sessions.bbox_history.torso_visibility_ratio was always null.
    """
    loop = asyncio.get_running_loop()
    executor = get_inference_executor()
    if kwargs:
        return await loop.run_in_executor(executor, partial(func, *args, **kwargs))
    return await loop.run_in_executor(executor, func, *args)


def shutdown_inference_executor():
    """Shut down the inference executor (called on app shutdown)."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)
        _executor = None
        logger.info("Inference executor shut down")
