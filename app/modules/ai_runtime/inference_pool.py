"""Dedicated, capped thread pool for AI inference.

All camera workers route detection / embedding extraction through this
single small executor so the CPU/GPU is never oversubscribed when many
cameras submit work at the same time (the default asyncio.to_thread pool
is unbounded for practical purposes).
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from loguru import logger

_executor: Optional[ThreadPoolExecutor] = None
_MAX_INFERENCE_THREADS = 2


def get_inference_executor() -> ThreadPoolExecutor:
    """Get (or lazily create) the shared inference executor."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=_MAX_INFERENCE_THREADS,
            thread_name_prefix="inference",
        )
        logger.info(f"Inference executor started (max_workers={_MAX_INFERENCE_THREADS})")
    return _executor


async def run_inference(func, *args):
    """Run a blocking inference call on the capped executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(get_inference_executor(), func, *args)


def shutdown_inference_executor():
    """Shut down the inference executor (called on app shutdown)."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)
        _executor = None
        logger.info("Inference executor shut down")