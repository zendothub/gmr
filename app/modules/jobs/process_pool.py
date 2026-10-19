"""Spawn process-pool offload for heavy background-job bodies.

The background worker (``retail-ai-worker.service``) is a single asyncio
process. Heavy job bodies must not run on that process:

* MinIO listing/purge (minio-py is sync; 715k objects ≈ 72s per sweep)
* numpy contamination cleanup (median-outlier loops hold the GIL)
* billing-visit stitch re-embedding (lazily loads OSNet + InsightFace)

Those run as SYNC entry points in a spawn-context ``ProcessPoolExecutor``;
the event-loop process only awaits a Future and stays responsive.

**Spawn, not fork:** forking a running asyncio process that already has
executor threads is unsafe. Spawned children re-import this module tree and
call ``setup_logging()`` so job logs still land in ``logs/ai_processing.log``.

**When the pool breaks (``BrokenProcessPool``):** ANY dead child poisons the
whole pool (CPython fails every pending future — more ``max_workers`` does not
provide failover). A child dies on: native crash in torch/cv2/pgvector (the
stitch path loads OSNet + InsightFace), OOM-kill during model load or the
embedding parse, external ``kill -9``, or a spawn-time import failure. Broken
pools are handled by REBUILD + RETRY ONCE (a fresh worker survives transient
deaths like a previous task's OOM). If the retry also breaks the pool, the job
body is assumed fatal (e.g. it segfaults) and is ABORTED with an error — never
re-run in a thread, which would take down the worker process with it. Job
bodies must therefore be safe to run at most twice (they are: merges are
SAVEPOINT-guarded, the sweep only deletes unreferenced objects).

**Pool cannot start (``OSError``, e.g. process-spawn resource exhaustion):**
nothing has executed in a child, so falling back to a worker thread is safe —
the caller's event loop still never blocks.

**Proactive recycling:** the pool is rebuilt after ``JOB_POOL_MAX_TASKS``
completed jobs so child memory growth (model loads) stays bounded. A timed-out
or broken pool is torn down (``shutdown(wait=False)``) and rebuilt on the next
call so a stuck child cannot poison later runs.

Usage::

    from app.modules.jobs.process_pool import run_in_subprocess

    result = await run_in_subprocess(_blocking_entry, arg1, timeout=1800)
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Callable, Optional

from loguru import logger

_pool: Optional[ProcessPoolExecutor] = None
_tasks_completed: int = 0

# Default guard: no job body should run longer than this in the pool.
DEFAULT_TIMEOUT_SECONDS = 30 * 60


def _get_pool() -> ProcessPoolExecutor:
    """Lazily create the shared spawn-context pool."""
    global _pool, _tasks_completed
    if _pool is None:
        from app.config import get_settings
        settings = get_settings()
        workers = max(1, int(getattr(settings, "JOB_POOL_WORKERS", 3)))
        ctx = mp.get_context("spawn")
        _pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
        _tasks_completed = 0
        logger.info(f"Process pool: created (spawn, workers={workers})")
    return _pool


def _reset_pool(reason: str = "") -> None:
    """Detach and forget the current pool (its workers may be stuck or dead)."""
    global _pool
    pool, _pool = _pool, None
    if pool is not None:
        try:
            pool.shutdown(wait=False)
        except Exception:
            pass
        logger.info(f"Process pool: reset{' — ' + reason if reason else ''}")


def _note_task_done() -> None:
    """Recycle the pool after JOB_POOL_MAX_TASKS jobs to bound child memory."""
    global _tasks_completed
    _tasks_completed += 1
    from app.config import get_settings
    max_tasks = int(getattr(get_settings(), "JOB_POOL_MAX_TASKS", 20))
    if max_tasks > 0 and _tasks_completed >= max_tasks:
        _reset_pool(f"recycled after {_tasks_completed} tasks")


def _call_with_kwargs(fn: Callable, args: tuple, kwargs: dict) -> Any:
    """Module-level trampoline — process pools must pickle the callable, so no lambdas."""
    return fn(*args, **kwargs)


async def run_in_subprocess(
    fn: Callable,
    *args: Any,
    timeout: Optional[float] = DEFAULT_TIMEOUT_SECONDS,
    **kwargs: Any,
) -> Any:
    """Run the SYNC entry point ``fn(*args, **kwargs)`` in the process pool.

    ``fn`` must be a module-level function (picklable for spawn), self-contained
    (sets up its own logging and event loop), must not rely on parent-process
    state, and must be safe to run at most twice (broken-pool retry).

    Failure policy:
      * BrokenProcessPool → rebuild the pool, retry once on a fresh worker.
      * Still broken      → abort (RuntimeError). NOT re-run in a thread: if
        the body itself segfaults/OOMs a child it would kill the worker too.
      * OSError (no pool) → thread fallback (body never ran in a child).
      * Timeout           → reset pool, abandon the stuck child, raise.
    """
    name = getattr(fn, "__name__", repr(fn))
    loop = asyncio.get_running_loop()

    for attempt in (1, 2):
        try:
            fut = loop.run_in_executor(_get_pool(), _call_with_kwargs, fn, args, kwargs)
            result = await asyncio.wait_for(fut, timeout=timeout)
            _note_task_done()
            return result
        except asyncio.TimeoutError:
            _reset_pool(f"{name} exceeded {timeout}s")
            logger.error(
                f"Process pool: {name} exceeded {timeout}s — "
                "pool reset; the stuck child is abandoned"
            )
            raise
        except BrokenProcessPool as e:
            _reset_pool(f"broken after {name} (attempt {attempt})")
            logger.warning(
                f"Process pool: broken during {name} (attempt {attempt}/2): {e}"
            )
            if attempt == 2:
                logger.error(
                    f"Process pool: {name} broke the pool twice — aborting this "
                    "run (the body may crash its child; not retrying in-thread)"
                )
                raise RuntimeError(
                    f"job {name} aborted: process pool workers died twice"
                ) from e
            continue
        except OSError as e:
            # Pool could not start/run at all (spawn resource exhaustion).
            _reset_pool(f"cannot start ({e})")
            logger.warning(
                f"Process pool unavailable ({e}); falling back to worker thread"
            )
            return await asyncio.to_thread(fn, *args, **kwargs)
