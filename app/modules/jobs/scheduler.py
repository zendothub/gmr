"""Background job scheduler using APScheduler (in-process, async)."""

from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from app.modules.jobs.tasks import (
    aggregate_daily_analytics,
    close_stale_track_sessions,
    cleanup_old_storage,
    probe_camera_statuses,
    deduplicate_persons,
    cleanup_stale_sessions,
    minio_sweep_job,
)

_scheduler: Optional[AsyncIOScheduler] = None


def get_scheduler() -> Optional[AsyncIOScheduler]:
    """Get the scheduler instance (may be None if not started)."""
    return _scheduler


def start_scheduler() -> AsyncIOScheduler:
    """Create and start the background job scheduler."""
    global _scheduler
    if _scheduler and _scheduler.running:
        return _scheduler

    _scheduler = AsyncIOScheduler()

    # Daily analytics aggregation at 00:15 every day
    _scheduler.add_job(
        aggregate_daily_analytics,
        CronTrigger(hour=0, minute=15),
        id="daily_analytics_aggregation",
        replace_existing=True,
    )

    # Close stale track sessions every 5 minutes
    _scheduler.add_job(
        close_stale_track_sessions,
        IntervalTrigger(minutes=5),
        id="close_stale_track_sessions",
        replace_existing=True,
    )

    # Storage cleanup daily at 02:00
    _scheduler.add_job(
        cleanup_old_storage,
        CronTrigger(hour=2, minute=0),
        id="storage_cleanup",
        replace_existing=True,
    )

    # Camera RTSP status probe every 2 minutes
    # Updates camera.status → ACTIVE or INACTIVE based on live RTSP connectivity.
    # Cameras with MAINTENANCE status are skipped so operators are not overridden.
    _scheduler.add_job(
        probe_camera_statuses,
        IntervalTrigger(minutes=2),
        id="camera_status_probe",
        replace_existing=True,
    )

    # Periodic person-identity deduplication every 6 minutes.
    # Merges cross-camera duplicates that the real-time matcher missed
    # (cross-angle face similarity just below FACE_MATCH_THRESHOLD).
    # Incremental: only probes embeddings created in DEDUP_PROBE_WINDOW_MINUTES
    # against the full pgvector index (2026-10-08 — full-DB probing was the
    # 5.5-6 min bottleneck). The daily full sweep below covers the rest.
    _scheduler.add_job(
        deduplicate_persons,
        IntervalTrigger(minutes=6),
        id="deduplicate_persons",
        replace_existing=True,
    )

    # Daily whole-DB dedup pair scan (safety net for the incremental window):
    # catches pairs skipped during worker downtime, IVFFlat recall misses, and
    # pairs unblocked when the same-camera overlap lookback expires.
    from app.config import get_settings
    _settings = get_settings()
    _scheduler.add_job(
        deduplicate_persons,
        CronTrigger(
            hour=int(_settings.DEDUP_FULL_SWEEP_HOUR_IST),
            minute=30,
        ),
        kwargs={"full": True},
        id="deduplicate_persons_full_sweep",
        replace_existing=True,
    )

    # MinIO orphan-crop sweep (decoupled from the dedup cycle 2026-10-08 —
    # it used to hold the dedup DB transaction open through the bucket listing
    # and could stretch a cycle to 47 min on a delete backlog).
    _scheduler.add_job(
        minio_sweep_job,
        IntervalTrigger(minutes=int(_settings.MINIO_SWEEP_INTERVAL_MINUTES)),
        id="minio_sweep",
        replace_existing=True,
    )

    # Device session cleanup every 2 minutes
    _scheduler.add_job(
        cleanup_stale_sessions,
        IntervalTrigger(minutes=2),
        id="cleanup_stale_sessions",
        replace_existing=True,
    )

    _scheduler.start()
    logger.info("Background job scheduler started (8 jobs registered)")
    return _scheduler


def stop_scheduler():
    """Stop the background job scheduler."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Background job scheduler stopped")
    _scheduler = None