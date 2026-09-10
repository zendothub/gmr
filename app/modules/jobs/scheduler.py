"""Background job scheduler using APScheduler (in-process, async)."""

from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from app.config import get_settings
from app.modules.jobs.tasks import (
    aggregate_daily_analytics,
    close_stale_track_sessions,
    cleanup_old_storage,
    probe_camera_statuses,
    deduplicate_persons,
    cleanup_stale_sessions,
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

    settings = get_settings()
    attendance_mode = settings.ATTENDANCE_MODE

    _scheduler = AsyncIOScheduler()
    job_count = 0

    # ── Jobs SKIPPED in attendance mode ───────────────────────────────
    # Attendance mode only matches registered employees — no anonymous
    # PersonIdentity creation, so dedup / analytics are unnecessary.

    if not attendance_mode:
        # Daily analytics aggregation at 00:15 every day
        _scheduler.add_job(
            aggregate_daily_analytics,
            CronTrigger(hour=0, minute=15),
            id="daily_analytics_aggregation",
            replace_existing=True,
        )
        job_count += 1

        # Periodic person-identity deduplication every 3 minutes.
        # Merges cross-camera duplicates that the real-time matcher missed.
        _scheduler.add_job(
            deduplicate_persons,
            IntervalTrigger(minutes=3),
            id="deduplicate_persons",
            replace_existing=True,
        )
        job_count += 1
    else:
        logger.info(
            "ATTENDANCE_MODE=True — skipping deduplicate_persons and "
            "aggregate_daily_analytics jobs (not needed for attendance)"
        )

    # ── Jobs ALWAYS needed ────────────────────────────────────────────

    # Close stale track sessions every 5 minutes
    _scheduler.add_job(
        close_stale_track_sessions,
        IntervalTrigger(minutes=5),
        id="close_stale_track_sessions",
        replace_existing=True,
    )
    job_count += 1

    # Storage cleanup daily at 02:00
    _scheduler.add_job(
        cleanup_old_storage,
        CronTrigger(hour=2, minute=0),
        id="storage_cleanup",
        replace_existing=True,
    )
    job_count += 1

    # Camera RTSP status probe every 2 minutes
    _scheduler.add_job(
        probe_camera_statuses,
        IntervalTrigger(minutes=2),
        id="camera_status_probe",
        replace_existing=True,
    )
    job_count += 1

    # Device session cleanup every 2 minutes
    _scheduler.add_job(
        cleanup_stale_sessions,
        IntervalTrigger(minutes=2),
        id="cleanup_stale_sessions",
        replace_existing=True,
    )
    job_count += 1

    _scheduler.start()
    logger.info(
        f"Background job scheduler started ({job_count} jobs registered"
        f"{', attendance_mode=ON' if attendance_mode else ''})"
    )
    return _scheduler


def stop_scheduler():
    """Stop the background job scheduler."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Background job scheduler stopped")
    _scheduler = None