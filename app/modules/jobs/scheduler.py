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

    # Periodic person-identity deduplication every 7 minutes.
    # Merges identities that were registered separately by different cameras for
    # the same physical person (cross-angle face similarity just below threshold).
    # Faster cadence reduces the window where duplicate person_ids inflate
    # the purchase count before the dashboard query aggregates them.
    _scheduler.add_job(
        deduplicate_persons,
        IntervalTrigger(minutes=7),
        id="deduplicate_persons",
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
    logger.info("Background job scheduler started (6 jobs registered)")
    return _scheduler


def stop_scheduler():
    """Stop the background job scheduler."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Background job scheduler stopped")
    _scheduler = None