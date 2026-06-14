"""Background job task implementations."""

from datetime import datetime, timedelta, date

from loguru import logger
from sqlalchemy import select, func, update

from app.core.db.session import AsyncSessionLocal
from app.core.db.models.event import Event
from app.core.db.models.billing import BillingInteraction
from app.core.db.models.tracking import TrackSession
from app.core.db.models.analytics import DailyAnalyticsSummary
from app.modules.storage.service import cleanup_old_objects as s3_cleanup_old_objects
from app.utils.time_utils import utc_now


async def aggregate_daily_analytics():
    """Aggregate yesterday's metrics into a single global daily_analytics_summary row."""
    yesterday: date = (utc_now() - timedelta(days=1)).date()
    day_start = datetime.combine(yesterday, datetime.min.time())
    day_end = day_start + timedelta(days=1)

    async with AsyncSessionLocal() as db:
        try:
            # Footfall: entry line crossings (across all cameras)
            total_footfall = (
                await db.execute(
                    select(func.count()).where(
                        Event.event_type == "line_crossing",
                        Event.occurred_at >= day_start,
                        Event.occurred_at < day_end,
                        Event.is_false_positive.is_(False),
                    )
                )
            ).scalar() or 0

            # Unique visitors via distinct person identities
            unique_visitors = (
                await db.execute(
                    select(func.count(func.distinct(TrackSession.person_identity_id))).where(
                        TrackSession.started_at >= day_start,
                        TrackSession.started_at < day_end,
                        TrackSession.person_identity_id.isnot(None),
                    )
                )
            ).scalar() or 0

            # Avg dwell
            duration = func.extract("epoch", TrackSession.last_seen_at - TrackSession.started_at)
            avg_dwell = (
                await db.execute(
                    select(func.avg(duration)).where(
                        TrackSession.started_at >= day_start,
                        TrackSession.started_at < day_end,
                    )
                )
            ).scalar()

            # Billing interactions
            total_billing = (
                await db.execute(
                    select(func.count()).where(
                        BillingInteraction.entered_at >= day_start,
                        BillingInteraction.entered_at < day_end,
                    )
                )
            ).scalar() or 0

            # Total events
            total_events = (
                await db.execute(
                    select(func.count()).where(
                        Event.occurred_at >= day_start,
                        Event.occurred_at < day_end,
                    )
                )
            ).scalar() or 0

            # Hourly footfall breakdown
            hourly_q = (
                select(
                    func.extract("hour", Event.occurred_at).label("hour"),
                    func.count().label("count"),
                )
                .where(
                    Event.event_type == "line_crossing",
                    Event.occurred_at >= day_start,
                    Event.occurred_at < day_end,
                    Event.is_false_positive.is_(False),
                )
                .group_by("hour")
            )
            hourly = {str(int(r.hour)): r.count for r in (await db.execute(hourly_q)).all()}

            # Upsert the single daily summary row
            existing = (
                await db.execute(
                    select(DailyAnalyticsSummary).where(
                        DailyAnalyticsSummary.summary_date == yesterday,
                    )
                )
            ).scalar_one_or_none()

            if existing:
                existing.total_footfall = total_footfall
                existing.unique_visitors = unique_visitors
                existing.avg_dwell_seconds = float(avg_dwell) if avg_dwell else None
                existing.total_billing_interactions = total_billing
                existing.total_events = total_events
                existing.hourly_footfall = hourly
            else:
                db.add(
                    DailyAnalyticsSummary(
                        summary_date=yesterday,
                        total_footfall=total_footfall,
                        unique_visitors=unique_visitors,
                        avg_dwell_seconds=float(avg_dwell) if avg_dwell else None,
                        total_billing_interactions=total_billing,
                        total_events=total_events,
                        hourly_footfall=hourly,
                    )
                )

            await db.commit()
            logger.info(f"Daily analytics aggregated for {yesterday}")
        except Exception as e:
            await db.rollback()
            logger.error(f"Daily analytics aggregation failed: {e}")



async def close_stale_track_sessions():
    """Close track sessions that stopped receiving updates (e.g. after crash)."""
    cutoff = utc_now() - timedelta(minutes=10)
    async with AsyncSessionLocal() as db:
        try:
            result = await db.execute(
                update(TrackSession)
                .where(TrackSession.is_active.is_(True), TrackSession.last_seen_at < cutoff)
                .values(is_active=False, ended_at=TrackSession.last_seen_at)
            )
            await db.commit()
            if result.rowcount:
                logger.info(f"Closed {result.rowcount} stale track sessions")
        except Exception as e:
            await db.rollback()
            logger.error(f"Stale track session cleanup failed: {e}")


async def cleanup_old_storage(retention_days: int = 30):
    """Delete snapshots/crops older than the retention period."""
    older_than = utc_now() - timedelta(days=retention_days)
    async with AsyncSessionLocal() as db:
        try:
            removed = await s3_cleanup_old_objects(db, older_than)
            await db.commit()
            logger.info(f"Storage cleanup job removed {removed} old objects")
        except Exception as e:
            await db.rollback()
            logger.error(f"Storage cleanup job failed: {e}")