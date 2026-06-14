"""Analytics service - aggregated metrics for the dashboard."""

from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from fastapi import HTTPException, status
from loguru import logger
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db.models.camera import Camera, Zone
from app.core.db.models.event import Event
from app.core.db.models.billing import BillingInteraction
from app.core.db.models.tracking import TrackSession
from app.core.db.models.person import PersonIdentity
from app.modules.analytics.schemas import (
    FootfallPoint,
    FootfallResponse,
    BillingAnalyticsResponse,
    DwellAnalyticsResponse,
    ZoneOccupancyItem,
    ZoneOccupancyResponse,
    JourneyStep,
    PersonJourneyResponse,
    DemographicsBreakdown,
    DashboardSummaryResponse,
)
from app.utils.time_utils import utc_now


def _default_range(start_time: Optional[datetime], end_time: Optional[datetime]):
    """Default to the last 24 hours if no range is given."""
    end = end_time or utc_now()
    start = start_time or (end - timedelta(hours=24))
    return start, end


class AnalyticsService:

    @staticmethod
    async def get_footfall(
        db: AsyncSession,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        camera_id: Optional[UUID] = None,
        interval: str = "hour",
    ) -> FootfallResponse:
        """Footfall from entry-line crossing events (entry cameras)."""
        start, end = _default_range(start_time, end_time)

        query = select(Event).where(
            Event.event_type == "line_crossing",
            Event.occurred_at >= start,
            Event.occurred_at <= end,
            Event.is_false_positive.is_(False),
        )
        if camera_id:
            query = query.where(Event.camera_id == camera_id)

        total = (
            await db.execute(select(func.count()).select_from(query.subquery()))
        ).scalar() or 0

        # Unique visitors (distinct person identities seen in range)
        unique_q = select(func.count(func.distinct(Event.person_identity_id))).where(
            Event.event_type == "line_crossing",
            Event.occurred_at >= start,
            Event.occurred_at <= end,
            Event.person_identity_id.isnot(None),
            Event.is_false_positive.is_(False),
        )
        if camera_id:
            unique_q = unique_q.where(Event.camera_id == camera_id)
        unique_visitors = (await db.execute(unique_q)).scalar() or 0

        # Timeline buckets
        bucket = func.date_trunc(interval, Event.occurred_at)
        timeline_q = (
            select(bucket.label("bucket"), func.count().label("count"))
            .where(
                Event.event_type == "line_crossing",
                Event.occurred_at >= start,
                Event.occurred_at <= end,
                Event.is_false_positive.is_(False),
            )
            .group_by("bucket")
            .order_by("bucket")
        )
        if camera_id:
            timeline_q = timeline_q.where(Event.camera_id == camera_id)
        rows = (await db.execute(timeline_q)).all()

        return FootfallResponse(
            start_time=start,
            end_time=end,
            total_entries=total,
            unique_visitors=unique_visitors,
            timeline=[FootfallPoint(bucket=r.bucket, count=r.count) for r in rows],
        )

    @staticmethod
    async def get_billing_analytics(
        db: AsyncSession,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        camera_id: Optional[UUID] = None,
        interval: str = "hour",
    ) -> BillingAnalyticsResponse:
        """Billing counter interaction analytics."""
        start, end = _default_range(start_time, end_time)

        base = select(BillingInteraction).where(
            BillingInteraction.entered_at >= start,
            BillingInteraction.entered_at <= end,
        )
        if camera_id:
            base = base.where(BillingInteraction.camera_id == camera_id)

        total = (
            await db.execute(select(func.count()).select_from(base.subquery()))
        ).scalar() or 0

        agg_q = select(
            func.avg(BillingInteraction.dwell_seconds),
            func.max(BillingInteraction.dwell_seconds),
        ).where(
            BillingInteraction.entered_at >= start,
            BillingInteraction.entered_at <= end,
        )
        if camera_id:
            agg_q = agg_q.where(BillingInteraction.camera_id == camera_id)
        avg_dwell, max_dwell = (await db.execute(agg_q)).one()

        bucket = func.date_trunc(interval, BillingInteraction.entered_at)
        timeline_q = (
            select(bucket.label("bucket"), func.count().label("count"))
            .where(
                BillingInteraction.entered_at >= start,
                BillingInteraction.entered_at <= end,
            )
            .group_by("bucket")
            .order_by("bucket")
        )
        if camera_id:
            timeline_q = timeline_q.where(BillingInteraction.camera_id == camera_id)
        rows = (await db.execute(timeline_q)).all()

        return BillingAnalyticsResponse(
            start_time=start,
            end_time=end,
            total_interactions=total,
            avg_dwell_seconds=round(avg_dwell, 1) if avg_dwell else None,
            max_dwell_seconds=round(max_dwell, 1) if max_dwell else None,
            timeline=[FootfallPoint(bucket=r.bucket, count=r.count) for r in rows],
        )

    @staticmethod
    async def get_dwell_analytics(
        db: AsyncSession,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        camera_id: Optional[UUID] = None,
    ) -> DwellAnalyticsResponse:
        """Average dwell (track session duration) analytics."""
        start, end = _default_range(start_time, end_time)

        duration = func.extract(
            "epoch", TrackSession.last_seen_at - TrackSession.started_at
        )

        agg_q = select(func.avg(duration), func.count()).where(
            TrackSession.started_at >= start,
            TrackSession.started_at <= end,
        )
        if camera_id:
            agg_q = agg_q.where(TrackSession.camera_id == camera_id)
        avg_duration, total = (await db.execute(agg_q)).one()

        by_camera_q = (
            select(Camera.name, func.avg(duration))
            .join(Camera, Camera.id == TrackSession.camera_id)
            .where(
                TrackSession.started_at >= start,
                TrackSession.started_at <= end,
            )
            .group_by(Camera.name)
        )
        rows = (await db.execute(by_camera_q)).all()

        return DwellAnalyticsResponse(
            start_time=start,
            end_time=end,
            avg_track_duration_seconds=round(avg_duration, 1) if avg_duration else None,
            total_tracks=total or 0,
            by_camera={name: round(float(avg or 0), 1) for name, avg in rows},
        )

    @staticmethod
    async def get_zone_occupancy(
        db: AsyncSession,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> ZoneOccupancyResponse:
        """Event activity per zone in the given range."""
        start, end = _default_range(start_time, end_time)

        query = (
            select(
                Zone.id,
                Zone.name,
                Zone.zone_type,
                func.count(Event.id).label("event_count"),
            )
            .join(Event, Event.zone_id == Zone.id)
            .where(
                Event.occurred_at >= start,
                Event.occurred_at <= end,
                Event.is_false_positive.is_(False),
            )
            .group_by(Zone.id, Zone.name, Zone.zone_type)
            .order_by(func.count(Event.id).desc())
        )
        rows = (await db.execute(query)).all()

        return ZoneOccupancyResponse(
            start_time=start,
            end_time=end,
            zones=[
                ZoneOccupancyItem(
                    zone_id=r.id,
                    zone_name=r.name,
                    zone_type=r.zone_type.value if hasattr(r.zone_type, "value") else r.zone_type,
                    event_count=r.event_count,
                )
                for r in rows
            ],
        )

    @staticmethod
    async def get_dashboard_summary(
        db: AsyncSession,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        camera_id: Optional[UUID] = None,
    ) -> DashboardSummaryResponse:
        """
        Unified dashboard summary within a datetime range.

        Returns:
        - unique_persons: count of distinct person_identity_id in track sessions
        - total_entries: count of line_crossing events (true footfall, excludes false positives)
        - total_purchases: count of billing interactions
        - demographics: age-group and gender breakdown from PersonIdentity
        """
        start, end = _default_range(start_time, end_time)

        # --- Unique persons (distinct person_identity_id in TrackSession) ---
        unique_q = select(func.count(func.distinct(TrackSession.person_identity_id))).where(
            TrackSession.started_at >= start,
            TrackSession.started_at <= end,
            TrackSession.person_identity_id.isnot(None),
        )
        if camera_id:
            unique_q = unique_q.where(TrackSession.camera_id == camera_id)
        unique_persons = (await db.execute(unique_q)).scalar() or 0

        # --- Total entries (line_crossing events — true footfall count) ---
        entries_q = select(func.count(Event.id)).where(
            Event.event_type == "line_crossing",
            Event.occurred_at >= start,
            Event.occurred_at <= end,
            Event.is_false_positive.is_(False),
        )
        if camera_id:
            entries_q = entries_q.where(Event.camera_id == camera_id)
        total_entries = (await db.execute(entries_q)).scalar() or 0

        # --- Total purchases (billing interactions) ---
        purchases_q = select(func.count(BillingInteraction.id)).where(
            BillingInteraction.entered_at >= start,
            BillingInteraction.entered_at <= end,
        )
        if camera_id:
            purchases_q = purchases_q.where(BillingInteraction.camera_id == camera_id)
        total_purchases = (await db.execute(purchases_q)).scalar() or 0

        # --- Demographics: age-group & gender counts from PersonIdentity ---
        # We get distinct person_identity_ids seen in TrackSessions within the range,
        # then join to PersonIdentity to aggregate their demographic fields.
        distinct_persons_subq = (
            select(TrackSession.person_identity_id)
            .where(
                TrackSession.started_at >= start,
                TrackSession.started_at <= end,
                TrackSession.person_identity_id.isnot(None),
            )
            .distinct()
        )
        if camera_id:
            distinct_persons_subq = distinct_persons_subq.where(
                TrackSession.camera_id == camera_id
            )
        distinct_persons_subq = distinct_persons_subq.subquery()

        # Count per age_group
        age_q = select(
            PersonIdentity.age_group,
            func.count(PersonIdentity.id),
        ).where(
            PersonIdentity.id.in_(select(distinct_persons_subq.c.person_identity_id))
        ).group_by(PersonIdentity.age_group)

        age_rows = (await db.execute(age_q)).all()
        age_counts: dict[str, int] = {}
        for row in age_rows:
            key = (row[0] or "").strip().lower()
            count = row[1] or 0
            age_counts[key] = age_counts.get(key, 0) + count

        # Count per gender
        gender_q = select(
            PersonIdentity.gender,
            func.count(PersonIdentity.id),
        ).where(
            PersonIdentity.id.in_(select(distinct_persons_subq.c.person_identity_id))
        ).group_by(PersonIdentity.gender)

        gender_rows = (await db.execute(gender_q)).all()
        gender_counts: dict[str, int] = {}
        for row in gender_rows:
            key = (row[0] or "").strip().lower()
            count = row[1] or 0
            gender_counts[key] = gender_counts.get(key, 0) + count

        demographics = DemographicsBreakdown(
            children=age_counts.get("child", 0),
            teenager=age_counts.get("teen", 0) + age_counts.get("teenager", 0),
            adult=age_counts.get("adult", 0),
            senior_citizen=age_counts.get("senior", 0) + age_counts.get("senior citizen", 0),
            male=gender_counts.get("male", 0),
            female=gender_counts.get("female", 0),
        )

        return DashboardSummaryResponse(
            start_time=start,
            end_time=end,
            unique_persons=unique_persons,
            total_entries=total_entries,
            total_purchases=total_purchases,
            demographics=demographics,
        )

    @staticmethod
    async def get_person_journey(
        db: AsyncSession, person_id: UUID
    ) -> PersonJourneyResponse:
        """Reconstruct a person's journey across cameras from track sessions."""
        result = await db.execute(
            select(PersonIdentity).where(PersonIdentity.id == person_id)
        )
        person = result.scalar_one_or_none()
        if not person:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Person identity not found")

        sessions_q = (
            select(TrackSession, Camera.name)
            .join(Camera, Camera.id == TrackSession.camera_id)
            .where(TrackSession.person_identity_id == person_id)
            .order_by(TrackSession.started_at)
        )
        rows = (await db.execute(sessions_q)).all()

        journey = []
        for session, camera_name in rows:
            end_ts = session.ended_at or session.last_seen_at
            duration = (end_ts - session.started_at).total_seconds() if end_ts else None
            journey.append(
                JourneyStep(
                    camera_id=session.camera_id,
                    camera_name=camera_name,
                    track_session_id=session.id,
                    started_at=session.started_at,
                    ended_at=session.ended_at,
                    duration_seconds=round(duration, 1) if duration else None,
                )
            )

        events_q = (
            select(Event)
            .where(
                Event.person_identity_id == person_id,
                Event.is_false_positive.is_(False),
            )
            .order_by(Event.occurred_at)
            .limit(200)
        )
        events = (await db.execute(events_q)).scalars().all()

        return PersonJourneyResponse(
            person_identity_id=person.id,
            first_seen_at=person.first_seen_at,
            last_seen_at=person.last_seen_at,
            visit_count=person.visit_count,
            total_sessions=len(journey),
            journey=journey,
            events=[
                {
                    "id": str(e.id),
                    "event_type": e.event_type,
                    "camera_id": str(e.camera_id),
                    "occurred_at": e.occurred_at.isoformat(),
                    "description": e.description,
                }
                for e in events
            ],
        )