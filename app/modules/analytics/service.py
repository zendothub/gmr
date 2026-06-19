"""Analytics service - aggregated metrics for the dashboard."""

from datetime import datetime, timedelta
from typing import List, Optional
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
from collections import defaultdict
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
    DemographicsTableRow,
    DemographicsTableSummary,
    DemographicsTableResponse,
    VisitorEntryExitPoint,
    VisitorEntryExitResponse,
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

    # Age-group definitions and display labels (remapped from estimated_age)
    _AGE_GROUPS = [
        ("child", "Child (0–10)", 0, 10),
        ("teenager", "Teenager (11–17)", 11, 17),
        ("young_adult", "Young Adult (18–35)", 18, 35),
        ("middle_adult", "Middle Adult (36–55)", 36, 55),
        ("senior", "Senior (55+)", 55, 999),
    ]

    @classmethod
    def _remap_age_group(cls, estimated_age: Optional[int]) -> str:
        """Map raw estimated_age to one of our 5 standard age groups."""
        if estimated_age is None:
            return "unidentified"
        for ag_key, _label, lo, hi in cls._AGE_GROUPS:
            if lo <= estimated_age <= hi:
                return ag_key
        return "unidentified"

    @classmethod
    def _gender_key(cls, raw_gender: Optional[str]) -> str:
        """Normalise gender to 'male', 'female', or 'unidentified'."""
        if raw_gender:
            g = raw_gender.strip().upper()
            if g == "M":
                return "male"
            if g == "F":
                return "female"
        return "unidentified"

    @classmethod
    async def get_demographics_table(
        cls,
        db: AsyncSession,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        camera_id: Optional[UUID] = None,
    ) -> DemographicsTableResponse:
        """
        Cross-tabulated demographics: age_group × gender, plus per-group purchases.

        Counts unique persons (by person_identity_id) seen in TrackSessions within the
        time range.  ``summary.total_visitors`` is the total track session count.
        """
        start, end = _default_range(start_time, end_time)

        # ── 1. Total track sessions (visitors) ──────────────────────────
        sessions_q = select(func.count(TrackSession.id)).where(
            TrackSession.started_at >= start,
            TrackSession.started_at <= end,
        )
        if camera_id:
            sessions_q = sessions_q.where(TrackSession.camera_id == camera_id)
        total_visitors = (await db.execute(sessions_q)).scalar() or 0

        # ── 2. Distinct person IDs seen in range ────────────────────────
        person_subq = (
            select(TrackSession.person_identity_id)
            .where(
                TrackSession.started_at >= start,
                TrackSession.started_at <= end,
                TrackSession.person_identity_id.isnot(None),
            )
            .distinct()
        )
        if camera_id:
            person_subq = person_subq.where(TrackSession.camera_id == camera_id)
        person_subq = person_subq.subquery()

        # ── 3. Fetch demographics for those persons ─────────────────────
        rows_q = select(
            PersonIdentity.id,
            PersonIdentity.gender,
            PersonIdentity.estimated_age,
        ).where(PersonIdentity.id.in_(select(person_subq.c.person_identity_id)))
        
        rows = (await db.execute(rows_q)).all()

        # ── 4. Fetch per-person purchase counts ─────────────────────────
        purchase_subq = (
            select(BillingInteraction.person_identity_id)
            .where(
                BillingInteraction.entered_at >= start,
                BillingInteraction.entered_at <= end,
                BillingInteraction.person_identity_id.isnot(None),
            )
        )
        if camera_id:
            purchase_subq = purchase_subq.where(BillingInteraction.camera_id == camera_id)
        purchase_subq = purchase_subq.subquery()

        purchase_q = select(
            purchase_subq.c.person_identity_id,
            func.count().label("purchase_count"),
        ).group_by(purchase_subq.c.person_identity_id)

        purchase_rows = (await db.execute(purchase_q)).all()
        purchase_map: dict[str, int] = {}
        for pid, cnt in purchase_rows:
            purchase_map[str(pid)] = cnt

        # ── 5. Cross-tabulate age_group × gender ────────────────────────
        # structure: matrix[age_group][gender] = count
        matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # Track which person IDs we've already counted (unique persons)
        seen_persons: set[UUID] = set()

        for person_id, raw_gender, estimated_age in rows:
            if person_id in seen_persons:
                continue
            seen_persons.add(person_id)

            ag = cls._remap_age_group(estimated_age)
            g = cls._gender_key(raw_gender)
            matrix[ag][g] += 1

        # Also count purchase totals per age group
        age_purchase_totals: dict[str, int] = defaultdict(int)
        for person_id, raw_gender, estimated_age in rows:
            ag = cls._remap_age_group(estimated_age)
            purchases = purchase_map.get(str(person_id), 0)
            age_purchase_totals[ag] += purchases

        # ── 6. Build ordered response rows ──────────────────────────────
        demographics: List[DemographicsTableRow] = []
        for ag_key, label, _lo, _hi in cls._AGE_GROUPS:
            males = matrix[ag_key].get("male", 0)
            females = matrix[ag_key].get("female", 0)
            unidentified = matrix[ag_key].get("unidentified", 0)
            total = males + females + unidentified
            purchases = age_purchase_totals.get(ag_key, 0)

            demographics.append(
                DemographicsTableRow(
                    age_group=ag_key,
                    label=label,
                    male_count=males,
                    female_count=females,
                    unidentified_count=unidentified,
                    total_count=total,
                    total_purchase_count=purchases,
                )
            )

        # ── 7. Summary ──────────────────────────────────────────────────
        summary = DemographicsTableSummary(
            total_male=sum(r.male_count for r in demographics),
            total_female=sum(r.female_count for r in demographics),
            total_unidentified=sum(r.unidentified_count for r in demographics),
            total_visitors=total_visitors,
            total_purchases=sum(r.total_purchase_count for r in demographics),
        )

        logger.info(
            f"Demographics table: {len(demographics)} rows, "
            f"unique_persons={summary.total_male + summary.total_female + summary.total_unidentified}, "
            f"visitors={total_visitors}, purchases={summary.total_purchases}"
        )

        return DemographicsTableResponse(demographics=demographics, summary=summary)

    @staticmethod
    def _resolve_group_by(group_by: str, range_days: float) -> str:
        """
        Resolve 'auto' to a concrete granularity based on range length.

        Auto rules:
          ≤ 2 days   → hour   (24–48 points)
          ≤ 30 days  → day    (up to 30 points)
          ≤ 180 days → week   (up to ~26 points)
          > 180 days → month  (monthly aggregation)
        """
        if group_by != "auto":
            return group_by
        if range_days <= 2:
            return "hour"
        if range_days <= 30:
            return "day"
        if range_days <= 180:
            return "week"
        return "month"

    @staticmethod
    def _truncate_slot(dt: datetime, resolved: str) -> datetime:
        """Truncate a datetime to the start of its slot (hour/day/week/month)."""
        if resolved == "hour":
            return dt.replace(minute=0, second=0, microsecond=0)
        if resolved == "day":
            return dt.replace(hour=0, minute=0, second=0, microsecond=0)
        if resolved == "week":
            # ISO week starts on Monday
            day_start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
            return day_start - timedelta(days=day_start.weekday())
        # month
        return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    @staticmethod
    def _next_slot(dt: datetime, resolved: str) -> datetime:
        """Advance a slot cursor by one slot."""
        if resolved == "hour":
            return dt + timedelta(hours=1)
        if resolved == "day":
            return dt + timedelta(days=1)
        if resolved == "week":
            return dt + timedelta(weeks=1)
        # month – advance to 1st of next month
        if dt.month == 12:
            return dt.replace(year=dt.year + 1, month=1, day=1)
        return dt.replace(month=dt.month + 1, day=1)

    @staticmethod
    def _slot_label(dt: datetime, resolved: str) -> str:
        """Format the display label for a slot."""
        if resolved == "hour":
            return dt.strftime("%H:%M")
        if resolved == "day":
            return dt.strftime("%Y-%m-%d")
        if resolved == "week":
            return dt.strftime("%Y-W%W")   # e.g. "2026-W25"
        return dt.strftime("%Y-%m")        # e.g. "2026-06"

    @staticmethod
    async def get_entry_exit_hourly(
        db: AsyncSession,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        camera_id: Optional[UUID] = None,
        group_by: str = "auto",
    ) -> VisitorEntryExitResponse:
        """
        Entry/exit counts for the requested date range, grouped by hour / day / week / month.

        - entry  → line_crossing_forward  events
        - exit   → line_crossing_backward events
        - group_by auto-detection:
            ≤ 2 days   → hour  (great for today / yesterday)
            ≤ 30 days  → day   (great for last-7 / last-30)
            ≤ 180 days → week  (great for last-90 / last-6-months)
            > 180 days → month (great for full-year view)

        All slots between start and end are always returned (zeros for empty
        slots) so the frontend gets a gapless series for the graph.
        """
        start, end = _default_range(start_time, end_time)

        range_days = (end - start).total_seconds() / 86400
        resolved = AnalyticsService._resolve_group_by(group_by, range_days)

        # -- build DB query with the appropriate PostgreSQL bucket -------
        # For week/month, date_trunc uses "week" / "month" in Postgres
        pg_trunc = resolved  # hour/day/week/month all valid in date_trunc
        bucket_expr = func.date_trunc(pg_trunc, Event.occurred_at)

        q = (
            select(
                bucket_expr.label("bucket"),
                Event.event_type,
                func.count(Event.id).label("cnt"),
            )
            .where(
                Event.event_type.in_(["line_crossing_forward", "line_crossing_backward"]),
                Event.occurred_at >= start,
                Event.occurred_at <= end,
                Event.is_false_positive.is_(False),
            )
            .group_by("bucket", Event.event_type)
            .order_by("bucket")
        )
        if camera_id:
            q = q.where(Event.camera_id == camera_id)

        rows = (await db.execute(q)).all()

        # -- index results by bucket datetime ----------------------------
        counts: dict = defaultdict(
            lambda: {"line_crossing_forward": 0, "line_crossing_backward": 0}
        )
        for row in rows:
            counts[row.bucket][row.event_type] = row.cnt

        # -- generate all slots between start and end --------------------
        slot_cursor = AnalyticsService._truncate_slot(start, resolved)
        data: List[VisitorEntryExitPoint] = []

        while slot_cursor <= end:
            slot_end = AnalyticsService._next_slot(slot_cursor, resolved)
            bucket_data = counts.get(slot_cursor, {})
            data.append(
                VisitorEntryExitPoint(
                    label=AnalyticsService._slot_label(slot_cursor, resolved),
                    slot_start=slot_cursor,
                    slot_end=slot_end,
                    entry=bucket_data.get("line_crossing_forward", 0),
                    exit=bucket_data.get("line_crossing_backward", 0),
                )
            )
            slot_cursor = slot_end

        total_entry = sum(p.entry for p in data)
        total_exit = sum(p.exit for p in data)

        logger.info(
            f"Entry/Exit ({resolved}): range={start}–{end}, "
            f"total_entry={total_entry}, total_exit={total_exit}, slots={len(data)}"
        )

        return VisitorEntryExitResponse(
            start_time=start,
            end_time=end,
            group_by=resolved,
            total_entry=total_entry,
            total_exit=total_exit,
            data=data,
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