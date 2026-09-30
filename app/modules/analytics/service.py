"""Analytics service - aggregated metrics for the dashboard."""

from datetime import datetime, timedelta, timezone
from typing import List, Optional
from uuid import UUID

from fastapi import HTTPException, status
from loguru import logger
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings

settings = get_settings()

from app.core.db.models.camera import Camera, Zone
from app.core.db.models.event import Event
from app.core.db.models.billing import BillingInteraction
from app.core.db.models.tracking import TrackSession
from app.core.db.models.person import PersonIdentity, PersonEmbedding
from collections import defaultdict

# ── Shared staff exclusion ───────────────────────────────────────────────
# Used by V2 dashboard + metrics (footfall, gender, age, purchase) and
# billing counts so employees don't inflate customer analytics.
# Staff classification is a live lazy check (see staff_classifier.py).
_STAFF_IDS = select(PersonIdentity.id).where(PersonIdentity.is_staff.is_(True))
_NOT_STAFF = PersonIdentity.is_staff.is_(False)
# ── Seen-in-window person semantics ──────────────────────────────────────
# BUG FIX (2026-09-21): person counting previously keyed on
# PersonIdentity.last_seen_at — a moving attribute that is overwritten on
# every repeat visit. A visitor seen on Sep 1 who returned on Sep 10 was
# retroactively REMOVED from Sep 1's historical counts and re-attributed to
# Sep 10. All person-based counts (footfall, gender, age, demographics)
# must use "had a track session starting within the window" semantics
# (first-seen per day), matching the debug date filter and the
# store-accuracy methodology. first_seen_at/last_seen_at remain
# informational fields on responses — only COUNT/BUCKET logic changed.

def _seen_in_window(start: datetime, end: datetime, cam_ids=None):
    """Person qualifies if they have ≥1 track session starting in [start, end].

    Returns a SQLAlchemy IN-clause condition on PersonIdentity.id.
    """
    subq = select(TrackSession.person_identity_id).where(
        TrackSession.started_at >= start,
        TrackSession.started_at <= end,
        TrackSession.person_identity_id.isnot(None),
    )
    if cam_ids:
        subq = subq.where(TrackSession.camera_id.in_(cam_ids))
    return PersonIdentity.id.in_(subq)

def _ist_slot_expr(expr, slot: str):
    """date_trunc(slot, expr in IST) for time-series bucketing."""
    return func.date_trunc(slot, func.timezone("Asia/Kolkata", expr))

from app.modules.analytics.schemas import (
    AnalyticsMetricsResponse,
    FootfallMetricData,
    GenderMetricData,
    AgeGroupsMetricData,
    PurchaseMetricData,
    PeakHourInfo,
    BusiestDayInfo,
    CameraBreakdownPoint,
    PeriodComparisonPoint,
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
    DashboardV2Response,
    DashboardV2FootfallMetric,
    DashboardV2GenderMetric,
    DashboardV2AgeGroupsMetric,
    DashboardV2PurchaseMetric,
    DashboardV2FootfallPoint,
    DashboardV2GenderTrendPoint,
    DashboardV2AgeGroupDistributionPoint,
)
from app.utils.time_utils import utc_now

# IST timezone (UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))


def _default_range(start_time: Optional[datetime], end_time: Optional[datetime]):
    """Default to the last 24 hours if no range is given."""
    end = end_time or utc_now()
    start = start_time or (end - timedelta(hours=24))
    return start, end


async def _resolve_camera_ids(
    db: AsyncSession,
    camera_id: Optional[UUID] = None,
    store_id: Optional[UUID] = None,
) -> Optional[list[UUID]]:
    """Resolve camera_id/store_id to a list of camera UUIDs for filtering.

    - If camera_id is provided, returns [camera_id] (explicit camera filter).
    - If store_id is provided, returns all camera IDs linked to that store.
    - If both are provided, camera_id takes precedence.
    - If neither is provided, returns None (no camera filter).
    """
    if camera_id:
        return [camera_id]
    if store_id:
        result = await db.execute(
            select(Camera.id).where(Camera.store_id == store_id)
        )
        cam_ids = [row[0] for row in result.all()]
        return cam_ids if cam_ids else None  # None if store has no cameras → no results
    return None


class AnalyticsService:

    @staticmethod
    async def get_footfall(
        db: AsyncSession,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        camera_id: Optional[UUID] = None,
        store_id: Optional[UUID] = None,
        interval: str = "hour",
    ) -> FootfallResponse:
        """Footfall from entry-line crossing events (entry cameras)."""
        start, end = _default_range(start_time, end_time)
        cam_ids = await _resolve_camera_ids(db, camera_id=camera_id, store_id=store_id)

        query = select(Event).where(
            Event.event_type == "line_crossing",
            Event.occurred_at >= start,
            Event.occurred_at <= end,
            Event.is_false_positive.is_(False),
        )
        if cam_ids:
            query = query.where(Event.camera_id.in_(cam_ids))

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
        if cam_ids:
            unique_q = unique_q.where(Event.camera_id.in_(cam_ids))
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
        if cam_ids:
            timeline_q = timeline_q.where(Event.camera_id.in_(cam_ids))
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
        store_id: Optional[UUID] = None,
        interval: str = "hour",
    ) -> BillingAnalyticsResponse:
        """Billing counter interaction analytics."""
        start, end = _default_range(start_time, end_time)
        cam_ids = await _resolve_camera_ids(db, camera_id=camera_id, store_id=store_id)

        base = select(BillingInteraction).where(
            BillingInteraction.entered_at >= start,
            BillingInteraction.entered_at <= end,
            BillingInteraction.person_identity_id.notin_(_STAFF_IDS),
        )
        if cam_ids:
            base = base.where(BillingInteraction.camera_id.in_(cam_ids))

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
        if cam_ids:
            agg_q = agg_q.where(BillingInteraction.camera_id.in_(cam_ids))
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
        if cam_ids:
            timeline_q = timeline_q.where(BillingInteraction.camera_id.in_(cam_ids))
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
        store_id: Optional[UUID] = None,
    ) -> DwellAnalyticsResponse:
        """Average dwell (track session duration) analytics."""
        start, end = _default_range(start_time, end_time)
        cam_ids = await _resolve_camera_ids(db, camera_id=camera_id, store_id=store_id)

        duration = func.extract(
            "epoch", TrackSession.last_seen_at - TrackSession.started_at
        )

        agg_q = select(func.avg(duration), func.count()).where(
            TrackSession.started_at >= start,
            TrackSession.started_at <= end,
        )
        if cam_ids:
            agg_q = agg_q.where(TrackSession.camera_id.in_(cam_ids))
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
        store_id: Optional[UUID] = None,
    ) -> DashboardSummaryResponse:
        """
        Unified dashboard summary within a datetime range.

        Returns:
        Returns:
        - unique_persons: count of persons SEEN in range (track session started
          in range — first-seen semantics; NOT last_seen_at, which shifts on
          repeat visits)
        - total_entries: count of line_crossing events (true footfall, excludes false positives)
        - total_purchases: count of billing interactions
        - demographics: age-group and gender breakdown from PersonIdentity
        """
        start, end = _default_range(start_time, end_time)
        cam_ids = await _resolve_camera_ids(db, camera_id=camera_id, store_id=store_id)

        seen = _seen_in_window(start, end, cam_ids)
        unique_q = select(func.count(PersonIdentity.id)).where(seen)
        unique_persons = (await db.execute(unique_q)).scalar() or 0

        # --- Total entries (line_crossing events — true footfall count) ---
        entries_q = select(func.count(Event.id)).where(
            Event.event_type == "line_crossing",
            Event.occurred_at >= start,
            Event.occurred_at <= end,
            Event.is_false_positive.is_(False),
        )
        if cam_ids:
            entries_q = entries_q.where(Event.camera_id.in_(cam_ids))
        total_entries = (await db.execute(entries_q)).scalar() or 0

        # --- Total purchases (billing interactions, excl. staff) ---
        purchases_q = select(func.count(func.distinct(BillingInteraction.person_identity_id))).where(
            BillingInteraction.entered_at >= start,
            BillingInteraction.entered_at <= end,
            BillingInteraction.person_identity_id.notin_(_STAFF_IDS),
        )
        if cam_ids:
            purchases_q = purchases_q.where(BillingInteraction.camera_id.in_(cam_ids))
        total_purchases = (await db.execute(purchases_q)).scalar() or 0

        demo_base = [
            _seen_in_window(start, end, cam_ids),
        ]

        age_q = select(
            PersonIdentity.age_group,
            func.count(PersonIdentity.id),
        ).where(*demo_base).group_by(PersonIdentity.age_group)

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
        ).where(*demo_base).group_by(PersonIdentity.gender)

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
        store_id: Optional[UUID] = None,
    ) -> DemographicsTableResponse:
        """
        Cross-tabulated demographics: age_group × gender, plus per-group purchases.

        Counts unique persons SEEN in range (track session started in range —
        first-seen semantics; NOT last_seen_at, which shifts on repeat visits).
        """
        start, end = _default_range(start_time, end_time)
        cam_ids = await _resolve_camera_ids(db, camera_id=camera_id, store_id=store_id)

        rows_q = select(
            PersonIdentity.id,
            PersonIdentity.gender,
            PersonIdentity.estimated_age,
        ).where(_seen_in_window(start, end, cam_ids))
        
        rows = (await db.execute(rows_q)).all()
        total_visitors = len(rows)

        # ── 4. Fetch per-person purchase counts ─────────────────────────
        purchase_subq = (
            select(BillingInteraction.person_identity_id)
            .where(
                BillingInteraction.entered_at >= start,
                BillingInteraction.entered_at <= end,
                BillingInteraction.person_identity_id.isnot(None),
            )
        )
        if cam_ids:
            purchase_subq = purchase_subq.where(BillingInteraction.camera_id.in_(cam_ids))
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
          ≤ 1 day  → hour  (24 data points for "today")
          > 1 day  → day   (one data point per calendar day)
        """
        if group_by != "auto":
            return group_by
        if range_days <= 1:
            return "hour"
        return "day"

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
        """Format the display label for a slot.

        hour → "2 PM", "12 AM", "12 PM"   (12-hour clock + AM/PM, no leading zero)
        day  → "14th Jul", "1st Aug"       (ordinal day + abbreviated month)
        """
        if resolved == "hour":
            h = dt.hour
            if h == 0:
                return "12 AM"
            if h < 12:
                return f"{h} AM"
            if h == 12:
                return "12 PM"
            return f"{h - 12} PM"
        if resolved == "day":
            day = dt.day
            # Ordinal suffix — 11th/12th/13th are special cases
            if 11 <= day <= 13:
                suffix = "th"
            else:
                suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
            month_abbr = dt.strftime("%b")  # "Jan", "Feb", …, "Dec"
            return f"{day}{suffix} {month_abbr}"
        if resolved == "week":
            return dt.strftime("%Y-W%W")   # e.g. "2026-W25"
        return dt.strftime("%Y-%m")        # e.g. "2026-06"

    @staticmethod
    async def get_entry_exit_hourly(
        db: AsyncSession,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        camera_id: Optional[UUID] = None,
        store_id: Optional[UUID] = None,
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
        cam_ids = await _resolve_camera_ids(db, camera_id=camera_id, store_id=store_id)

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
        if cam_ids:
            q = q.where(Event.camera_id.in_(cam_ids))

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

    # ── V2 Dashboard age-group bins (matches UI card) ─────────────────────────
    _V2_AGE_BINS = [
        ("under_18",   "Under 18",  0,   17),
        ("age_18_24",  "18-24",    18,   24),
        ("age_25_34",  "25-34",    25,   34),
        ("age_35_44",  "35-44",    35,   44),
        ("age_45_60",  "45-60",    45,   60),
        ("age_60_plus","60+",      61,  999),
    ]

    @classmethod
    def _v2_age_bin(cls, estimated_age: Optional[int]) -> str:
        """Map estimated_age to the 6 UI age-group keys (or 'unidentified')."""
        if estimated_age is None:
            return "unidentified"
        for key, _label, lo, hi in cls._V2_AGE_BINS:
            if lo <= estimated_age <= hi:
                return key
        return "unidentified"

    @staticmethod
    def _v2_gender(raw: Optional[str]) -> str:
        if not raw:
            return "unidentified"
        g = raw.strip().upper()
        if g in ("M", "MALE"):
            return "male"
        if g in ("F", "FEMALE"):
            return "female"
        return "unidentified"

    @staticmethod
    def _v2_pct_change(current: int, prev: int) -> Optional[float]:
        """Return % change rounded to 1 dp, or None if previous period is 0."""
        if prev == 0:
            return None
        return round((current - prev) / prev * 100, 1)

    @staticmethod
    def _v2_resolve_range(time_range: str, start_time: Optional[datetime], end_time: Optional[datetime]):
        """Return (start, end, prev_start, prev_end) for the requested time_range.
        
        All times are computed in IST (Asia/Kolkata, UTC+5:30) to match user expectations.
        """
        now = datetime.now(IST)  # Current time in IST
        
        if time_range == "today":
            # Start of today in IST (00:00 IST)
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = now
            duration = end - start
            return start, end, start - duration, start
        
        if time_range == "weekly":
            # Last 7 days: 00:00 IST exactly 7 days ago → now
            start = (now - timedelta(days=7)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            end = now
            return start, end, start - timedelta(days=7), start
        
        # custom range: treat naive datetimes as IST
        if end_time is None:
            end = now
        elif end_time.tzinfo is None:
            # Naive datetime from frontend → assume IST
            end = end_time.replace(tzinfo=IST)
        else:
            end = end_time
        
        if start_time is None:
            start = end - timedelta(hours=24)
        elif start_time.tzinfo is None:
            # Naive datetime from frontend → assume IST
            start = start_time.replace(tzinfo=IST)
        else:
            start = start_time
        
        duration = end - start
        return start, end, start - duration, start

    @classmethod
    async def get_dashboard_v2(
        cls,
        db: AsyncSession,
        store_id: Optional[UUID] = None,
        time_range: str = "today",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> DashboardV2Response:
        """
        Single consolidated V2 dashboard response.

        Powers the Retail Intelligence page:
        - Footfall card (total visitors + % vs prev period)
        - Gender card   (male / female / unidentified %)
        - Age Groups card (6 bins + peak label)
        - Purchase Count card (total + conversion % + % vs prev)
        - Foot Fall Over Time chart (hourly counts)
        - Gender Classification Trend chart (hourly male/female/unidentified stacked bars)
        - Camera badge (total / active cameras)
        """
        start, end, prev_start, prev_end = cls._v2_resolve_range(time_range, start_time, end_time)
        cam_ids = await _resolve_camera_ids(db, store_id=store_id)

        # ── 1. Store name ────────────────────────────────────────────────
        store_name = "All Stores"
        if store_id:
            store_row = await db.execute(select(Store.name).where(Store.id == store_id))
            sn = store_row.scalar_one_or_none()
            if sn:
                store_name = sn

        # ── 2. Camera counts ─────────────────────────────────────────────
        cam_base = select(func.count(Camera.id)).where(Camera.is_active.is_(True))
        active_base = cam_base.where(Camera.status == "active")
        if store_id:
            cam_base = cam_base.where(Camera.store_id == store_id)
            active_base = active_base.where(Camera.store_id == store_id)
        total_cameras = (await db.execute(cam_base)).scalar() or 0
        active_cameras = (await db.execute(active_base)).scalar() or 0

        # ── 3. Footfall (unique non-staff persons SEEN in range) ─────────
        # First-seen semantics: ≥1 track session started in the window.
        def _ff_q(s, e):
            q = select(func.count(PersonIdentity.id)).where(
                _seen_in_window(s, e, cam_ids),
                _NOT_STAFF,
            )
            return q

        total_visitors = (await db.execute(_ff_q(start, end))).scalar() or 0
        prev_visitors  = (await db.execute(_ff_q(prev_start, prev_end))).scalar() or 0

        # ── 4. Fetch demographics (distinct non-staff persons in range) ───
        # SEEN in range (track started in range) — NOT last_seen_at
        demo_q = select(PersonIdentity.id, PersonIdentity.gender, PersonIdentity.estimated_age).where(
            _seen_in_window(start, end, cam_ids),
            _NOT_STAFF,
        )

        demo_rows = (await db.execute(demo_q)).all()

        # Gender counts
        gender_cnt: dict = {"male": 0, "female": 0, "unidentified": 0}
        age_cnt: dict = {k: 0 for k, *_ in cls._V2_AGE_BINS}
        age_cnt["unidentified"] = 0
        seen: set = set()

        for pid, raw_gender, estimated_age in demo_rows:
            if pid in seen:
                continue
            seen.add(pid)
            gender_cnt[cls._v2_gender(raw_gender)] += 1
            age_cnt[cls._v2_age_bin(estimated_age)] += 1

        total_demo = sum(gender_cnt.values()) or 1  # avoid div/0

        def _pct(n: int) -> float:
            return round(n / total_demo * 100, 1)

        # Peak age group label
        named_bins = {k: age_cnt[k] for k, *_ in cls._V2_AGE_BINS}
        peak_key = max(named_bins, key=lambda k: named_bins[k]) if any(named_bins.values()) else None
        peak_label = None
        if peak_key:
            for k, label, *_ in cls._V2_AGE_BINS:
                if k == peak_key:
                    peak_label = f"{label} dominant"
                    break

        # ── 5. Purchases ──────────────────────────────────────────────────
        def _purchase_q(s, e):
            q = select(func.count(func.distinct(BillingInteraction.person_identity_id))).where(
                BillingInteraction.entered_at >= s,
                BillingInteraction.entered_at <= e,
                BillingInteraction.person_identity_id.notin_(_STAFF_IDS),
            )
            if cam_ids:
                q = q.where(BillingInteraction.camera_id.in_(cam_ids))
            return q

        total_purchases = (await db.execute(_purchase_q(start, end))).scalar() or 0
        prev_purchases  = (await db.execute(_purchase_q(prev_start, prev_end))).scalar() or 0
        conversion_pct = round(total_purchases / max(total_visitors, 1) * 100, 1)

        # ── 6. Footfall Over Time (same population as total_visitors) ──
        # Bucketed by track START time in IST (first-seen per slot), not
        # last_seen_at (which moves on repeat visits and re-writes history).
        range_days = (end - start).total_seconds() / 86400
        resolved = cls._resolve_group_by("auto", range_days)
        slot_bexpr = _ist_slot_expr(TrackSession.started_at, resolved)

        def _person_cam_filter(q):
            if cam_ids:
                q = q.where(TrackSession.camera_id.in_(cam_ids))
            return q

        ffq = _person_cam_filter(
            select(slot_bexpr.label("b"), func.count(func.distinct(TrackSession.person_identity_id)).label("c"))
            .join(PersonIdentity, PersonIdentity.id == TrackSession.person_identity_id)
            .where(
                TrackSession.started_at >= start,
                TrackSession.started_at <= end,
                _NOT_STAFF,
            )
            .group_by("b")
            .order_by("b")
        )
        ff_rows = (await db.execute(ffq)).all()
        ff_map = {}
        for row in ff_rows:
            if row.b is not None:
                bucket_ist = row.b.replace(tzinfo=IST) if row.b.tzinfo is None else row.b
                ff_map[bucket_ist] = row.c

        footfall_over_time: List[DashboardV2FootfallPoint] = []
        slot = cls._truncate_slot(start, resolved)
        display_end = end
        if resolved == "hour" and time_range in ("today", "weekly"):
            display_end = end.replace(hour=23, minute=59, second=59, microsecond=999999)

        while slot <= display_end:
            next_slot = cls._next_slot(slot, resolved)
            footfall_over_time.append(DashboardV2FootfallPoint(
                label=cls._slot_label(slot, resolved),
                slot_start=slot,
                slot_end=next_slot,
                count=ff_map.get(slot, 0),
            ))
            slot = next_slot

        # ── 7. Gender Trend (same track-start buckets as footfall_over_time) ─
        gt_q = _person_cam_filter(
            select(
                slot_bexpr.label("b"),
                PersonIdentity.gender,
                func.count(func.distinct(TrackSession.person_identity_id)).label("c"),
            )
            .join(PersonIdentity, PersonIdentity.id == TrackSession.person_identity_id)
            .where(
                TrackSession.started_at >= start,
                TrackSession.started_at <= end,
                _NOT_STAFF,
            )
            .group_by("b", PersonIdentity.gender)
            .order_by("b")
        )
        gt_rows = (await db.execute(gt_q)).all()
        gt_map: dict = defaultdict(lambda: {"male": 0, "female": 0, "unidentified": 0})
        for row in gt_rows:
            g = cls._v2_gender(row.gender)
            bucket_ist = row.b.replace(tzinfo=IST) if row.b and row.b.tzinfo is None else row.b
            gt_map[bucket_ist][g] += row.c

        gender_trend: List[DashboardV2GenderTrendPoint] = []
        slot = cls._truncate_slot(start, resolved)
        while slot <= display_end:
            next_slot = cls._next_slot(slot, resolved)
            bucket_data = gt_map.get(slot, {})
            gender_trend.append(DashboardV2GenderTrendPoint(
                label=cls._slot_label(slot, resolved),
                slot_start=slot,
                slot_end=next_slot,
                male=bucket_data.get("male", 0),
                female=bucket_data.get("female", 0),
                unidentified=bucket_data.get("unidentified", 0),
            ))
            slot = next_slot

        logger.info(
            f"V2 Dashboard: store={store_id or 'all'}, range={time_range}, "
            f"visitors={total_visitors}, purchases={total_purchases}, "
            f"cameras={active_cameras}/{total_cameras}"
        )

        return DashboardV2Response(
            store_id=store_id,
            store_name=store_name,
            time_range=time_range,
            start_time=start,
            end_time=end,
            total_cameras=total_cameras,
            active_cameras=active_cameras,
            footfall=DashboardV2FootfallMetric(
                total_visitors=total_visitors,
                vs_prev_pct=cls._v2_pct_change(total_visitors, prev_visitors),
            ),
            gender=DashboardV2GenderMetric(
                male=gender_cnt["male"],
                female=gender_cnt["female"],
                unidentified=gender_cnt["unidentified"],
                male_pct=_pct(gender_cnt["male"]),
                female_pct=_pct(gender_cnt["female"]),
                unidentified_pct=_pct(gender_cnt["unidentified"]),
            ),
            age_groups=DashboardV2AgeGroupsMetric(
                under_18=age_cnt["under_18"],
                age_18_24=age_cnt["age_18_24"],
                age_25_34=age_cnt["age_25_34"],
                age_35_44=age_cnt["age_35_44"],
                age_45_60=age_cnt["age_45_60"],
                age_60_plus=age_cnt["age_60_plus"],
                unidentified=age_cnt["unidentified"],
                peak_group=peak_label,
            ),
            purchase_count=DashboardV2PurchaseMetric(
                total=total_purchases,
                conversion_pct=conversion_pct,
                vs_prev_pct=cls._v2_pct_change(total_purchases, prev_purchases),
            ),
            footfall_over_time=footfall_over_time,
            gender_trend=gender_trend,
            age_group_distribution=[
                DashboardV2AgeGroupDistributionPoint(key=k, label=lbl, count=age_cnt[k])
                for k, lbl, *_ in cls._V2_AGE_BINS
            ] + [
                DashboardV2AgeGroupDistributionPoint(
                    key="unidentified", label="Unidentified", count=age_cnt["unidentified"]
                )
            ],
        )

    # ── helpers shared by get_analytics_metrics ───────────────────────────────

    @classmethod
    def _build_ff_slots(cls, start, end, resolved, ff_map) -> List[DashboardV2FootfallPoint]:
        slots: List[DashboardV2FootfallPoint] = []
        slot = cls._truncate_slot(start, resolved)
        while slot <= end:
            nxt = cls._next_slot(slot, resolved)
            slots.append(DashboardV2FootfallPoint(
                label=cls._slot_label(slot, resolved),
                slot_start=slot, slot_end=nxt,
                count=ff_map.get(slot, 0),
            ))
            slot = nxt
        return slots

    @staticmethod
    def _peak_hours_label(hourly_map: dict) -> Optional[str]:
        """Convert {hour: count} → 'HH PM – HH PM and HH PM – HH PM'."""
        if not hourly_map:
            return None
        max_count = max(hourly_map.values()) or 1
        threshold = max_count * 0.7          # hours with ≥70% of peak count
        hot = sorted(h for h, c in hourly_map.items() if c >= threshold)
        if not hot:
            return None
        # group consecutive hours
        groups = []
        group = [hot[0]]
        for h in hot[1:]:
            if h == group[-1] + 1:
                group.append(h)
            else:
                groups.append(group)
                group = [h]
        groups.append(group)

        def _fmt(h: int) -> str:
            if h == 0:    return "12 AM"
            if h < 12:    return f"{h} AM"
            if h == 12:   return "12 PM"
            return f"{h - 12} PM"

        parts = [f"{_fmt(g[0])} – {_fmt(g[-1] + 1)}" for g in groups[:2]]
        return " and ".join(parts)

    @classmethod
    async def get_analytics_metrics(
        cls,
        db: AsyncSession,
        metric: str,
        store_id: Optional[UUID] = None,
        time_range: str = "today",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> AnalyticsMetricsResponse:
        """
        Detailed analytics for the Analytics page, scoped to one metric tab.

        `metric` must be one of: footfall | gender | age_groups | purchase

        Common for all metrics:
        - period_comparison  → This Period vs Last Period dual-line chart
        - per_camera_breakdown → horizontal bar chart

        Footfall-specific: total_visitors, peak_hour, avg_daily, busiest_day,
                           footfall_over_time, peak_hours_label
        Gender-specific:   total_male/female/unidentified + %, gender_trend
        Age-Groups:        age_group_distribution, peak_group
        Purchase:          total_purchases, conversion_pct, avg_daily, busiest_day,
                           purchases_over_time, peak_hours_label
        """
        start, end, prev_start, prev_end = cls._v2_resolve_range(time_range, start_time, end_time)
        cam_ids = await _resolve_camera_ids(db, store_id=store_id)
        range_days = (end - start).total_seconds() / 86400
        resolved = cls._resolve_group_by("auto", range_days)
        duration = end - start  # length of one period

        # ── store name ─────────────────────────────────────────────────
        store_name = "All Stores"
        if store_id:
            row = await db.execute(select(Store.name).where(Store.id == store_id))
            sn = row.scalar_one_or_none()
            if sn:
                store_name = sn

        base_resp = dict(
            store_id=store_id, store_name=store_name,
            time_range=time_range, start_time=start, end_time=end,
            metric=metric,
        )

        # ── Shared helpers ─────────────────────────────────────────────

        def _unique_persons_q(s, e):
            """Count unique non-staff persons SEEN in [s, e].

            Track-session-start semantics (first-seen per window). NOT
            last_seen_at — that shifts on repeat visits and retroactively
            rewrites historical counts.
            """
            q = select(func.count(PersonIdentity.id)).where(
                _seen_in_window(s, e, cam_ids),
                _NOT_STAFF,
            )
            return q

        def _purchase_q(s, e):
            q = select(func.count(func.distinct(BillingInteraction.person_identity_id))).where(
                BillingInteraction.entered_at >= s,
                BillingInteraction.entered_at <= e,
                BillingInteraction.person_identity_id.notin_(_STAFF_IDS),
            )
            if cam_ids:
                q = q.where(BillingInteraction.camera_id.in_(cam_ids))
            return q

        def _person_cam(q):
            # Legacy helper retained for no-op compatibility; seen-window
            # filtering now happens inside each query via _seen_in_window.
            if cam_ids:
                q = q.where(
                    PersonIdentity.id.in_(
                        select(TrackSession.person_identity_id).where(
                            TrackSession.camera_id.in_(cam_ids),
                            TrackSession.started_at >= start,
                            TrackSession.started_at <= end,
                            TrackSession.person_identity_id.isnot(None),
                        )
                    )
                )
            return q

        def _ist_map(rows) -> dict:
            result = {}
            for r in rows:
                if r.b is not None:
                    result[r.b.replace(tzinfo=IST) if r.b.tzinfo is None else r.b] = r.c
            return result

        async def _seen_slot_map(s, e) -> dict:
            """Per-slot distinct non-staff persons, bucketed by track START (IST)."""
            bexpr = _ist_slot_expr(TrackSession.started_at, resolved)
            q = (
                select(bexpr.label("b"), func.count(func.distinct(TrackSession.person_identity_id)).label("c"))
                .join(PersonIdentity, PersonIdentity.id == TrackSession.person_identity_id)
                .where(
                    TrackSession.started_at >= s,
                    TrackSession.started_at <= e,
                    _NOT_STAFF,
                )
            )
            if cam_ids:
                q = q.where(TrackSession.camera_id.in_(cam_ids))
            q = q.group_by("b").order_by("b")
            return _ist_map((await db.execute(q)).all())

        async def _last_seen_period_comparison(curr_map: dict) -> List[PeriodComparisonPoint]:
            prev_map = await _seen_slot_map(prev_start, prev_end)
            points: List[PeriodComparisonPoint] = []
            slot = cls._truncate_slot(start, resolved)
            prev_slot = cls._truncate_slot(prev_start, resolved)
            while slot <= end:
                nxt = cls._next_slot(slot, resolved)
                points.append(PeriodComparisonPoint(
                    label=cls._slot_label(slot, resolved),
                    slot_start=slot, slot_end=nxt,
                    current=curr_map.get(slot, 0),
                    previous=prev_map.get(prev_slot, 0),
                ))
                slot = nxt
                prev_slot = cls._next_slot(prev_slot, resolved)
            return points

        async def _last_seen_per_camera() -> List[CameraBreakdownPoint]:
            """Per-camera distinct non-staff persons SEEN in range (track-start)."""
            q = (
                select(
                    Camera.id,
                    Camera.name,
                    func.count(func.distinct(TrackSession.person_identity_id)).label("cnt"),
                )
                .select_from(TrackSession)
                .join(PersonIdentity, PersonIdentity.id == TrackSession.person_identity_id)
                .join(Camera, Camera.id == TrackSession.camera_id)
                .where(
                    TrackSession.started_at >= start,
                    TrackSession.started_at <= end,
                    _NOT_STAFF,
                )
                .group_by(Camera.id, Camera.name)
                .order_by(func.count(func.distinct(TrackSession.person_identity_id)).desc())
            )
            if cam_ids:
                q = q.where(TrackSession.camera_id.in_(cam_ids))
            rows = (await db.execute(q)).all()
            return [CameraBreakdownPoint(camera_id=r[0], camera_name=r[1], count=r[2]) for r in rows]

        # ── FOOTFALL ───────────────────────────────────────────────────
        if metric == "footfall":
            # Count unique persons (distinct person_identity_id)
            total_visitors = (await db.execute(_unique_persons_q(start, end))).scalar() or 0

            # Hourly map for peak hour (unique non-staff persons per hour,
            # bucketed by track START in IST)
            h_bexpr = _ist_slot_expr(TrackSession.started_at, "hour")

            hq = (
                select(
                    h_bexpr.label("b"),
                    func.count(func.distinct(TrackSession.person_identity_id)).label("c")
                )
                .join(PersonIdentity, PersonIdentity.id == TrackSession.person_identity_id)
                .where(
                    TrackSession.started_at >= start,
                    TrackSession.started_at <= end,
                    _NOT_STAFF,
                )
                .group_by("b")
                .order_by("b")
            )
            if cam_ids:
                hq = hq.where(TrackSession.camera_id.in_(cam_ids))
            h_rows = (await db.execute(hq)).all()
            # Convert timezone-naive buckets to IST timezone-aware
            hourly_cnt = {}
            hourly_int = {}
            for r in h_rows:
                if r.b is not None:
                    bucket_ist = r.b.replace(tzinfo=IST) if r.b.tzinfo is None else r.b
                    hourly_cnt[bucket_ist] = r.c
                    # Extract hour as integer for peak_hours_label
                    hourly_int[bucket_ist.hour] = r.c

            # daily map for avg_daily + busiest_day (unique non-staff persons per day,
            # bucketed by track START in IST)
            d_bexpr = _ist_slot_expr(TrackSession.started_at, "day")

            dq = (
                select(
                    d_bexpr.label("b"),
                    func.count(func.distinct(TrackSession.person_identity_id)).label("c")
                )
                .join(PersonIdentity, PersonIdentity.id == TrackSession.person_identity_id)
                .where(
                    TrackSession.started_at >= start,
                    TrackSession.started_at <= end,
                    _NOT_STAFF,
                )
                .group_by("b")
                .order_by("b")
            )
            if cam_ids:
                dq = dq.where(TrackSession.camera_id.in_(cam_ids))
            d_rows = (await db.execute(dq)).all()
            # Convert timezone-naive buckets to IST timezone-aware
            daily_cnt = {}
            for r in d_rows:
                if r.b is not None:
                    bucket_ist = r.b.replace(tzinfo=IST) if r.b.tzinfo is None else r.b
                    daily_cnt[bucket_ist] = r.c

            peak_h_dt = max(hourly_cnt, key=lambda k: hourly_cnt[k]) if hourly_cnt else None
            busiest_day_dt = max(daily_cnt, key=lambda k: daily_cnt[k]) if daily_cnt else None
            avg_daily = (total_visitors // max(len(daily_cnt), 1)) if daily_cnt else 0
            # Today / single-day: total ÷ hours elapsed (empty hours included)
            avg_hourly = None
            if resolved == "hour":
                hours_so_far = max(1, int((end - start).total_seconds() // 3600) + 1)
                avg_hourly = total_visitors // hours_so_far

            # footfall_over_time (resolved granularity) - unique non-staff persons,
            # bucketed by track START in IST
            slot_bexpr = _ist_slot_expr(TrackSession.started_at, resolved)

            ffq = (
                select(
                    slot_bexpr.label("b"),
                    func.count(func.distinct(TrackSession.person_identity_id)).label("c")
                )
                .join(PersonIdentity, PersonIdentity.id == TrackSession.person_identity_id)
                .where(
                    TrackSession.started_at >= start,
                    TrackSession.started_at <= end,
                    _NOT_STAFF,
                )
                .group_by("b")
                .order_by("b")
            )
            if cam_ids:
                ffq = ffq.where(TrackSession.camera_id.in_(cam_ids))
            ff_rows = (await db.execute(ffq)).all()
            # Convert timezone-naive buckets to IST timezone-aware
            ff_map = {}
            for r in ff_rows:
                if r.b is not None:
                    bucket_ist = r.b.replace(tzinfo=IST) if r.b.tzinfo is None else r.b
                    ff_map[bucket_ist] = r.c
            
            footfall_over_time = cls._build_ff_slots(start, end, resolved, ff_map)
            period_comp = await _last_seen_period_comparison(ff_map)
            cam_breakdown = await _last_seen_per_camera()

            return AnalyticsMetricsResponse(
                **base_resp,
                footfall_data=FootfallMetricData(
                    total_visitors=total_visitors,
                    peak_hour=PeakHourInfo(
                        count=hourly_cnt[peak_h_dt],
                        time=peak_h_dt.strftime("%H:%M"),
                    ) if peak_h_dt else None,
                    avg_daily=avg_daily,
                    avg_hourly=avg_hourly,
                    busiest_day=BusiestDayInfo(
                        count=daily_cnt[busiest_day_dt],
                        date=busiest_day_dt.strftime("%m-%d"),
                    ) if busiest_day_dt else None,
                    footfall_over_time=footfall_over_time,
                    period_comparison=period_comp,
                    per_camera_breakdown=cam_breakdown,
                    peak_hours_label=cls._peak_hours_label(hourly_int),
                ),
            )

        # ── GENDER ─────────────────────────────────────────────────────
        if metric == "gender":
            # Distinct non-staff persons — mirrors footfall total_visitors.
            demo_q = select(PersonIdentity.id, PersonIdentity.gender).where(
                _seen_in_window(start, end, cam_ids),
                _NOT_STAFF,
            )
            demo = (await db.execute(demo_q)).all()

            gcnt: dict = {"male": 0, "female": 0, "unidentified": 0}
            seen2: set = set()
            for pid, raw in demo:
                if pid in seen2: continue
                seen2.add(pid)
                gcnt[cls._v2_gender(raw)] += 1
            total_g = sum(gcnt.values()) or 1

            bexpr = _ist_slot_expr(TrackSession.started_at, resolved)
            gt_q = (
                select(
                    bexpr.label("b"),
                    PersonIdentity.gender,
                    func.count(func.distinct(TrackSession.person_identity_id)).label("c"),
                )
                .join(PersonIdentity, PersonIdentity.id == TrackSession.person_identity_id)
                .where(
                    TrackSession.started_at >= start,
                    TrackSession.started_at <= end,
                    _NOT_STAFF,
                )
                .group_by("b", PersonIdentity.gender)
                .order_by("b")
            )
            if cam_ids:
                gt_q = gt_q.where(TrackSession.camera_id.in_(cam_ids))
            gt_rows = (await db.execute(gt_q)).all()
            gt_map: dict = defaultdict(lambda: {"male": 0, "female": 0, "unidentified": 0})
            curr_totals: dict = {}
            for r in gt_rows:
                bucket_ist = r.b.replace(tzinfo=IST) if r.b and r.b.tzinfo is None else r.b
                gt_map[bucket_ist][cls._v2_gender(r.gender)] += r.c
                curr_totals[bucket_ist] = curr_totals.get(bucket_ist, 0) + r.c

            gender_trend: List[DashboardV2GenderTrendPoint] = []
            slot = cls._truncate_slot(start, resolved)
            while slot <= end:
                nxt = cls._next_slot(slot, resolved)
                bd = gt_map.get(slot, {})
                gender_trend.append(DashboardV2GenderTrendPoint(
                    label=cls._slot_label(slot, resolved), slot_start=slot, slot_end=nxt,
                    male=bd.get("male", 0), female=bd.get("female", 0),
                    unidentified=bd.get("unidentified", 0),
                ))
                slot = nxt

            period_comp = await _last_seen_period_comparison(curr_totals)
            cam_breakdown = await _last_seen_per_camera()

            return AnalyticsMetricsResponse(
                **base_resp,
                gender_data=GenderMetricData(
                    total_male=gcnt["male"],
                    total_female=gcnt["female"],
                    total_unidentified=gcnt["unidentified"],
                    male_pct=round(gcnt["male"] / total_g * 100, 1),
                    female_pct=round(gcnt["female"] / total_g * 100, 1),
                    unidentified_pct=round(gcnt["unidentified"] / total_g * 100, 1),
                    gender_trend=gender_trend,
                    period_comparison=period_comp,
                    per_camera_breakdown=cam_breakdown,
                ),
            )

        # ── AGE GROUPS ─────────────────────────────────────────────────
        if metric == "age_groups":
            # Distinct non-staff persons — mirrors footfall total_visitors.
            age_demo_q = select(PersonIdentity.id, PersonIdentity.estimated_age).where(
                _seen_in_window(start, end, cam_ids),
                _NOT_STAFF,
            )
            age_demo = (await db.execute(age_demo_q)).all()

            age_cnt2: dict = {k: 0 for k, *_ in cls._V2_AGE_BINS}
            age_cnt2["unidentified"] = 0
            seen3: set = set()
            for pid, estimated_age in age_demo:
                if pid in seen3: continue
                seen3.add(pid)
                age_cnt2[cls._v2_age_bin(estimated_age)] += 1

            named2 = {k: age_cnt2[k] for k, *_ in cls._V2_AGE_BINS}
            pk = max(named2, key=lambda k: named2[k]) if any(named2.values()) else None
            peak_lbl = None
            if pk:
                for k, lbl, *_ in cls._V2_AGE_BINS:
                    if k == pk:
                        peak_lbl = f"{lbl} dominant"
                        break

            total_identified = sum(named2.values())
            distribution = [
                DashboardV2AgeGroupDistributionPoint(key=k, label=lbl, count=age_cnt2[k])
                for k, lbl, *_ in cls._V2_AGE_BINS
            ] + [DashboardV2AgeGroupDistributionPoint(
                key="unidentified", label="Unidentified", count=age_cnt2["unidentified"]
            )]

            period_comp = await _last_seen_period_comparison(
                await _seen_slot_map(start, end)
            )
            cam_breakdown = await _last_seen_per_camera()

            return AnalyticsMetricsResponse(
                **base_resp,
                age_groups_data=AgeGroupsMetricData(
                    total_identified=total_identified,
                    total_unidentified=age_cnt2["unidentified"],
                    peak_group=peak_lbl,
                    age_group_distribution=distribution,
                    period_comparison=period_comp,
                    per_camera_breakdown=cam_breakdown,
                ),
            )

        # ── PURCHASE ───────────────────────────────────────────────────
        if metric == "purchase":
            total_purchases = (await db.execute(_purchase_q(start, end))).scalar() or 0
            total_visitors_p = (await db.execute(_unique_persons_q(start, end))).scalar() or 0
            conv_pct = round(total_purchases / max(total_visitors_p, 1) * 100, 1)

            # Daily map - use IST timezone
            d_bexpr2 = func.date_trunc("day", func.timezone('Asia/Kolkata', BillingInteraction.entered_at))
            dq2 = (
                select(d_bexpr2.label("b"), func.count(func.distinct(BillingInteraction.person_identity_id)).label("c"))
                .where(BillingInteraction.entered_at >= start, BillingInteraction.entered_at <= end,
                       BillingInteraction.person_identity_id.notin_(_STAFF_IDS))
                .group_by("b").order_by("b")
            )
            if cam_ids:
                dq2 = dq2.where(BillingInteraction.camera_id.in_(cam_ids))
            d_rows2 = (await db.execute(dq2)).all()
            # Convert timezone-naive buckets to IST timezone-aware
            daily2 = {}
            for r in d_rows2:
                if r.b is not None:
                    bucket_ist = r.b.replace(tzinfo=IST) if r.b.tzinfo is None else r.b
                    daily2[bucket_ist] = r.c

            busiest2 = max(daily2, key=lambda k: daily2[k]) if daily2 else None
            avg_daily2 = (total_purchases // max(len(daily2), 1)) if daily2 else 0
            avg_hourly2 = None
            if resolved == "hour":
                hours_so_far2 = max(1, int((end - start).total_seconds() // 3600) + 1)
                avg_hourly2 = total_purchases // hours_so_far2

            # Hourly map for peak hour + peak_hours_label - use IST timezone
            h_bexpr2 = func.date_trunc("hour", func.timezone('Asia/Kolkata', BillingInteraction.entered_at))
            hq2 = (
                select(h_bexpr2.label("b"), func.count(func.distinct(BillingInteraction.person_identity_id)).label("c"))
                .where(BillingInteraction.entered_at >= start, BillingInteraction.entered_at <= end,
                       BillingInteraction.person_identity_id.notin_(_STAFF_IDS))
                .group_by("b").order_by("b")
            )
            if cam_ids:
                hq2 = hq2.where(BillingInteraction.camera_id.in_(cam_ids))
            h_rows2 = (await db.execute(hq2)).all()
            # Convert timezone-naive buckets to IST timezone-aware for hour extraction
            hourly_cnt2 = {}
            hourly_int2 = {}
            for r in h_rows2:
                if r.b is not None:
                    bucket_ist = r.b.replace(tzinfo=IST) if r.b.tzinfo is None else r.b
                    hourly_cnt2[bucket_ist] = r.c
                    hourly_int2[bucket_ist.hour] = r.c
            peak_h2_dt = max(hourly_cnt2, key=lambda k: hourly_cnt2[k]) if hourly_cnt2 else None

            # Slot maps must match cards: distinct non-staff persons (not raw row counts)
            async def _purchase_slot_map(s, e) -> dict:
                bexpr = func.date_trunc(resolved, func.timezone('Asia/Kolkata', BillingInteraction.entered_at))
                q = (
                    select(
                        bexpr.label("b"),
                        func.count(func.distinct(BillingInteraction.person_identity_id)).label("c"),
                    )
                    .where(
                        BillingInteraction.entered_at >= s,
                        BillingInteraction.entered_at <= e,
                        BillingInteraction.person_identity_id.notin_(_STAFF_IDS),
                    )
                    .group_by("b")
                    .order_by("b")
                )
                if cam_ids:
                    q = q.where(BillingInteraction.camera_id.in_(cam_ids))
                rows = (await db.execute(q)).all()
                result = {}
                for r in rows:
                    if r.b is not None:
                        bucket_ist = r.b.replace(tzinfo=IST) if r.b.tzinfo is None else r.b
                        result[bucket_ist] = r.c
                return result

            async def _purchase_period_comparison() -> List[PeriodComparisonPoint]:
                curr_map = await _purchase_slot_map(start, end)
                prev_map = await _purchase_slot_map(prev_start, prev_end)
                points: List[PeriodComparisonPoint] = []
                slot = cls._truncate_slot(start, resolved)
                prev_slot = cls._truncate_slot(prev_start, resolved)
                while slot <= end:
                    nxt = cls._next_slot(slot, resolved)
                    points.append(PeriodComparisonPoint(
                        label=cls._slot_label(slot, resolved),
                        slot_start=slot, slot_end=nxt,
                        current=curr_map.get(slot, 0),
                        previous=prev_map.get(prev_slot, 0),
                    ))
                    slot = nxt
                    prev_slot = cls._next_slot(prev_slot, resolved)
                return points

            # Per-camera: distinct non-staff purchasers
            cam_q2 = (
                select(
                    Camera.id,
                    Camera.name,
                    func.count(func.distinct(BillingInteraction.person_identity_id)).label("cnt"),
                )
                .join(Camera, Camera.id == BillingInteraction.camera_id)
                .where(
                    BillingInteraction.entered_at >= start,
                    BillingInteraction.entered_at <= end,
                    BillingInteraction.person_identity_id.notin_(_STAFF_IDS),
                )
                .group_by(Camera.id, Camera.name)
                .order_by(func.count(func.distinct(BillingInteraction.person_identity_id)).desc())
            )
            if cam_ids:
                cam_q2 = cam_q2.where(BillingInteraction.camera_id.in_(cam_ids))
            cam_rows2 = (await db.execute(cam_q2)).all()
            cam_breakdown2 = [
                CameraBreakdownPoint(camera_id=r[0], camera_name=r[1], count=r[2]) for r in cam_rows2
            ]

            pur_map = await _purchase_slot_map(start, end)
            purchases_ot = cls._build_ff_slots(start, end, resolved, pur_map)
            period_comp = await _purchase_period_comparison()

            return AnalyticsMetricsResponse(
                **base_resp,
                purchase_data=PurchaseMetricData(
                    total_purchases=total_purchases,
                    conversion_pct=conv_pct,
                    avg_daily=avg_daily2,
                    avg_hourly=avg_hourly2,
                    peak_hour=PeakHourInfo(
                        count=hourly_cnt2[peak_h2_dt],
                        time=peak_h2_dt.strftime("%H:%M"),
                    ) if peak_h2_dt else None,
                    busiest_day=BusiestDayInfo(
                        count=daily2[busiest2],
                        date=busiest2.strftime("%m-%d"),
                    ) if busiest2 else None,
                    purchases_over_time=purchases_ot,
                    period_comparison=period_comp,
                    per_camera_breakdown=cam_breakdown2,
                    peak_hours_label=cls._peak_hours_label(hourly_int2),
                ),
            )

        raise HTTPException(
            status_code=400,
            detail=f"Invalid metric '{metric}'. Choose one of: footfall, gender, age_groups, purchase",
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

    # ── Live Viewers (real-time device tracking) ──────────────────────

    @staticmethod
    async def get_live_viewers(
        db: AsyncSession,
        store_id: Optional[UUID] = None,
    ) -> "LiveViewersResponse":
        """Real-time snapshot of devices watching live camera feeds."""
        from datetime import datetime, timezone
        from app.core.db.models.stream_viewer import StreamViewerSession
        from app.core.db.models.device_session import DeviceSession
        from app.modules.analytics.schemas import (
            LiveViewersResponse,
            LiveViewerCamera,
            LiveViewerEntry,
        )
        from app.utils.device_fingerprint import device_label

        now = datetime.now(timezone.utc)

        # Active device sessions (logged in, active in last 30 min)
        device_query = select(func.count(DeviceSession.id)).where(
            DeviceSession.is_active == True,
            DeviceSession.last_active_at >= now - timedelta(seconds=settings.SESSION_IDLE_TIMEOUT_SECONDS),
        )
        device_result = await db.execute(device_query)
        total_devices_connected = device_result.scalar() or 0

        # Active stream viewers (ended_at is NULL = still watching)
        viewer_query = select(StreamViewerSession).where(
            StreamViewerSession.ended_at.is_(None),
        )
        viewer_result = await db.execute(viewer_query)
        active_viewers = viewer_result.scalars().all()

        # Group by camera
        camera_map: dict[UUID, dict] = {}
        for v in active_viewers:
            cid = v.camera_id
            if cid not in camera_map:
                cam = await db.get(Camera, cid)
                camera_map[cid] = {
                    "camera_id": cid,
                    "camera_name": cam.name if cam else "Unknown",
                    "viewers": [],
                }
            duration_minutes = (now - v.started_at).total_seconds() / 60.0
            camera_map[cid]["viewers"].append(
                LiveViewerEntry(
                    device_hash=v.device_hash,
                    device_label=device_label(v.user_agent or ""),
                    ip_address=v.ip_address,
                    camera_id=cid,
                    camera_name=camera_map[cid]["camera_name"],
                    viewing_since=v.started_at,
                    duration_minutes=round(duration_minutes, 1),
                )
            )

        # Build camera list
        cameras = []
        for cid, data in camera_map.items():
            cameras.append(
                LiveViewerCamera(
                    camera_id=data["camera_id"],
                    camera_name=data["camera_name"],
                    active_viewers=len(data["viewers"]),
                    viewers=data["viewers"],
                )
            )

        total_watching = len(active_viewers)

        return LiveViewersResponse(
            total_devices_connected=total_devices_connected,
            total_devices_watching_feeds=total_watching,
            cameras=cameras,
        )
