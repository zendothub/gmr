"""Analytics API routes."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_user
from app.core.db.models.user import User
from app.modules.analytics.schemas import (
    AnalyticsMetricsResponse,
    FootfallResponse,
    BillingAnalyticsResponse,
    DwellAnalyticsResponse,
    ZoneOccupancyResponse,
    PersonJourneyResponse,
    DashboardSummaryResponse,
    DemographicsTableResponse,
    VisitorEntryExitResponse,
    DashboardV2Response,
)
from app.modules.analytics.service import AnalyticsService

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])
v2_router = APIRouter(prefix="/api/v2/analytics", tags=["Analytics V2"])


@router.get("/footfall", response_model=FootfallResponse)
async def footfall_analytics(
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    camera_id: Optional[UUID] = Query(None),
    store_id: Optional[UUID] = Query(None, description="Filter by store — overrides camera_id"),
    interval: str = Query("hour", pattern="^(hour|day|week)$"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get footfall (entry-line crossings) analytics."""
    return await AnalyticsService.get_footfall(db, start_time, end_time, camera_id, store_id, interval)


@router.get("/billing", response_model=BillingAnalyticsResponse)
async def billing_analytics(
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    camera_id: Optional[UUID] = Query(None),
    store_id: Optional[UUID] = Query(None, description="Filter by store — overrides camera_id"),
    interval: str = Query("hour", pattern="^(hour|day|week)$"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get billing counter interaction analytics."""
    return await AnalyticsService.get_billing_analytics(db, start_time, end_time, camera_id, store_id, interval)


@router.get("/dwell", response_model=DwellAnalyticsResponse)
async def dwell_analytics(
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    camera_id: Optional[UUID] = Query(None),
    store_id: Optional[UUID] = Query(None, description="Filter by store — overrides camera_id"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get average dwell time analytics."""
    return await AnalyticsService.get_dwell_analytics(db, start_time, end_time, camera_id, store_id)


@router.get("/zone-occupancy", response_model=ZoneOccupancyResponse)
async def zone_occupancy_analytics(
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get per-zone event activity."""
    return await AnalyticsService.get_zone_occupancy(db, start_time, end_time)


@router.get("/events", response_model=DashboardSummaryResponse)
async def dashboard_summary(
    start_time: Optional[datetime] = Query(None, description="Range start (default: 24h ago)"),
    end_time: Optional[datetime] = Query(None, description="Range end (default: now)"),
    camera_id: Optional[UUID] = Query(None, description="Optional camera filter"),
    store_id: Optional[UUID] = Query(None, description="Filter by store — overrides camera_id"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Unified dashboard summary for a given datetime range.

    Returns:
    - unique_persons: count of distinct persons detected
    - total_entries: total track sessions including duplicates
    - total_purchases: count of billing interactions
    - demographics: age-group (children, teenager, adult, senior citizen)
                    and gender (male, female) breakdown
    """
    return await AnalyticsService.get_dashboard_summary(db, start_time, end_time, camera_id, store_id)


@router.get("/demographics", response_model=DemographicsTableResponse)
async def demographics_table(
    start_time: Optional[datetime] = Query(None, description="Range start (default: 24h ago)"),
    end_time: Optional[datetime] = Query(None, description="Range end (default: now)"),
    camera_id: Optional[UUID] = Query(None, description="Optional camera filter"),
    store_id: Optional[UUID] = Query(None, description="Filter by store — overrides camera_id"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Cross-tabulated demographics table: age-group × gender with per-group purchases.

    Returns 5 standard age groups (child, teenager, young adult, middle adult, senior)
    each broken down into male / female / unidentified counts, plus purchase totals.
    Includes a summary row with overall totals.
    """
    return await AnalyticsService.get_demographics_table(
        db, start_time=start_time, end_time=end_time, camera_id=camera_id, store_id=store_id
    )


@router.get("/person-journey/{person_id}", response_model=PersonJourneyResponse)
async def person_journey(
    person_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a person's journey across cameras."""
    return await AnalyticsService.get_person_journey(db, person_id)


@router.get("/visitors/entry-exit", response_model=VisitorEntryExitResponse)
async def visitor_entry_exit(
    start_time: Optional[datetime] = Query(None, description="Range start (default: 24 h ago)"),
    end_time: Optional[datetime] = Query(None, description="Range end (default: now)"),
    camera_id: Optional[UUID] = Query(None, description="Optional camera filter"),
    store_id: Optional[UUID] = Query(None, description="Filter by store — overrides camera_id"),
    group_by: str = Query(
        "auto",
        pattern="^(hour|day|week|month|auto)$",
        description=(
            "Granularity of each data point. "
            "'hour'  → one slot per hour   (ideal for ≤ 2-day views); "
            "'day'   → one slot per day    (ideal for last-7 / last-30 views); "
            "'week'  → one slot per week   (ideal for last-90 / 6-month views); "
            "'month' → one slot per month  (ideal for full-year views); "
            "'auto'  → auto-selects based on range: "
            "≤2d→hour, ≤30d→day, ≤180d→week, >180d→month."
        ),
    ),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Visitor entry / exit counts for a given date range, grouped by hour or day.

    Each data point includes:
    - **label**      – `"09:00"` (hour mode) or `"2026-06-19"` (day mode)
    - **slot_start** – ISO timestamp of the slot's start
    - **slot_end**   – ISO timestamp of the slot's end
    - **entry**      – count of `line_crossing_forward` events in that slot
    - **exit**       – count of `line_crossing_backward` events in that slot

    All slots between start and end are returned (zeros filled in for empty
    slots) so the frontend always gets a gapless series for the graph.

    **Auto granularity rules:**
    - range ≤ 2 days → `hour` (24–48 points)
    - range > 2 days → `day`  (e.g. 7 points for last-7-days)

    **Store filter:** When store_id is provided, only cameras linked to that
    store are included — enables store-wise analytics.
    """
    return await AnalyticsService.get_entry_exit_hourly(
        db,
        start_time=start_time,
        end_time=end_time,
        camera_id=camera_id,
        store_id=store_id,
        group_by=group_by,
    )


# ---------------------------------------------------------------------------
# V2 Dashboard
# ---------------------------------------------------------------------------

@v2_router.get("/dashboard", response_model=DashboardV2Response)
async def dashboard_v2(
    store_id: Optional[UUID] = Query(
        None,
        description="Filter by store UUID. Omit (or leave blank) to include all stores.",
    ),
    time_range: str = Query(
        "today",
        pattern="^(today|this_week|custom)$",
        description=(
            "'today'     → from 00:00 today to now (hourly slots). "
            "'this_week' → from Monday 00:00 to now (hourly slots). "
            "'custom'    → use start_time / end_time; granularity auto-selected."
        ),
    ),
    start_time: Optional[datetime] = Query(
        None, description="Range start — only used when time_range='custom'."
    ),
    end_time: Optional[datetime] = Query(
        None, description="Range end — only used when time_range='custom'."
    ),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Unified V2 dashboard — single endpoint that powers the Retail Intelligence page.

    **Filters**
    - `store_id`   → omit for All Stores, pass a UUID to scope to one store
    - `time_range` → `today` | `this_week` | `custom`

    **Response highlights**
    - `total_cameras` / `active_cameras` — camera badge ("6 / 8 Cameras")
    - `footfall`      — total visitors + % change vs previous equivalent period
    - `gender`        — male / female / unidentified counts and percentages
    - `age_groups`    — 6 bins (Under 18, 18-24, 25-34, 35-44, 45-60, 60+) + peak label
    - `purchase_count`— total purchases, conversion %, % vs previous period
    - `footfall_over_time` — gapless hourly (or daily for long ranges) visitor counts
    - `gender_trend`  — gapless hourly male / female / unidentified stacked bar data
    """
    return await AnalyticsService.get_dashboard_v2(
        db,
        store_id=store_id,
        time_range=time_range,
        start_time=start_time,
        end_time=end_time,
    )


# ---------------------------------------------------------------------------
# V2 Analytics — per-tab metrics (Foot Fall / Gender / Age Groups / Purchase)
# ---------------------------------------------------------------------------

@v2_router.get("/metrics", response_model=AnalyticsMetricsResponse)
async def analytics_metrics(
    metric: str = Query(
        ...,
        pattern="^(footfall|gender|age_groups|purchase)$",
        description=(
            "Which metric tab to load — **exactly one** must be chosen:\n"
            "- `footfall`   → Total Visitors, Peak Hour, Avg Daily, Busiest Day, "
            "Foot Fall Over Time chart, This Period vs Last Period chart, Per-Camera Breakdown\n"
            "- `gender`     → Male / Female / Unidentified counts & %, "
            "Gender Trend stacked bars, comparison chart, Per-Camera Breakdown\n"
            "- `age_groups` → Age Group Distribution bar chart, peak group label, "
            "comparison chart, Per-Camera Breakdown\n"
            "- `purchase`   → Total Purchases, Conversion %, Avg Daily, Busiest Day, "
            "Purchases Over Time chart, comparison chart, Per-Camera Breakdown, Peak Hours banner"
        ),
    ),
    store_id: Optional[UUID] = Query(
        None,
        description="All Stores when omitted; pass a UUID to scope to one store.",
    ),
    time_range: str = Query(
        "today",
        pattern="^(today|this_week|custom)$",
        description="'today' | 'this_week' | 'custom' (requires start_time + end_time)",
    ),
    start_time: Optional[datetime] = Query(
        None, description="Range start — only used when time_range='custom'."
    ),
    end_time: Optional[datetime] = Query(
        None, description="Range end — only used when time_range='custom'."
    ),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Analytics page — detailed data for the selected metric tab.

    **Filters (top bar)**
    | Param        | Values                                    |
    |---|---|
    | `store_id`   | omit = All Stores, UUID = specific store  |
    | `time_range` | `today` \\| `this_week` \\| `custom`      |
    | `metric`     | `footfall` \\| `gender` \\| `age_groups` \\| `purchase` |

    **Common charts in every metric:**
    - `period_comparison[]`     → This Period (solid line) vs Last Period (dashed) dual-line chart
    - `per_camera_breakdown[]`  → Horizontal bar chart sorted by count descending

    **Footfall tab fields** (`footfall_data`):
    - `total_visitors`, `peak_hour` `{ count, time }`, `avg_daily`, `busiest_day` `{ count, date }`
    - `footfall_over_time[]`, `peak_hours_label` (e.g. "12 PM – 2 PM and 6 PM – 8 PM")

    **Gender tab fields** (`gender_data`):
    - `total_male`, `total_female`, `total_unidentified`, `male_pct`, `female_pct`, `unidentified_pct`
    - `gender_trend[]` (stacked bars)

    **Age Groups tab fields** (`age_groups_data`):
    - `total_identified`, `total_unidentified`, `peak_group`
    - `age_group_distribution[]` (horizontal bar chart)

    **Purchase tab fields** (`purchase_data`):
    - `total_purchases`, `conversion_pct`, `avg_daily`, `busiest_day`
    - `purchases_over_time[]`, `peak_hours_label`
    """
    return await AnalyticsService.get_analytics_metrics(
        db,
        metric=metric,
        store_id=store_id,
        time_range=time_range,
        start_time=start_time,
        end_time=end_time,
    )
