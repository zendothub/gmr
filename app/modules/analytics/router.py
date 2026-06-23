"""Analytics API routes."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_user
from app.core.db.models.user import User
from app.modules.analytics.schemas import (
    FootfallResponse,
    BillingAnalyticsResponse,
    DwellAnalyticsResponse,
    ZoneOccupancyResponse,
    PersonJourneyResponse,
    DashboardSummaryResponse,
    DemographicsTableResponse,
    VisitorEntryExitResponse,
)
from app.modules.analytics.service import AnalyticsService

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])


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
