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
)
from app.modules.analytics.service import AnalyticsService

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])


@router.get("/footfall", response_model=FootfallResponse)
async def footfall_analytics(
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    camera_id: Optional[UUID] = Query(None),
    interval: str = Query("hour", pattern="^(hour|day|week)$"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get footfall (entry-line crossings) analytics."""
    return await AnalyticsService.get_footfall(db, start_time, end_time, camera_id, interval)


@router.get("/billing", response_model=BillingAnalyticsResponse)
async def billing_analytics(
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    camera_id: Optional[UUID] = Query(None),
    interval: str = Query("hour", pattern="^(hour|day|week)$"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get billing counter interaction analytics."""
    return await AnalyticsService.get_billing_analytics(db, start_time, end_time, camera_id, interval)


@router.get("/dwell", response_model=DwellAnalyticsResponse)
async def dwell_analytics(
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    camera_id: Optional[UUID] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get average dwell time analytics."""
    return await AnalyticsService.get_dwell_analytics(db, start_time, end_time, camera_id)


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
    return await AnalyticsService.get_dashboard_summary(db, start_time, end_time, camera_id)


@router.get("/demographics", response_model=DemographicsTableResponse)
async def demographics_table(
    start_time: Optional[datetime] = Query(None, description="Range start (default: 24h ago)"),
    end_time: Optional[datetime] = Query(None, description="Range end (default: now)"),
    camera_id: Optional[UUID] = Query(None, description="Optional camera filter"),
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
        db, start_time=start_time, end_time=end_time, camera_id=camera_id
    )


@router.get("/person-journey/{person_id}", response_model=PersonJourneyResponse)
async def person_journey(
    person_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a person's journey across cameras."""
    return await AnalyticsService.get_person_journey(db, person_id)
