"""Analytics module Pydantic schemas."""

from datetime import datetime, date
from typing import Optional, List, Dict
from uuid import UUID
from pydantic import BaseModel


class FootfallPoint(BaseModel):
    bucket: datetime
    count: int


class FootfallResponse(BaseModel):
    start_time: datetime
    end_time: datetime
    total_entries: int
    unique_visitors: int
    timeline: List[FootfallPoint]


class BillingAnalyticsResponse(BaseModel):
    start_time: datetime
    end_time: datetime
    total_interactions: int
    avg_dwell_seconds: Optional[float] = None
    max_dwell_seconds: Optional[float] = None
    timeline: List[FootfallPoint]


class DwellAnalyticsResponse(BaseModel):
    start_time: datetime
    end_time: datetime
    avg_track_duration_seconds: Optional[float] = None
    total_tracks: int
    by_camera: Dict[str, float]


class ZoneOccupancyItem(BaseModel):
    zone_id: UUID
    zone_name: str
    zone_type: str
    event_count: int


class ZoneOccupancyResponse(BaseModel):
    start_time: datetime
    end_time: datetime
    zones: List[ZoneOccupancyItem]


class JourneyStep(BaseModel):
    camera_id: UUID
    camera_name: Optional[str] = None
    track_session_id: UUID
    started_at: datetime
    ended_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None


class PersonJourneyResponse(BaseModel):
    person_identity_id: UUID
    first_seen_at: datetime
    last_seen_at: datetime
    visit_count: int
    total_sessions: int
    journey: List[JourneyStep]
    events: List[dict]


# ---------------------------------------------------------------------------
# Dashboard Summary (single endpoint aggregating all key metrics)
# ---------------------------------------------------------------------------

class DemographicsBreakdown(BaseModel):
    """Age-group and gender breakdown from track sessions."""
    children: int = 0
    teenager: int = 0
    adult: int = 0
    senior_citizen: int = 0
    male: int = 0
    female: int = 0


class DashboardSummaryResponse(BaseModel):
    """Unified dashboard summary for a given datetime range."""
    start_time: datetime
    end_time: datetime
    unique_persons: int
    total_entries: int
    total_purchases: int
    demographics: DemographicsBreakdown


# ---------------------------------------------------------------------------
# Demographics Table (cross-tabulated age-group × gender with purchases)
# ---------------------------------------------------------------------------

class DemographicsTableRow(BaseModel):
    """Single row in the demographics cross-table."""
    age_group: str
    label: str
    male_count: int = 0
    female_count: int = 0
    unidentified_count: int = 0
    total_count: int = 0
    total_purchase_count: int = 0


class DemographicsTableSummary(BaseModel):
    """Summary rollup for the demographics table."""
    total_male: int = 0
    total_female: int = 0
    total_unidentified: int = 0
    total_visitors: int = 0
    total_purchases: int = 0


class DemographicsTableResponse(BaseModel):
    """Full demographics table with age-group × gender breakdown."""
    demographics: List[DemographicsTableRow]
    summary: DemographicsTableSummary


# ---------------------------------------------------------------------------
# Visitors Entry / Exit  (hourly breakdown for graph)
# ---------------------------------------------------------------------------

class VisitorEntryExitPoint(BaseModel):
    """Single data point (hour or day) for the entry/exit graph."""
    label: str          # "09:00" for hour-mode, "2026-06-19" for day-mode
    slot_start: datetime
    slot_end: datetime
    entry: int = 0
    exit: int = 0


class VisitorEntryExitResponse(BaseModel):
    """Entry/exit counts grouped by hour or day for the requested date range."""
    start_time: datetime
    end_time: datetime
    group_by: str           # "hour" or "day" (resolved from "auto")
    total_entry: int
    total_exit: int
    data: List[VisitorEntryExitPoint]
