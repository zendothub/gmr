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