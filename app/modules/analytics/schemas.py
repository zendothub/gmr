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


# ---------------------------------------------------------------------------
# V2 Dashboard — single consolidated response used by the Retail Intelligence
# dashboard page (store filter + time-range selector).
# ---------------------------------------------------------------------------

class DashboardV2FootfallMetric(BaseModel):
    """Footfall (total visitors) with % change vs previous period."""
    total_visitors: int = 0
    vs_prev_pct: Optional[float] = None   # +12.0 means +12 % vs prev period


class DashboardV2GenderMetric(BaseModel):
    """Gender distribution counts and percentages."""
    male: int = 0
    female: int = 0
    unidentified: int = 0
    male_pct: float = 0.0
    female_pct: float = 0.0
    unidentified_pct: float = 0.0


class DashboardV2AgeGroupsMetric(BaseModel):
    """Age-group breakdown matching the dashboard age-group card."""
    under_18: int = 0
    age_18_24: int = 0
    age_25_34: int = 0
    age_35_44: int = 0
    age_45_60: int = 0
    age_60_plus: int = 0
    unidentified: int = 0
    peak_group: Optional[str] = None   # e.g. "25-34 dominant"


class DashboardV2PurchaseMetric(BaseModel):
    """Purchase (billing interaction) count, conversion rate, and % change."""
    total: int = 0
    conversion_pct: float = 0.0        # total_purchases / total_visitors * 100
    vs_prev_pct: Optional[float] = None


class DashboardV2FootfallPoint(BaseModel):
    """Single hourly slot for the Foot Fall Over Time chart."""
    label: str              # "00:00", "01:00", … or "Mon", "2026-06-19"
    slot_start: datetime
    slot_end: datetime
    count: int = 0


class DashboardV2GenderTrendPoint(BaseModel):
    """Single hourly slot for the Gender Classification Trend stacked bar."""
    label: str
    slot_start: datetime
    slot_end: datetime
    male: int = 0
    female: int = 0
    unidentified: int = 0


class DashboardV2AgeGroupDistributionPoint(BaseModel):
    """Single bar in the Age Group Distribution horizontal bar chart.

    Ordered from Under 18 → 18-24 → 25-34 → 35-44 → 45-60 → 60+ → Unidentified.
    """
    key: str          # "under_18" | "age_18_24" | … (machine-readable key)
    label: str        # "Under 18" | "18-24" | … (display label)
    count: int = 0


# ── Analytics Metrics (per-tab Analytics page) ──────────────────────────────

class PeakHourInfo(BaseModel):
    """Top-traffic hour summary card."""
    count: int = 0
    time: str = ""          # e.g. "18:00"


class BusiestDayInfo(BaseModel):
    """Top-traffic day summary card."""
    count: int = 0
    date: str = ""          # e.g. "06-21"


class CameraBreakdownPoint(BaseModel):
    """Single bar in the Per-Camera Breakdown horizontal bar chart."""
    camera_id: UUID
    camera_name: str
    count: int = 0


class PeriodComparisonPoint(BaseModel):
    """Single slot for the This Period vs Last Period dual-line chart."""
    label: str
    slot_start: datetime
    slot_end: datetime
    current: int = 0
    previous: int = 0


class FootfallMetricData(BaseModel):
    """Data payload when metric='footfall'."""
    # Summary cards
    total_visitors: int = 0
    peak_hour: Optional[PeakHourInfo] = None
    avg_daily: int = 0
    # Set when range is hourly (today / custom ≤1 day): total // hours_so_far (empty hours included).
    avg_hourly: Optional[int] = None
    busiest_day: Optional[BusiestDayInfo] = None
    # Charts
    footfall_over_time: List[DashboardV2FootfallPoint] = []
    period_comparison: List[PeriodComparisonPoint] = []
    per_camera_breakdown: List[CameraBreakdownPoint] = []
    # Banner
    peak_hours_label: Optional[str] = None


class GenderMetricData(BaseModel):
    """Data payload when metric='gender'."""
    # Cards
    total_male: int = 0
    total_female: int = 0
    total_unidentified: int = 0
    male_pct: float = 0.0
    female_pct: float = 0.0
    unidentified_pct: float = 0.0
    # Charts
    gender_trend: List[DashboardV2GenderTrendPoint] = []
    period_comparison: List[PeriodComparisonPoint] = []
    per_camera_breakdown: List[CameraBreakdownPoint] = []


class AgeGroupsMetricData(BaseModel):
    """Data payload when metric='age_groups'."""
    # Cards
    total_identified: int = 0
    total_unidentified: int = 0
    peak_group: Optional[str] = None
    # Charts
    age_group_distribution: List[DashboardV2AgeGroupDistributionPoint] = []
    period_comparison: List[PeriodComparisonPoint] = []
    per_camera_breakdown: List[CameraBreakdownPoint] = []


class PurchaseMetricData(BaseModel):
    """Data payload when metric='purchase'."""
    # Cards
    total_purchases: int = 0
    conversion_pct: float = 0.0
    avg_daily: int = 0
    # Set when range is hourly (today / custom ≤1 day): total // hours_so_far (empty hours included).
    avg_hourly: Optional[int] = None
    peak_hour: Optional[PeakHourInfo] = None
    busiest_day: Optional[BusiestDayInfo] = None
    # Charts
    purchases_over_time: List[DashboardV2FootfallPoint] = []
    period_comparison: List[PeriodComparisonPoint] = []
    per_camera_breakdown: List[CameraBreakdownPoint] = []
    # Banner
    peak_hours_label: Optional[str] = None


class AnalyticsMetricsResponse(BaseModel):
    """Top-level response for GET /api/v2/analytics/metrics.

    Only one of the four *_data fields will be populated — the one matching
    the requested `metric` parameter.
    """
    store_id: Optional[UUID] = None
    store_name: str = "All Stores"
    time_range: str
    start_time: datetime
    end_time: datetime
    metric: str   # "footfall" | "gender" | "age_groups" | "purchase"

    footfall_data: Optional[FootfallMetricData] = None
    gender_data: Optional[GenderMetricData] = None
    age_groups_data: Optional[AgeGroupsMetricData] = None
    purchase_data: Optional[PurchaseMetricData] = None


# ── Live Viewers (real-time device tracking) ──────────────────────────────

class LiveViewerEntry(BaseModel):
    """A single device watching a live camera feed."""
    device_hash: str
    device_label: str
    ip_address: str | None = None
    camera_id: UUID
    camera_name: str
    viewing_since: datetime
    duration_minutes: float


class LiveViewerCamera(BaseModel):
    """Per-camera breakdown of active viewers."""
    camera_id: UUID
    camera_name: str
    active_viewers: int
    viewers: list[LiveViewerEntry]


class LiveViewersResponse(BaseModel):
    """Real-time snapshot of devices watching live feeds."""
    total_devices_connected: int
    total_devices_watching_feeds: int
    cameras: list[LiveViewerCamera]


class DashboardV2Response(BaseModel):
    """Unified V2 dashboard response.

    Drives the Retail Intelligence page:
    - Top-right camera badge  → total_cameras / active_cameras
    - FOOT FALL card          → footfall
    - GENDER card             → gender
    - AGE GROUPS card         → age_groups
    - PURCHASE COUNT card     → purchase_count
    - Foot Fall Over Time     → footfall_over_time
    - Gender Classification   → gender_trend
    """
    # Filter context echoed back
    store_id: Optional[UUID] = None
    store_name: str = "All Stores"
    time_range: str                # "today" | "weekly" | "custom"
    start_time: datetime
    end_time: datetime

    # Camera badge
    total_cameras: int = 0
    active_cameras: int = 0

    # Summary cards
    footfall: DashboardV2FootfallMetric
    gender: DashboardV2GenderMetric
    age_groups: DashboardV2AgeGroupsMetric
    purchase_count: DashboardV2PurchaseMetric

    # Charts
    footfall_over_time: List[DashboardV2FootfallPoint] = []
    gender_trend: List[DashboardV2GenderTrendPoint] = []
    age_group_distribution: List[DashboardV2AgeGroupDistributionPoint] = []
