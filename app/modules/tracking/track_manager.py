"""Track manager - maintains active track state in memory."""

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field

from loguru import logger

from app.utils.geometry import bbox_center, bbox_height, point_in_polygon, polygon_from_json
from app.utils.time_utils import utc_now, seconds_since


@dataclass
class ActiveTrack:
    """In-memory representation of an active track."""
    local_track_id: int
    track_session_id: Optional[uuid.UUID] = None
    camera_id: Optional[uuid.UUID] = None
    person_identity_id: Optional[uuid.UUID] = None
    started_at: datetime = field(default_factory=utc_now)
    last_seen_at: datetime = field(default_factory=utc_now)
    bbox: Optional[dict] = None
    prev_bbox: Optional[dict] = None
    bbox_history: List[dict] = field(default_factory=list)
    current_zones: Set[str] = field(default_factory=set)
    zone_enter_times: Dict[str, datetime] = field(default_factory=dict)
    dwell_seconds: Dict[str, float] = field(default_factory=dict)
    total_frames: int = 0
    confidence_sum: float = 0.0
    stability_score: float = 0.0
    last_reid_time: Optional[datetime] = None
    reid_attempted: bool = False

    # Refined ReID Accumulation & Confidence State
    reid_resolved: bool = False
    reid_confident: bool = False
    reid_score: float = 0.0
    reid_frame_count: int = 0

    # Demographics and Face crop tracking
    face_analysis_count: int = 0
    best_demographics: Optional[dict] = None

    # Track's best crop (body)
    best_crop_quality: float = 0.0
    best_crop_path: Optional[str] = None

    @property
    def track_age_seconds(self) -> float:
        return seconds_since(self.started_at)

    @property
    def avg_confidence(self) -> float:
        return self.confidence_sum / self.total_frames if self.total_frames > 0 else 0.0

    def should_run_reid(self) -> bool:
        """Check if ReID should be triggered for this track.

        Runs continuously throughout the track's lifetime.  When the identity is
        already confident we still run but at a reduced cadence (every 5th
        eligible candidate) to save compute while allowing correction if a
        higher-quality match appears later.
        """
        if not self.bbox:
            return False
        # Minimum crop size (pixels)
        if bbox_height(self.bbox) < 100:
            return False
        # Continuous mode: never fully stop.  If confident, only sample
        # occasionally to catch better-quality matches.
        if self.reid_confident:
            return self.reid_frame_count % 5 == 0
        return True


class TrackManager:
    """Manages active tracks in memory for a single camera."""

    def __init__(self, camera_id: uuid.UUID):
        self.camera_id = camera_id
        self.tracks: Dict[int, ActiveTrack] = {}  # local_track_id -> ActiveTrack
        self._stale_threshold = 5.0  # seconds before removing stale tracks

    def update_track(self, local_track_id: int, bbox: dict, confidence: float) -> ActiveTrack:
        """Update or create a track with new detection data."""
        now = utc_now()

        if local_track_id in self.tracks:
            track = self.tracks[local_track_id]
            track.prev_bbox = track.bbox
            track.bbox = bbox
            track.last_seen_at = now
            track.total_frames += 1
            track.confidence_sum += confidence

            # Keep limited bbox history
            track.bbox_history.append(bbox)
            if len(track.bbox_history) > 30:
                track.bbox_history = track.bbox_history[-30:]

            # Update stability score
            track.stability_score = self._compute_stability(track)
        else:
            track = ActiveTrack(
                local_track_id=local_track_id,
                camera_id=self.camera_id,
                started_at=now,
                last_seen_at=now,
                bbox=bbox,
                total_frames=1,
                confidence_sum=confidence,
            )
            track.bbox_history.append(bbox)
            self.tracks[local_track_id] = track

        return track

    def update_zones(self, track: ActiveTrack, zones_data: List[dict],
                     frame_width: int = 1920, frame_height: int = 1080):
        """
        Update which zones a track is currently in.

        Zone polygons are stored as 0–100 percentage coordinates (camera-agnostic).
        They are converted to pixel coordinates using the actual frame dimensions
        before comparing with the track's bounding-box positions.

        A track is considered "in" a zone when EITHER its bottom-centre (feet)
        OR its bbox-centre falls inside the zone polygon.  This allows both
        foot-level and body-level zone detection (e.g. a person standing behind
        a billing counter may have feet outside the drawn zone but their body
        clearly visible inside it).

        Args:
            track: The active track
            zones_data: List of zone dicts with id, polygon, zone_type, etc.
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
        """
        now = utc_now()
        if not track.bbox:
            return

        from app.utils.geometry import bbox_bottom_center, bbox_center

        # Raw pixel positions — we compare in pixel space, not normalised space
        bottom = bbox_bottom_center(track.bbox)
        centre = bbox_center(track.bbox)

        current_zone_ids = set()

        for zone in zones_data:
            zone_id = str(zone["id"])
            poly_points = polygon_from_json(zone.get("polygon"))

            if not poly_points:
                continue

            # Convert polygon from 0-100 percentage → pixel coordinates
            pixel_poly = [
                (p[0] * frame_width / 100.0, p[1] * frame_height / 100.0)
                for p in poly_points
            ]

            # Check both foot (bottom-centre) and body (bbox-centre)
            in_zone = point_in_polygon(bottom, pixel_poly) or point_in_polygon(centre, pixel_poly)

            if in_zone:
                current_zone_ids.add(zone_id)

                # Track zone entry time
                if zone_id not in track.zone_enter_times:
                    track.zone_enter_times[zone_id] = now

                # Update dwell time
                enter_time = track.zone_enter_times[zone_id]
                track.dwell_seconds[zone_id] = seconds_since(enter_time)
            else:
                # Exited zone
                if zone_id in track.zone_enter_times:
                    del track.zone_enter_times[zone_id]

        track.current_zones = current_zone_ids

    def get_active_tracks(self) -> List[ActiveTrack]:
        """Get all active tracks."""
        return list(self.tracks.values())

    def get_track(self, local_track_id: int) -> Optional[ActiveTrack]:
        """Get a specific track."""
        return self.tracks.get(local_track_id)

    def cleanup_stale_tracks(self) -> List[ActiveTrack]:
        """Remove tracks that haven't been seen recently. Returns removed tracks."""
        stale = []
        to_remove = []

        for tid, track in self.tracks.items():
            if seconds_since(track.last_seen_at) > self._stale_threshold:
                stale.append(track)
                to_remove.append(tid)

        for tid in to_remove:
            del self.tracks[tid]

        if stale:
            logger.debug(f"Cleaned up {len(stale)} stale tracks for camera {self.camera_id}")

        return stale

    def _compute_stability(self, track: ActiveTrack) -> float:
        """Compute track stability based on bbox consistency."""
        if len(track.bbox_history) < 3:
            return 0.0

        # Compute variance of bbox centers
        centers = [bbox_center(b) for b in track.bbox_history[-10:]]
        if len(centers) < 2:
            return 0.0

        x_coords = [c[0] for c in centers]
        y_coords = [c[1] for c in centers]

        x_var = max(x_coords) - min(x_coords)
        y_var = max(y_coords) - min(y_coords)

        # Lower variance = higher stability (normalized 0-1)
        max_var = 200.0  # pixels
        stability = max(0.0, 1.0 - ((x_var + y_var) / (2 * max_var)))

        # Factor in track age
        age_factor = min(1.0, track.track_age_seconds / 3.0)

        return stability * 0.7 + age_factor * 0.3

    def reset(self):
        """Clear all tracks."""
        self.tracks.clear()
