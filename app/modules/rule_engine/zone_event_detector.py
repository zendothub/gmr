"""Zone event detector for automatic enter, exit, and dwell milestone events."""

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from loguru import logger

from app.utils.time_utils import utc_now
from app.modules.tracking.track_manager import ActiveTrack


@dataclass
class ZoneEvent:
    """Represents an automatic zone transition or dwell event."""
    event_type: str  # "zone_enter", "zone_exit", "zone_dwell_milestone"
    camera_id: uuid.UUID
    zone_id: uuid.UUID
    track_session_id: Optional[uuid.UUID]
    person_identity_id: Optional[uuid.UUID]
    description: str
    metadata: dict = field(default_factory=dict)


class ZoneEventDetector:
    """
    Detects and generates automatic events for zone entries, exits, and dwell milestones
    by comparing active track states frame-over-frame.
    """

    def __init__(self, camera_id: uuid.UUID):
        self.camera_id = camera_id
        self._last_zones: Dict[int, Set[str]] = {}  # local_track_id -> set of zone_ids
        self._fired_milestones: Dict[Tuple[int, str], Set[int]] = {}  # (local_track_id, zone_id) -> set of milestones (30, 60, 120)

    def detect(self, active_tracks: List[ActiveTrack]) -> List[ZoneEvent]:
        """
        Compare current track zones against previous frame to detect events.
        
        Args:
            active_tracks: List of currently tracked entities
            
        Returns:
            List of generated ZoneEvent objects
        """
        events = []
        current_track_ids = set()

        for track in active_tracks:
            track_id = track.local_track_id
            current_track_ids.add(track_id)

            # Retrieve current zones for the track (convert string IDs to UUID format if needed)
            current_zones = track.current_zones
            prev_zones = self._last_zones.get(track_id, set())

            # Detect zone entry (entered this frame)
            entered_zones = current_zones - prev_zones
            for zone_str in entered_zones:
                try:
                    zone_uuid = uuid.UUID(zone_str)
                    events.append(
                        ZoneEvent(
                            event_type="zone_enter",
                            camera_id=self.camera_id,
                            zone_id=zone_uuid,
                            track_session_id=track.track_session_id,
                            person_identity_id=track.person_identity_id,
                            description=f"Person entered zone {zone_str}",
                            metadata={"local_track_id": track_id},
                        )
                    )
                except Exception as e:
                    logger.error(f"Failed to trigger zone_enter: invalid zone UUID {zone_str}: {e}")

            # Detect zone exit (exited this frame)
            exited_zones = prev_zones - current_zones
            for zone_str in exited_zones:
                try:
                    zone_uuid = uuid.UUID(zone_str)
                    # Read dwell time for the exited zone
                    dwell_seconds = track.dwell_seconds.get(zone_str, 0.0)
                    events.append(
                        ZoneEvent(
                            event_type="zone_exit",
                            camera_id=self.camera_id,
                            zone_id=zone_uuid,
                            track_session_id=track.track_session_id,
                            person_identity_id=track.person_identity_id,
                            description=f"Person exited zone {zone_str} after {dwell_seconds:.1f}s",
                            metadata={
                                "local_track_id": track_id,
                                "dwell_seconds": dwell_seconds,
                            },
                        )
                    )
                    # Clean up milestones for this track/zone
                    self._fired_milestones.pop((track_id, zone_str), None)
                except Exception as e:
                    logger.error(f"Failed to trigger zone_exit: invalid zone UUID {zone_str}: {e}")

            # Detect dwell milestones (30s, 60s, 120s) for zones the person is currently in
            for zone_str in current_zones:
                try:
                    zone_uuid = uuid.UUID(zone_str)
                    dwell_seconds = track.dwell_seconds.get(zone_str, 0.0)
                    
                    # Milestone check thresholds
                    milestone_thresholds = [120, 60, 30]
                    fired = self._fired_milestones.setdefault((track_id, zone_str), set())

                    for threshold in milestone_thresholds:
                        if dwell_seconds >= threshold and threshold not in fired:
                            fired.add(threshold)
                            events.append(
                                ZoneEvent(
                                    event_type="zone_dwell_milestone",
                                    camera_id=self.camera_id,
                                    zone_id=zone_uuid,
                                    track_session_id=track.track_session_id,
                                    person_identity_id=track.person_identity_id,
                                    description=f"Person reached {threshold}s dwell milestone in zone {zone_str}",
                                    metadata={
                                        "local_track_id": track_id,
                                        "dwell_seconds": dwell_seconds,
                                        "threshold_reached": threshold,
                                    },
                                )
                            )
                            break  # fire only the highest matching milestone in one frame
                except Exception as e:
                    logger.error(f"Failed to process milestones for zone {zone_str}: {e}")

            # Keep trace of the track's zones for the next frame
            self._last_zones[track_id] = current_zones

        # Clean up untracked/stale tracks to avoid memory leaks
        stale_track_ids = set(self._last_zones.keys()) - current_track_ids
        for track_id in stale_track_ids:
            self._last_zones.pop(track_id, None)
            # Prune all matching track ID keys in milestones
            keys_to_remove = [k for k in self._fired_milestones.keys() if k[0] == track_id]
            for key in keys_to_remove:
                self._fired_milestones.pop(key, None)

        return events
