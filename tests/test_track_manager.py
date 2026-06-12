"""Tests for the in-memory track manager and ReID gating logic."""

import uuid
from datetime import timedelta

import pytest

pytest.importorskip("loguru")

from app.modules.tracking.track_manager import TrackManager, ActiveTrack
from app.utils.time_utils import utc_now


CAMERA_ID = uuid.uuid4()


def make_bbox(x1=0, y1=0, x2=60, y2=160):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


class TestTrackManager:
    def test_create_new_track(self):
        tm = TrackManager(CAMERA_ID)
        track = tm.update_track(1, make_bbox(), 0.9)
        assert track.local_track_id == 1
        assert track.total_frames == 1
        assert len(tm.get_active_tracks()) == 1

    def test_update_existing_track(self):
        tm = TrackManager(CAMERA_ID)
        tm.update_track(1, make_bbox(), 0.9)
        track = tm.update_track(1, make_bbox(x1=5), 0.8)
        assert track.total_frames == 2
        assert track.prev_bbox is not None

    def test_zone_dwell_tracking(self):
        tm = TrackManager(CAMERA_ID)
        track = tm.update_track(1, make_bbox(x1=40, y1=40, x2=60, y2=60), 0.9)
        zones = [
            {
                "id": str(uuid.uuid4()),
                "polygon": {"points": [[0, 0], [100, 0], [100, 100], [0, 100]]},
                "zone_type": "billing_zone",
            }
        ]
        tm.update_zones(track, zones)
        assert len(track.current_zones) == 1

    def test_zone_exit(self):
        tm = TrackManager(CAMERA_ID)
        track = tm.update_track(1, make_bbox(x1=400, y1=400, x2=420, y2=460), 0.9)
        zones = [
            {
                "id": str(uuid.uuid4()),
                "polygon": {"points": [[0, 0], [100, 0], [100, 100], [0, 100]]},
                "zone_type": "billing_zone",
            }
        ]
        tm.update_zones(track, zones)
        assert len(track.current_zones) == 0


class TestShouldRunReid:
    def _stable_track(self) -> ActiveTrack:
        track = ActiveTrack(
            local_track_id=1,
            camera_id=CAMERA_ID,
            bbox=make_bbox(y1=0, y2=200),  # height 200 > 120
        )
        track.started_at = utc_now() - timedelta(seconds=3)  # age > 1.5s
        track.stability_score = 0.8  # > 0.65
        track.last_reid_time = None
        return track

    def test_eligible_track(self):
        assert self._stable_track().should_run_reid() is True

    def test_too_young(self):
        track = self._stable_track()
        track.started_at = utc_now()
        assert track.should_run_reid() is False

    def test_bbox_too_small(self):
        track = self._stable_track()
        track.bbox = make_bbox(y1=0, y2=100)  # height 100 < 120
        assert track.should_run_reid() is False

    def test_unstable(self):
        track = self._stable_track()
        track.stability_score = 0.5
        assert track.should_run_reid() is False

    def test_recent_reid(self):
        track = self._stable_track()
        track.last_reid_time = utc_now()
        assert track.should_run_reid() is False