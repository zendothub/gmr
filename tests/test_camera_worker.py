"""Tests for CameraWorker GUI display and helper methods."""

import uuid
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import numpy as np

from app.modules.ai_runtime.camera_worker import CameraWorker
from app.modules.tracking.track_manager import ActiveTrack
from app.modules.rule_engine.rule_evaluator import RuleEvent
from app.modules.rule_engine.zone_event_detector import ZoneEvent


CAMERA_CONFIG = {
    "id": uuid.uuid4(),
    "rtsp_url": "rtsp://localhost:8554/test",
    "role": "general",
    "fps_target": 10,
    "reid_enabled": False,
    "demographic_enabled": False,
    "frame_rotation": 0,
}

RUNTIME_CONFIG = {
    "zones": [],
    "views": [],
    "rules": [],
}


@pytest.fixture
def mock_worker():
    with patch("app.modules.ai_runtime.camera_worker.LatestFrameBuffer"), \
         patch("app.modules.ai_runtime.camera_worker.get_camera_detector"):
        worker = CameraWorker(CAMERA_CONFIG, RUNTIME_CONFIG)
        yield worker


def test_check_gui_availability_default_false(mock_worker):
    """By default RUNTIME_SHOW_GUI is False, so GUI is not available."""
    mock_worker.settings.RUNTIME_SHOW_GUI = False
    assert mock_worker._check_gui_availability() is False


@patch("app.modules.ai_runtime.camera_worker.cv2")
def test_check_gui_availability_true_when_supported(mock_cv2, mock_worker):
    """When RUNTIME_SHOW_GUI is True and cv2 works, GUI is available."""
    mock_worker.settings.RUNTIME_SHOW_GUI = True
    
    # Reset cached attribute if any
    if hasattr(mock_worker, "_gui_available"):
        delattr(mock_worker, "_gui_available")
        
    mock_cv2.namedWindow.return_value = None
    mock_cv2.destroyWindow.return_value = None
    
    assert mock_worker._check_gui_availability() is True
    assert mock_worker._gui_available is True
    mock_cv2.namedWindow.assert_called_once_with("GUI_Check", mock_cv2.WINDOW_AUTOSIZE)
    mock_cv2.destroyWindow.assert_called_once_with("GUI_Check")


@patch("app.modules.ai_runtime.camera_worker.cv2")
def test_check_gui_availability_fails_gracefully(mock_cv2, mock_worker):
    """When RUNTIME_SHOW_GUI is True but cv2 fails, GUI is marked as unavailable."""
    mock_worker.settings.RUNTIME_SHOW_GUI = True
    
    # Reset cached attribute if any
    if hasattr(mock_worker, "_gui_available"):
        delattr(mock_worker, "_gui_available")
        
    mock_cv2.namedWindow.side_effect = Exception("No display")
    
    assert mock_worker._check_gui_availability() is False
    assert mock_worker._gui_available is False


@patch("app.modules.ai_runtime.camera_worker.cv2")
def test_display_gui_frame_no_gui(mock_cv2, mock_worker):
    """When GUI is not available, we should not call imshow."""
    mock_worker.settings.RUNTIME_SHOW_GUI = False
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    mock_worker._display_gui_frame(frame, [])
    mock_cv2.imshow.assert_not_called()


@patch("app.modules.ai_runtime.camera_worker.cv2")
def test_display_gui_frame_with_tracks(mock_cv2, mock_worker):
    """When GUI is available, draw overlays and call imshow."""
    mock_worker.settings.RUNTIME_SHOW_GUI = True
    mock_worker._gui_available = True
    mock_cv2.getTextSize.return_value = ((50, 15), 5)
    
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    track = ActiveTrack(
        local_track_id=42,
        bbox={"x1": 100, "y1": 100, "x2": 200, "y2": 300},
    )
    
    mock_worker._display_gui_frame(frame, [track])
    
    # We should have drawn a rectangle and put text on the frame
    assert mock_cv2.rectangle.called
    assert mock_cv2.putText.called
    assert mock_cv2.imshow.called
    assert mock_cv2.waitKey.called


@pytest.mark.asyncio
async def test_close_track_session_fires_exit_event(mock_worker):
    """Test that closing a track session fires a person_left_view event."""
    from unittest.mock import AsyncMock
    mock_db = AsyncMock()
    
    track = ActiveTrack(local_track_id=1, bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10})
    track.track_session_id = uuid.uuid4()
    track.person_identity_id = uuid.uuid4()
    
    await mock_worker._close_track_session(mock_db, track)
    
    # Verify db.add was called with the exit event
    added_items = [call.args[0] for call in mock_db.add.call_args_list]
    exit_events = [item for item in added_items if getattr(item, "event_type", None) == "person_left_view"]
    
    assert len(exit_events) == 1
    assert exit_events[0].person_identity_id == track.person_identity_id
    assert exit_events[0].track_session_id == track.track_session_id


def test_refresh_event_person_ids_fills_null_from_active_track(mock_worker):
    """Same-frame ReID: rule events snapshot null; refresh from live tracks."""
    sid = uuid.uuid4()
    pid = uuid.uuid4()
    track = ActiveTrack(local_track_id=7, track_session_id=sid, person_identity_id=pid)
    mock_worker.track_manager.get_active_tracks = MagicMock(return_value=[track])

    rule = RuleEvent(
        rule_id=uuid.uuid4(),
        rule_type="billing_interaction",
        camera_id=uuid.uuid4(),
        zone_id=uuid.uuid4(),
        track_session_id=sid,
        person_identity_id=None,
        event_type="billing_interaction",
        severity="info",
        description="test",
        metadata={},
    )
    zone = ZoneEvent(
        event_type="zone_dwell_milestone",
        camera_id=uuid.uuid4(),
        zone_id=uuid.uuid4(),
        track_session_id=sid,
        person_identity_id=None,
        description="dwell",
        metadata={},
    )
    mock_worker._refresh_event_person_ids([rule], [zone])
    assert rule.person_identity_id == pid
    assert zone.person_identity_id == pid


def test_refresh_event_person_ids_keeps_existing(mock_worker):
    sid = uuid.uuid4()
    existing = uuid.uuid4()
    other = uuid.uuid4()
    track = ActiveTrack(local_track_id=1, track_session_id=sid, person_identity_id=other)
    mock_worker.track_manager.get_active_tracks = MagicMock(return_value=[track])
    rule = RuleEvent(
        rule_id=uuid.uuid4(),
        rule_type="billing_interaction",
        camera_id=uuid.uuid4(),
        zone_id=uuid.uuid4(),
        track_session_id=sid,
        person_identity_id=existing,
        event_type="billing_interaction",
        severity="info",
        description="test",
        metadata={},
    )
    mock_worker._refresh_event_person_ids([rule], [])
    assert rule.person_identity_id == existing


@pytest.mark.asyncio
async def test_backfill_null_person_fks_noops_without_ids(mock_worker):
    db = AsyncMock()
    await mock_worker._backfill_null_person_fks(db, None, uuid.uuid4())
    await mock_worker._backfill_null_person_fks(db, uuid.uuid4(), None)
    db.execute.assert_not_called()
