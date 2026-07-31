"""Unit tests for fragmented billing visit clustering / temporal stitch."""
from datetime import datetime, timedelta, timezone

from app.modules.jobs.tasks import cluster_sessions_into_visits, temporal_gap_seconds


def _s(start_off: float, dur: float, dwell: float = 10.0) -> dict:
    base = datetime(2026, 7, 31, 10, 0, 0, tzinfo=timezone.utc)
    start = base + timedelta(seconds=start_off)
    return {
        "sid": f"s{start_off}",
        "started_at": start,
        "ended_at": start + timedelta(seconds=dur),
        "max_dwell": dwell,
        "is_staff": False,
    }


def test_cluster_single_session():
    visits = cluster_sessions_into_visits([_s(0, 20, 20)], gap_seconds=60)
    assert len(visits) == 1
    assert len(visits[0]) == 1


def test_cluster_gap_within_threshold_sums_fragments():
    sessions = [_s(0, 20, 20), _s(25, 20, 20), _s(50, 15, 15)]
    visits = cluster_sessions_into_visits(sessions, gap_seconds=60)
    assert len(visits) == 1
    assert len(visits[0]) == 3
    total = sum(s["max_dwell"] for s in visits[0])
    assert total == 55


def test_cluster_large_gap_splits_visits():
    sessions = [_s(0, 20, 20), _s(200, 20, 20)]
    visits = cluster_sessions_into_visits(sessions, gap_seconds=60)
    assert len(visits) == 2


def test_cluster_same_cam_overlap_does_not_mix():
    a = _s(0, 60, 30)
    b = _s(10, 60, 30)
    visits = cluster_sessions_into_visits([a, b], gap_seconds=60)
    assert len(visits) == 2


def test_temporal_gap_within_one_minute():
    base = datetime(2026, 7, 31, 10, 0, 0, tzinfo=timezone.utc)
    a0, a1 = base, base + timedelta(seconds=13)
    b0, b1 = base + timedelta(seconds=20), base + timedelta(seconds=44)
    g = temporal_gap_seconds(a0, a1, b0, b1)
    assert g == 7.0
    assert g <= 60


def test_temporal_gap_overlap_is_none():
    base = datetime(2026, 7, 31, 10, 0, 0, tzinfo=timezone.utc)
    a0, a1 = base, base + timedelta(seconds=60)
    b0, b1 = base + timedelta(seconds=10), base + timedelta(seconds=70)
    assert temporal_gap_seconds(a0, a1, b0, b1) is None
