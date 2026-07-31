"""Unit tests for fragmented billing visit clustering."""
from datetime import datetime, timedelta, timezone

from app.modules.jobs.tasks import cluster_sessions_into_visits


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
    # 20s + gap5 + 20s + gap5 + 15s → one visit (fragmentation case)
    sessions = [_s(0, 20, 20), _s(25, 20, 20), _s(50, 15, 15)]
    visits = cluster_sessions_into_visits(sessions, gap_seconds=60)
    assert len(visits) == 1
    assert len(visits[0]) == 3
    total = sum(s["max_dwell"] for s in visits[0])
    assert total == 55


def test_cluster_large_gap_splits_visits():
    sessions = [_s(0, 20, 20), _s(200, 20, 20)]  # gap 180s > 60
    visits = cluster_sessions_into_visits(sessions, gap_seconds=60)
    assert len(visits) == 2


def test_cluster_same_cam_overlap_does_not_mix():
    # Concurrent on same camera cannot be same person → new visit
    a = _s(0, 60, 30)
    b = _s(10, 60, 30)  # starts while a still open
    visits = cluster_sessions_into_visits([a, b], gap_seconds=60)
    assert len(visits) == 2
