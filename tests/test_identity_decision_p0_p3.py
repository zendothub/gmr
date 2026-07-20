"""Unit tests for identity engine P0–P3 fixes (recent face/body + same-cam).

Mocks DB search helpers so decide_identity can be exercised without pgvector.
"""

from __future__ import annotations

import uuid
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.config import get_settings
from app.modules.reid.identity_decision_engine import IdentityDecisionEngine
from app.utils.time_utils import utc_now


def _face(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def _body(seed: int = 1) -> np.ndarray:
    return _face(seed + 100)


def _cand(pid, dist: float, recent: bool = True):
    last = utc_now() - timedelta(seconds=30 if recent else 3600)
    return {
        "person_identity_id": pid,
        "camera_id": uuid.uuid4(),
        "distance": dist,
        "similarity": 1.0 - dist,
        "last_seen_at": last,
        "first_seen_at": last - timedelta(minutes=10),
        "captured_at": last,
        "crop_quality": 0.9,
    }


@pytest.fixture
def engine():
    return IdentityDecisionEngine()


@pytest.fixture
def db():
    session = AsyncMock()
    session.execute = AsyncMock(return_value=MagicMock())
    nested = AsyncMock()
    nested.__aenter__ = AsyncMock(return_value=None)
    nested.__aexit__ = AsyncMock(return_value=False)
    session.begin_nested = MagicMock(return_value=nested)
    return session


@pytest.mark.asyncio
async def test_accept_threshold_tiers(engine):
    s = get_settings()
    assert engine._accept_threshold("face_recent", True) == s.FACE_MATCH_THRESHOLD_RECENT
    assert engine._accept_threshold("face_strict", True) == s.FACE_MATCH_THRESHOLD
    assert engine._accept_threshold("body_recent", False) == s.RECENT_BODY_SINGLE_MATCH_THRESHOLD
    assert engine._accept_threshold("body", False) == s.REID_MATCH_THRESHOLD
    assert engine._accept_threshold("staff_reattach", False) == s.STAFF_REATTACH_BODY_MEDIAN


@pytest.mark.asyncio
async def test_face_recent_grey_zone_matches(engine, db):
    """P1: grey-zone face 0.37 + recent must attach (not recreate at thr 0.40)."""
    pid = uuid.uuid4()
    cam = uuid.uuid4()
    face = _face(1)
    body = _body(1)

    with (
        patch.object(engine, "_search_similar_face", new_callable=AsyncMock) as face_search,
        patch.object(
            engine, "_face_match_passes_cluster_median", new_callable=AsyncMock, return_value=True
        ),
        patch.object(engine, "_has_same_camera_overlap", new_callable=AsyncMock, return_value=False),
        patch.object(engine, "_person_exists", new_callable=AsyncMock, return_value=True),
        patch.object(engine, "_attach_embeddings", new_callable=AsyncMock, return_value=True),
        patch.object(engine, "_update_person", new_callable=AsyncMock, return_value=True),
        patch.object(engine, "_store_embedding", new_callable=AsyncMock),
        patch.object(engine, "_store_face_embedding", new_callable=AsyncMock),
        patch.object(engine, "_search_similar", new_callable=AsyncMock, return_value=[]),
        patch.object(engine, "_try_staff_reattach", new_callable=AsyncMock, return_value=None),
    ):
        face_search.return_value = _cand(pid, dist=1.0 - 0.37, recent=True)
        person_id, score, conf, is_new, prune = await engine.decide_identity(
            db=db,
            mean_embedding=body,
            camera_id=cam,
            crop_quality_score=0.9,
            current_person_id=None,
            face_embedding=face,
            face_score=0.8,
            good_face_count=2,
            face_embedding_list=[(face, 0.8, None)],
            track_started_at=utc_now(),
            track_session_id=uuid.uuid4(),
        )
    assert person_id == pid
    assert is_new is False
    assert abs(score - 0.37) < 1e-6
    assert conf is False  # below confidence_limit


@pytest.mark.asyncio
async def test_face_recent_non_recent_does_not_use_grey(engine, db):
    """Non-recent face at 0.37 should not match (falls through; create may run)."""
    pid = uuid.uuid4()
    cam = uuid.uuid4()
    face = _face(2)
    body = _body(2)

    with (
        patch.object(engine, "_search_similar_face", new_callable=AsyncMock) as face_search,
        patch.object(
            engine, "_face_match_passes_cluster_median", new_callable=AsyncMock, return_value=True
        ),
        patch.object(engine, "_person_is_activity_recent", new_callable=AsyncMock, return_value=False),
        patch.object(engine, "_has_same_camera_overlap", new_callable=AsyncMock, return_value=False),
        patch.object(engine, "_search_similar", new_callable=AsyncMock, return_value=[]),
        patch.object(engine, "_try_staff_reattach", new_callable=AsyncMock, return_value=None),
        patch.object(engine, "_create_new_person", new_callable=AsyncMock, return_value=None),
        patch.object(engine, "_try_body_only_create", new_callable=AsyncMock, return_value=None),
    ):
        face_search.return_value = _cand(pid, dist=1.0 - 0.37, recent=False)
        person_id, score, conf, is_new, prune = await engine.decide_identity(
            db=db,
            mean_embedding=body,
            camera_id=cam,
            crop_quality_score=0.9,
            current_person_id=None,
            face_embedding=face,
            face_score=0.8,
            good_face_count=2,
            face_embedding_list=[(face, 0.8, None)],
            track_started_at=utc_now(),
            track_session_id=uuid.uuid4(),
        )
    assert person_id is None  # create blocked / no match


@pytest.mark.asyncio
async def test_body_recent_single_matches_without_consensus_votes(engine, db):
    """P0: unique-person body median ≥ 0.55 recent attaches (no 2-of-3 votes)."""
    pid = uuid.uuid4()
    cam = uuid.uuid4()
    body = _body(3)

    with (
        patch.object(engine, "_search_similar", new_callable=AsyncMock) as body_search,
        patch.object(engine, "_person_body_count", new_callable=AsyncMock, return_value=3),
        patch.object(engine, "_person_body_median_sim", new_callable=AsyncMock, return_value=0.58),
        patch.object(engine, "_person_is_activity_recent", new_callable=AsyncMock, return_value=True),
        patch.object(engine, "_has_same_camera_overlap", new_callable=AsyncMock, return_value=False),
        patch.object(engine, "_person_exists", new_callable=AsyncMock, return_value=True),
        patch.object(engine, "_attach_embeddings", new_callable=AsyncMock, return_value=True),
        patch.object(engine, "_update_person", new_callable=AsyncMock, return_value=True),
        patch.object(engine, "_store_embedding", new_callable=AsyncMock),
        patch.object(engine, "_try_staff_reattach", new_callable=AsyncMock, return_value=None),
    ):
        # distance from pgvector best-emb; media gating uses median override
        body_search.return_value = [_cand(pid, dist=0.48, recent=True)]
        person_id, score, conf, is_new, prune = await engine.decide_identity(
            db=db,
            mean_embedding=body,
            camera_id=cam,
            crop_quality_score=0.9,
            current_person_id=None,
            face_embedding=None,
            face_score=0.0,
            good_face_count=0,
            track_started_at=utc_now(),
            track_session_id=uuid.uuid4(),
        )
    assert person_id == pid
    assert abs(score - 0.58) < 1e-6
    assert is_new is False
    assert conf is False


@pytest.mark.asyncio
async def test_body_strict_median_match(engine, db):
    """Strong recent-gallery body match (median ≥ 0.50) attaches."""
    pid = uuid.uuid4()
    cam = uuid.uuid4()
    body = _body(4)

    with (
        patch.object(engine, "_search_similar", new_callable=AsyncMock) as body_search,
        patch.object(engine, "_person_body_count", new_callable=AsyncMock, return_value=4),
        patch.object(engine, "_person_body_median_sim", new_callable=AsyncMock, return_value=0.62),
        patch.object(engine, "_person_is_activity_recent", new_callable=AsyncMock, return_value=True),
        patch.object(engine, "_has_same_camera_overlap", new_callable=AsyncMock, return_value=False),
        patch.object(engine, "_person_exists", new_callable=AsyncMock, return_value=True),
        patch.object(engine, "_attach_embeddings", new_callable=AsyncMock, return_value=True),
        patch.object(engine, "_update_person", new_callable=AsyncMock, return_value=True),
        patch.object(engine, "_store_embedding", new_callable=AsyncMock),
        patch.object(engine, "_try_staff_reattach", new_callable=AsyncMock, return_value=None),
    ):
        # Must be activity-recent: customer body gallery is recent-window only
        body_search.return_value = [_cand(pid, dist=0.35, recent=True)]
        person_id, score, conf, is_new, _ = await engine.decide_identity(
            db=db,
            mean_embedding=body,
            camera_id=cam,
            crop_quality_score=0.9,
            current_person_id=None,
            face_embedding=None,
        )
    assert person_id == pid
    assert abs(score - 0.62) < 1e-6


@pytest.mark.asyncio
async def test_body_ambiguous_top_two_rejected(engine, db):
    pid_a = uuid.uuid4()
    pid_b = uuid.uuid4()
    cam = uuid.uuid4()
    body = _body(5)

    async def median_sim(db_, pid, emb, recent_only=False):
        return 0.70 if pid == pid_a else 0.69

    with (
        patch.object(engine, "_search_similar", new_callable=AsyncMock) as body_search,
        patch.object(engine, "_person_body_count", new_callable=AsyncMock, return_value=3),
        patch.object(engine, "_person_body_median_sim", new_callable=AsyncMock, side_effect=median_sim),
        patch.object(engine, "_person_is_activity_recent", new_callable=AsyncMock, return_value=True),
        patch.object(engine, "_has_same_camera_overlap", new_callable=AsyncMock, return_value=False),
        patch.object(engine, "_try_staff_reattach", new_callable=AsyncMock, return_value=None),
        patch.object(engine, "_create_new_person", new_callable=AsyncMock, return_value=None),
        patch.object(engine, "_try_body_only_create", new_callable=AsyncMock, return_value=None),
    ):
        body_search.return_value = [
            _cand(pid_a, dist=0.30, recent=True),
            _cand(pid_b, dist=0.31, recent=True),
        ]
        person_id, score, conf, is_new, _ = await engine.decide_identity(
            db=db,
            mean_embedding=body,
            camera_id=cam,
            crop_quality_score=0.9,
            current_person_id=None,
            face_embedding=None,
        )
    assert person_id is None


@pytest.mark.asyncio
async def test_same_cam_overlap_suppresses_create(engine, db):
    """P2: after SAME_CAM reject, do not create a new clone person."""
    pid = uuid.uuid4()
    cam = uuid.uuid4()
    face = _face(6)
    body = _body(6)

    with (
        patch.object(engine, "_search_similar_face", new_callable=AsyncMock) as face_search,
        patch.object(
            engine, "_face_match_passes_cluster_median", new_callable=AsyncMock, return_value=True
        ),
        patch.object(engine, "_has_same_camera_overlap", new_callable=AsyncMock, return_value=True),
        patch.object(engine, "_create_new_person", new_callable=AsyncMock) as create,
        patch.object(engine, "_search_similar", new_callable=AsyncMock, return_value=[]),
        patch.object(engine, "_try_staff_reattach", new_callable=AsyncMock, return_value=None),
    ):
        face_search.return_value = _cand(pid, dist=1.0 - 0.60, recent=True)
        person_id, score, conf, is_new, prune = await engine.decide_identity(
            db=db,
            mean_embedding=body,
            camera_id=cam,
            crop_quality_score=0.9,
            current_person_id=None,
            face_embedding=face,
            face_score=0.85,
            good_face_count=3,
            face_embedding_list=[(face, 0.85, None)],
            track_started_at=utc_now(),
            track_session_id=uuid.uuid4(),
        )
    assert person_id is None
    assert is_new is False
    create.assert_not_called()


@pytest.mark.asyncio
async def test_create_blocked_low_score_logs_and_returns_none(engine, db):
    """P3: face score below min → no create (body-only also mocked off)."""
    cam = uuid.uuid4()
    face = _face(7)
    body = _body(7)

    with (
        patch.object(engine, "_search_similar_face", new_callable=AsyncMock, return_value=None),
        patch.object(engine, "_search_similar", new_callable=AsyncMock, return_value=[]),
        patch.object(engine, "_try_staff_reattach", new_callable=AsyncMock, return_value=None),
        patch.object(engine, "_try_body_only_create", new_callable=AsyncMock, return_value=None),
        patch.object(engine, "_has_same_camera_overlap", new_callable=AsyncMock, return_value=False),
    ):
        person_id, score, conf, is_new, _ = await engine.decide_identity(
            db=db,
            mean_embedding=body,
            camera_id=cam,
            crop_quality_score=0.9,
            current_person_id=None,
            face_embedding=face,
            face_score=0.40,  # < FACE_IDENTITY_MIN_SCORE 0.60
            good_face_count=5,
            face_embedding_list=[(face, 0.40, None)],
            track_started_at=utc_now(),
            track_session_id=uuid.uuid4(),
        )
    assert person_id is None
