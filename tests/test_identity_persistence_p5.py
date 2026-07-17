"""P5: identity store safety — person-gone / FK race / session not poisoned."""

from __future__ import annotations

import uuid
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from sqlalchemy.exc import IntegrityError

from app.modules.reid.identity_decision_engine import (
    IdentityDecisionEngine,
    IdentityStoreError,
)
from app.utils.time_utils import utc_now


def _body(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def _cand(pid, dist: float = 0.3, recent: bool = True):
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
    session.execute = AsyncMock(return_value=MagicMock(scalar=MagicMock(return_value=1), fetchall=MagicMock(return_value=[]), fetchone=MagicMock(return_value=None)))
    # begin_nested as async context manager
    nested = AsyncMock()
    nested.__aenter__ = AsyncMock(return_value=None)
    nested.__aexit__ = AsyncMock(return_value=False)
    session.begin_nested = MagicMock(return_value=nested)
    return session


@pytest.mark.asyncio
async def test_person_exists_false(engine, db):
    db.execute = AsyncMock(return_value=MagicMock(scalar=MagicMock(return_value=None)))
    assert await engine._person_exists(db, uuid.uuid4()) is False


@pytest.mark.asyncio
async def test_stale_match_no_attach_returns_none(engine, db):
    """Body finds person that is gone at FOR SHARE — no create after match (unassigned)."""
    pid = uuid.uuid4()
    cam = uuid.uuid4()
    body = _body(1)

    with (
        patch.object(engine, "_search_similar", new_callable=AsyncMock) as body_search,
        patch.object(engine, "_person_body_count", new_callable=AsyncMock, return_value=3),
        patch.object(engine, "_person_body_median_sim", new_callable=AsyncMock, return_value=0.70),
        patch.object(engine, "_has_same_camera_overlap", new_callable=AsyncMock, return_value=False),
        patch.object(engine, "_person_exists", new_callable=AsyncMock, return_value=False),
        patch.object(engine, "_try_staff_reattach", new_callable=AsyncMock, return_value=None),
        patch.object(engine, "_create_new_person", new_callable=AsyncMock) as create,
        patch.object(engine, "_attach_embeddings", new_callable=AsyncMock) as attach,
    ):
        body_search.return_value = [_cand(pid, dist=0.30)]
        person_id, score, conf, is_new, _ = await engine.decide_identity(
            db=db,
            mean_embedding=body,
            camera_id=cam,
            crop_quality_score=0.9,
            current_person_id=None,
            face_embedding=None,
        )
    assert person_id is None
    attach.assert_not_called()
    # May still create if no match and face gates pass - here no face so create blocked
    # Important: must not call attach on stale


@pytest.mark.asyncio
async def test_attach_persistence_error_no_create(engine, db):
    """IdentityStoreError on attach → unassigned, never create clone."""
    pid = uuid.uuid4()
    cam = uuid.uuid4()
    body = _body(2)

    with (
        patch.object(engine, "_search_similar", new_callable=AsyncMock) as body_search,
        patch.object(engine, "_person_body_count", new_callable=AsyncMock, return_value=3),
        patch.object(engine, "_person_body_median_sim", new_callable=AsyncMock, return_value=0.70),
        patch.object(engine, "_has_same_camera_overlap", new_callable=AsyncMock, return_value=False),
        patch.object(engine, "_person_exists", new_callable=AsyncMock, return_value=True),
        patch.object(engine, "_try_staff_reattach", new_callable=AsyncMock, return_value=None),
        patch.object(
            engine, "_attach_embeddings", new_callable=AsyncMock,
            side_effect=IdentityStoreError("FK")
        ),
        patch.object(engine, "_create_new_person", new_callable=AsyncMock) as create,
    ):
        body_search.return_value = [_cand(pid, dist=0.30)]
        person_id, score, conf, is_new, _ = await engine.decide_identity(
            db=db,
            mean_embedding=body,
            camera_id=cam,
            crop_quality_score=0.9,
            current_person_id=None,
            face_embedding=None,
        )
    assert person_id is None
    assert is_new is False
    create.assert_not_called()


@pytest.mark.asyncio
async def test_store_embedding_raises_on_integrity(engine, db):
    pid = uuid.uuid4()
    cam = uuid.uuid4()
    body = _body(3)

    with patch.object(engine, "_person_exists", new_callable=AsyncMock, return_value=True):
        # first execute (existing body rows) returns empty; flush raises
        call_n = {"n": 0}

        async def exec_side(*a, **k):
            call_n["n"] += 1
            m = MagicMock()
            m.fetchall = MagicMock(return_value=[])
            m.scalar = MagicMock(return_value=1)
            return m

        db.execute = AsyncMock(side_effect=exec_side)
        db.add = MagicMock()
        db.flush = AsyncMock(side_effect=IntegrityError("stmt", {}, Exception("fk")))
        with pytest.raises(IdentityStoreError):
            await engine._store_embedding(db, pid, body, cam, 0.9, None)


@pytest.mark.asyncio
async def test_outer_exception_does_not_create(engine, db):
    """Any unexpected error: keep current / None — no create-on-poison fallback."""
    cam = uuid.uuid4()
    body = _body(4)
    with (
        patch.object(engine, "_search_similar", new_callable=AsyncMock, side_effect=RuntimeError("boom")),
        patch.object(engine, "_create_new_person", new_callable=AsyncMock) as create,
    ):
        person_id, score, conf, is_new, _ = await engine.decide_identity(
            db=db,
            mean_embedding=body,
            camera_id=cam,
            crop_quality_score=0.9,
            current_person_id=None,
            face_embedding=None,
        )
    assert person_id is None
    create.assert_not_called()
