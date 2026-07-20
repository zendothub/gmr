import uuid
from datetime import timedelta
from unittest.mock import AsyncMock

import numpy as np
import pytest

from app.modules.reid.identity_decision_engine import IdentityDecisionEngine
from app.utils.time_utils import utc_now


@pytest.fixture
def mock_db():
    return AsyncMock()


@pytest.fixture
def engine():
    return IdentityDecisionEngine()


@pytest.mark.asyncio
async def test_decide_identity_with_face_match(engine, mock_db):
    """Test that face matching takes precedence over body ReID."""
    mean_body_embedding = np.random.rand(512).astype(np.float32)
    mean_body_embedding /= np.linalg.norm(mean_body_embedding) + 1e-9
    face_embedding = np.random.rand(512).astype(np.float32)
    face_embedding /= np.linalg.norm(face_embedding) + 1e-9
    camera_id = uuid.uuid4()
    person_id = uuid.uuid4()
    now = utc_now()

    async def mock_search_similar_face(db, face_emb):
        return {
            "person_identity_id": person_id,
            "camera_id": camera_id,
            "face_score": 0.9,
            "captured_at": now,
            "last_seen_at": now,
            "first_seen_at": now - timedelta(minutes=5),
            "distance": 0.1,  # similarity = 0.9
        }

    engine._search_similar_face = mock_search_similar_face
    engine._search_similar = AsyncMock()  # Should not be called
    engine._face_match_passes_cluster_median = AsyncMock(return_value=True)
    engine._has_same_camera_overlap = AsyncMock(return_value=False)
    engine._person_exists = AsyncMock(return_value=True)
    engine._attach_embeddings = AsyncMock(return_value=True)
    engine._update_person = AsyncMock(return_value=True)
    engine._store_embedding = AsyncMock(return_value=True)
    engine._store_face_embedding = AsyncMock(return_value=True)

    pid, score, is_confident, is_new, prune_old = await engine.decide_identity(
        db=mock_db,
        mean_embedding=mean_body_embedding,
        camera_id=camera_id,
        crop_quality_score=0.8,
        crop_path="/tmp/crop.jpg",
        current_person_id=None,
        face_embedding=face_embedding,
        face_score=0.9,
        face_crop_path="/tmp/face.jpg",
        good_face_count=2,
        face_embedding_list=[(face_embedding, 0.9, None)],
    )

    assert pid == person_id
    assert abs(score - 0.9) < 1e-6
    assert is_confident is True
    assert is_new is False
    engine._search_similar.assert_not_called()
    # Face may be stored via _attach_embeddings or _store_face_embedding
    assert (
        engine._store_face_embedding.called
        or engine._attach_embeddings.called
    )


@pytest.mark.asyncio
async def test_decide_identity_fallback_to_body(engine, mock_db):
    """Test that if face doesn't match, it falls back to body ReID."""
    mean_body_embedding = np.random.rand(512).astype(np.float32)
    mean_body_embedding /= np.linalg.norm(mean_body_embedding) + 1e-9
    face_embedding = np.random.rand(512).astype(np.float32)
    face_embedding /= np.linalg.norm(face_embedding) + 1e-9
    camera_id = uuid.uuid4()
    person_id = uuid.uuid4()
    now = utc_now()

    engine._search_similar_face = AsyncMock(return_value=None)

    async def mock_search_similar(db, emb, top_k=5, recent_body_only=False):
        return [{
            "person_identity_id": person_id,
            "camera_id": camera_id,
            "crop_quality": 0.8,
            "captured_at": now,
            "last_seen_at": now,
            "first_seen_at": now - timedelta(minutes=5),
            "distance": 0.2,  # ANN hint only
            "similarity": 0.8,
        }]

    engine._search_similar = mock_search_similar
    engine._person_body_count = AsyncMock(return_value=3)
    engine._person_body_median_sim = AsyncMock(return_value=0.80)
    engine._person_is_activity_recent = AsyncMock(return_value=True)
    engine._has_same_camera_overlap = AsyncMock(return_value=False)
    engine._person_exists = AsyncMock(return_value=True)
    engine._attach_embeddings = AsyncMock(return_value=True)
    engine._update_person = AsyncMock(return_value=True)
    engine._store_embedding = AsyncMock(return_value=True)
    engine._store_face_embedding = AsyncMock(return_value=True)
    engine._try_staff_reattach = AsyncMock(return_value=None)
    engine._get_person_face_embeddings = AsyncMock(return_value=[])

    pid, score, is_confident, is_new, prune_old = await engine.decide_identity(
        db=mock_db,
        mean_embedding=mean_body_embedding,
        camera_id=camera_id,
        crop_quality_score=0.8,
        face_embedding=face_embedding,
        face_score=0.60,
        good_face_count=1,
    )

    assert pid == person_id
    assert abs(score - 0.80) < 1e-6
    assert is_new is False
    engine._search_similar_face.assert_called_once()
