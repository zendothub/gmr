import uuid
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import numpy as np

from app.modules.reid.identity_decision_engine import IdentityDecisionEngine


@pytest.fixture
def mock_db():
    return AsyncMock()


@pytest.fixture
def engine():
    return IdentityDecisionEngine()


@pytest.mark.asyncio
async def test_decide_identity_with_face_match(engine, mock_db):
    """Test that face matching takes precedence over body ReID."""
    mean_body_embedding = np.random.rand(512)
    face_embedding = np.random.rand(512)
    camera_id = uuid.uuid4()
    person_id = uuid.uuid4()
    
    # Mock face match success
    async def mock_search_similar_face(db, face_emb):
        return {
            "person_identity_id": person_id,
            "camera_id": camera_id,
            "face_score": 0.9,
            "captured_at": "now",
            "last_seen_at": "now",
            "distance": 0.1  # similarity = 0.9
        }
    
    engine._search_similar_face = mock_search_similar_face
    engine._search_similar = AsyncMock()  # Should not be called
    engine._update_person = AsyncMock()
    engine._store_embedding = AsyncMock()
    engine._store_face_embedding = AsyncMock()
    
    pid, score, is_confident, is_new, prune_old = await engine.decide_identity(
        db=mock_db,
        mean_embedding=mean_body_embedding,
        camera_id=camera_id,
        crop_quality_score=0.8,
        crop_path="/tmp/crop.jpg",
        current_person_id=None,
        face_embedding=face_embedding,
        face_score=0.9,
        face_crop_path="/tmp/face.jpg"
    )
    
    assert pid == person_id
    assert score == 0.9
    assert is_confident is True
    assert is_new is False
    engine._search_similar.assert_not_called()
    engine._store_face_embedding.assert_called_once()


@pytest.mark.asyncio
async def test_decide_identity_fallback_to_body(engine, mock_db):
    """Test that if face doesn't match, it falls back to body ReID."""
    mean_body_embedding = np.random.rand(512)
    face_embedding = np.random.rand(512)
    camera_id = uuid.uuid4()
    person_id = uuid.uuid4()
    
    # Mock face match failure
    engine._search_similar_face = AsyncMock(return_value=None)
    
    # Mock body match success
    async def mock_search_similar(db, emb, top_k):
        return [{
            "person_identity_id": person_id,
            "camera_id": camera_id,
            "crop_quality": 0.8,
            "captured_at": "now",
            "last_seen_at": "now",
            "distance": 0.2  # similarity = 0.8
        }]
        
    engine._search_similar = mock_search_similar
    engine._update_person = AsyncMock()
    engine._store_embedding = AsyncMock()
    engine._store_face_embedding = AsyncMock()
    
    pid, score, is_confident, is_new, prune_old = await engine.decide_identity(
        db=mock_db,
        mean_embedding=mean_body_embedding,
        camera_id=camera_id,
        crop_quality_score=0.8,
        face_embedding=face_embedding,
        face_score=0.60
    )
    
    assert pid == person_id
    assert score == 0.8
    assert is_confident is True
    assert is_new is False
    engine._search_similar_face.assert_called_once()
    engine._store_face_embedding.assert_called_once()
