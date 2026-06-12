"""Identity decision engine - matches embeddings to existing persons using pgvector."""

import uuid
from typing import Optional, Tuple
from datetime import datetime

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from loguru import logger

from app.config import get_settings
from app.core.db.models.person import PersonIdentity, PersonEmbedding
from app.utils.time_utils import utc_now, time_score


class IdentityDecisionEngine:
    """
    Decides whether a new embedding matches an existing person identity.

    final_score = 0.60 * visual_similarity
               + 0.15 * crop_quality
               + 0.10 * time_score
               + 0.10 * camera_transition_score
               + 0.05 * track_stability

    If final_score >= 0.78 -> match existing person
    Otherwise -> create new anonymous person
    """

    def __init__(self):
        self.settings = get_settings()
        self.match_threshold = self.settings.REID_MATCH_THRESHOLD

    async def decide_identity(
        self,
        db: AsyncSession,
        embedding: np.ndarray,
        crop_quality_score: float,
        camera_id: uuid.UUID,
        track_stability: float,
        crop_path: Optional[str] = None,
        top_k: int = 5,
    ) -> Tuple[uuid.UUID, bool]:
        """
        Decide identity for a new embedding.

        Args:
            db: Database session
            embedding: 512-dim embedding vector
            crop_quality_score: Quality score of the crop (0-1)
            camera_id: Camera ID where detected
            track_stability: Track stability score (0-1)
            crop_path: Path to saved crop image
            top_k: Number of top matches to consider

        Returns:
            Tuple of (person_identity_id, is_new_identity)
        """
        try:
            # Search for similar embeddings using pgvector cosine distance
            candidates = await self._search_similar(db, embedding, top_k)

            best_match = None
            best_score = 0.0

            for candidate in candidates:
                visual_sim = 1.0 - candidate["distance"]  # cosine distance -> similarity
                t_score = time_score(candidate["last_seen_at"])

                # Camera transition score: higher if detected on different camera
                cam_transition = 0.8 if str(candidate["camera_id"]) != str(camera_id) else 1.0

                final_score = (
                    0.60 * visual_sim
                    + 0.15 * crop_quality_score
                    + 0.10 * t_score
                    + 0.10 * cam_transition
                    + 0.05 * track_stability
                )

                if final_score > best_score:
                    best_score = final_score
                    best_match = candidate

            if best_match and best_score >= self.match_threshold:
                # Match existing person
                person_id = best_match["person_identity_id"]
                logger.info(
                    f"ReID match: person={person_id}, score={best_score:.3f}, "
                    f"visual_sim={1.0 - best_match['distance']:.3f}"
                )

                # Update last_seen_at and visit_count
                await self._update_person(db, person_id)

                # Store new embedding
                await self._store_embedding(
                    db, person_id, embedding, camera_id, crop_quality_score, crop_path
                )

                return person_id, False

            else:
                # Create new anonymous person
                person_id = await self._create_new_person(
                    db, embedding, camera_id, crop_quality_score, crop_path
                )
                logger.info(f"New person created: {person_id} (best_score={best_score:.3f})")
                return person_id, True

        except Exception as e:
            logger.error(f"Identity decision failed: {e}")
            # Fallback: create new person
            person_id = await self._create_new_person(
                db, embedding, camera_id, crop_quality_score, crop_path
            )
            return person_id, True

    async def _search_similar(
        self, db: AsyncSession, embedding: np.ndarray, top_k: int
    ) -> list:
        """Search for similar embeddings using pgvector cosine distance.

        Optimizations:
        - `ivfflat.probes` raised for better recall on the IVFFlat index.
        - Candidates limited to embeddings captured in the last 48 hours,
          keeping the search fast as the table grows (a same-day shopper
          is the realistic re-identification window for a pharmacy).
        """
        try:
            embedding_list = embedding.tolist()

            # Better recall on the IVFFlat index for this transaction
            await db.execute(text("SET LOCAL ivfflat.probes = 10"))

            query = text("""
                SELECT pe.person_identity_id, pe.camera_id, pe.crop_quality,
                       pe.captured_at,
                       pi.last_seen_at,
                       pe.embedding <=> :embedding AS distance
                FROM person_embeddings pe
                JOIN person_identities pi ON pe.person_identity_id = pi.id
                WHERE pe.captured_at > NOW() - INTERVAL '48 hours'
                ORDER BY pe.embedding <=> :embedding
                LIMIT :top_k
            """)

            result = await db.execute(
                query, {"embedding": str(embedding_list), "top_k": top_k}
            )
            rows = result.fetchall()

            candidates = []
            for row in rows:
                candidates.append({
                    "person_identity_id": row[0],
                    "camera_id": row[1],
                    "crop_quality": row[2],
                    "captured_at": row[3],
                    "last_seen_at": row[4],
                    "distance": float(row[5]),
                })

            return candidates

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    async def _update_person(self, db: AsyncSession, person_id: uuid.UUID):
        """Update existing person's last_seen and visit count."""
        result = await db.execute(
            select(PersonIdentity).where(PersonIdentity.id == person_id)
        )
        person = result.scalar_one_or_none()
        if person:
            person.last_seen_at = utc_now()
            person.visit_count += 1

    # Maximum stored embeddings per identity; lowest-quality extras are pruned
    MAX_EMBEDDINGS_PER_PERSON = 10

    async def _store_embedding(
        self,
        db: AsyncSession,
        person_id: uuid.UUID,
        embedding: np.ndarray,
        camera_id: uuid.UUID,
        crop_quality_score: float,
        crop_path: Optional[str],
    ):
        """Store a new embedding for an existing person (capped per identity)."""
        emb = PersonEmbedding(
            person_identity_id=person_id,
            embedding=embedding.tolist(),
            camera_id=camera_id,
            crop_quality=crop_quality_score,
            crop_path=crop_path,
            captured_at=utc_now(),
        )
        db.add(emb)
        await db.flush()
        await self._prune_embeddings(db, person_id)

    async def _prune_embeddings(self, db: AsyncSession, person_id: uuid.UUID):
        """Keep only the best-quality K embeddings per identity (unbounded growth guard)."""
        try:
            await db.execute(
                text("""
                    DELETE FROM person_embeddings
                    WHERE id IN (
                        SELECT id FROM person_embeddings
                        WHERE person_identity_id = :pid
                        ORDER BY crop_quality DESC, captured_at DESC
                        OFFSET :keep
                    )
                """),
                {"pid": str(person_id), "keep": self.MAX_EMBEDDINGS_PER_PERSON},
            )
        except Exception as e:
            logger.warning(f"Embedding pruning failed for person {person_id}: {e}")

    async def _create_new_person(
        self,
        db: AsyncSession,
        embedding: np.ndarray,
        camera_id: uuid.UUID,
        crop_quality_score: float,
        crop_path: Optional[str],
    ) -> uuid.UUID:
        """Create a new anonymous person identity."""
        now = utc_now()
        person = PersonIdentity(
            label=None,
            first_seen_at=now,
            last_seen_at=now,
            visit_count=1,
            is_anonymous=True,
        )
        db.add(person)
        await db.flush()

        emb = PersonEmbedding(
            person_identity_id=person.id,
            embedding=embedding.tolist(),
            camera_id=camera_id,
            crop_quality=crop_quality_score,
            crop_path=crop_path,
            captured_at=now,
        )
        db.add(emb)
        await db.flush()

        return person.id
