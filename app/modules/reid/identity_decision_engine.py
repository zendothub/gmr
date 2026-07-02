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
    Decides whether a new accumulated embedding matches an existing person identity.
    Uses direct cosine similarity from pgvector candidates within the last 48 hours.
    """

    def __init__(self):
        self.settings = get_settings()
        self.match_threshold = self.settings.REID_MATCH_THRESHOLD

    async def decide_identity(
        self,
        db: AsyncSession,
        mean_embedding: Optional[np.ndarray],
        camera_id: uuid.UUID,
        crop_quality_score: float,
        crop_path: Optional[str] = None,
        current_person_id: Optional[uuid.UUID] = None,
        previous_score: float = 0.0,
        is_temporary: bool = False,
        face_embedding: Optional[np.ndarray] = None,
        face_score: float = 0.0,
        face_crop_path: Optional[str] = None,
    ) -> Tuple[uuid.UUID, float, bool, bool, Optional[uuid.UUID]]:
        """
        Decide identity with face contradiction gate and disassociation logic.
        
        Enhanced logic:
        1. Face Contradiction Check: If track has face, check against all DB faces.
        2. Face Matching Priority: Face similarity checked first (higher confidence).
        3. Body ReID Matching: With face contradiction gate (exclude contradicting candidates).
        4. Refinement Disassociation: If assigned ID face contradicts -> disassociate -> re-search.
        """
        try:
            best_candidate = None
            best_similarity = -1.0
            used_face = False
            
            # === FACE CONTRADICTION GATE ===
            # Check if current_person_id face embedding contradicts track face
            current_id_contradicted = False
            if current_person_id is not None and face_embedding is not None:
                current_face = await self._get_person_face_embedding(db, current_person_id)
                if current_face is not None:
                    sim = float(np.dot(current_face, face_embedding))
                    if sim < self.settings.FACE_MATCH_THRESHOLD:
                        current_id_contradicted = True
                        logger.warning(
                            f"[CONTRADICTION] Current ID {str(current_person_id)[:8]} face contradicts track face! "
                            f"Sim={sim:.3f} < {self.settings.FACE_MATCH_THRESHOLD}"
                        )
            
            # Step 1: Face matching (highest priority)
            if face_embedding is not None:
                face_candidate = await self._search_similar_face(db, face_embedding)
                if face_candidate:
                    face_sim = 1.0 - face_candidate["distance"]
                    # If face match is strong enough
                    if face_sim >= self.settings.FACE_MATCH_THRESHOLD:
                        best_candidate = face_candidate
                        best_similarity = face_sim
                        used_face = True
                        logger.info(f"[Face Match] Score: {face_sim:.3f}, Person: {str(face_candidate['person_identity_id'])[:8]}")

            # Step 2: Fallback to Body ReID matching (with face contradiction gate)
            skip_body_reid = False
            if face_embedding is not None and not used_face:
                if face_score >= self.settings.FACE_SEARCH_THRESHOLD:
                    skip_body_reid = True
                    logger.info(f"High-quality face (score: {face_score:.2f}) did not match in database. Skipping body ReID to avoid false merges.")

            if not used_face and not skip_body_reid and mean_embedding is not None:
                # Search for similar embeddings using pgvector cosine distance (limit 5 candidates)
                candidates = await self._search_similar(db, mean_embedding, top_k=5)
                for candidate in candidates:
                    candidate_id = candidate["person_identity_id"]
                    
                    # === FACE CONTRADICTION GATE FOR BODY MATCHING ===
                    # Skip candidates whose face contradicts the track's face
                    if face_embedding is not None:
                        candidate_face = await self._get_person_face_embedding(db, candidate_id)
                        if candidate_face is not None:
                            f_sim = float(np.dot(candidate_face, face_embedding))
                            if f_sim < self.settings.FACE_MATCH_THRESHOLD:
                                logger.debug(
                                    f"[Body Match] Candidate {str(candidate_id)[:8]} EXCLUDED due to face contradiction "
                                    f"(Face Sim={f_sim:.3f} < {self.settings.FACE_MATCH_THRESHOLD})"
                                )
                                continue  # Skip this candidate
                    
                    similarity = 1.0 - candidate["distance"]  # cosine distance -> similarity
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_candidate = candidate

            confidence_limit = self.settings.REID_CONFIDENCE_LIMIT  # 0.75
            required_threshold = self.settings.FACE_MATCH_THRESHOLD if used_face else self.settings.REID_MATCH_THRESHOLD

            # CASE 1: Initial resolution (no current_person_id assigned yet)
            if current_person_id is None:
                if best_candidate and best_similarity >= required_threshold:
                    person_id = best_candidate["person_identity_id"]
                    is_confident = (best_similarity >= confidence_limit)
                    
                    # Update person last seen / visit count
                    await self._update_person(db, person_id)
                    # Store this embedding
                    await self._store_embedding(
                        db, person_id, mean_embedding, camera_id, crop_quality_score, crop_path
                    )
                    if face_embedding is not None and face_score > 0:
                        await self._store_face_embedding(db, person_id, face_embedding, camera_id, face_score, face_crop_path)
                    
                    logger.info(f"[Initial ReID] Matched track to existing person ID {person_id} with score {best_similarity:.3f}")
                    return person_id, best_similarity, is_confident, False, None
                else:
                    # Create new anonymous person
                    person_id = await self._create_new_person(
                        db, mean_embedding, camera_id, crop_quality_score, crop_path, face_embedding, face_score, face_crop_path
                    )
                    logger.info(f"[Initial ReID] Created new anonymous person ID {person_id} (best score: {best_similarity:.3f} < {required_threshold})")
                    return person_id, 0.0, False, True, None

            # CASE 2: Refinement of an existing identity
            else:
                # If current ID contradicts track face, it MUST be disassociated
                if current_id_contradicted:
                    if best_candidate and best_similarity >= required_threshold:
                        matched_id = best_candidate["person_identity_id"]
                        
                        prune_old_id = current_person_id if is_temporary else None
                        if prune_old_id:
                            await self._delete_person(db, prune_old_id, matched_id)
                        
                        is_confident = (best_similarity >= confidence_limit)
                        await self._update_person(db, matched_id)
                        await self._store_embedding(
                            db, matched_id, mean_embedding, camera_id, crop_quality_score, crop_path
                        )
                        if face_embedding is not None and face_score > 0:
                            await self._store_face_embedding(db, matched_id, face_embedding, camera_id, face_score, face_crop_path)
                        logger.info(f"[ReID Refined] Identity switched due to contradiction: {current_person_id} -> {matched_id} (score: {best_similarity:.3f}, used_face={used_face})")
                        return matched_id, best_similarity, is_confident, False, prune_old_id
                    else:
                        # No other match found; create a new person ID entirely
                        new_pid = await self._create_new_person(
                            db, mean_embedding, camera_id, crop_quality_score, crop_path, face_embedding, face_score, face_crop_path
                        )
                        
                        prune_old_id = current_person_id if is_temporary else None
                        if prune_old_id:
                            await self._delete_person(db, prune_old_id, new_pid)
                        logger.info(f"[ReID Refined] Contradiction forced new ID: {current_person_id} -> {new_pid}")
                        return new_pid, 0.0, False, True, prune_old_id

                is_valid_match = False
                if best_candidate and best_similarity >= required_threshold:
                    # If we used face and it's a different ID, allow switch regardless of previous body score
                    if used_face and best_candidate["person_identity_id"] != current_person_id:
                        is_valid_match = True
                    # Otherwise, only switch/update if the new similarity is better than the previous one
                    elif best_similarity > previous_score:
                        is_valid_match = True
                    # Or if it's the same ID, just allow updating the profile
                    elif best_candidate["person_identity_id"] == current_person_id:
                        is_valid_match = True

                if is_valid_match:
                    matched_id = best_candidate["person_identity_id"]
                    
                    if matched_id != current_person_id:
                        # Gating on crop quality should only apply for body ReID, not face matches
                        if not used_face and crop_quality_score < self.settings.REID_MIN_QUALITY_FOR_SWITCH:
                            logger.debug(
                                f"[ReID Refined] Skipping identity switch for track (quality={crop_quality_score:.2f} < {self.settings.REID_MIN_QUALITY_FOR_SWITCH})"
                            )
                            # Still store the embedding to refine the current identity's signature
                            await self._store_embedding(
                                db, current_person_id, mean_embedding, camera_id, crop_quality_score, crop_path
                            )
                            if face_embedding is not None and face_score > 0:
                                await self._store_face_embedding(db, current_person_id, face_embedding, camera_id, face_score, face_crop_path)
                            return current_person_id, best_similarity, False, False, None

                        # We are switching identities!
                        prune_old_id = current_person_id if is_temporary else None
                        if prune_old_id:
                            await self._delete_person(db, prune_old_id, matched_id)
                            
                        is_confident = (best_similarity >= confidence_limit)
                        await self._update_person(db, matched_id)
                        await self._store_embedding(
                            db, matched_id, mean_embedding, camera_id, crop_quality_score, crop_path
                        )
                        if face_embedding is not None and face_score > 0:
                            await self._store_face_embedding(db, matched_id, face_embedding, camera_id, face_score, face_crop_path)
                        logger.info(f"[ReID Refined] Identity switched: {current_person_id} -> {matched_id} (score: {best_similarity:.3f}, quality={crop_quality_score:.2f}, used_face={used_face})")
                        return matched_id, best_similarity, is_confident, False, prune_old_id
                    else:
                        # Same ID, but score upgraded
                        is_confident = (best_similarity >= confidence_limit)
                        await self._store_embedding(
                            db, current_person_id, mean_embedding, camera_id, crop_quality_score, crop_path
                        )
                        if face_embedding is not None and face_score > 0:
                            await self._store_face_embedding(db, current_person_id, face_embedding, camera_id, face_score, face_crop_path)
                        new_score = max(previous_score, best_similarity)
                        if new_score > previous_score:
                            logger.info(f"[ReID Refined] Score upgraded for ID {current_person_id}: {previous_score:.3f} -> {new_score:.3f}")
                        return current_person_id, new_score, is_confident, False, None
                else:
                    # No better match found. If this is a temporary/unconfident ID, refine its signature
                    if is_temporary:
                        await self._store_embedding(
                            db, current_person_id, mean_embedding, camera_id, crop_quality_score, crop_path
                        )
                        logger.debug(f"[ReID Refined] Refined embedding signature for temporary ID {current_person_id}")
                    if face_embedding is not None and face_score > 0:
                        await self._store_face_embedding(db, current_person_id, face_embedding, camera_id, face_score, face_crop_path)
                    return current_person_id, previous_score, False, False, None

        except Exception as e:
            logger.error(f"Identity decision failed: {e}")
            if current_person_id is not None:
                return current_person_id, previous_score, False, False, None
            # Fallback
            person_id = await self._create_new_person(
                db, mean_embedding, camera_id, crop_quality_score, crop_path, face_embedding, face_score, face_crop_path
            )
            return person_id, 0.0, False, True, None

    async def _search_similar(
        self, db: AsyncSession, embedding: np.ndarray, top_k: int
    ) -> list:
        """Search for similar body embeddings using pgvector cosine distance."""
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

    async def _search_similar_face(self, db: AsyncSession, face_embedding: np.ndarray) -> Optional[dict]:
        """Search for similar face embeddings using pgvector cosine distance."""
        try:
            embedding_list = face_embedding.tolist()
            
            # Use smaller top_k for face matching since it's more definitive
            query = text("""
                SELECT pfe.person_identity_id, pfe.camera_id, pfe.face_score,
                       pfe.captured_at,
                       pi.last_seen_at,
                       pfe.embedding <=> :embedding AS distance
                FROM person_face_embeddings pfe
                JOIN person_identities pi ON pfe.person_identity_id = pi.id
                ORDER BY pfe.embedding <=> :embedding
                LIMIT 1
            """)

            result = await db.execute(query, {"embedding": str(embedding_list)})
            row = result.fetchone()
            
            if row:
                return {
                    "person_identity_id": row[0],
                    "camera_id": row[1],
                    "face_score": row[2],
                    "captured_at": row[3],
                    "last_seen_at": row[4],
                    "distance": float(row[5]),
                }
            return None
        except Exception as e:
            logger.error(f"Face similarity search failed: {e}")
            return None

    async def _get_person_face_embedding(self, db: AsyncSession, person_id: uuid.UUID) -> Optional[np.ndarray]:
        """
        Retrieve stored face embedding for a person (for contradiction checking).
        
        Args:
            db: Database session
            person_id: Person identity ID
            
        Returns:
            Face embedding as numpy array, or None if not found
        """
        try:
            from app.core.db.models.person import PersonFaceEmbedding
            
            result = await db.execute(
                select(PersonFaceEmbedding).where(PersonFaceEmbedding.person_identity_id == person_id)
            )
            face_emb = result.scalar_one_or_none()
            
            if face_emb and face_emb.embedding:
                return np.array(face_emb.embedding, dtype=np.float32)
            return None
        except Exception as e:
            logger.debug(f"Failed to retrieve face embedding for person {person_id}: {e}")
            return None

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
        embedding: Optional[np.ndarray],
        camera_id: uuid.UUID,
        crop_quality_score: float,
        crop_path: Optional[str],
    ):
        """Store a new embedding for an existing person (capped per identity)."""
        if embedding is None:
            return
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

    async def _store_face_embedding(
        self,
        db: AsyncSession,
        person_id: uuid.UUID,
        face_embedding: np.ndarray,
        camera_id: uuid.UUID,
        face_score: float,
        face_crop_path: Optional[str],
    ):
        """Store or update face embedding. Keeps only the highest score embedding."""
        try:
            from app.core.db.models.person import PersonFaceEmbedding
            
            # Check if existing face embedding exists and compare scores
            result = await db.execute(
                select(PersonFaceEmbedding).where(PersonFaceEmbedding.person_identity_id == person_id)
            )
            existing_face = result.scalar_one_or_none()

            if existing_face:
                if face_score > existing_face.face_score:
                    # Update to the better face crop
                    old_path = existing_face.face_crop_path
                    existing_face.embedding = face_embedding.tolist()
                    existing_face.face_score = face_score
                    existing_face.face_crop_path = face_crop_path
                    existing_face.camera_id = camera_id
                    existing_face.captured_at = utc_now()
                    
                    # Clean up old face crop from MinIO if it changed
                    if old_path and old_path != face_crop_path:
                        from app.modules.storage.minio_client import delete_object as minio_delete
                        try:
                            obj_name = old_path.split("/", 1)[1] if "/" in old_path else old_path
                            minio_delete(obj_name)
                        except Exception as e:
                            logger.warning(f"Failed to delete old face crop from MinIO: {e}")
            else:
                # Insert new face embedding
                face_emb = PersonFaceEmbedding(
                    person_identity_id=person_id,
                    embedding=face_embedding.tolist(),
                    camera_id=camera_id,
                    face_score=face_score,
                    face_crop_path=face_crop_path,
                    captured_at=utc_now(),
                )
                db.add(face_emb)
            
            await db.flush()
        except Exception as e:
            logger.error(f"Failed to store face embedding: {e}")

    async def _prune_embeddings(self, db: AsyncSession, person_id: uuid.UUID):
        """Keep only the best-quality K embeddings per identity (unbounded growth guard)."""
        try:
            # Query the rows that exceed the limit to retrieve their crop paths before deletion
            query = text("""
                SELECT id, crop_path FROM person_embeddings
                WHERE person_identity_id = :pid
                ORDER BY crop_quality DESC, captured_at DESC
                OFFSET :keep
            """)
            result = await db.execute(query, {"pid": str(person_id), "keep": self.MAX_EMBEDDINGS_PER_PERSON})
            rows = result.fetchall()

            if rows:
                ids_to_delete = [row[0] for row in rows]
                paths_to_delete = [row[1] for row in rows if row[1]]

                # Delete from database
                await db.execute(
                    text("DELETE FROM person_embeddings WHERE id = ANY(:ids)"),
                    {"ids": ids_to_delete}
                )

                # Delete files from MinIO
                from app.modules.storage.minio_client import delete_object as minio_delete
                for path in paths_to_delete:
                    try:
                        obj_name = path.split("/", 1)[1] if "/" in path else path
                        minio_delete(obj_name)
                        logger.debug(f"Pruned crop file deleted: {path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete pruned crop file {path}: {e}")
        except Exception as e:
            logger.warning(f"Embedding pruning failed for person {person_id}: {e}")

    async def _delete_person(self, db: AsyncSession, person_id: uuid.UUID, matched_id: uuid.UUID):
        """Delete a temporary person identity and merge all their references into matched_id."""
        try:
            # Update references in other tables pointing to person_id to point to matched_id
            await db.execute(
                text("UPDATE track_sessions SET person_identity_id = :matched_id WHERE person_identity_id = :pid"),
                {"pid": str(person_id), "matched_id": str(matched_id)}
            )
            await db.execute(
                text("UPDATE events SET person_identity_id = :matched_id WHERE person_identity_id = :pid"),
                {"pid": str(person_id), "matched_id": str(matched_id)}
            )
            await db.execute(
                text("UPDATE billing_interactions SET person_identity_id = :matched_id WHERE person_identity_id = :pid"),
                {"pid": str(person_id), "matched_id": str(matched_id)}
            )
            await db.execute(
                text("UPDATE storage_objects SET person_identity_id = :matched_id WHERE person_identity_id = :pid"),
                {"pid": str(person_id), "matched_id": str(matched_id)}
            )

            # Get all crop paths first
            query = text("SELECT crop_path FROM person_embeddings WHERE person_identity_id = :pid")
            res = await db.execute(query, {"pid": str(person_id)})
            paths = [row[0] for row in res.fetchall() if row[0]]

            # Also delete face crop from PersonIdentity
            query_face = text("SELECT face_crop_path FROM person_identities WHERE id = :pid")
            res_face = await db.execute(query_face, {"pid": str(person_id)})
            face_row = res_face.fetchone()
            if face_row and face_row[0]:
                paths.append(face_row[0])

            # Delete the identity (cascades to person_embeddings)
            await db.execute(
                text("DELETE FROM person_identities WHERE id = :pid"),
                {"pid": str(person_id)}
            )
            logger.info(f"Deleted temporary person identity from database: {person_id} (merged into {matched_id})")

            # Delete files from MinIO
            from app.modules.storage.minio_client import delete_object as minio_delete
            for path in set(paths):  # set to avoid duplicate removals
                try:
                    obj_name = path.split("/", 1)[1] if "/" in path else path
                    minio_delete(obj_name)
                    logger.debug(f"Temporary crop file deleted: {path}")
                except Exception as e:
                    logger.warning(f"Failed to delete temporary crop file {path}: {e}")
        except Exception as e:
            logger.warning(f"Failed to delete temporary person {person_id}: {e}")

    async def _create_new_person(
        self,
        db: AsyncSession,
        embedding: Optional[np.ndarray],
        camera_id: uuid.UUID,
        crop_quality_score: float,
        crop_path: Optional[str],
        face_embedding: Optional[np.ndarray] = None,
        face_score: float = 0.0,
        face_crop_path: Optional[str] = None,
    ) -> uuid.UUID:
        """Create a new anonymous person identity and fire registration event."""
        now = utc_now()
        person = PersonIdentity(
            label=None,
            first_seen_at=now,
            last_seen_at=now,
            visit_count=1,
            is_anonymous=True,
            best_face_score=face_score if face_score > 0 else None,
            face_crop_path=face_crop_path,
        )
        db.add(person)
        await db.flush()

        if embedding is not None:
            emb = PersonEmbedding(
                person_identity_id=person.id,
                embedding=embedding.tolist(),
                camera_id=camera_id,
                crop_quality=crop_quality_score,
                crop_path=crop_path,
                captured_at=now,
            )
            db.add(emb)
        
        if face_embedding is not None and face_score > 0:
            from app.core.db.models.person import PersonFaceEmbedding
            face_emb = PersonFaceEmbedding(
                person_identity_id=person.id,
                embedding=face_embedding.tolist(),
                camera_id=camera_id,
                face_score=face_score,
                face_crop_path=face_crop_path,
                captured_at=now,
            )
            db.add(face_emb)

        # Fire core lifecycle event: new_person_registered
        from app.core.db.models.event import Event, EventSeverity
        reg_event = Event(
            camera_id=camera_id,
            person_identity_id=person.id,
            event_type="new_person_registered",
            severity=EventSeverity.LOW,
            description=f"New person {person.id} registered.",
            occurred_at=now,
            metadata_json={"face_score": face_score, "crop_quality_score": crop_quality_score}
        )
        db.add(reg_event)
        
        await db.flush()

        return person.id
