"""Identity decision engine - matches embeddings to existing persons using pgvector."""

import uuid
from typing import Optional, Tuple
from datetime import datetime, timedelta

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

    @staticmethod
    def _face_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two face embeddings.

        InsightFace buffalo_l embeddings are NOT L2-normalized (norms 12-27),
        so raw np.dot() gives values in the hundreds, not [-1, +1].
        This helper normalizes both vectors before the dot product so the
        result is a proper cosine similarity in [-1, +1].
        """
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

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
        good_face_count: int = 0,
        face_embedding_list: Optional[list] = None,
    ) -> Tuple[uuid.UUID, float, bool, bool, Optional[uuid.UUID]]:
        """
        Decide identity with face contradiction gate and disassociation logic.
        
        Enhanced logic:
        1. Face Contradiction Check: If track has face, check against all DB faces.
        2. Face Matching Priority: Face similarity checked first (higher confidence).
           Searches with ALL accumulated faces (not just best) for multi-angle matching.
        3. Body ReID Matching: With face contradiction gate (exclude contradicting candidates).
        4. Refinement Disassociation: If assigned ID face contradicts -> disassociate -> re-search.
        5. Body-only matches are demoted in confidence (BODY_ONLY_CONFIDENCE_LIMIT).
        """
        try:
            # === RACE CONDITION PREVENTION ===
            # Acquire transaction-level exclusive lock to serialize ReID decisions across all camera workers
            await db.execute(text("SELECT pg_advisory_xact_lock(1001)"))

            best_candidate = None
            best_similarity = -1.0
            used_face = False
            
            # === FACE CONTRADICTION GATE ===
            # Check if current_person_id face embeddings contradict track face.
            # Uses FACE_CONTRADICTION_THRESHOLD (0.25) — much lower than the match
            # threshold so that same-person cross-angle similarity (0.40-0.47) does NOT
            # trigger disassociation.  Only truly different faces (< 0.25) disassociate.
            current_id_contradicted = False
            if current_person_id is not None and face_embedding is not None:
                current_faces = await self._get_person_face_embeddings(db, current_person_id)
                if current_faces:
                    best_face_sim = max(self._face_sim(f, face_embedding) for f in current_faces)
                    if best_face_sim < self.settings.FACE_CONTRADICTION_THRESHOLD:
                        current_id_contradicted = True
                        logger.warning(
                            f"[CONTRADICTION] Current ID {str(current_person_id)[:8]} face contradicts track face! "
                            f"BestSim={best_face_sim:.3f} < {self.settings.FACE_CONTRADICTION_THRESHOLD}"
                        )
            
            # Step 1: Face matching (highest priority)
            # Search with ALL accumulated faces (different angles), not just the best.
            # A face with lower detection score may still produce a higher match
            # similarity against stored embeddings from a different angle.
            if face_embedding is not None:
                # Build list of faces to search: prefer full list, fallback to best only
                search_faces = []
                if face_embedding_list:
                    for item in face_embedding_list:
                        if isinstance(item, tuple) and item[0] is not None:
                            search_faces.append(item[0])
                if not search_faces and face_embedding is not None:
                    search_faces.append(face_embedding)

                for search_emb in search_faces:
                    face_candidate = await self._search_similar_face(db, search_emb)
                    if face_candidate:
                        face_sim = 1.0 - face_candidate["distance"]
                        # Strict face match (unchanged, works for any age of candidate)
                        if face_sim >= self.settings.FACE_MATCH_THRESHOLD and face_sim > best_similarity:
                            best_candidate = face_candidate
                            best_similarity = face_sim
                            used_face = True
                            logger.info(f"[Face Match] Score: {face_sim:.3f}, Person: {str(face_candidate['person_identity_id'])[:8]}")
                        # Recent-window relaxed face match: same-visit cross-angle handoff
                        # whose best face pair falls in [0.35, 0.40) — would be missed by the
                        # strict threshold and create a duplicate. Only for candidates whose
                        # first_seen_at is within RECENT_WINDOW_MINUTES (same visit, small pool).
                        elif (
                            self.settings.ENABLE_RECENT_WINDOW_MATCHING
                            and face_sim >= self.settings.FACE_MATCH_THRESHOLD_RECENT
                            and face_sim > best_similarity
                            and self._is_recent(face_candidate.get("last_seen_at"))
                        ):
                            # Grey-zone face match [0.35, 0.40): validate with
                            # median of ALL cross-pairs. A single lucky crop can
                            # hit 0.35+ for different people; median catches this.
                            # Same-person min median=0.401, diff-person p50=0.200.
                            # Only check when >= 3 total cross-pairs (meaningful median).
                            cand_id = face_candidate["person_identity_id"]
                            track_face_list = [f for f in search_faces if f is not None]
                            n_track = len(track_face_list)
                            r = await db.execute(text(
                                "SELECT COUNT(*) FROM person_face_embeddings"
                                " WHERE person_identity_id = :pid AND embedding IS NOT NULL"
                            ), {"pid": str(cand_id)})
                            n_cand = int(r.scalar() or 0)
                            n_cross = n_track * n_cand

                            if n_cross >= 3:
                                face_median = await self._person_face_median_sim(db, cand_id, track_face_list)
                                if face_median is not None and face_median < self.settings.FACE_MATCH_MEDIAN_THRESHOLD:
                                    logger.info(
                                        f"[Face Match RECENT REJECTED] Score: {face_sim:.3f} but median={face_median:.3f} "
                                        f"< {self.settings.FACE_MATCH_MEDIAN_THRESHOLD} (cross-pairs={n_cross}), "
                                        f"Person: {str(cand_id)[:8]} — single lucky pair, different person"
                                    )
                                    continue

                            best_candidate = face_candidate
                            best_similarity = face_sim
                            used_face = True
                            logger.info(
                                f"[Face Match RECENT] Score: {face_sim:.3f} (≥{self.settings.FACE_MATCH_THRESHOLD_RECENT}, "
                                f"relaxed from {self.settings.FACE_MATCH_THRESHOLD}), Person: "
                                f"{str(face_candidate['person_identity_id'])[:8]}"
                            )

            # Step 2: Fallback to Body ReID matching (with face contradiction gate).
            # skip_body_reid logic intentionally removed — high-quality face that misses
            # in face search (cross-angle, same person) should still fall through to body
            # ReID rather than creating a duplicate identity.
            if not used_face and mean_embedding is not None:
                # Search for similar embeddings using pgvector cosine distance.
                # Candidates are now per-unique-person (deduplicated in _search_similar).
                candidates = await self._search_similar(db, mean_embedding, top_k=5)

                # ── Body consensus gate ──────────────────────────────────
                # A single body match can be a false positive (OSNet same-person
                # median=0.680, diff-person p90=0.534 — overlap at 0.50–0.55).
                # Require at least 2 of the top-3 unique-identity candidates to
                # agree AND exceed REID_MATCH_THRESHOLD before accepting.
                body_votes: dict = {}
                body_best_by_id: dict = {}
                for candidate in candidates:
                    candidate_id = candidate["person_identity_id"]
                    is_recent_candidate = self._is_recent(candidate.get("last_seen_at"))

                    if face_embedding is not None:
                        candidate_faces = await self._get_person_face_embeddings(db, candidate_id)
                        if candidate_faces:
                            best_f_sim = max(self._face_sim(f, face_embedding) for f in candidate_faces)
                            # Face-contradiction gate. Use the stricter 0.30 (FACE_BODY_EXCLUSION_THRESHOLD)
                            # for older candidates; relax to 0.25 (FACE_CONTRADICTION_THRESHOLD) for
                            # recent candidates so same-visit cross-angle faces in [0.25, 0.30) don't
                            # block the body-only recent-window path.
                            exclusion_bar = (
                                self.settings.FACE_CONTRADICTION_THRESHOLD
                                if is_recent_candidate
                                else self.settings.FACE_BODY_EXCLUSION_THRESHOLD
                            )
                            if best_f_sim < exclusion_bar:
                                continue  # face contradicts → skip

                    similarity = 1.0 - candidate["distance"]
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_candidate = candidate

                    # Collect consensus votes from top-3 unique candidates
                    if similarity >= self.settings.REID_MATCH_THRESHOLD:
                        body_votes[candidate_id] = body_votes.get(candidate_id, 0) + 1
                        if candidate_id not in body_best_by_id or similarity > body_best_by_id[candidate_id]["similarity"]:
                            body_best_by_id[candidate_id] = candidate

                    # Stop after evaluating 3 unique candidates
                    if len(body_votes) >= 3 or (len(body_votes) >= 2 and max(body_votes.values()) >= 2):
                        break

                # Apply consensus: only accept if 2+ of top-3 agree
                consensus_id = None
                if body_votes:
                    max_votes = max(body_votes.values())
                    if max_votes >= 2:
                        # Find the ID with the most votes (and best similarity as tiebreaker)
                        consensus_id = max(
                            [k for k, v in body_votes.items() if v == max_votes],
                            key=lambda k: body_best_by_id[k]["similarity"]
                        )
                        best_candidate = body_best_by_id[consensus_id]
                        best_similarity = body_best_by_id[consensus_id]["similarity"]
                        logger.info(
                            f"[Body Consensus] {max_votes}/{min(3, len(candidates))} candidates "
                            f"agree on ID {str(consensus_id)[:8]} (sim={best_similarity:.3f})"
                        )
                    else:
                        best_candidate = None
                        best_similarity = -1.0
                        logger.debug(
                            f"[Body Consensus] No consensus: top votes={body_votes}"
                        )

                # ── Recent-window single-candidate body override ────────────
                # When the store is quiet (1-2 people in frame), the 2-of-3
                # consensus is structurally impossible — only 1 candidate exists.
                # Within a same-visit window the candidate pool is small AND body
                # ReID is reliable (same clothing). Accept a single body candidate
                # whose median body sim ≥ 0.60, candidate is recent, has ≥2 body
                # embeddings, and whose face does NOT contradict the track's face
                # (face exclusion already applied above, so candidates reaching
                # here are face-compatible or faceless). This catches the
                # cross-camera handoff / brief re-appearance cases that otherwise
                # create duplicates. Outside the window: strict consensus only.
                if (
                    self.settings.ENABLE_RECENT_WINDOW_MATCHING
                    and consensus_id is None
                    and best_candidate is not None
                    and self._is_recent(best_candidate.get("last_seen_at"))
                ):
                    cand_id = best_candidate["person_identity_id"]
                    body_median = await self._person_body_median_sim(db, cand_id, mean_embedding)
                    n_bodies = await self._person_body_count(db, cand_id)
                    if (
                        body_median is not None
                        and body_median >= self.settings.RECENT_BODY_SINGLE_MATCH_THRESHOLD
                        and n_bodies >= 2
                    ):
                        best_similarity = body_median  # report the median we actually gated on
                        logger.info(
                            f"[Body RECENT single] ID {str(cand_id)[:8]} body_median={body_median:.3f} "
                            f"≥{self.settings.RECENT_BODY_SINGLE_MATCH_THRESHOLD} (n_bodies={n_bodies}, "
                            f"window={self.settings.RECENT_WINDOW_MINUTES}min) — accepting without consensus"
                        )
                        # used_face stays False — body-only match, demoted to non-confident downstream
                    else:
                        # Recent candidate but body evidence insufficient → reject
                        best_candidate = None
                        best_similarity = -1.0
                        logger.debug(
                            f"[Body RECENT] candidate {str(cand_id)[:8]} body_median={body_median} "
                            f"n_bodies={n_bodies} — below recent body bar, rejected"
                        )

            # Step 2b: Staff reattach — last resort before create-new.
            # Staff-only, recent window, strong body median. Soft face floor only
            # (hard different-face veto). Blurry/side faces that pass the floor still
            # go through store contamination gate (may drop face without rejecting attach).
            staff_drop_face = False
            if (
                self.settings.ENABLE_STAFF_REATTACH
                and not used_face
                and mean_embedding is not None
            ):
                provisional_threshold = (
                    self.settings.FACE_MATCH_THRESHOLD if used_face else self.settings.REID_MATCH_THRESHOLD
                )
                needs_staff = (
                    best_candidate is None
                    or best_similarity < provisional_threshold
                )
                if needs_staff:
                    staff_hit = await self._try_staff_reattach(
                        db, mean_embedding, face_embedding, face_score
                    )
                    if staff_hit is not None:
                        best_candidate = staff_hit
                        best_similarity = staff_hit["similarity"]
                        staff_drop_face = bool(staff_hit.get("drop_face", False))

            confidence_limit = self.settings.REID_CONFIDENCE_LIMIT  # 0.75
            required_threshold = self.settings.FACE_MATCH_THRESHOLD if used_face else self.settings.REID_MATCH_THRESHOLD

            # CASE 1: Initial resolution (no current_person_id assigned yet)
            if current_person_id is None:
                if best_candidate and best_similarity >= required_threshold:
                    person_id = best_candidate["person_identity_id"]
                    raw_confident = (best_similarity >= confidence_limit)
                    # Demote body-only matches: only mark confident if face matched
                    is_confident = raw_confident and used_face
                    
                    # Update person last seen / visit count
                    await self._update_person(db, person_id)
                    # Store this embedding
                    await self._store_embedding(
                        db, person_id, mean_embedding, camera_id, crop_quality_score, crop_path
                    )
                    if face_embedding is not None and face_score > 0 and not staff_drop_face:
                        await self._store_face_embedding(db, person_id, face_embedding, camera_id, face_score, face_crop_path)
                    elif staff_drop_face and face_embedding is not None:
                        logger.info(
                            f"[Staff REATTACH] face DROPPED (low sim vs staff gallery) "
                            f"person={str(person_id)[:8]}"
                        )

                    logger.info(
                        f"[Initial ReID] Matched track to existing person ID {person_id} "
                        f"with score {best_similarity:.3f} (used_face={used_face}, "
                        f"staff_reattach_drop_face={staff_drop_face})"
                    )
                    return person_id, best_similarity, is_confident, False, None
                else:
                    # Create new anonymous person
                    person_id = await self._create_new_person(
                        db, mean_embedding, camera_id, crop_quality_score, crop_path,
                        face_embedding, face_score, face_crop_path, good_face_count
                    )
                    if person_id is None:
                        logger.info("[Initial ReID] Identity creation blocked (face quality gate).")
                        return None, 0.0, False, False, None
                        
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
                        if face_embedding is not None and face_score > 0 and not staff_drop_face:
                            await self._store_face_embedding(db, matched_id, face_embedding, camera_id, face_score, face_crop_path)
                        logger.info(f"[ReID Refined] Identity switched due to contradiction: {current_person_id} -> {matched_id} (score: {best_similarity:.3f}, used_face={used_face})")
                        return matched_id, best_similarity, is_confident, False, prune_old_id
                    else:
                        # No other match found; create a new person ID entirely
                        new_pid = await self._create_new_person(
                            db, mean_embedding, camera_id, crop_quality_score, crop_path,
                            face_embedding, face_score, face_crop_path, good_face_count
                        )
                        if new_pid is None:
                            logger.info("[ReID Refined] Contradiction occurred but couldn't create new ID (face quality gate). Returning None.")
                            prune_old_id = current_person_id if is_temporary else None
                            return None, 0.0, False, False, prune_old_id
                            
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
                        # Gating on crop quality should only apply for body ReID, not face matches.
                        # Staff reattach already required median body gate — allow switch.
                        if (
                            not used_face
                            and not best_candidate.get("staff_reattach")
                            and crop_quality_score < self.settings.REID_MIN_QUALITY_FOR_SWITCH
                        ):
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
                        elif not is_temporary:
                            # Even for confident (non-temporary) identities, if the old identity
                            # has no other active track sessions it is effectively orphaned.
                            # Mark it for cleanup by the periodic dedup job — don't hard-delete
                            # here because another camera may still reference it.
                            logger.info(
                                f"[ReID Refined] Non-temporary ID {str(current_person_id)[:8]} superseded by "
                                f"{str(matched_id)[:8]} (score={best_similarity:.3f}). "
                                f"Old identity left for dedup job to clean up."
                            )
                            
                        raw_confident = (best_similarity >= confidence_limit)
                        is_confident = raw_confident and used_face
                        await self._update_person(db, matched_id)
                        await self._store_embedding(
                            db, matched_id, mean_embedding, camera_id, crop_quality_score, crop_path
                        )
                        if face_embedding is not None and face_score > 0 and not staff_drop_face:
                            await self._store_face_embedding(db, matched_id, face_embedding, camera_id, face_score, face_crop_path)
                        logger.info(f"[ReID Refined] Identity switched: {current_person_id} -> {matched_id} (score: {best_similarity:.3f}, quality={crop_quality_score:.2f}, used_face={used_face})")
                        return matched_id, best_similarity, is_confident, False, prune_old_id
                    else:
                        # Same ID, but score upgraded
                        raw_confident = (best_similarity >= confidence_limit)
                        is_confident = raw_confident and used_face
                        await self._store_embedding(
                            db, current_person_id, mean_embedding, camera_id, crop_quality_score, crop_path
                        )
                        if face_embedding is not None and face_score > 0 and not staff_drop_face:
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
                db, mean_embedding, camera_id, crop_quality_score, crop_path,
                face_embedding, face_score, face_crop_path, good_face_count
            )
            if person_id is None:
                return None, 0.0, False, False, None
            return person_id, 0.0, False, True, None

    async def _search_similar(
        self, db: AsyncSession, embedding: np.ndarray, top_k: int
    ) -> list:
        """Search for similar body embeddings using pgvector cosine distance.

        Deduplicates by person_identity_id — returns the best match per unique
        person, rather than raw embeddings.  This prevents a single person with
        10 stored body embeddings from dominating the top-K results.
        """
        try:
            embedding_list = embedding.tolist()

            # Better recall on the IVFFlat index for this transaction
            await db.execute(text("SET LOCAL ivfflat.probes = 10"))

            # Fetch more candidates than needed for dedup by person
            fetch_limit = max(top_k * 5, 25)

            query = text("""
                SELECT pe.person_identity_id, pe.camera_id, pe.crop_quality,
                       pe.captured_at,
                       pi.last_seen_at, pi.first_seen_at,
                       pe.embedding <=> :embedding AS distance
                FROM person_embeddings pe
                JOIN person_identities pi ON pe.person_identity_id = pi.id
                WHERE pe.captured_at > NOW() - INTERVAL '48 hours'
                ORDER BY pe.embedding <=> :embedding
                LIMIT :fetch_limit
            """)

            result = await db.execute(
                query, {"embedding": str(embedding_list), "fetch_limit": fetch_limit}
            )
            rows = result.fetchall()

            # Deduplicate: keep only the best match per person_identity_id
            best_by_person: dict = {}
            for row in rows:
                pid = row[0]
                dist = float(row[6])
                sim = 1.0 - dist
                if pid not in best_by_person or sim > best_by_person[pid]["similarity"]:
                    best_by_person[pid] = {
                        "person_identity_id": pid,
                        "camera_id": row[1],
                        "crop_quality": row[2],
                        "captured_at": row[3],
                        "last_seen_at": row[4],
                        "first_seen_at": row[5],
                        "distance": dist,
                        "similarity": sim,
                    }

            candidates = sorted(
                best_by_person.values(), key=lambda c: c["distance"]
            )[:top_k]

            return candidates

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    async def _search_similar_face(self, db: AsyncSession, face_embedding: np.ndarray) -> Optional[dict]:
        """Search for similar face embeddings using pgvector cosine distance.

        Searches ALL stored face embeddings per person and returns the best
        per-person similarity via multi-face matching (aggregated by person_id).

        Uses ivfflat.probes=50 for maximum recall (checks all index buckets).
        Without this, default probes=1 only scans 1/50 buckets, missing real
        matches that fall in other buckets.
        """
        try:
            embedding_list = face_embedding.tolist()

            # Increase IVFFlat probes for maximum recall on face search.
            # Default probes=1 checks only 1 of 50 buckets → misses 98% of vectors.
            await db.execute(text("SET LOCAL ivfflat.probes = 50"))

            query = text("""
                SELECT pfe.person_identity_id, pfe.camera_id, pfe.face_score,
                       pfe.captured_at,
                       pi.last_seen_at, pi.first_seen_at,
                       pfe.embedding <=> :embedding AS distance
                FROM person_face_embeddings pfe
                JOIN person_identities pi ON pfe.person_identity_id = pi.id
                ORDER BY pfe.embedding <=> :embedding
                LIMIT :top_k
            """)

            result = await db.execute(query, {
                "embedding": str(embedding_list),
                "top_k": 100,
            })
            rows = result.fetchall()

            if not rows:
                return None

            best_by_person = {}
            for row in rows:
                pid = row[0]
                dist = float(row[6])
                sim = 1.0 - dist
                if pid not in best_by_person or sim > best_by_person[pid]["similarity"]:
                    best_by_person[pid] = {
                        "person_identity_id": pid,
                        "camera_id": row[1],
                        "face_score": row[2],
                        "captured_at": row[3],
                        "last_seen_at": row[4],
                        "first_seen_at": row[5],
                        "distance": dist,
                        "similarity": sim,
                    }

            return max(best_by_person.values(), key=lambda c: c["similarity"])
        except Exception as e:
            logger.error(f"Face similarity search failed: {e}")
            return None

    async def _get_person_face_embedding(self, db: AsyncSession, person_id: uuid.UUID) -> Optional[np.ndarray]:
        """
        Retrieve the best-matching stored face embedding for a person.

        With multi-face support (MAX_FACE_EMBEDDINGS_PER_PERSON), multiple face
        embeddings may exist per person. All are checked for contradiction.
        Returns a list instead of a single embedding.
        """
        return await self._get_person_face_embeddings(db, person_id)

    async def _get_person_face_embeddings(self, db: AsyncSession, person_id: uuid.UUID) -> list[np.ndarray]:
        try:
            from app.core.db.models.person import PersonFaceEmbedding

            result = await db.execute(
                select(PersonFaceEmbedding)
                .where(PersonFaceEmbedding.person_identity_id == person_id)
            )
            rows = result.scalars().all()
            return [np.array(row.embedding, dtype=np.float32) for row in rows if row.embedding is not None]
        except Exception as e:
            logger.debug(f"Failed to retrieve face embeddings for person {person_id}: {e}")
            return []

    # ── Recent-window matching helpers (plan 2026-07-09) ────────────────────
    def _is_recent(self, last_seen_at) -> bool:
        """True if a candidate's last_seen_at is within RECENT_WINDOW_MINUTES.

        Uses last_seen_at (not first_seen_at) because the question is 'is this
        person currently in the store?' — a staff member who arrived 6 hours
        ago but was tracked 30 seconds ago IS recent.  first_seen_at would
        incorrectly mark them as non-recent.
        """
        if last_seen_at is None:
            return False
        try:
            if isinstance(last_seen_at, str):
                fs = datetime.fromisoformat(last_seen_at.replace("Z", "+00:00"))
            else:
                fs = last_seen_at
            if fs.tzinfo is None:
                fs = fs.replace(tzinfo=utc_now().tzinfo)
            return (utc_now() - fs) <= timedelta(minutes=self.settings.RECENT_WINDOW_MINUTES)
        except Exception:
            return False

    async def _try_staff_reattach(
        self,
        db: AsyncSession,
        body_embedding: np.ndarray,
        face_embedding: Optional[np.ndarray],
        face_score: float = 0.0,
    ) -> Optional[dict]:
        """Reattach track to a recent is_staff identity via strong body match.

        Used when normal face/body paths fail (blur/side face staff fragments).
        Returns a candidate dict with person_identity_id, similarity (body median),
        drop_face flag, or None.

        Gates (tightened after uniform body FPs, 2026-07-10):
          - body_median >= STAFF_REATTACH_BODY_MEDIAN (0.70)
          - staff bodies >= STAFF_REATTACH_MIN_BODIES
          - STAFF_REATTACH_REQUIRE_FACE: faceless track rejected
          - face_sim < STAFF_REATTACH_FACE_MIN (0.30) → reject
          - face_sim >= FACE_CONTAMINATION_THRESHOLD → may store face later
          - otherwise → reattach but drop_face=True (avoid gallery pollution)

        FUTURE face-quality-aware veto (disabled until quality calc is calibrated):
          # high_q = face_score >= settings.STAFF_REATTACH_FACE_QUALITY_HIGH  # e.g. 0.75
          # if face_sim < STAFF_REATTACH_FACE_MIN:
          #     if high_q:
          #         return None  # sharp face that is clearly not this staff
          #     else:
          #         drop_face = True  # low-quality face: body merge, discard face
        """
        try:
            if self.settings.STAFF_REATTACH_REQUIRE_FACE and face_embedding is None:
                logger.debug("[Staff REATTACH REJECT] faceless track (REQUIRE_FACE=True)")
                return None

            candidates = await self._search_similar_staff(db, body_embedding, top_k=5)
            if not candidates:
                return None

            scored: list[tuple] = []  # (body_median, cand)
            for cand in candidates:
                if not self._is_recent(cand.get("last_seen_at")):
                    continue
                pid = cand["person_identity_id"]
                n_bodies = await self._person_body_count(db, pid)
                if n_bodies < self.settings.STAFF_REATTACH_MIN_BODIES:
                    continue
                body_median = await self._person_body_median_sim(db, pid, body_embedding)
                if body_median is None or body_median < self.settings.STAFF_REATTACH_BODY_MEDIAN:
                    continue
                scored.append((body_median, cand))

            if not scored:
                return None

            scored.sort(key=lambda x: x[0], reverse=True)
            top_median, top_cand = scored[0]
            if len(scored) >= 2:
                second_median = scored[1][0]
                if (top_median - second_median) < self.settings.STAFF_REATTACH_AMBIGUITY:
                    logger.info(
                        f"[Staff REATTACH REJECT] ambiguous staff bodies "
                        f"top={top_median:.3f} second={second_median:.3f} gap<{self.settings.STAFF_REATTACH_AMBIGUITY}"
                    )
                    return None

            drop_face = False
            face_sim = None
            staff_faces = await self._get_person_face_embeddings(
                db, top_cand["person_identity_id"]
            )
            if self.settings.STAFF_REATTACH_REQUIRE_FACE and not staff_faces:
                logger.info(
                    f"[Staff REATTACH REJECT] staff has no faces "
                    f"person={str(top_cand['person_identity_id'])[:8]}"
                )
                return None
            if face_embedding is not None and staff_faces:
                face_sim = max(self._face_sim(face_embedding, f) for f in staff_faces)
                # See method docstring for FUTURE quality-aware branch.
                if face_sim < self.settings.STAFF_REATTACH_FACE_MIN:
                    logger.info(
                        f"[Staff REATTACH REJECT] face_sim={face_sim:.3f} < "
                        f"{self.settings.STAFF_REATTACH_FACE_MIN} "
                        f"(face_score={face_score:.2f}) "
                        f"person={str(top_cand['person_identity_id'])[:8]}"
                    )
                    return None
                if face_sim < self.settings.FACE_CONTAMINATION_THRESHOLD:
                    drop_face = True
            elif face_embedding is None and self.settings.STAFF_REATTACH_REQUIRE_FACE:
                return None

            top_cand = dict(top_cand)
            top_cand["similarity"] = top_median
            top_cand["distance"] = 1.0 - top_median
            top_cand["drop_face"] = drop_face
            top_cand["staff_reattach"] = True
            logger.info(
                f"[Staff REATTACH] person={str(top_cand['person_identity_id'])[:8]} "
                f"body_median={top_median:.3f} face_sim={face_sim if face_sim is not None else 'n/a'} "
                f"drop_face={drop_face} face_score={face_score:.2f}"
            )
            return top_cand
        except Exception as e:
            logger.error(f"Staff reattach failed: {e}")
            return None

    async def _search_similar_staff(
        self, db: AsyncSession, embedding: np.ndarray, top_k: int = 5
    ) -> list:
        """Body pgvector search restricted to is_staff=True identities."""
        try:
            embedding_list = embedding.tolist()
            await db.execute(text("SET LOCAL ivfflat.probes = 10"))
            fetch_limit = max(top_k * 5, 25)
            query = text("""
                SELECT pe.person_identity_id, pe.camera_id, pe.crop_quality,
                       pe.captured_at,
                       pi.last_seen_at, pi.first_seen_at,
                       pe.embedding <=> :embedding AS distance
                FROM person_embeddings pe
                JOIN person_identities pi ON pe.person_identity_id = pi.id
                WHERE pi.is_staff = TRUE
                  AND pe.captured_at > NOW() - INTERVAL '48 hours'
                ORDER BY pe.embedding <=> :embedding
                LIMIT :fetch_limit
            """)
            result = await db.execute(
                query, {"embedding": str(embedding_list), "fetch_limit": fetch_limit}
            )
            rows = result.fetchall()
            best_by_person: dict = {}
            for row in rows:
                pid = row[0]
                dist = float(row[6])
                sim = 1.0 - dist
                if pid not in best_by_person or sim > best_by_person[pid]["similarity"]:
                    best_by_person[pid] = {
                        "person_identity_id": pid,
                        "camera_id": row[1],
                        "crop_quality": row[2],
                        "captured_at": row[3],
                        "last_seen_at": row[4],
                        "first_seen_at": row[5],
                        "distance": dist,
                        "similarity": sim,
                    }
            return sorted(best_by_person.values(), key=lambda c: c["distance"])[:top_k]
        except Exception as e:
            logger.error(f"Staff body search failed: {e}")
            return []

    async def _person_body_count(self, db: AsyncSession, person_id) -> int:
        try:
            r = await db.execute(text(
                "SELECT COUNT(*) FROM person_embeddings"
                " WHERE person_identity_id = :pid AND embedding IS NOT NULL"
            ), {"pid": str(person_id)})
            return int(r.scalar() or 0)
        except Exception:
            return 0

    async def _person_body_median_sim(self, db: AsyncSession, person_id, query_embedding: np.ndarray) -> Optional[float]:
        """Median cosine similarity of `query_embedding` to ALL of a person's
        stored body embeddings (consistency check — a single lucky crop is not
        enough). OSNet embeddings are L2-normalized at extract; query is
        normalized here defensively."""
        try:
            q = np.asarray(query_embedding, dtype=np.float32)
            nq = np.linalg.norm(q)
            if nq > 0:
                q = q / nq
            r = await db.execute(text(
                "SELECT embedding FROM person_embeddings"
                " WHERE person_identity_id = :pid AND embedding IS NOT NULL"
            ), {"pid": str(person_id)})
            sims = []
            for row in r.fetchall():
                raw = row[0]
                if isinstance(raw, str):
                    e = np.array(eval(raw), dtype=np.float32)
                else:
                    e = np.array(raw, dtype=np.float32)
                ne = np.linalg.norm(e)
                if ne > 0:
                    e = e / ne
                sims.append(float(np.dot(q, e)))
            if not sims:
                return None
            return float(np.median(sims))
        except Exception as e:
            logger.debug(f"body median sim failed for {person_id}: {e}")
            return None

    async def _person_face_median_sim(self, db: AsyncSession, person_id, track_faces: list) -> Optional[float]:
        """Median cosine similarity of ALL track faces vs ALL stored candidate
        faces. Used to validate grey-zone recent face matches (best-pair in
        [0.35, 0.40)) — a single lucky cross-pair can hit 0.35+ while the rest
        are low (different person). The median catches this: same-person
        medians start at 0.40, diff-person p50 is 0.20.

        track_faces: list of face embeddings (np.ndarray) accumulated for the
        current track (face_embedding_list from decide_identity).
        Returns median of all cross-pair sims, or None if insufficient data.
        """
        try:
            r = await db.execute(text(
                "SELECT embedding FROM person_face_embeddings"
                " WHERE person_identity_id = :pid AND embedding IS NOT NULL"
            ), {"pid": str(person_id)})
            candidate_faces = []
            for row in r.fetchall():
                raw = row[0]
                if isinstance(raw, str):
                    e = np.array(eval(raw), dtype=np.float32)
                else:
                    e = np.array(raw, dtype=np.float32)
                candidate_faces.append(e)

            if not candidate_faces or not track_faces:
                return None

            sims = []
            for tf in track_faces:
                for cf in candidate_faces:
                    sims.append(self._face_sim(tf, cf))
            if not sims:
                return None
            return float(np.median(sims))
        except Exception as e:
            logger.debug(f"face median sim failed for {person_id}: {e}")
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
        """Store a new embedding for an existing person (capped per identity).

        Body contamination gate: if this person already has >=3 stored body
        embeddings, checks the median cosine similarity of the new embedding
        to existing ones. If median < BODY_CONTAMINATION_THRESHOLD (0.50),
        reject it — it belongs to a different person whose body ReID falsely
        matched. Uses median instead of min to avoid single-edge false positives
        (OSNet chains different-person clusters via weak edges around 0.66-0.75).
        """
        if embedding is None:
            return

        # ── Body contamination gate ───────────────────────────────────────
        import numpy as np
        new_emb = np.array(embedding.tolist(), dtype=np.float32)
        existing_body = await db.execute(text(
            "SELECT embedding FROM person_embeddings"
            " WHERE person_identity_id = :pid AND embedding IS NOT NULL"
            " ORDER BY crop_quality DESC LIMIT 10"
        ), {"pid": str(person_id)})
        existing_rows = existing_body.fetchall()
        if len(existing_rows) >= 3:
            sims = []
            for row in existing_rows:
                raw = row[0]
                if isinstance(raw, str):
                    emb = np.array(eval(raw), dtype=np.float32)
                else:
                    emb = np.array(raw, dtype=np.float32)
                sim = float(np.dot(emb, new_emb))
                sims.append(sim)
            median_sim = float(np.median(sims))
            if median_sim < self.settings.BODY_CONTAMINATION_THRESHOLD:
                logger.warning(
                    f"Body embedding CONTAMINATION rejected for person {person_id}: "
                    f"median_sim_to_existing={median_sim:.3f} < "
                    f"{self.settings.BODY_CONTAMINATION_THRESHOLD} "
                    f"samples={[f'{s:.3f}' for s in sorted(sims)[:5]]} "
                    f"(different person's body — OSNet false positive)"
                )
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
        """Store a face embedding. Keeps up to MAX_FACE_EMBEDDINGS_PER_PERSON per identity.

        Multiple face embeddings per person capture different angles and lighting
        conditions. Low-quality embeddings are pruned when the cap is exceeded.

        Duplication guard: if this face_crop_path is already stored for this person,
        the existing row's score and embedding are updated (upgraded if the new score
        is higher) instead of creating a redundant row.  This prevents the per-window
        accumulation pipeline from inserting the same crop 5+ times.
        """
        try:
            from app.core.db.models.person import PersonFaceEmbedding

            # ── Dedup by crop path ─────────────────────────────────────────
            if face_crop_path:
                existing_result = await db.execute(text(
                    "SELECT id, face_score FROM person_face_embeddings"
                    " WHERE person_identity_id = :pid AND face_crop_path = :path"
                    " ORDER BY face_score DESC LIMIT 1"
                ), {"pid": str(person_id), "path": face_crop_path})
                existing_row = existing_result.fetchone()
                if existing_row:
                    existing_score = float(existing_row[1]) if existing_row[1] else 0.0
                    if face_score > existing_score:
                        # Upgrade the existing row's score + embedding
                        await db.execute(text(
                            "UPDATE person_face_embeddings"
                            " SET face_score = :score, embedding = :emb, captured_at = NOW()"
                            " WHERE id = :row_id"
                        ), {
                            "score": face_score,
                            "emb": face_embedding.tolist(),
                            "row_id": existing_row[0],
                        })
                        logger.debug(
                            f"Upgraded existing face embedding {existing_row[0]} for person "
                            f"{person_id}: score {existing_score:.3f} → {face_score:.3f}"
                        )
                    else:
                        logger.debug(
                            f"Skipped duplicate face embedding for person {person_id} "
                            f"(path already stored, score {existing_score:.3f} ≥ {face_score:.3f})"
                        )
                    return

            # ── Contamination gate ──────────────────────────────────────────
            # If existing face embeddings for this person form a cluster, the
            # new face must belong to that cluster.  A cosine similarity below
            # FACE_CONTAMINATION_THRESHOLD (0.35) to the cluster means it's from
            # a DIFFERENT person whose body crop overlapped ours.
            # Uses _face_sim() which normalizes both vectors first — InsightFace
            # embeddings are NOT L2-normalized (norms 12-27).
            existing_result = await db.execute(text(
                "SELECT embedding FROM person_face_embeddings"
                " WHERE person_identity_id = :pid AND embedding IS NOT NULL"
                " ORDER BY face_score DESC LIMIT 5"
            ), {"pid": str(person_id)})
            existing_faces = existing_result.fetchall()
            if existing_faces:
                import numpy as np
                new_emb = np.array(face_embedding.tolist(), dtype=np.float32)
                min_sim = float('inf')
                for row in existing_faces:
                    raw = row[0]
                    if isinstance(raw, str):
                        emb = np.array(eval(raw), dtype=np.float32)
                    else:
                        emb = np.array(raw, dtype=np.float32)
                    sim = self._face_sim(emb, new_emb)
                    if sim < min_sim:
                        min_sim = sim
                if min_sim < self.settings.FACE_CONTAMINATION_THRESHOLD:
                    logger.warning(
                        f"Face embedding CONTAMINATION rejected for person {person_id}: "
                        f"min_sim_to_existing={min_sim:.3f} < "
                        f"{self.settings.FACE_CONTAMINATION_THRESHOLD} (different person's face)"
                    )
                    return  # skip — this face doesn't belong to this person

            # Normalize the face embedding before storing so future comparisons
            # can use raw np.dot() as cosine similarity.  Existing DB rows that
            # were stored before this fix are handled by the migration script
            # (danger/normalize_face_embeddings.py) and by _face_sim() which
            # normalizes at comparison time as a safety net.
            _face_emb_to_store = face_embedding
            _norm = np.linalg.norm(face_embedding)
            if _norm > 0:
                _face_emb_to_store = face_embedding / _norm

            face_emb = PersonFaceEmbedding(
                person_identity_id=person_id,
                embedding=_face_emb_to_store.tolist(),
                camera_id=camera_id,
                face_score=face_score,
                face_crop_path=face_crop_path,
                captured_at=utc_now(),
            )
            db.add(face_emb)
            await db.flush()
            await self._prune_face_embeddings(db, person_id)
        except Exception as e:
            logger.error(f"Failed to store face embedding: {e}")

    async def _prune_face_embeddings(self, db: AsyncSession, person_id: uuid.UUID):
        """Keep only the best-quality K face embeddings per identity.

        Excess rows are deleted from the DB.  MinIO file deletion is deferred to
        the periodic sweep in ``deduplicate_persons()`` (every 10 min) which
        cross-references files against all live DB paths before deleting.
        """
        try:
            query = text("""
                SELECT id, face_crop_path FROM person_face_embeddings
                WHERE person_identity_id = :pid
                ORDER BY face_score DESC, captured_at DESC
                OFFSET :keep
            """)
            result = await db.execute(query, {
                "pid": str(person_id),
                "keep": self.settings.MAX_FACE_EMBEDDINGS_PER_PERSON,
            })
            rows = result.fetchall()

            if rows:
                ids_to_delete = [row[0] for row in rows]
                paths_deferred = sum(1 for row in rows if row[1])
                await db.execute(
                    text("DELETE FROM person_face_embeddings WHERE id = ANY(:ids)"),
                    {"ids": ids_to_delete},
                )
                if paths_deferred:
                    logger.debug(
                        f"Pruned {paths_deferred} face embedding(s) for person {person_id} "
                        f"(MinIO cleanup deferred to dedup-job sweep)"
                    )

        except Exception as e:
            logger.warning(f"Face embedding pruning failed for person {person_id}: {e}")

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
                paths_deferred = sum(1 for row in rows if row[1])

                await db.execute(
                    text("DELETE FROM person_embeddings WHERE id = ANY(:ids)"),
                    {"ids": ids_to_delete}
                )

                if paths_deferred:
                    logger.debug(
                        f"Pruned {paths_deferred} body embedding(s) for person {person_id} "
                        f"(MinIO cleanup deferred to dedup-job sweep)"
                    )
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

            # Delete the identity (cascades to person_embeddings)
            await db.execute(
                text("DELETE FROM person_identities WHERE id = :pid"),
                {"pid": str(person_id)}
            )
            logger.info(f"Deleted temporary person identity from database: {person_id} (merged into {matched_id})")
            # MinIO files for the merged identity are NOT deleted immediately.
            # The periodic dedup-job sweep (every 10 min) cross-references all MinIO
            # ``crops/`` objects against live DB paths and removes only truly
            # unreferenced files.
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
        good_face_count: int = 0,
    ) -> Optional[uuid.UUID]:
        """Create a new anonymous person identity and fire registration event.

        Identity creation is gated on face quality:
        1. Face must be present (REQUIRE_FACE_FOR_IDENTITY).
        2. Face quality must meet FACE_IDENTITY_MIN_SCORE.
        3. At least FACE_IDENTITY_MIN_DETECTIONS good face captures across the track.
        """
        if self.settings.REQUIRE_FACE_FOR_IDENTITY and face_embedding is None:
            logger.debug(f"[_create_new_person] Blocked: REQUIRE_FACE_FOR_IDENTITY is True, but no face detected on camera {camera_id}.")
            return None

        if face_score < self.settings.FACE_IDENTITY_MIN_SCORE:
            logger.debug(
                f"[_create_new_person] Blocked: face score {face_score:.3f} < "
                f"FACE_IDENTITY_MIN_SCORE ({self.settings.FACE_IDENTITY_MIN_SCORE})"
            )
            return None

        if good_face_count < self.settings.FACE_IDENTITY_MIN_DETECTIONS:
            logger.debug(
                f"[_create_new_person] Blocked: good_face_count {good_face_count} < "
                f"FACE_IDENTITY_MIN_DETECTIONS ({self.settings.FACE_IDENTITY_MIN_DETECTIONS})"
            )
            return None

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
            _face_emb = face_embedding
            _norm = np.linalg.norm(face_embedding)
            if _norm > 0:
                _face_emb = face_embedding / _norm
            face_emb = PersonFaceEmbedding(
                person_identity_id=person.id,
                embedding=_face_emb.tolist(),
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
