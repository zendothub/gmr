"""Background job task implementations."""

import asyncio
from datetime import datetime, timedelta, date

import anyio
from loguru import logger
from sqlalchemy import select, func, update, text

from app.core.db.session import AsyncSessionLocal
from app.core.db.models.camera import Camera, CameraStatus
from app.core.db.models.event import Event
from app.core.db.models.billing import BillingInteraction
from app.core.db.models.tracking import TrackSession
from app.core.db.models.person import PersonIdentity
from app.core.db.models.analytics import DailyAnalyticsSummary
from app.modules.storage.service import cleanup_old_objects as s3_cleanup_old_objects
from app.utils.time_utils import utc_now


async def aggregate_daily_analytics():
    """Aggregate yesterday's metrics into a single global daily_analytics_summary row."""
    yesterday: date = (utc_now() - timedelta(days=1)).date()
    day_start = datetime.combine(yesterday, datetime.min.time())
    day_end = day_start + timedelta(days=1)

    async with AsyncSessionLocal() as db:
        try:
            # Footfall: entry line crossings (across all cameras)
            total_footfall = (
                await db.execute(
                    select(func.count()).where(
                        Event.event_type == "line_crossing",
                        Event.occurred_at >= day_start,
                        Event.occurred_at < day_end,
                        Event.is_false_positive.is_(False),
                    )
                )
            ).scalar() or 0

            # Unique visitors via distinct person identities
            unique_visitors = (
                await db.execute(
                    select(func.count(func.distinct(TrackSession.person_identity_id))).where(
                        TrackSession.started_at >= day_start,
                        TrackSession.started_at < day_end,
                        TrackSession.person_identity_id.isnot(None),
                    )
                )
            ).scalar() or 0

            # Avg dwell
            duration = func.extract("epoch", TrackSession.last_seen_at - TrackSession.started_at)
            avg_dwell = (
                await db.execute(
                    select(func.avg(duration)).where(
                        TrackSession.started_at >= day_start,
                        TrackSession.started_at < day_end,
                    )
                )
            ).scalar()

            # Billing interactions (unique purchasers excl. staff)
            total_billing = (
                await db.execute(
                    select(func.count(func.distinct(BillingInteraction.person_identity_id))).where(
                        BillingInteraction.entered_at >= day_start,
                        BillingInteraction.entered_at < day_end,
                        BillingInteraction.person_identity_id.notin_(
                            select(PersonIdentity.id).where(PersonIdentity.is_staff.is_(True))
                        ),
                    )
                )
            ).scalar() or 0

            # Total events
            total_events = (
                await db.execute(
                    select(func.count()).where(
                        Event.occurred_at >= day_start,
                        Event.occurred_at < day_end,
                    )
                )
            ).scalar() or 0

            # Hourly footfall breakdown
            hourly_q = (
                select(
                    func.extract("hour", Event.occurred_at).label("hour"),
                    func.count().label("count"),
                )
                .where(
                    Event.event_type == "line_crossing",
                    Event.occurred_at >= day_start,
                    Event.occurred_at < day_end,
                    Event.is_false_positive.is_(False),
                )
                .group_by("hour")
            )
            hourly = {str(int(r.hour)): r.count for r in (await db.execute(hourly_q)).all()}

            # Upsert the single daily summary row
            existing = (
                await db.execute(
                    select(DailyAnalyticsSummary).where(
                        DailyAnalyticsSummary.summary_date == yesterday,
                    )
                )
            ).scalar_one_or_none()

            if existing:
                existing.total_footfall = total_footfall
                existing.unique_visitors = unique_visitors
                existing.avg_dwell_seconds = float(avg_dwell) if avg_dwell else None
                existing.total_billing_interactions = total_billing
                existing.total_events = total_events
                existing.hourly_footfall = hourly
            else:
                db.add(
                    DailyAnalyticsSummary(
                        summary_date=yesterday,
                        total_footfall=total_footfall,
                        unique_visitors=unique_visitors,
                        avg_dwell_seconds=float(avg_dwell) if avg_dwell else None,
                        total_billing_interactions=total_billing,
                        total_events=total_events,
                        hourly_footfall=hourly,
                    )
                )

            await db.commit()
            logger.info(f"Daily analytics aggregated for {yesterday}")
        except Exception as e:
            await db.rollback()
            logger.error(f"Daily analytics aggregation failed: {e}")



async def close_stale_track_sessions():
    """Close track sessions that stopped receiving updates (e.g. after crash)."""
    cutoff = utc_now() - timedelta(minutes=10)
    async with AsyncSessionLocal() as db:
        try:
            result = await db.execute(
                update(TrackSession)
                .where(TrackSession.is_active.is_(True), TrackSession.last_seen_at < cutoff)
                .values(is_active=False, ended_at=TrackSession.last_seen_at)
            )
            await db.commit()
            if result.rowcount:
                logger.info(f"Closed {result.rowcount} stale track sessions")
        except Exception as e:
            await db.rollback()
            logger.error(f"Stale track session cleanup failed: {e}")


async def cleanup_old_storage(retention_days: int = 30):
    """Delete snapshots/crops older than the retention period."""
    older_than = utc_now() - timedelta(days=retention_days)
    async with AsyncSessionLocal() as db:
        try:
            removed = await s3_cleanup_old_objects(db, older_than)
            await db.commit()
            logger.info(f"Storage cleanup job removed {removed} old objects")
        except Exception as e:
            await db.rollback()
            logger.error(f"Storage cleanup job failed: {e}")


async def deduplicate_persons():
    """
    Merge duplicate PersonIdentity records created when two cameras register the
    same physical person separately (cross-angle face similarity just below the
    match threshold at registration time).

    Algorithm (O(N·log N), index-assisted via pgvector LATERAL):
    1. For every stored face embedding, find the nearest neighbours in *other*
       identities that exceed FACE_MATCH_THRESHOLD.
    2. Group valid pairs so the identity with the better face score (or earlier
       first_seen_at as tiebreaker) is the "winner".
    3. Merge the loser into the winner:
         • Reassign track_sessions, events, billing_interactions, storage_objects
         • Absorb first_seen_at and visit_count into winner
         • DELETE loser (cascades to person_embeddings / person_face_embeddings)
    4. Log a summary.

    This job runs every 10 minutes.  It does NOT modify any config or realtime
    state — only the PostgreSQL person tables.
    """
    from app.config import get_settings
    settings = get_settings()
    threshold = 0.40  # empirically determined from retail CCTV face distribution

    async with AsyncSessionLocal() as db:
        try:
            # ── Step 1: efficient duplicate-pair discovery ──────────────────
            # LATERAL lets pgvector's IVFFlat index handle each probe in
            # O(log N) instead of a full O(N²) cross-join.
            await db.execute(text("SET LOCAL ivfflat.probes = 50"))

            pairs_result = await db.execute(text("""
                SELECT DISTINCT
                    LEAST(a.person_identity_id::text, b_near.person_identity_id::text)  AS pid_a,
                    GREATEST(a.person_identity_id::text, b_near.person_identity_id::text) AS pid_b,
                    MAX(1.0 - (b_near.dist)) AS max_sim
                FROM person_face_embeddings a
                CROSS JOIN LATERAL (
                    SELECT pfe.person_identity_id,
                           pfe.embedding <=> a.embedding AS dist
                    FROM   person_face_embeddings pfe
                    WHERE  pfe.person_identity_id != a.person_identity_id
                      AND  (1.0 - (pfe.embedding <=> a.embedding)) >= :threshold
                    ORDER  BY dist
                    LIMIT  5
                ) b_near
                GROUP  BY pid_a, pid_b
                HAVING MAX(1.0 - (b_near.dist)) >= :threshold
            """), {"threshold": threshold})

            pairs = pairs_result.fetchall()

            if not pairs:
                logger.debug("Dedup job: no duplicate pairs found.")
                return

            logger.info(f"Dedup job: found {len(pairs)} duplicate pair(s) — merging...")

            # ── Same-camera temporal overlap gate ──────────────────────────
            # Drop face-similar pairs whose tracks already overlap on the same
            # camera — those cannot be the same physical person.
            if settings.ENABLE_SAME_CAMERA_OVERLAP_GATE and pairs:
                min_sec = float(settings.SAME_CAMERA_OVERLAP_MIN_SECONDS)
                pair_list = [(str(r[0]), str(r[1]), float(r[2])) for r in pairs]
                all_pair_ids = list({p for a, b, _ in pair_list for p in (a, b)})
                win_result = await db.execute(text("""
                    SELECT person_identity_id::text, camera_id::text,
                           started_at, COALESCE(ended_at, last_seen_at) AS ended
                    FROM track_sessions
                    WHERE person_identity_id::text = ANY(:ids)
                      AND started_at IS NOT NULL
                      AND COALESCE(ended_at, last_seen_at) IS NOT NULL
                """), {"ids": all_pair_ids})
                by_pid: dict[str, list] = {}
                for pid, cam, start, end in win_result.fetchall():
                    by_pid.setdefault(pid, []).append((cam, start, end))

                def _same_cam_overlap(wa, wb) -> bool:
                    for a_cam, a0, a1 in wa:
                        for b_cam, b0, b1 in wb:
                            if a_cam != b_cam:
                                continue
                            start = max(a0, b0)
                            end = min(a1, b1)
                            if (end - start).total_seconds() >= min_sec:
                                return True
                    return False

                kept = []
                dropped = 0
                for a, b, sim in pair_list:
                    if _same_cam_overlap(by_pid.get(a, []), by_pid.get(b, [])):
                        dropped += 1
                        logger.info(
                            f"Dedup: drop pair {a[:8]}↔{b[:8]} sim={sim:.3f} "
                            f"(same-camera track overlap)"
                        )
                        continue
                    kept.append((a, b, sim))
                if dropped:
                    logger.info(
                        f"Dedup: dropped {dropped}/{len(pair_list)} pair(s) "
                        f"due to same-camera track overlap; {len(kept)} remain"
                    )
                pairs = kept  # list of (a,b,sim) tuples
                if not pairs:
                    logger.debug("Dedup job: all pairs blocked by same-camera overlap.")
                    # Still run contamination/staff steps below — fall through but
                    # skip merge section when pairs empty.
            else:
                # Normalize row access to (a,b,sim) triples when gate off
                pairs = [(str(r[0]), str(r[1]), float(r[2])) for r in pairs]

            if not pairs:
                logger.debug("Dedup job: no mergeable pairs after overlap filter.")
            else:
                logger.info(f"Dedup job: merging {len(pairs)} pair(s) after overlap filter...")

            # ── Step 2–3: merge only when pairs remain after filter ─────────
            merged_count = 0
            failed_count = 0
            deferred_minio_paths: list[str] = []
            if pairs:
                # Build union-find so A=B and B=C → all three merge into one winner.
                parent: dict[str, str] = {}

                def find(x: str) -> str:
                    p = parent.get(x, x)
                    if p != x:
                        parent[x] = find(p)
                    return parent.get(x, x)

                def union(x: str, y: str):
                    parent[find(x)] = find(y)

                for row in pairs:
                    union(str(row[0]), str(row[1]))

                all_ids = set()
                for row in pairs:
                    all_ids.add(str(row[0]))
                    all_ids.add(str(row[1]))

                # Fetch identity metadata to pick the best winner per component
                id_list = list(all_ids)
                meta_result = await db.execute(text("""
                    SELECT id::text, best_face_score, first_seen_at, visit_count
                    FROM   person_identities
                    WHERE  id::text = ANY(:ids)
                """), {"ids": id_list})
                meta = {r[0]: {"score": r[1] or 0.0, "first_seen": r[2], "visits": r[3]}
                        for r in meta_result.fetchall()}

                # Group IDs by their representative (root of union-find tree)
                components: dict[str, list[str]] = {}
                for pid in all_ids:
                    root = find(pid)
                    components.setdefault(root, []).append(pid)

                # For each component pick the winner = highest face score, tie-break by earliest first_seen
                merges: list[tuple[str, str]] = []  # (winner_id, loser_id)
                for root, members in components.items():
                    if len(members) < 2:
                        continue
                    winner = max(
                        members,
                        key=lambda pid: (
                            meta.get(pid, {}).get("score", 0.0),
                            -(meta.get(pid, {}).get("first_seen") or utc_now()).timestamp(),
                        )
                    )
                    for loser in members:
                        if loser != winner:
                            merges.append((winner, loser))

                if not merges:
                    logger.debug("Dedup job: all pairs already resolved.")
                else:
                    # ── Step 3: merge each loser into its winner ────────────────
                    # Each merge runs in its own SAVEPOINT (begin_nested). A single bad
                    # pair rolls back ONLY that merge and the batch continues.

                    # ── Pre-merge: count purchase impact for each loser ─────────
                    # The dashboard query is COUNT(DISTINCT person_identity_id)
                    # WHERE NOT is_staff. Merging two non-staff persons → purchase
                    # count drops by 1. Merging staff into non-staff → no change
                    # (staff purchases were already excluded). Merging non-staff
                    # into staff → purchases LOST (they become staff-associated).
                    loser_ids_for_impact = [lid for _, lid in merges]
                    winner_ids_for_impact = [wid for wid, _ in merges]
                    # Fetch is_staff flags for all involved persons
                    staff_lookup: dict[str, bool] = {}
                    all_impact_ids = list(set(loser_ids_for_impact + winner_ids_for_impact))
                    if all_impact_ids:
                        staff_rows = await db.execute(text("""
                            SELECT id::text, COALESCE(is_staff, false) AS is_staff
                            FROM person_identities
                            WHERE id::text = ANY(:ids)
                        """), {"ids": all_impact_ids})
                        staff_lookup = {r[0]: bool(r[1]) for r in staff_rows.fetchall()}
                    # Count billing rows per loser (before merge)
                    bi_counts = {}
                    if loser_ids_for_impact:
                        bi_rows = await db.execute(text("""
                            SELECT person_identity_id::text, COUNT(*) AS n
                            FROM billing_interactions
                            WHERE person_identity_id::text = ANY(:ids)
                            GROUP BY person_identity_id::text
                        """), {"ids": loser_ids_for_impact})
                        bi_counts = {r[0]: r[1] for r in bi_rows.fetchall()}

                    for winner_id, loser_id in merges:
                        loser_staff = staff_lookup.get(loser_id, False)
                        winner_staff = staff_lookup.get(winner_id, False)
                        loser_bis = bi_counts.get(loser_id, 0)
                        winner_bis = bi_counts.get(winner_id, 0)
                        # After merge: loser's BIs will be reassigned to winner.
                        # Impact: if both non-staff → DISTINCT drops by 1 (one person_id).
                        # If loser is staff → no change (staff BIs already excluded).
                        # If loser non-staff, winner staff → loser's purchases become
                        # staff-associated → they DROP from analytics.
                        will_decrease = (
                            loser_bis > 0
                            and winner_staff
                            and not loser_staff
                        )
                        purchase_impact = "COUNT DECREASE" if will_decrease else "no change"

                        try:
                            extra_visits = meta.get(loser_id, {}).get("visits", 0)
                            async with db.begin_nested():
                                # Reassign FK references
                                for tbl, col in [
                                    ("track_sessions",      "person_identity_id"),
                                    ("events",              "person_identity_id"),
                                    ("billing_interactions","person_identity_id"),
                                    ("storage_objects",     "person_identity_id"),
                                ]:
                                    await db.execute(text(
                                        f"UPDATE {tbl} SET {col} = :winner WHERE {col}::text = :loser"
                                    ), {"winner": winner_id, "loser": loser_id})

                                # Absorb visit_count and first_seen_at into winner
                                loser_meta  = meta.get(loser_id, {})
                                winner_meta = meta.get(winner_id, {})
                                loser_first  = loser_meta.get("first_seen")
                                winner_first = winner_meta.get("first_seen")

                                update_parts = ["visit_count = visit_count + :extra_visits"]
                                params: dict = {"extra_visits": extra_visits, "winner": winner_id}

                                if loser_first and winner_first and loser_first < winner_first:
                                    update_parts.append("first_seen_at = :loser_first")
                                    params["loser_first"] = loser_first

                                await db.execute(text(
                                    f"UPDATE person_identities SET {', '.join(update_parts)} WHERE id::text = :winner"
                                ), params)

                                # ── Absorb face/body embeddings (FIXED, contamination-gated) ──
                                max_faces = settings.MAX_FACE_EMBEDDINGS_PER_PERSON  # 5
                                await _absorb_face_embeddings(db, winner_id, loser_id, max_faces)
                                max_bodies = 10  # MAX_EMBEDDINGS_PER_PERSON
                                await _absorb_body_embeddings(db, winner_id, loser_id, max_bodies)

                                # ── Re-vote gender from ALL tracks ──────────────────────
                                await _revote_person_gender(db, winner_id)

                                # ── Collect loser's crop paths for deferred MinIO cleanup ─
                                paths_result = await db.execute(text("""
                                    SELECT crop_path        FROM person_embeddings      WHERE person_identity_id::text = :loser
                                    UNION ALL
                                    SELECT face_crop_path   FROM person_face_embeddings WHERE person_identity_id::text = :loser
                                    UNION ALL
                                    SELECT face_crop_path   FROM person_identities      WHERE id::text = :loser
                                """), {"loser": loser_id})
                                deferred_minio_paths.extend(r[0] for r in paths_result.fetchall() if r[0])

                                # Delete loser (embeddings already absorbed → no cascade)
                                await db.execute(
                                    text("DELETE FROM person_identities WHERE id::text = :loser"),
                                    {"loser": loser_id}
                                )

                            merged_count += 1
                            logger.info(
                                f"Dedup: merged {loser_id[:8]} → {winner_id[:8]} "
                                f"(sim>{threshold:.2f}, +{extra_visits} visits, "
                                f"loser_staff={loser_staff}, winner_staff={winner_staff}, "
                                f"loser_BI_rows={loser_bis}, winner_BI_rows={winner_bis}, "
                                f"purchase_impact={purchase_impact}, "
                                f"faces+body absorbed, gender re-voted)"
                            )

                        except Exception as e:
                            failed_count += 1
                            logger.error(
                                f"Dedup: failed to merge {loser_id[:8]} → {winner_id[:8]}: {e} "
                                f"— skipping (batch continues)"
                            )

                    await db.commit()

                    # ── Deferred MinIO cleanup for merged losers ──────────────
                    if deferred_minio_paths:
                        from app.modules.storage.minio_client import delete_object as minio_del
                        for path in set(deferred_minio_paths):
                            try:
                                key = path.split("/", 1)[1] if "/" in path else path
                                minio_del(key)
                            except Exception as e:
                                logger.warning(f"Dedup: MinIO delete failed for {path}: {e}")

                    logger.info(
                        f"Dedup job: merged {merged_count} duplicate identit(ies)"
                        f"{f', {failed_count} failed (skipped)' if failed_count else ''}."
                    )

            # ── Step 4: clean contaminated face embeddings ──────────────────
            face_removed = await _clean_contaminated_face_embeddings(db, settings)

            if face_removed > 0:
                await db.commit()
                logger.info(f"Face contamination cleanup: removed {face_removed} face embedding(s).")

            # ── Step 5: clean contaminated body embeddings ──────────────────
            body_removed = await _clean_contaminated_body_embeddings(db, settings)
            if body_removed > 0:
                await db.commit()
                logger.info(f"Body contamination cleanup: removed {body_removed} body embedding(s).")

            # ── Step 6: sweep orphaned MinIO objects ─────────────────────────
            swept = await _sweep_orphaned_crops(db)
            if swept > 0:
                logger.info(f"MinIO sweep: removed {swept} unreferenced crop file(s).")

            # ── Step 7: classify staff ────────────────────────────────────────
            # A person is flagged as staff if their TOTAL visible session time
            # across all cameras exceeds the configurable duration threshold
            # (default 30 min) OR if they have appeared on 3+ distinct days.
            # Staff persons are excluded from all purchase/billing analytics.
            _dur = settings.STAFF_DURATION_THRESHOLD_SECONDS
            _days = settings.STAFF_DISTINCT_DAYS_THRESHOLD
            promote_result = await db.execute(text("""
                UPDATE person_identities SET is_staff = TRUE WHERE is_staff = FALSE AND id IN (
                    SELECT pi.id FROM person_identities pi
                    LEFT JOIN track_sessions ts ON ts.person_identity_id = pi.id
                    GROUP BY pi.id
                    HAVING COALESCE(SUM(EXTRACT(epoch FROM COALESCE(ts.ended_at, ts.last_seen_at) - ts.started_at)), 0) > :dur
                        OR COUNT(DISTINCT DATE(ts.started_at)) >= :days
                )
            """), {"dur": _dur, "days": _days})
            # Demote persons who no longer meet criteria (e.g., after data reset)
            # BUT only if they have NO active tracks — staff might be mid-shift.
            demote_result = await db.execute(text("""
                UPDATE person_identities SET is_staff = FALSE WHERE is_staff = TRUE AND id NOT IN (
                    SELECT pi.id FROM person_identities pi
                    LEFT JOIN track_sessions ts ON ts.person_identity_id = pi.id
                    GROUP BY pi.id
                    HAVING COALESCE(SUM(EXTRACT(epoch FROM COALESCE(ts.ended_at, ts.last_seen_at) - ts.started_at)), 0) > :dur
                        OR COUNT(DISTINCT DATE(ts.started_at)) >= :days
                )
            """), {"dur": _dur, "days": _days})
            await db.commit()
            if promote_result.rowcount or demote_result.rowcount:
                logger.info(
                    f"Staff classification: +{promote_result.rowcount} promoted, "
                    f"-{demote_result.rowcount} demoted "
                    f"(dur>{_dur}s OR days>={_days})"
                )

        except Exception as e:
            await db.rollback()
            logger.error(f"Dedup job failed: {e}")


async def _clean_contaminated_face_embeddings(db, settings) -> int:
    """
    Remove face embeddings that don't match the face cluster for a person.

    Uses iterative median-based outlier removal (the same approach as the body
    version): for each person with >=2 face embeddings, repeatedly removes the
    embedding with the lowest median similarity to the rest until all remaining
    embeddings have median >= FACE_CONTAMINATION_THRESHOLD (0.35) or the
    cluster drops below 2.

    Why median, not the previous "compatible with ANY kept" greedy:
    single-linkage greedy clustering chains through borderline bridge
    embeddings (similar ~0.35-0.49 to BOTH real people), allowing contamination
    to slip through undetected. The median approach correctly isolates the
    majority cluster and rejects outliers — verified by hand against the two
    real contaminated identities found on 2026-07-09 (9b6053ac: 2-person split
    with a bridge embedding; cf793282: single outlier with greedy-chain link).

    Aggressive-reject tuning: any embedding whose median similarity to the
    rest of the cluster is < threshold is removed — no leniency, no "keep if
    borderline" behaviour. This is intentional: storing a contaminated face
    pollutes the person's identity and causes future false merges; deleting a
    borderline-same-person face is recoverable (reextract_or_delete_faceless.py
    will re-extract from track crops on the next 20-min cycle).

    The numpy computation runs in a thread pool (asyncio.to_thread) to avoid
    blocking the FastAPI event loop at 1k+ persons scale.

    Returns the number of embeddings removed.
    """
    import numpy as np

    r = await db.execute(text("""
        SELECT pi.id FROM person_identities pi
        WHERE (SELECT COUNT(*) FROM person_face_embeddings
               WHERE person_identity_id = pi.id) >= 2
    """))
    person_ids = [row[0] for row in r.fetchall()]

    # Fetch all embeddings upfront (avoids interleaving DB + numpy work)
    person_data = []
    for pid in person_ids:
        r2 = await db.execute(text("""
            SELECT id, embedding, face_score FROM person_face_embeddings
            WHERE person_identity_id = :pid AND embedding IS NOT NULL
            ORDER BY face_score DESC
        """), {"pid": str(pid)})
        rows = r2.fetchall()
        if len(rows) < 2:
            continue
        ids = [r[0] for r in rows]
        embs = []
        for row in rows:
            if isinstance(row[1], str):
                embs.append(np.array(eval(row[1]), dtype=np.float32))
            else:
                embs.append(np.array(row[1], dtype=np.float32))
        person_data.append((str(pid), ids, embs))

    threshold = settings.FACE_CONTAMINATION_THRESHOLD

    # Run the numpy-heavy iterative median computation in a thread
    def _compute_face_removals():
        results = []
        for pid, ids, embs in person_data:
            N = len(embs)
            # Normalize each embedding (InsightFace embeddings are NOT L2-normalized)
            for emb in embs:
                _n = np.linalg.norm(emb)
                if _n > 0:
                    emb /= _n

            remove_idx = set()
            active = set(range(N))

            while len(active) >= 2:
                medians = []
                for i in active:
                    sims = []
                    for j in active:
                        if i != j:
                            sims.append(float(np.dot(embs[i], embs[j])))
                    # With 1 other member, median == that single sim.
                    # With 2+, median is the middle value — robust to a single
                    # borderline-bridge edge.
                    medians.append((i, float(np.median(sims)) if sims else 0.0))

                worst_idx, worst_median = min(medians, key=lambda x: x[1])

                if worst_median >= threshold:
                    break

                remove_idx.add(worst_idx)
                active.discard(worst_idx)

            if remove_idx:
                remove_ids = [ids[i] for i in sorted(remove_idx)]
                results.append((pid, remove_ids, len(active), N))
        return results

    removal_results = await asyncio.to_thread(_compute_face_removals)

    total_removed = 0
    for pid, remove_ids, keep_count, total_count in removal_results:
        logger.info(
            f"Face contamination cleanup: person {pid[:12]} "
            f"keeping {keep_count}/{total_count} faces, removing {len(remove_ids)} "
            f"(threshold={threshold})"
        )
        await db.execute(text(
            "DELETE FROM person_face_embeddings WHERE id = ANY(:ids)"
        ), {"ids": remove_ids})
        total_removed += len(remove_ids)

    return total_removed


async def _clean_contaminated_body_embeddings(db, settings) -> int:
    """
    Remove body embeddings that don't match the body cluster for a person.

    Uses iterative median-based outlier removal: for each person with >=3 body
    embeddings, repeatedly removes the embedding with the lowest median
    similarity to the rest until all remaining embeddings have median
    >= BODY_CONTAMINATION_THRESHOLD (0.50) or the cluster drops below 3.

    The numpy computation runs in a thread pool (asyncio.to_thread) to avoid
    blocking the FastAPI event loop at 1k+ persons scale.

    Returns the number of embeddings removed.
    """
    import numpy as np

    r = await db.execute(text("""
        SELECT pi.id FROM person_identities pi
        WHERE (SELECT COUNT(*) FROM person_embeddings
               WHERE person_identity_id = pi.id) >= 3
    """))
    person_ids = [row[0] for row in r.fetchall()]

    # Fetch all embeddings upfront
    person_data = []
    for pid in person_ids:
        r2 = await db.execute(text("""
            SELECT id, embedding, crop_quality FROM person_embeddings
            WHERE person_identity_id = :pid AND embedding IS NOT NULL
            ORDER BY crop_quality DESC
        """), {"pid": str(pid)})
        rows = r2.fetchall()
        if len(rows) < 3:
            continue
        ids = [r[0] for r in rows]
        embs = []
        for row in rows:
            if isinstance(row[1], str):
                embs.append(np.array(eval(row[1]), dtype=np.float32))
            else:
                embs.append(np.array(row[1], dtype=np.float32))
        person_data.append((str(pid), ids, embs))

    threshold = settings.BODY_CONTAMINATION_THRESHOLD

    # Run the numpy-heavy iterative median computation in a thread
    def _compute_body_removals():
        results = []
        for pid, ids, embs in person_data:
            N = len(embs)
            remove_idx = set()
            active = set(range(N))

            while len(active) >= 3:
                medians = []
                for i in active:
                    sims = []
                    for j in active:
                        if i != j:
                            sims.append(float(np.dot(embs[i], embs[j])))
                    medians.append((i, float(np.median(sims))))

                worst_idx, worst_median = min(medians, key=lambda x: x[1])

                if worst_median >= threshold:
                    break

                remove_idx.add(worst_idx)
                active.discard(worst_idx)

            if remove_idx:
                remove_ids = [ids[i] for i in sorted(remove_idx)]
                results.append((pid, remove_ids, len(active), N))
        return results

    removal_results = await asyncio.to_thread(_compute_body_removals)

    total_removed = 0
    for pid, remove_ids, active_count, total_count in removal_results:
        logger.info(
            f"Body contamination cleanup: person {pid[:12]} "
            f"keeping {active_count}/{total_count} bodies, removing {len(remove_ids)} "
            f"(threshold={threshold})"
        )
        await db.execute(text(
            "DELETE FROM person_embeddings WHERE id = ANY(:ids)"
        ), {"ids": remove_ids})
        total_removed += len(remove_ids)

    return total_removed


async def _absorb_face_embeddings(db, winner_id: str, loser_id: str, max_faces: int):
    """Move face embeddings from loser to winner, skipping duplicate angles and contamination.

    For each loser face, three gates before moving it to the winner:
      1. Duplicate angle (sim > 0.95 to an existing winner face) → skip.
      2. Contamination gate — median similarity to the winner's *existing* face
         cluster must be >= FACE_CONTAMINATION_THRESHOLD (0.35). If below, the
         face belongs to a different person (the merge was likely a false positive
         on a borderline face pair); DROP it instead of moving it. This is the
         fix for the previously-unchecked absorption path that was silently
         re-injecting contamination into winner identities on every dedup merge
         (root cause of the staff-identity pollution seen on 2026-07-09).
      3. Winner is full (already has max_faces) → prune lowest-score after move.

    Note: gate 2 uses median (not max) similarity so a single borderline-bridge
    face on the winner side cannot chain a stranger's face in. With <2 existing
    winner faces, falls back to max similarity (can't compute a median of 1).
    """
    import numpy as np

    threshold = 0.35  # FACE_CONTAMINATION_THRESHOLD — inlined to avoid settings re-fetch

    # Get winner's existing face embeddings
    winner_faces = await db.execute(text("""
        SELECT embedding FROM person_face_embeddings
        WHERE person_identity_id::text = :pid AND embedding IS NOT NULL
    """), {"pid": winner_id})
    winner_embs = []
    for row in winner_faces.fetchall():
        if isinstance(row[0], str):
            winner_embs.append(np.array(eval(row[0]), dtype=np.float32))
        else:
            winner_embs.append(np.array(row[0], dtype=np.float32))

    # Normalize winner embeddings (InsightFace embeddings are NOT L2-normalized)
    for w in winner_embs:
        _n = np.linalg.norm(w)
        if _n > 0:
            w /= _n

    # Get loser's face embeddings
    loser_faces = await db.execute(text("""
        SELECT id, embedding, face_score FROM person_face_embeddings
        WHERE person_identity_id::text = :pid AND embedding IS NOT NULL
        ORDER BY face_score DESC
    """), {"pid": loser_id})
    loser_rows = loser_faces.fetchall()

    if not loser_rows:
        return

    moved = 0
    rejected = 0
    for row in loser_rows:
        loser_emb = np.array(row[1], dtype=np.float32) if not isinstance(row[1], str) else np.array(eval(row[1]), dtype=np.float32)
        _n = np.linalg.norm(loser_emb)
        if _n > 0:
            loser_emb_norm = loser_emb / _n
        else:
            loser_emb_norm = loser_emb

        # Gate 1: duplicate angle
        is_dup = False
        for w_emb in winner_embs:
            sim = float(np.dot(w_emb, loser_emb_norm))
            if sim > 0.95:
                is_dup = True
                break
        if is_dup:
            continue

        # Gate 2: contamination — must fit the winner's existing cluster
        if winner_embs:
            sims_to_winner = [float(np.dot(w_emb, loser_emb_norm)) for w_emb in winner_embs]
            if len(sims_to_winner) >= 2:
                fit_sim = float(np.median(sims_to_winner))
            else:
                fit_sim = max(sims_to_winner)
            if fit_sim < threshold:
                # Different person's face — drop instead of move
                await db.execute(text(
                    "DELETE FROM person_face_embeddings WHERE id = :row_id"
                ), {"row_id": row[0]})
                rejected += 1
                logger.info(
                    f"Dedup absorb: REJECTED contaminated face from {loser_id[:8]} → "
                    f"{winner_id[:8]} (cluster_fit={fit_sim:.3f} < {threshold}, dropped)"
                )
                continue

        # Move to winner
        await db.execute(text("""
            UPDATE person_face_embeddings SET person_identity_id = :winner
            WHERE id = :row_id
        """), {"winner": winner_id, "row_id": row[0]})
        winner_embs.append(loser_emb_norm)
        moved += 1

    if moved > 0:
        # Prune winner to max_faces (keep highest face_score)
        await db.execute(text("""
            DELETE FROM person_face_embeddings WHERE id IN (
                SELECT id FROM person_face_embeddings
                WHERE person_identity_id = :pid
                ORDER BY face_score DESC OFFSET :keep
            )
        """), {"pid": winner_id, "keep": max_faces})
        logger.info(
            f"Dedup: absorbed {moved} face embedding(s) from {loser_id[:8]} → {winner_id[:8]} "
            f"(winner now has up to {max_faces} faces, rejected {rejected} contaminated)"
        )
    elif rejected > 0:
        logger.info(
            f"Dedup: rejected all {rejected} face embedding(s) from {loser_id[:8]} as "
            f"contamination (none absorbed into {winner_id[:8]})"
        )


async def _absorb_body_embeddings(db, winner_id: str, loser_id: str, max_bodies: int):
    """Move body embeddings from loser to winner, skipping duplicate angles and contamination.

    Same contamination-gate logic as _absorb_face_embeddings but using
    BODY_CONTAMINATION_THRESHOLD (0.50) — median similarity to the winner's
    existing body cluster must clear it before a loser body is moved. Previously
    this function had NO contamination check (only a >0.95 duplicate-angle
    check), silently absorbing a stranger's body into the winner on every
    dedup merge.
    """
    import numpy as np

    threshold = 0.50  # BODY_CONTAMINATION_THRESHOLD

    # Get winner's existing body embeddings
    winner_bodies = await db.execute(text("""
        SELECT embedding FROM person_embeddings
        WHERE person_identity_id::text = :pid AND embedding IS NOT NULL
    """), {"pid": winner_id})
    winner_embs = []
    for row in winner_bodies.fetchall():
        if isinstance(row[0], str):
            winner_embs.append(np.array(eval(row[0]), dtype=np.float32))
        else:
            winner_embs.append(np.array(row[0], dtype=np.float32))

    # OSNet embeddings are L2-normalized at extract time, but be defensive
    for w in winner_embs:
        _n = np.linalg.norm(w)
        if _n > 0:
            w /= _n

    # Get loser's body embeddings
    loser_bodies = await db.execute(text("""
        SELECT id, embedding, crop_quality FROM person_embeddings
        WHERE person_identity_id::text = :pid AND embedding IS NOT NULL
        ORDER BY crop_quality DESC
    """), {"pid": loser_id})
    loser_rows = loser_bodies.fetchall()

    if not loser_rows:
        return

    moved = 0
    rejected = 0
    for row in loser_rows:
        loser_emb = np.array(row[1], dtype=np.float32) if not isinstance(row[1], str) else np.array(eval(row[1]), dtype=np.float32)
        _n = np.linalg.norm(loser_emb)
        if _n > 0:
            loser_emb_norm = loser_emb / _n
        else:
            loser_emb_norm = loser_emb

        # Gate 1: near-duplicate
        is_dup = False
        for w_emb in winner_embs:
            sim = float(np.dot(w_emb, loser_emb_norm))
            if sim > 0.95:
                is_dup = True
                break
        if is_dup:
            continue

        # Gate 2: contamination — must fit the winner's existing body cluster
        if winner_embs:
            sims_to_winner = [float(np.dot(w_emb, loser_emb_norm)) for w_emb in winner_embs]
            if len(sims_to_winner) >= 2:
                fit_sim = float(np.median(sims_to_winner))
            else:
                fit_sim = max(sims_to_winner)
            if fit_sim < threshold:
                # Different person's body — drop instead of move
                await db.execute(text(
                    "DELETE FROM person_embeddings WHERE id = :row_id"
                ), {"row_id": row[0]})
                rejected += 1
                logger.info(
                    f"Dedup absorb: REJECTED contaminated body from {loser_id[:8]} → "
                    f"{winner_id[:8]} (cluster_fit={fit_sim:.3f} < {threshold}, dropped)"
                )
                continue

        # Move to winner
        await db.execute(text("""
            UPDATE person_embeddings SET person_identity_id = :winner
            WHERE id = :row_id
        """), {"winner": winner_id, "row_id": row[0]})
        winner_embs.append(loser_emb_norm)
        moved += 1

    if moved > 0:
        # Prune winner to max_bodies (keep highest crop_quality)
        await db.execute(text("""
            DELETE FROM person_embeddings WHERE id IN (
                SELECT id FROM person_embeddings
                WHERE person_identity_id = :pid
                ORDER BY crop_quality DESC OFFSET :keep
            )
        """), {"pid": winner_id, "keep": max_bodies})
        logger.info(
            f"Dedup: absorbed {moved} body embedding(s) from {loser_id[:8]} → {winner_id[:8]} "
            f"(winner now has up to {max_bodies} bodies, rejected {rejected} contaminated)"
        )
    elif rejected > 0:
        logger.info(
            f"Dedup: rejected all {rejected} body embedding(s) from {loser_id[:8]} as "
            f"contamination (none absorbed into {winner_id[:8]})"
        )


async def _revote_person_gender(db, person_id: str):
    """Re-vote person-level gender from ALL assigned track sessions.

    Uses majority vote across all tracks that have a non-null gender.
    If tied or no votes, keeps the existing gender.
    """
    r = await db.execute(text("""
        SELECT ts.gender, COUNT(*) as votes
        FROM track_sessions ts
        WHERE ts.person_identity_id::text = :pid
          AND ts.gender IS NOT NULL
        GROUP BY ts.gender
        ORDER BY votes DESC
    """), {"pid": person_id})
    votes = r.fetchall()

    if not votes:
        return

    majority = votes[0][0]
    majority_count = votes[0][1]

    # Check if there's a tie (M and F have same count)
    if len(votes) > 1 and votes[0][1] == votes[1][1]:
        logger.debug(
            f"Dedup: gender vote tied for {person_id[:8]} — keeping existing gender"
        )
        return

    await db.execute(text("""
        UPDATE person_identities SET gender = :gender WHERE id::text = :pid
    """), {"gender": majority, "pid": person_id})

    logger.info(
        f"Dedup: gender re-voted for {person_id[:8]} → {majority} "
        f"({majority_count} track votes)"
    )


async def _sweep_orphaned_crops(db) -> int:
    """
    Delete MinIO objects under the ``crops/`` prefix that are NOT referenced by
    any live DB row.

    Referenced paths are collected from:
      • person_face_embeddings.face_crop_path
      • person_identities.face_crop_path
      • person_embeddings.crop_path
      • track_sessions.best_crop_path
      • track_sessions.bbox_history->>'best_face_crop_path' (debug object only)

    When running in the same process as the API server, this also drains
    ``CameraWorker._pending_minio_deletes``. When running in the separate
    worker process, the pending set is not accessible — unreferenced crops
    are still deleted by the DB cross-reference, just one sweep cycle later.

    Returns the number of objects removed from MinIO.
    """
    from app.config import get_settings
    from app.modules.storage.minio_client import get_client, BUCKET_PREFIX

    settings = get_settings()
    client = get_client()
    bucket = BUCKET_PREFIX

    # ── 1. Collect every known-referenced path from the DB ───────────────────
    known: set[str] = set()

    # face_crop_path (person_face_embeddings)
    r = await db.execute(text(
        "SELECT face_crop_path FROM person_face_embeddings WHERE face_crop_path IS NOT NULL"
    ))
    for row in r.fetchall():
        known.add(row[0])

    # face_crop_path (person_identities)
    r = await db.execute(text(
        "SELECT face_crop_path FROM person_identities WHERE face_crop_path IS NOT NULL"
    ))
    for row in r.fetchall():
        known.add(row[0])

    # crop_path (person_embeddings)
    r = await db.execute(text(
        "SELECT crop_path FROM person_embeddings WHERE crop_path IS NOT NULL"
    ))
    for row in r.fetchall():
        known.add(row[0])

    # best_crop_path (track_sessions)
    r = await db.execute(text(
        "SELECT best_crop_path FROM track_sessions WHERE best_crop_path IS NOT NULL"
    ))
    for row in r.fetchall():
        known.add(row[0])

    # best_face_crop_path nested in track_sessions.bbox_history debug object
    # (legacy rows are a JSON array with no face key — COALESCE/jsonb skips them)
    r = await db.execute(text("""
        SELECT bbox_history->>'best_face_crop_path'
        FROM track_sessions
        WHERE jsonb_typeof(bbox_history) = 'object'
          AND bbox_history ? 'best_face_crop_path'
          AND NULLIF(bbox_history->>'best_face_crop_path', '') IS NOT NULL
    """))
    for row in r.fetchall():
        if row[0]:
            known.add(row[0])

    # ── 2. Drain the pending MinIO deletes queue (if in-process) ────────────
    # When running in the separate worker process, CameraWorker is not
    # importable (no API server). The sweep still works — it just relies
    # on the DB cross-reference alone instead of the "hint" set.
    pending: set[str] = set()
    try:
        from app.modules.ai_runtime.camera_worker import CameraWorker
        pending = CameraWorker._pending_minio_deletes.copy()
        CameraWorker._pending_minio_deletes.clear()
    except Exception:
        pass  # running in separate worker process — no in-memory state

    if not known and not pending:
        # DB is empty and no pending deletes — don't sweep
        return 0

    # Normalise: "retaileye/crops/foo.jpg" → "crops/foo.jpg"
    def _normalise(path: str) -> str:
        if path.startswith(f"{bucket}/"):
            return path[len(bucket) + 1:]
        return path

    objects_to_check: set[str] = {_normalise(p) for p in pending}

    removed = 0
    try:
        obj_list = client.list_objects(bucket, prefix="crops/", recursive=True)
        for obj in obj_list:
            key = obj.object_name
            full_key = f"{bucket}/{key}"
            objects_to_check.add(key)

            if full_key not in known and key not in known:
                try:
                    client.remove_object(bucket, key)
                    removed += 1
                    logger.debug(f"MinIO sweep: deleted unreferenced object {full_key}")
                except Exception as e:
                    logger.warning(f"MinIO sweep: failed to delete {full_key}: {e}")

    except Exception as e:
        logger.error(f"MinIO sweep: failed to list/delete objects: {e}")
        return removed

    return removed


# ---------------------------------------------------------------------------
# Camera status probe (runs every 2 minutes)
# ---------------------------------------------------------------------------

def _probe_rtsp(rtsp_url: str, timeout: int = 8) -> bool:
    """Synchronous RTSP connectivity probe. Returns True if stream is reachable."""
    import cv2
    cap = None
    try:
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout * 1000)
        if not cap.isOpened():
            return False
        ret, frame = cap.read()
        return ret and frame is not None
    except Exception:
        return False
    finally:
        if cap is not None:
            cap.release()


async def probe_camera_statuses():
    """Check every camera's RTSP stream and update its status in the database.

    Runs every 2 minutes via APScheduler.

    Status update rules:
    - Camera in MAINTENANCE → skipped (operator-set status, do not override).
    - RTSP reachable → ACTIVE
    - RTSP unreachable → INACTIVE

    Uses anyio.to_thread.run_sync so the blocking cv2 probe does not stall
    the async event loop.
    """
    async with AsyncSessionLocal() as db:
        try:
            # Fetch all cameras except those in MAINTENANCE status
            result = await db.execute(
                select(Camera).where(Camera.status != CameraStatus.MAINTENANCE)
            )
            cameras = list(result.scalars().all())

            if not cameras:
                return

            active_count = 0
            inactive_count = 0

            for camera in cameras:
                try:
                    # Run blocking RTSP probe in a thread
                    reachable: bool = await anyio.to_thread.run_sync(
                        lambda url=camera.rtsp_url: _probe_rtsp(url)
                    )
                    new_status = CameraStatus.ACTIVE if reachable else CameraStatus.INACTIVE

                    if camera.status != new_status:
                        camera.status = new_status
                        logger.info(
                            f"Camera status updated: {camera.name} (id={camera.id}) "
                            f"→ {new_status}"
                        )

                    if new_status == CameraStatus.ACTIVE:
                        active_count += 1
                    else:
                        inactive_count += 1

                except Exception as cam_err:
                    logger.warning(
                        f"RTSP probe error for camera {camera.id} ({camera.name}): {cam_err}"
                    )
                    # Mark as inactive on probe error
                    if camera.status not in (CameraStatus.INACTIVE, CameraStatus.ERROR):
                        camera.status = CameraStatus.INACTIVE

            await db.commit()
            logger.debug(
                f"Camera status probe complete: {len(cameras)} checked, "
                f"{active_count} active, {inactive_count} inactive"
            )

        except Exception as e:
            await db.rollback()
            logger.error(f"Camera status probe job failed: {e}")


# ── Device Session Cleanup ────────────────────────────────────────────────

async def cleanup_stale_sessions():
    """Deactivate device sessions that have expired or been idle too long.

    Also marks stale stream viewer sessions as ended (stream reaped but
    viewer row never cleaned up).
    """
    from app.config import get_settings
    from app.core.db.models.device_session import DeviceSession
    from app.core.db.models.stream_viewer import StreamViewerSession

    settings = get_settings()
    now = utc_now()
    idle_cutoff = now - timedelta(seconds=settings.SESSION_IDLE_TIMEOUT_SECONDS)

    async with AsyncSessionLocal() as db:
        try:
            # 1. Deactivate expired or idle device sessions
            result = await db.execute(
                update(DeviceSession)
                .where(
                    DeviceSession.is_active == True,
                    (
                        (DeviceSession.expires_at < now)
                        | (DeviceSession.last_active_at < idle_cutoff)
                    ),
                )
                .values(is_active=False)
            )
            device_count = result.rowcount
            if device_count:
                logger.info(f"Cleaned up {device_count} stale device sessions")

            # 2. Mark stale stream viewer sessions as ended
            stream_idle_cutoff = now - timedelta(hours=2)  # 2h idle = dead
            result = await db.execute(
                update(StreamViewerSession)
                .where(
                    StreamViewerSession.ended_at.is_(None),
                    StreamViewerSession.last_heartbeat_at < stream_idle_cutoff,
                )
                .values(ended_at=now)
            )
            stream_count = result.rowcount
            if stream_count:
                logger.info(f"Marked {stream_count} stale stream viewer sessions as ended")

            await db.commit()

        except Exception as e:
            await db.rollback()
            logger.error(f"Session cleanup job failed: {e}")
