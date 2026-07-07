"""Background job task implementations."""

from datetime import datetime, timedelta, date

import anyio
from loguru import logger
from sqlalchemy import select, func, update, text

from app.core.db.session import AsyncSessionLocal
from app.core.db.models.camera import Camera, CameraStatus
from app.core.db.models.event import Event
from app.core.db.models.billing import BillingInteraction
from app.core.db.models.tracking import TrackSession
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

            # Billing interactions
            total_billing = (
                await db.execute(
                    select(func.count()).where(
                        BillingInteraction.entered_at >= day_start,
                        BillingInteraction.entered_at < day_end,
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
    threshold = settings.FACE_MATCH_THRESHOLD

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

            # ── Step 2: resolve winners per connected component ─────────────
            # Build union-find so A=B and B=C → all three merge into one winner.
            parent: dict[str, str] = {}

            def find(x: str) -> str:
                while parent.get(x, x) != x:
                    parent[x] = parent.get(parent.get(x, x), x)  # path compression
                    x = parent.get(x, x)
                return x

            def union(x: str, y: str):
                parent[find(x)] = find(y)

            for row in pairs:
                union(str(row[0]), str(row[1]))

            # Collect all unique IDs involved
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
                return

            # ── Step 3: merge each loser into its winner ────────────────────
            merged_count = 0
            for winner_id, loser_id in merges:
                try:
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
                    extra_visits = loser_meta.get("visits", 0)
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

                    # Collect paths to delete from MinIO BEFORE cascade-deleting the loser row
                    paths_result = await db.execute(text("""
                        SELECT crop_path        FROM person_embeddings      WHERE person_identity_id::text = :loser
                        UNION ALL
                        SELECT face_crop_path   FROM person_face_embeddings WHERE person_identity_id::text = :loser
                        UNION ALL
                        SELECT face_crop_path   FROM person_identities      WHERE id::text = :loser
                    """), {"loser": loser_id})
                    paths_to_remove = [r[0] for r in paths_result.fetchall() if r[0]]

                    # Delete loser (CASCADE removes person_embeddings + person_face_embeddings)
                    await db.execute(
                        text("DELETE FROM person_identities WHERE id::text = :loser"),
                        {"loser": loser_id}
                    )

                    # Delete MinIO files for the loser's crops
                    from app.modules.storage.minio_client import delete_object as minio_del
                    for path in set(paths_to_remove):
                        try:
                            key = path.split("/", 1)[1] if "/" in path else path
                            minio_del(key)
                        except Exception as e:
                            logger.warning(f"Dedup: MinIO delete failed for {path}: {e}")

                    merged_count += 1
                    logger.info(
                        f"Dedup: merged {loser_id[:8]} → {winner_id[:8]} "
                        f"(sim>{threshold:.2f}, +{extra_visits} visits)"
                    )

                except Exception as e:
                    logger.error(f"Dedup: failed to merge {loser_id[:8]} → {winner_id[:8]}: {e}")
                    await db.rollback()
                    return  # abort this run; retry in 10 min

            await db.commit()
            logger.info(f"Dedup job complete: merged {merged_count} duplicate identit(ies).")

            # ── Step 4: sweep orphaned MinIO objects ─────────────────────────
            swept = await _sweep_orphaned_crops(db)
            if swept > 0:
                logger.info(f"MinIO sweep: removed {swept} unreferenced crop file(s).")

        except Exception as e:
            await db.rollback()
            logger.error(f"Dedup job failed: {e}")


async def _sweep_orphaned_crops(db) -> int:
    """
    Delete MinIO objects under the ``crops/`` prefix that are NOT referenced by
    any live DB row.

    Referenced paths are collected from:
      • person_face_embeddings.face_crop_path
      • person_identities.face_crop_path
      • person_embeddings.crop_path
      • track_sessions.best_crop_path

    The function also drains ``CameraWorker._pending_minio_deletes`` — every
    path that was queued for deferred deletion by the AI runtime is also
    processed here so the set stays bounded.

    Returns the number of objects removed from MinIO.
    """
    from app.config import get_settings
    from app.modules.storage.minio_client import get_client, BUCKET_PREFIX
    from app.modules.ai_runtime.camera_worker import CameraWorker

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

    if not known:
        # DB is empty — don't sweep (avoids accidentally deleting everything
        # during/after a reset)
        CameraWorker._pending_minio_deletes.clear()
        return 0

    # ── 2. Build a set of MinIO object keys (strip bucket prefix) ───────────
    # Also drain the pending queue into the check set so those paths are
    # processed even if they aren't referenced by the DB.
    pending = CameraWorker._pending_minio_deletes.copy()
    CameraWorker._pending_minio_deletes.clear()

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
                # Also check: is this key referenced by a pending path?
                pending_match = any(
                    _normalise(p) == key for p in pending
                )
                # Only delete if it was explicitly queued for deletion
                # (pending) OR it's been unreferenced for a full sweep cycle.
                # For safety, we DO delete all truly unreferenced objects.
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
