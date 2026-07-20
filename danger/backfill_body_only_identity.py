#!/usr/bin/env python3
"""
backfill_body_only_identity.py

Apply body-only identity attach/create for unassigned track_sessions
(yesterday+today Asia/Kolkata by default, or --days).

Default is dry-run (no writes). Pass --apply to write DB.
On --apply takes pg_advisory_xact_lock(IDENTITY_ADVISORY_LOCK_KEY=1001).

Customer body match uses ONLY body embeddings with captured_at inside the
recent window. Staff: activity-recent + full body gallery. Never deletes
old body embeddings.

Usage:
    PYTHONPATH=/gmr/gmr venv/bin/python danger/backfill_body_only_identity.py
    PYTHONPATH=/gmr/gmr venv/bin/python danger/backfill_body_only_identity.py --apply
    PYTHONPATH=/gmr/gmr venv/bin/python danger/backfill_body_only_identity.py --days 2026-07-19,2026-07-20 --apply
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, time, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import text

from app.config import get_settings
from app.core.db.session import AsyncSessionLocal
from app.modules.storage.minio_client import BUCKET_PREFIX, get_client
from app.modules.reid.osnet_extractor import get_shared_extractor
from app.modules.reid.identity_decision_engine import IDENTITY_ADVISORY_LOCK_KEY

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore

IST = ZoneInfo("Asia/Kolkata")

# Staff attach exclusion (not in settings; matches dry_run / handoff)
BODY_ONLY_ATTACH_STAFF_EXCLUSION = 0.50
BODY_ONLY_ATTACH_STAFF_GAP = 0.05
BODY_ONLY_MIN_BODIES_GALLERY = 2
LEGACY_MIN_FRAMES = 4
LEGACY_DEFAULT_QUALITY = 0.60


def _aware(dt: datetime) -> datetime:
    if dt is None:
        return dt
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_embedding(raw) -> Optional[np.ndarray]:
    if raw is None:
        return None
    try:
        if isinstance(raw, str):
            v = np.array(eval(raw), dtype=np.float32)
        else:
            v = np.asarray(raw, dtype=np.float32)
        v = v.flatten()
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
        return v
    except Exception:
        return None


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _median_sims(query: np.ndarray, gallery: List[np.ndarray]) -> Optional[float]:
    if not gallery:
        return None
    return float(np.median([_cos(query, g) for g in gallery]))


def _download_crop(minio_client, crop_path: str):
    if not crop_path:
        return None
    try:
        key = crop_path
        if key.startswith(f"{BUCKET_PREFIX}/"):
            key = key[len(BUCKET_PREFIX) + 1 :]
        if "/" in key and not key.startswith("crops/"):
            key = key.split("/", 1)[1]
        resp = minio_client.get_object(BUCKET_PREFIX, key)
        data = resp.read()
        resp.close()
        resp.release_conn()
        import cv2

        arr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"  download failed {crop_path[:70]}: {e}")
        return None


def _quality_from_row(total_frames: int, bbox_history) -> Optional[float]:
    if isinstance(bbox_history, dict):
        q = bbox_history.get("best_crop_quality")
        if q is not None:
            try:
                return float(q)
            except (TypeError, ValueError):
                pass
    if total_frames is not None and int(total_frames) >= LEGACY_MIN_FRAMES:
        return LEGACY_DEFAULT_QUALITY
    return None


@dataclass
class BodyEmb:
    vec: np.ndarray
    t: datetime


@dataclass
class PersonState:
    pid: str
    is_staff: bool
    bodies: List[BodyEmb] = field(default_factory=list)
    # (camera_id, start, end)
    track_windows: List[Tuple[Optional[str], datetime, datetime]] = field(
        default_factory=list
    )
    last_seen_at: Optional[datetime] = None
    visit_count: int = 1
    is_virtual: bool = False


@dataclass
class NullTrack:
    track_id: str
    camera_id: Optional[str]
    started_at: datetime
    ended_at: datetime
    crop_path: str
    total_frames: int
    bbox_history: object


@dataclass
class Counters:
    null_tracks: int = 0
    attach_nonstaff: int = 0
    create_new: int = 0
    left_unassigned: int = 0
    no_crop: int = 0
    extract_fail: int = 0
    quality_gate_fail: int = 0
    attach_reject_staff: int = 0
    attach_reject_ambig: int = 0
    attach_reject_same_cam: int = 0
    attach_no_recent_gallery: int = 0
    create_reject_near: int = 0
    create_reject_staff: int = 0
    billing_updated: int = 0


def _window_bounds(now: datetime, recent_minutes: int) -> Tuple[datetime, datetime]:
    w = timedelta(minutes=recent_minutes)
    return now - w, now


def _is_activity_recent(
    st: PersonState, now: datetime, recent_minutes: int
) -> bool:
    t0, t1 = _window_bounds(now, recent_minutes)
    for _cam, a, b in st.track_windows:
        if a <= t1 and b >= t0:
            return True
    for be in st.bodies:
        if t0 <= be.t <= t1:
            return True
    return False


def _customer_recent_bodies(
    st: PersonState, now: datetime, recent_minutes: int
) -> List[np.ndarray]:
    t0, t1 = _window_bounds(now, recent_minutes)
    return [be.vec for be in st.bodies if t0 <= be.t <= t1]


def _staff_bodies(
    st: PersonState,
    now: datetime,
    recent_minutes: int,
    staff_full: bool,
) -> List[np.ndarray]:
    if not _is_activity_recent(st, now, recent_minutes):
        return []
    if staff_full:
        return [be.vec for be in st.bodies]
    return _customer_recent_bodies(st, now, recent_minutes)


def _rank_customers(
    query: np.ndarray,
    persons: Dict[str, PersonState],
    now: datetime,
    recent_minutes: int,
    min_bodies: int = BODY_ONLY_MIN_BODIES_GALLERY,
) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for pid, st in persons.items():
        if st.is_staff:
            continue
        gal = _customer_recent_bodies(st, now, recent_minutes)
        if len(gal) < min_bodies:
            continue
        med = _median_sims(query, gal)
        if med is not None:
            out.append((pid, med))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def _rank_staff(
    query: np.ndarray,
    persons: Dict[str, PersonState],
    now: datetime,
    recent_minutes: int,
    staff_full: bool,
    min_bodies: int = 1,
) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for pid, st in persons.items():
        if not st.is_staff:
            continue
        gal = _staff_bodies(st, now, recent_minutes, staff_full)
        if len(gal) < min_bodies:
            continue
        med = _median_sims(query, gal)
        if med is not None:
            out.append((pid, med))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def _nearest_any(
    query: np.ndarray,
    persons: Dict[str, PersonState],
    now: datetime,
    recent_minutes: int,
    staff_full: bool,
) -> float:
    best = -1.0
    for pid, st in persons.items():
        if st.is_staff:
            gal = _staff_bodies(st, now, recent_minutes, staff_full)
        else:
            gal = _customer_recent_bodies(st, now, recent_minutes)
        if not gal:
            continue
        med = _median_sims(query, gal)
        if med is not None and med > best:
            best = med
    return best


def _best_staff_sim(
    query: np.ndarray,
    persons: Dict[str, PersonState],
    now: datetime,
    recent_minutes: int,
    staff_full: bool,
) -> float:
    ranked = _rank_staff(query, persons, now, recent_minutes, staff_full, min_bodies=1)
    return ranked[0][1] if ranked else -1.0


def _overlap_seconds(
    a0: datetime, a1: datetime, b0: datetime, b1: datetime
) -> float:
    start = max(a0, b0)
    end = min(a1, b1)
    if end <= start:
        return 0.0
    return (end - start).total_seconds()


def _has_same_camera_overlap(
    st: PersonState,
    camera_id: Optional[str],
    probe_start: datetime,
    probe_end: datetime,
    min_sec: float,
) -> bool:
    if not camera_id:
        return False
    for cam, a, b in st.track_windows:
        if cam is None or cam != camera_id:
            continue
        if _overlap_seconds(a, b, probe_start, probe_end) >= min_sec:
            return True
    return False


def _add_body(st: PersonState, vec: np.ndarray, t: datetime) -> None:
    st.bodies.append(BodyEmb(vec=vec, t=_aware(t)))


def _add_track_window(
    st: PersonState,
    camera_id: Optional[str],
    start: datetime,
    end: datetime,
) -> None:
    a, b = _aware(start), _aware(end)
    if a is None or b is None:
        return
    if b < a:
        a, b = b, a
    st.track_windows.append((camera_id, a, b))
    if st.last_seen_at is None or b > st.last_seen_at:
        st.last_seen_at = b


async def load_galleries(db) -> Dict[str, PersonState]:
    r = await db.execute(
        text(
            """
            SELECT pi.id::text, pi.is_staff, pi.last_seen_at, pi.visit_count
            FROM person_identities pi
            """
        )
    )
    persons: Dict[str, PersonState] = {}
    for pid, is_staff, last_seen, visit_count in r.fetchall():
        persons[pid] = PersonState(
            pid=pid,
            is_staff=bool(is_staff),
            last_seen_at=_aware(last_seen),
            visit_count=int(visit_count or 1),
        )

    r = await db.execute(
        text(
            """
            SELECT person_identity_id::text, embedding, captured_at
            FROM person_embeddings
            WHERE embedding IS NOT NULL
            ORDER BY captured_at
            """
        )
    )
    for pid, emb_raw, cap in r.fetchall():
        if pid not in persons:
            continue
        e = _parse_embedding(emb_raw)
        if e is not None:
            persons[pid].bodies.append(BodyEmb(vec=e, t=_aware(cap)))

    r = await db.execute(
        text(
            """
            SELECT person_identity_id::text,
                   camera_id::text,
                   started_at,
                   COALESCE(ended_at, last_seen_at) AS ended
            FROM track_sessions
            WHERE person_identity_id IS NOT NULL
              AND started_at IS NOT NULL
            """
        )
    )
    for pid, cam, st_at, en in r.fetchall():
        if pid not in persons:
            continue
        _add_track_window(persons[pid], cam, st_at, en)
    return persons


async def load_null_tracks(
    db, start: datetime, end: datetime
) -> List[NullTrack]:
    r = await db.execute(
        text(
            """
            SELECT
                id::text,
                camera_id::text,
                started_at,
                COALESCE(ended_at, last_seen_at) AS ended,
                best_crop_path,
                COALESCE(total_frames, 0) AS total_frames,
                bbox_history
            FROM track_sessions
            WHERE person_identity_id IS NULL
              AND best_crop_path IS NOT NULL
              AND started_at >= :start
              AND started_at < :end
            ORDER BY started_at ASC, id ASC
            """
        ),
        {"start": start.astimezone(timezone.utc), "end": end.astimezone(timezone.utc)},
    )
    out: List[NullTrack] = []
    for row in r.fetchall():
        tid, cam, st_at, en, crop, frames, bbox_h = row
        a, b = _aware(st_at), _aware(en)
        if a is None:
            continue
        if b is None:
            b = a
        if b < a:
            a, b = b, a
        out.append(
            NullTrack(
                track_id=tid,
                camera_id=cam,
                started_at=a,
                ended_at=b,
                crop_path=crop,
                total_frames=int(frames or 0),
                bbox_history=bbox_h,
            )
        )
    return out


async def load_billing_for_tracks(
    db, track_ids: List[str]
) -> Dict[str, List[str]]:
    """track_id -> list of billing_interaction ids with null person."""
    if not track_ids:
        return {}
    r = await db.execute(
        text(
            """
            SELECT id::text, track_session_id::text
            FROM billing_interactions
            WHERE track_session_id::text = ANY(:tids)
              AND person_identity_id IS NULL
            """
        ),
        {"tids": track_ids},
    )
    out: Dict[str, List[str]] = {}
    for bi_id, tid in r.fetchall():
        out.setdefault(tid, []).append(bi_id)
    return out


async def count_nonstaff_purchasers(
    db, start: datetime, end: datetime
) -> int:
    r = await db.execute(
        text(
            """
            SELECT COUNT(DISTINCT bi.person_identity_id)
            FROM billing_interactions bi
            JOIN person_identities pi ON pi.id = bi.person_identity_id
            WHERE bi.entered_at >= :start
              AND bi.entered_at < :end
              AND bi.person_identity_id IS NOT NULL
              AND COALESCE(pi.is_staff, false) = false
            """
        ),
        {"start": start.astimezone(timezone.utc), "end": end.astimezone(timezone.utc)},
    )
    return int(r.scalar() or 0)


def try_attach(
    query: np.ndarray,
    persons: Dict[str, PersonState],
    now: datetime,
    camera_id: Optional[str],
    probe_start: datetime,
    probe_end: datetime,
    cnt: Counters,
    recent_minutes: int,
    attach_med: float,
    ambig: float,
    same_cam_min_sec: float,
    enable_same_cam: bool,
    staff_full: bool,
) -> Optional[str]:
    ranked_ns = _rank_customers(query, persons, now, recent_minutes)
    if not ranked_ns:
        cnt.attach_no_recent_gallery += 1
        return None

    staff_sim = _best_staff_sim(query, persons, now, recent_minutes, staff_full)
    same_cam_hits = 0

    for i, (top_pid, top_med) in enumerate(ranked_ns):
        if top_med < attach_med:
            break
        if i + 1 < len(ranked_ns):
            gap = top_med - ranked_ns[i + 1][1]
            if gap < ambig and ranked_ns[i + 1][1] >= attach_med:
                cnt.attach_reject_ambig += 1
                return None
        if staff_sim >= BODY_ONLY_ATTACH_STAFF_EXCLUSION:
            if (top_med - staff_sim) < BODY_ONLY_ATTACH_STAFF_GAP:
                cnt.attach_reject_staff += 1
                return None
        st = persons[top_pid]
        if enable_same_cam and _has_same_camera_overlap(
            st, camera_id, probe_start, probe_end, same_cam_min_sec
        ):
            same_cam_hits += 1
            continue
        _add_body(st, query, now)
        _add_track_window(st, camera_id, probe_start, probe_end)
        st.visit_count = int(st.visit_count or 0) + 1
        cnt.attach_nonstaff += 1
        return top_pid

    if same_cam_hits:
        cnt.attach_reject_same_cam += 1
    return None


def try_create(
    query: np.ndarray,
    quality: Optional[float],
    persons: Dict[str, PersonState],
    now: datetime,
    camera_id: Optional[str],
    probe_start: datetime,
    probe_end: datetime,
    cnt: Counters,
    recent_minutes: int,
    create_min_q: float,
    create_max_near: float,
    create_max_staff: float,
    staff_full: bool,
) -> Optional[str]:
    if quality is None or quality < create_min_q:
        cnt.quality_gate_fail += 1
        return None
    nearest = _nearest_any(query, persons, now, recent_minutes, staff_full)
    if nearest >= create_max_near:
        cnt.create_reject_near += 1
        return None
    staff_sim = _best_staff_sim(query, persons, now, recent_minutes, staff_full)
    if staff_sim >= create_max_staff:
        cnt.create_reject_staff += 1
        return None
    pid = str(uuid.uuid4())
    st = PersonState(pid=pid, is_staff=False, is_virtual=True, visit_count=1)
    _add_body(st, query, now)
    _add_track_window(st, camera_id, probe_start, probe_end)
    persons[pid] = st
    cnt.create_new += 1
    return pid


async def db_attach(
    db,
    pid: str,
    track: NullTrack,
    emb: np.ndarray,
    quality: float,
    billing_ids: List[str],
    cnt: Counters,
) -> None:
    emb_str = str(emb.tolist())
    seen = track.ended_at
    await db.execute(
        text(
            """
            UPDATE person_identities
            SET last_seen_at = GREATEST(last_seen_at, :seen),
                visit_count = visit_count + 1,
                updated_at = now()
            WHERE id = CAST(:pid AS uuid)
            """
        ),
        {"pid": pid, "seen": seen},
    )
    await db.execute(
        text(
            """
            INSERT INTO person_embeddings (
                id, person_identity_id, embedding, camera_id,
                crop_quality, crop_path, captured_at, created_at, updated_at
            ) VALUES (
                gen_random_uuid(),
                CAST(:pid AS uuid),
                CAST(:emb AS vector),
                CAST(:cam AS uuid),
                :q,
                :path,
                :cap,
                now(),
                now()
            )
            """
        ),
        {
            "pid": pid,
            "emb": emb_str,
            "cam": track.camera_id,
            "q": float(quality if quality is not None else 0.0),
            "path": track.crop_path,
            "cap": track.started_at,
        },
    )
    await db.execute(
        text(
            """
            UPDATE track_sessions
            SET person_identity_id = CAST(:pid AS uuid)
            WHERE id = CAST(:tid AS uuid)
            """
        ),
        {"pid": pid, "tid": track.track_id},
    )
    if billing_ids:
        await db.execute(
            text(
                """
                UPDATE billing_interactions
                SET person_identity_id = CAST(:pid AS uuid),
                    updated_at = now()
                WHERE id::text = ANY(:bids)
                  AND person_identity_id IS NULL
                """
            ),
            {"pid": pid, "bids": billing_ids},
        )
        cnt.billing_updated += len(billing_ids)


async def db_create(
    db,
    pid: str,
    track: NullTrack,
    emb: np.ndarray,
    quality: float,
    billing_ids: List[str],
    cnt: Counters,
) -> None:
    emb_str = str(emb.tolist())
    meta = json.dumps({"body_only_create": True, "created_tier": "body_only"})
    await db.execute(
        text(
            """
            INSERT INTO person_identities (
                id, first_seen_at, last_seen_at, visit_count,
                is_anonymous, is_staff, metadata_json, created_at, updated_at
            ) VALUES (
                CAST(:pid AS uuid),
                :first_seen,
                :last_seen,
                1,
                true,
                false,
                CAST(:meta AS jsonb),
                now(),
                now()
            )
            """
        ),
        {
            "pid": pid,
            "first_seen": track.started_at,
            "last_seen": track.ended_at,
            "meta": meta,
        },
    )
    await db.execute(
        text(
            """
            INSERT INTO person_embeddings (
                id, person_identity_id, embedding, camera_id,
                crop_quality, crop_path, captured_at, created_at, updated_at
            ) VALUES (
                gen_random_uuid(),
                CAST(:pid AS uuid),
                CAST(:emb AS vector),
                CAST(:cam AS uuid),
                :q,
                :path,
                :cap,
                now(),
                now()
            )
            """
        ),
        {
            "pid": pid,
            "emb": emb_str,
            "cam": track.camera_id,
            "q": float(quality if quality is not None else 0.0),
            "path": track.crop_path,
            "cap": track.started_at,
        },
    )
    await db.execute(
        text(
            """
            UPDATE track_sessions
            SET person_identity_id = CAST(:pid AS uuid)
            WHERE id = CAST(:tid AS uuid)
            """
        ),
        {"pid": pid, "tid": track.track_id},
    )
    if billing_ids:
        await db.execute(
            text(
                """
                UPDATE billing_interactions
                SET person_identity_id = CAST(:pid AS uuid),
                    updated_at = now()
                WHERE id::text = ANY(:bids)
                  AND person_identity_id IS NULL
                """
            ),
            {"pid": pid, "bids": billing_ids},
        )
        cnt.billing_updated += len(billing_ids)


async def run(start: datetime, end: datetime, apply: bool) -> None:
    settings = get_settings()
    recent_minutes = int(settings.RECENT_WINDOW_MINUTES)
    attach_med = float(settings.RECENT_BODY_SINGLE_MATCH_THRESHOLD)
    ambig = float(settings.BODY_MATCH_AMBIGUITY)
    create_min_q = float(settings.BODY_ONLY_CREATE_MIN_QUALITY)
    create_max_near = float(settings.BODY_ONLY_CREATE_MAX_NEAREST_SIM)
    create_max_staff = float(settings.BODY_ONLY_CREATE_MAX_STAFF_SIM)
    staff_full = bool(settings.STAFF_BODY_USE_FULL_GALLERY)
    enable_same_cam = bool(settings.ENABLE_SAME_CAMERA_OVERLAP_GATE)
    same_cam_min_sec = float(settings.SAME_CAMERA_OVERLAP_MIN_SECONDS)

    mode = "APPLY" if apply else "DRY-RUN"
    print(f"\n{'=' * 78}")
    print(f"  body-only identity backfill -- {mode}")
    print(f"  Window: {start.isoformat()} -> {end.isoformat()}")
    print(f"  recent={recent_minutes}m attach_med>={attach_med} ambig={ambig}")
    print(
        f"  create_q>={create_min_q} near<{create_max_near} staff<{create_max_staff}"
    )
    print(
        f"  staff_full={staff_full} same_cam={enable_same_cam} "
        f"min_overlap={same_cam_min_sec}s lock={IDENTITY_ADVISORY_LOCK_KEY}"
    )
    print(f"{'=' * 78}\n")

    print("  Loading OSNet...")
    osnet = get_shared_extractor()
    minio = get_client()
    print("  OSNet OK.\n")

    cnt = Counters()
    emb_cache: Dict[str, Optional[np.ndarray]] = {}
    sim_new_person_ids: set = set()
    sim_billing_person: Dict[str, str] = {}

    async with AsyncSessionLocal() as db:
        if apply:
            await db.execute(
                text(
                    f"SELECT pg_advisory_xact_lock({int(IDENTITY_ADVISORY_LOCK_KEY)})"
                )
            )

        baseline = await count_nonstaff_purchasers(db, start, end)
        persons = await load_galleries(db)
        tracks = await load_null_tracks(db, start, end)
        cnt.null_tracks = len(tracks)
        track_ids = [t.track_id for t in tracks]
        billing_map = await load_billing_for_tracks(db, track_ids)

        # Billing rows already assigned in window (for after-count dry-run)
        r = await db.execute(
            text(
                """
                SELECT bi.id::text, bi.person_identity_id::text,
                       COALESCE(pi.is_staff, false)
                FROM billing_interactions bi
                LEFT JOIN person_identities pi ON pi.id = bi.person_identity_id
                WHERE bi.entered_at >= :start
                  AND bi.entered_at < :end
                  AND bi.person_identity_id IS NOT NULL
                """
            ),
            {
                "start": start.astimezone(timezone.utc),
                "end": end.astimezone(timezone.utc),
            },
        )
        assigned_billing_staff: Dict[str, bool] = {}
        for bi_id, pid, is_staff in r.fetchall():
            assigned_billing_staff[bi_id] = bool(is_staff)
            sim_billing_person[bi_id] = pid

        n_staff = sum(1 for p in persons.values() if p.is_staff)
        n_body = sum(len(p.bodies) for p in persons.values())
        print(
            f"  Gallery: persons={len(persons)} staff={n_staff} "
            f"body_embs={n_body}"
        )
        print(f"  Null tracks with crop in window: {len(tracks)}")
        print(f"  Baseline DISTINCT non-staff purchasers: {baseline}")
        print(
            f"  Null-person billing rows linked to those tracks: "
            f"{sum(len(v) for v in billing_map.values())}"
        )

        for i, tr in enumerate(tracks):
            if (i + 1) % 50 == 0 or i == 0:
                print(
                    f"  [{i+1}/{len(tracks)}] attach={cnt.attach_nonstaff} "
                    f"create={cnt.create_new} unass={cnt.left_unassigned} "
                    f"same_cam={cnt.attach_reject_same_cam}"
                )

            quality = _quality_from_row(tr.total_frames, tr.bbox_history)
            if not tr.crop_path:
                cnt.no_crop += 1
                cnt.left_unassigned += 1
                continue

            cache_key = tr.track_id
            if cache_key not in emb_cache:
                img = _download_crop(minio, tr.crop_path)
                emb_cache[cache_key] = (
                    osnet.extract(img) if img is not None else None
                )
            query = emb_cache[cache_key]
            if query is None:
                cnt.extract_fail += 1
                cnt.left_unassigned += 1
                continue
            q = _parse_embedding(query)
            if q is None:
                cnt.extract_fail += 1
                cnt.left_unassigned += 1
                continue
            query = q

            now = tr.started_at
            bid_list = billing_map.get(tr.track_id, [])

            attached = try_attach(
                query,
                persons,
                now,
                tr.camera_id,
                tr.started_at,
                tr.ended_at,
                cnt,
                recent_minutes,
                attach_med,
                ambig,
                same_cam_min_sec,
                enable_same_cam,
                staff_full,
            )
            if attached is not None:
                for bi in bid_list:
                    sim_billing_person[bi] = attached
                if apply:
                    await db_attach(
                        db,
                        attached,
                        tr,
                        query,
                        quality if quality is not None else 0.0,
                        bid_list,
                        cnt,
                    )
                continue

            created = try_create(
                query,
                quality,
                persons,
                now,
                tr.camera_id,
                tr.started_at,
                tr.ended_at,
                cnt,
                recent_minutes,
                create_min_q,
                create_max_near,
                create_max_staff,
                staff_full,
            )
            if created is not None:
                sim_new_person_ids.add(created)
                for bi in bid_list:
                    sim_billing_person[bi] = created
                if apply:
                    await db_create(
                        db,
                        created,
                        tr,
                        query,
                        quality if quality is not None else 0.0,
                        bid_list,
                        cnt,
                    )
                continue

            cnt.left_unassigned += 1

        if apply:
            await db.commit()
            after = await count_nonstaff_purchasers(db, start, end)
        else:
            # Simulated after: assigned non-staff + newly resolved null billing
            # to non-staff persons
            nonstaff_pids = set()
            for bi_id, pid in sim_billing_person.items():
                if bi_id in assigned_billing_staff:
                    if not assigned_billing_staff[bi_id]:
                        nonstaff_pids.add(pid)
                else:
                    st = persons.get(pid)
                    if st is not None and not st.is_staff:
                        nonstaff_pids.add(pid)
                    elif pid in sim_new_person_ids:
                        nonstaff_pids.add(pid)
            after = len(nonstaff_pids)

    print(f"\n{'-' * 78}")
    print(f"  RESULTS -- {mode}")
    print(f"{'-' * 78}")
    print(f"  Baseline DISTINCT non-staff purchasers: {baseline}")
    print(f"  After    DISTINCT non-staff purchasers: {after}")
    print(f"  Delta:                                  {after - baseline:+d}")
    print()
    print(f"  null_tracks:              {cnt.null_tracks}")
    print(f"  attach_nonstaff:          {cnt.attach_nonstaff}")
    print(f"  create_new:               {cnt.create_new}")
    print(f"  left_unassigned:          {cnt.left_unassigned}")
    print(f"  no_crop:                  {cnt.no_crop}")
    print(f"  extract_fail:             {cnt.extract_fail}")
    print(f"  quality_gate_fail:        {cnt.quality_gate_fail}")
    print(f"  attach_no_recent_gallery: {cnt.attach_no_recent_gallery}")
    print(f"  attach_reject_staff:      {cnt.attach_reject_staff}")
    print(f"  attach_reject_ambig:      {cnt.attach_reject_ambig}")
    print(f"  attach_reject_same_cam:   {cnt.attach_reject_same_cam}")
    print(f"  create_reject_near:       {cnt.create_reject_near}")
    print(f"  create_reject_staff:      {cnt.create_reject_staff}")
    if apply:
        print(f"  billing_updated:          {cnt.billing_updated}")
    else:
        print("  DRY-RUN -- no writes. Re-run with --apply.")
    print(f"{'=' * 78}\n")


def _parse_days(s: Optional[str]) -> Tuple[date, date]:
    now_ist = datetime.now(IST)
    today = now_ist.date()
    yesterday = today - timedelta(days=1)
    if not s:
        return yesterday, today
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) == 1:
        d = date.fromisoformat(parts[0])
        return d, d
    d0 = date.fromisoformat(parts[0])
    d1 = date.fromisoformat(parts[-1])
    if d1 < d0:
        d0, d1 = d1, d0
    return d0, d1


def main() -> None:
    p = argparse.ArgumentParser(
        description="Backfill body-only identity attach/create for null tracks"
    )
    p.add_argument(
        "--days",
        type=str,
        default=None,
        help="YYYY-MM-DD,YYYY-MM-DD IST inclusive (default: yesterday,today)",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Write DB changes (default dry-run)",
    )
    args = p.parse_args()
    d0, d1 = _parse_days(args.days)
    start = datetime.combine(d0, time.min, tzinfo=IST)
    end = datetime.combine(d1 + timedelta(days=1), time.min, tzinfo=IST)
    asyncio.run(run(start, end, apply=bool(args.apply)))


if __name__ == "__main__":
    main()
