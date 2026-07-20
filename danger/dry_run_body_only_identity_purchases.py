#!/usr/bin/env python3
"""
dry_run_body_only_identity_purchases.py

Retrospective dry-run of faceless body-only identity attach/create.

RECENT-WINDOW BODY MATCHING (2026-07-20):
  - Customer body match uses ONLY body embeddings with captured_at inside
    [T - RECENT_WINDOW, T]. Old clothing-day bodies stay in the gallery but
    are not queried and are never "rejected/deleted" for mismatch.
  - No match in recent body cluster → do not merge/attach (leave unassigned
    or create if create gates pass).
  - Staff: must be activity-recent (track or emb in window), but body gallery
    may use ALL stored bodies (uniform stable across days).

NO DB writes. Quality-only gates (no torso required).

Usage:
    PYTHONPATH=/gmr/gmr venv/bin/python danger/dry_run_body_only_identity_purchases.py --preset both
"""
from __future__ import annotations

import argparse
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import text

from app.core.db.session import AsyncSessionLocal
from app.modules.storage.minio_client import BUCKET_PREFIX, get_client
from app.modules.reid.osnet_extractor import get_shared_extractor

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore

IST = ZoneInfo("Asia/Kolkata")

# ── Proposed thresholds (SESSION_HANDOFF_2026-07-20) ──────────────────────
RECENT_WINDOW_MINUTES = 5
BODY_ONLY_CREATE_MIN_QUALITY = 0.55
BODY_ONLY_CREATE_MAX_NEAREST_SIM = 0.45
BODY_ONLY_CREATE_MAX_STAFF_SIM = 0.48
BODY_ONLY_ATTACH_NONSTAFF_MEDIAN = 0.55
BODY_ONLY_ATTACH_STAFF_EXCLUSION = 0.50
BODY_ONLY_ATTACH_STAFF_GAP = 0.05
BODY_MATCH_AMBIGUITY = 0.03
BODY_ONLY_MIN_BODIES_GALLERY = 2
# Staff reattach / staff exclusion: full lifetime body gallery if activity-recent
STAFF_USE_FULL_BODY_GALLERY = True
ENABLE_CUSTOMER_RECENT_MERGE = True
CUSTOMER_MERGE_MEDIAN = 0.60
LEGACY_MIN_FRAMES = 4
LEGACY_DEFAULT_QUALITY = 0.60

PRESETS = {
    "today_to_13": (
        datetime(2026, 7, 20, 0, 0, 0, tzinfo=IST),
        datetime(2026, 7, 20, 13, 0, 0, tzinfo=IST),
        "2026-07-20 00:00-13:00 IST (staff claimed 54, baseline 34)",
    ),
    "yesterday": (
        datetime(2026, 7, 19, 0, 0, 0, tzinfo=IST),
        datetime(2026, 7, 20, 0, 0, 0, tzinfo=IST),
        "2026-07-19 full day IST (staff claimed 110, baseline 46)",
    ),
}


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
    # ALL body embeddings kept forever (never deleted for mismatch)
    bodies: List[BodyEmb] = field(default_factory=list)
    # Track activity intervals (start, end) for true recent activity
    track_windows: List[Tuple[datetime, datetime]] = field(default_factory=list)
    is_virtual: bool = False


@dataclass
class Event:
    t: datetime
    bi_id: str
    track_id: Optional[str]
    person_id: Optional[str]
    is_staff: bool
    crop_path: Optional[str]
    total_frames: int
    bbox_history: object
    camera_id: Optional[str]


@dataclass
class Counters:
    baseline_assigned: int = 0
    null_billing: int = 0
    keep_assigned: int = 0
    customer_merge: int = 0
    attach_nonstaff: int = 0
    create_virtual: int = 0
    left_unassigned: int = 0
    no_crop: int = 0
    extract_fail: int = 0
    quality_gate_fail: int = 0
    attach_reject_staff: int = 0
    attach_reject_ambig: int = 0
    attach_no_recent_gallery: int = 0
    create_reject_near: int = 0
    create_reject_staff: int = 0
    sum_n_recent_persons: int = 0
    n_scored_events: int = 0


def _window_bounds(now: datetime) -> Tuple[datetime, datetime]:
    w = timedelta(minutes=RECENT_WINDOW_MINUTES)
    return now - w, now


def _is_activity_recent(st: PersonState, now: datetime) -> bool:
    """True if person has track activity or any body emb in [T-window, T]."""
    t0, t1 = _window_bounds(now)
    for a, b in st.track_windows:
        if a <= t1 and b >= t0:
            return True
    for be in st.bodies:
        if t0 <= be.t <= t1:
            return True
    return False


def _customer_recent_bodies(st: PersonState, now: datetime) -> List[np.ndarray]:
    """Bodies with captured_at in recent window only. Never drops old rows."""
    t0, t1 = _window_bounds(now)
    return [be.vec for be in st.bodies if t0 <= be.t <= t1]


def _staff_bodies(st: PersonState, now: datetime) -> List[np.ndarray]:
    """Staff: activity-recent required; gallery = full lifetime or recent."""
    if not _is_activity_recent(st, now):
        return []
    if STAFF_USE_FULL_BODY_GALLERY:
        return [be.vec for be in st.bodies]
    return _customer_recent_bodies(st, now)


def _rank_customers(
    query: np.ndarray,
    persons: Dict[str, PersonState],
    now: datetime,
    min_bodies: int = BODY_ONLY_MIN_BODIES_GALLERY,
) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for pid, st in persons.items():
        if st.is_staff:
            continue
        gal = _customer_recent_bodies(st, now)
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
    min_bodies: int = 1,
) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for pid, st in persons.items():
        if not st.is_staff:
            continue
        gal = _staff_bodies(st, now)
        if len(gal) < min_bodies:
            continue
        med = _median_sims(query, gal)
        if med is not None:
            out.append((pid, med))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def _nearest_any_customer_or_staff(
    query: np.ndarray,
    persons: Dict[str, PersonState],
    now: datetime,
) -> Tuple[Optional[str], float]:
    """Create gate: nearest among persons with usable recent(customer)/staff gallery."""
    best_pid, best = None, -1.0
    # customers: recent bodies only, n>=1 for CREATE nearest
    for pid, st in persons.items():
        if st.is_staff:
            continue
        gal = _customer_recent_bodies(st, now)
        if not gal:
            continue
        med = _median_sims(query, gal)
        if med is not None and med > best:
            best, best_pid = med, pid
    # staff with full gallery if active
    for pid, st in persons.items():
        if not st.is_staff:
            continue
        gal = _staff_bodies(st, now)
        if not gal:
            continue
        med = _median_sims(query, gal)
        if med is not None and med > best:
            best, best_pid = med, pid
    return best_pid, best


def _best_staff_sim(query: np.ndarray, persons: Dict[str, PersonState], now: datetime) -> float:
    ranked = _rank_staff(query, persons, now, min_bodies=1)
    return ranked[0][1] if ranked else -1.0


def _count_recent_persons(persons: Dict[str, PersonState], now: datetime) -> int:
    return sum(1 for st in persons.values() if _is_activity_recent(st, now))


def _add_body(st: PersonState, vec: np.ndarray, t: datetime) -> None:
    """Append body — never removes older bodies for mismatch."""
    st.bodies.append(BodyEmb(vec=vec, t=_aware(t)))


def _add_track_activity(st: PersonState, t: datetime) -> None:
    """Mark a short activity blip around event time (billing track proxy)."""
    t = _aware(t)
    st.track_windows.append((t - timedelta(seconds=30), t + timedelta(seconds=30)))


async def load_galleries(db) -> Dict[str, PersonState]:
    r = await db.execute(
        text(
            """
            SELECT pi.id::text, pi.is_staff
            FROM person_identities pi
            """
        )
    )
    persons: Dict[str, PersonState] = {}
    for pid, is_staff in r.fetchall():
        persons[pid] = PersonState(pid=pid, is_staff=bool(is_staff))

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

    # True activity from tracks (started_at .. ended/last_seen)
    r = await db.execute(
        text(
            """
            SELECT person_identity_id::text,
                   started_at,
                   COALESCE(ended_at, last_seen_at) AS ended
            FROM track_sessions
            WHERE person_identity_id IS NOT NULL
              AND started_at IS NOT NULL
            """
        )
    )
    for pid, st_at, en in r.fetchall():
        if pid not in persons:
            continue
        a, b = _aware(st_at), _aware(en)
        if a is None or b is None:
            continue
        if b < a:
            a, b = b, a
        persons[pid].track_windows.append((a, b))
    return persons


async def load_billing_events(db, start: datetime, end: datetime) -> List[Event]:
    r = await db.execute(
        text(
            """
            SELECT
                bi.id::text,
                bi.entered_at,
                bi.person_identity_id::text,
                COALESCE(pi.is_staff, false) AS is_staff,
                bi.track_session_id::text,
                ts.best_crop_path,
                COALESCE(ts.total_frames, 0) AS total_frames,
                ts.bbox_history,
                ts.camera_id::text
            FROM billing_interactions bi
            LEFT JOIN person_identities pi ON pi.id = bi.person_identity_id
            LEFT JOIN track_sessions ts ON ts.id = bi.track_session_id
            WHERE bi.entered_at >= :start
              AND bi.entered_at < :end
            ORDER BY bi.entered_at ASC, bi.id ASC
            """
        ),
        {"start": start.astimezone(timezone.utc), "end": end.astimezone(timezone.utc)},
    )
    events: List[Event] = []
    for row in r.fetchall():
        (
            bi_id,
            entered_at,
            person_id,
            is_staff,
            track_id,
            crop_path,
            total_frames,
            bbox_history,
            camera_id,
        ) = row
        t = _aware(entered_at)
        events.append(
            Event(
                t=t,
                bi_id=bi_id,
                track_id=track_id,
                person_id=person_id,
                is_staff=bool(is_staff),
                crop_path=crop_path,
                total_frames=int(total_frames or 0),
                bbox_history=bbox_history,
                camera_id=camera_id,
            )
        )
    return events


def try_customer_merge(
    query: np.ndarray,
    assigned_pid: str,
    persons: Dict[str, PersonState],
    now: datetime,
    cnt: Counters,
) -> str:
    """Merge only if recent-window body match is strong — never drop old bodies."""
    if not ENABLE_CUSTOMER_RECENT_MERGE:
        return assigned_pid
    st = persons.get(assigned_pid)
    if st is None or st.is_staff:
        return assigned_pid
    ranked = _rank_customers(query, persons, now)
    others = [(p, m) for p, m in ranked if p != assigned_pid]
    if not others:
        return assigned_pid
    top_pid, top_med = others[0]
    if top_med < CUSTOMER_MERGE_MEDIAN:
        return assigned_pid
    if len(others) >= 2 and (others[0][1] - others[1][1]) < BODY_MATCH_AMBIGUITY:
        return assigned_pid
    # Conceptual merge: move recent query onto winner; KEEP all old bodies on both
    # (sim only reassigns bill id). Append query to winner recent gallery.
    winner = persons[top_pid]
    _add_body(winner, query, now)
    _add_track_activity(winner, now)
    cnt.customer_merge += 1
    return top_pid


def try_attach(
    query: np.ndarray,
    persons: Dict[str, PersonState],
    now: datetime,
    cnt: Counters,
) -> Optional[str]:
    ranked_ns = _rank_customers(query, persons, now)
    if not ranked_ns:
        cnt.attach_no_recent_gallery += 1
        return None
    top_pid, top_med = ranked_ns[0]
    if top_med < BODY_ONLY_ATTACH_NONSTAFF_MEDIAN:
        # Recent-window body does not match → do not merge (no body "reject/delete")
        return None
    if len(ranked_ns) >= 2 and (ranked_ns[0][1] - ranked_ns[1][1]) < BODY_MATCH_AMBIGUITY:
        cnt.attach_reject_ambig += 1
        return None
    staff_sim = _best_staff_sim(query, persons, now)
    if staff_sim >= BODY_ONLY_ATTACH_STAFF_EXCLUSION:
        if (top_med - staff_sim) < BODY_ONLY_ATTACH_STAFF_GAP:
            cnt.attach_reject_staff += 1
            return None
    st = persons[top_pid]
    # Append; never remove older bodies
    _add_body(st, query, now)
    _add_track_activity(st, now)
    cnt.attach_nonstaff += 1
    return top_pid


def try_create(
    query: np.ndarray,
    quality: Optional[float],
    persons: Dict[str, PersonState],
    now: datetime,
    cnt: Counters,
) -> Optional[str]:
    if quality is None or quality < BODY_ONLY_CREATE_MIN_QUALITY:
        cnt.quality_gate_fail += 1
        return None
    _, nearest = _nearest_any_customer_or_staff(query, persons, now)
    if nearest >= BODY_ONLY_CREATE_MAX_NEAREST_SIM:
        cnt.create_reject_near += 1
        return None
    staff_sim = _best_staff_sim(query, persons, now)
    if staff_sim >= BODY_ONLY_CREATE_MAX_STAFF_SIM:
        cnt.create_reject_staff += 1
        return None
    vid = f"virt-{uuid.uuid4()}"
    persons[vid] = PersonState(pid=vid, is_staff=False, is_virtual=True)
    _add_body(persons[vid], query, now)
    _add_track_activity(persons[vid], now)
    cnt.create_virtual += 1
    return vid


async def run_window(start: datetime, end: datetime, label: str) -> None:
    print(f"\n{'=' * 78}")
    print(f"  DRY-RUN body-only identity / purchases")
    print(f"  MODE: recent-window body gallery only (staff full gallery if active)")
    print(f"  Window: {label}")
    print(f"  {start.isoformat()} → {end.isoformat()}")
    print(f"{'=' * 78}\n")

    print("  Loading OSNet...")
    osnet = get_shared_extractor()
    minio = get_client()
    print("  OSNet OK.\n")

    async with AsyncSessionLocal() as db:
        persons = await load_galleries(db)
        events = await load_billing_events(db, start, end)

    n_staff = sum(1 for p in persons.values() if p.is_staff)
    n_cust = sum(1 for p in persons.values() if not p.is_staff)
    n_body = sum(len(p.bodies) for p in persons.values())
    n_trk = sum(len(p.track_windows) for p in persons.values())
    print(
        f"  Gallery: persons={len(persons)} staff={n_staff} nonstaff={n_cust} "
        f"body_embs={n_body} track_windows={n_trk}"
    )
    print(f"  Billing rows in window: {len(events)}")
    print(
        f"  Logic: customer match body.captured_at in last {RECENT_WINDOW_MINUTES}m; "
        f"no merge if no recent match; never drop old bodies"
    )
    print(
        f"  Staff: activity-recent + "
        f"{'FULL body gallery' if STAFF_USE_FULL_BODY_GALLERY else 'recent bodies only'}"
    )

    baseline_ids = {e.person_id for e in events if e.person_id and not e.is_staff}
    baseline_staff = {e.person_id for e in events if e.person_id and e.is_staff}
    null_events = [e for e in events if not e.person_id]
    print(f"  Baseline DISTINCT non-staff purchasers: {len(baseline_ids)}")
    print(f"  Baseline DISTINCT staff on billing:   {len(baseline_staff)}")
    print(f"  Null-person billing rows:             {len(null_events)}")

    emb_cache: Dict[str, Optional[np.ndarray]] = {}
    cnt = Counters()
    cnt.baseline_assigned = len(baseline_ids)
    cnt.null_billing = len(null_events)
    resolved: List[Tuple[str, Optional[str], bool]] = []

    for i, ev in enumerate(events):
        if (i + 1) % 50 == 0 or i == 0:
            print(
                f"  [{i+1}/{len(events)}] attach={cnt.attach_nonstaff} "
                f"create={cnt.create_virtual} unass={cnt.left_unassigned} "
                f"near_rej={cnt.create_reject_near}"
            )

        if ev.person_id:
            is_staff = ev.is_staff or (
                ev.person_id in persons and persons[ev.person_id].is_staff
            )
            pid = ev.person_id
            if pid in persons:
                _add_track_activity(persons[pid], ev.t)
            if not is_staff and ev.crop_path:
                cache_key = ev.track_id or ev.crop_path
                if cache_key not in emb_cache:
                    img = _download_crop(minio, ev.crop_path)
                    emb_cache[cache_key] = (
                        osnet.extract(img) if img is not None else None
                    )
                q = emb_cache[cache_key]
                if q is not None and pid in persons:
                    # Grow recent gallery only (append timed body; keep history)
                    _add_body(persons[pid], q, ev.t)
                    pid = try_customer_merge(q, pid, persons, ev.t, cnt)
            cnt.keep_assigned += 1
            if pid in persons:
                is_staff = persons[pid].is_staff
            resolved.append((ev.bi_id, pid, is_staff))
            continue

        # NULL person path
        quality = _quality_from_row(ev.total_frames, ev.bbox_history)
        if not ev.crop_path:
            cnt.no_crop += 1
            cnt.left_unassigned += 1
            resolved.append((ev.bi_id, None, False))
            continue

        cache_key = ev.track_id or ev.crop_path
        if cache_key not in emb_cache:
            img = _download_crop(minio, ev.crop_path)
            emb_cache[cache_key] = osnet.extract(img) if img is not None else None
        query = emb_cache[cache_key]
        if query is None:
            cnt.extract_fail += 1
            cnt.left_unassigned += 1
            resolved.append((ev.bi_id, None, False))
            continue

        cnt.sum_n_recent_persons += _count_recent_persons(persons, ev.t)
        cnt.n_scored_events += 1

        attached = try_attach(query, persons, ev.t, cnt)
        if attached is not None:
            resolved.append((ev.bi_id, attached, False))
            continue

        created = try_create(query, quality, persons, ev.t, cnt)
        if created is not None:
            resolved.append((ev.bi_id, created, False))
            continue

        cnt.left_unassigned += 1
        resolved.append((ev.bi_id, None, False))

    sim_nonstaff = {pid for _, pid, is_staff in resolved if pid and not is_staff}
    still_null = sum(1 for _, pid, _ in resolved if pid is None)
    virtual_in = sum(1 for pid in sim_nonstaff if str(pid).startswith("virt-"))
    real_in = len(sim_nonstaff) - virtual_in
    avg_recent = (
        cnt.sum_n_recent_persons / cnt.n_scored_events if cnt.n_scored_events else 0.0
    )

    print(f"\n{'─' * 78}")
    print(f"  RESULTS — {label}")
    print(f"{'─' * 78}")
    print(f"  Baseline DISTINCT non-staff purchasers:  {len(baseline_ids)}")
    print(f"  Simulated DISTINCT non-staff purchasers: {len(sim_nonstaff)}")
    print(f"    of which real gallery ids:             {real_in}")
    print(f"    of which virtual created:              {virtual_in}")
    print(f"  Delta (sim - baseline):                  {len(sim_nonstaff) - len(baseline_ids):+d}")
    print(f"  Billing rows still null:                 {still_null}")
    print(f"  Avg activity-recent persons at null events: {avg_recent:.1f}")
    print()
    print(f"  Decisions:")
    print(f"    keep_assigned:           {cnt.keep_assigned}")
    print(f"    customer_merge:          {cnt.customer_merge}")
    print(f"    attach_nonstaff:         {cnt.attach_nonstaff}")
    print(f"    create_virtual:          {cnt.create_virtual}")
    print(f"    left_unassigned:         {cnt.left_unassigned}")
    print(f"    no_crop:                 {cnt.no_crop}")
    print(f"    extract_fail:            {cnt.extract_fail}")
    print(f"    quality_gate_fail:       {cnt.quality_gate_fail}")
    print(f"    attach_no_recent_gal:    {cnt.attach_no_recent_gallery}")
    print(f"    attach_reject_staff:     {cnt.attach_reject_staff}")
    print(f"    attach_reject_ambig:     {cnt.attach_reject_ambig}")
    print(f"    create_reject_near:      {cnt.create_reject_near}")
    print(f"    create_reject_staff:     {cnt.create_reject_staff}")
    print()
    print(f"  Thresholds:")
    print(f"    window={RECENT_WINDOW_MINUTES}m  attach_med>={BODY_ONLY_ATTACH_NONSTAFF_MEDIAN}")
    print(f"    staff_excl={BODY_ONLY_ATTACH_STAFF_EXCLUSION} gap={BODY_ONLY_ATTACH_STAFF_GAP}")
    print(f"    ambig={BODY_MATCH_AMBIGUITY}  create_q>={BODY_ONLY_CREATE_MIN_QUALITY}")
    print(f"    create_near<{BODY_ONLY_CREATE_MAX_NEAREST_SIM} staff<{BODY_ONLY_CREATE_MAX_STAFF_SIM}")
    print(f"    min_bodies={BODY_ONLY_MIN_BODIES_GALLERY} customer_merge>={CUSTOMER_MERGE_MEDIAN}")
    print(f"    staff_full_gallery={STAFF_USE_FULL_BODY_GALLERY}")
    print(f"{'=' * 78}\n")


async def main_async(args) -> None:
    windows: List[Tuple[datetime, datetime, str]] = []
    if args.preset == "both":
        for key in ("today_to_13", "yesterday"):
            windows.append(PRESETS[key])
    elif args.preset in PRESETS:
        windows.append(PRESETS[args.preset])
    elif args.start and args.end:
        start = datetime.fromisoformat(args.start).replace(tzinfo=IST)
        end = datetime.fromisoformat(args.end).replace(tzinfo=IST)
        windows.append((start, end, f"custom {args.start} → {args.end} IST"))
    else:
        print("Provide --preset both|today_to_13|yesterday or --start/--end")
        sys.exit(2)

    for start, end, label in windows:
        await run_window(start, end, label)


def main():
    global RECENT_WINDOW_MINUTES
    p = argparse.ArgumentParser(
        description="Dry-run faceless body-only identity (recent-window body gallery)"
    )
    p.add_argument(
        "--preset",
        choices=["both", "today_to_13", "yesterday"],
        default=None,
    )
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument(
        "--window-min",
        type=int,
        default=None,
        help="Override RECENT_WINDOW_MINUTES (default 5)",
    )
    args = p.parse_args()
    if args.window_min is not None:
        RECENT_WINDOW_MINUTES = int(args.window_min)
    if args.preset is None and not (args.start and args.end):
        args.preset = "both"
    import asyncio

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
