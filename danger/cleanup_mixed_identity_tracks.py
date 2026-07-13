#!/usr/bin/env python3
"""
cleanup_mixed_identity_tracks.py — orphan tracks that don't fit a person identity.

Beyond same-camera concurrent overlap (see cleanup_same_camera_overlap.py), this
handles sequential multi-person pollution:

1) Same-camera temporal primary cluster (longest-first independent set) — same as
   cleanup_same_camera_overlap.
2) Gender veto: majority M/F among remaining tracks; orphan opposite-gender tracks.
3) Face-cluster fit (--with-faces): InsightFace on track best_crop; orphan if no face
   or median sim to person face gallery < FACE_CONTAMINATION_THRESHOLD (0.35).

Never deletes person_identities. Dry-run default.

Usage:
  PYTHONPATH=/gmr/gmr venv/bin/python danger/cleanup_mixed_identity_tracks.py \\
    --ids c7bdce30-e1a6-4143-9dd1-614993bcbada
  PYTHONPATH=/gmr/gmr venv/bin/python danger/cleanup_mixed_identity_tracks.py \\
    --ids UUID --with-faces --apply
  PYTHONPATH=/gmr/gmr venv/bin/python danger/cleanup_mixed_identity_tracks.py --staff-only
"""

from __future__ import annotations

import argparse
import asyncio
import io
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import UUID

import numpy as np
from sqlalchemy import text

from app.config import get_settings
from app.core.db.session import AsyncSessionLocal

FACE_CONTAM_DEFAULT = 0.35
MIN_OVERLAP_DEFAULT = 1.0


@dataclass
class Track:
    id: str
    camera_id: str
    started_at: datetime
    ended_at: datetime
    gender: Optional[str]
    duration_s: float
    best_crop_path: Optional[str]


@dataclass
class PersonPlan:
    pid: str
    is_staff: bool
    gender: Optional[str]
    tracks: list[Track] = field(default_factory=list)
    keep_ids: list[str] = field(default_factory=list)
    orphan_ids: list[str] = field(default_factory=list)
    reasons: dict = field(default_factory=dict)  # track_id -> reason
    action: str = "skip"


def _overlap_s(a0, a1, b0, b1) -> float:
    return max(0.0, (min(a1, b1) - max(a0, b0)).total_seconds())


def _primary_cluster(tracks: list[Track], min_sec: float) -> set[str]:
    ordered = sorted(tracks, key=lambda t: (-t.duration_s, t.started_at))
    keep: list[Track] = []
    for t in ordered:
        conflict = False
        for k in keep:
            if t.camera_id != k.camera_id:
                continue
            if _overlap_s(t.started_at, t.ended_at, k.started_at, k.ended_at) >= min_sec:
                conflict = True
                break
        if not conflict:
            keep.append(t)
    return {t.id for t in keep}


def _norm_gender(g: Optional[str]) -> Optional[str]:
    if not g:
        return None
    u = g.strip().upper()
    if u in ("M", "MALE"):
        return "M"
    if u in ("F", "FEMALE"):
        return "F"
    return None


def _face_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _parse_emb(raw) -> Optional[np.ndarray]:
    if raw is None:
        return None
    if isinstance(raw, str):
        return np.array(eval(raw), dtype=np.float32)
    return np.array(raw, dtype=np.float32)


async def _discover_ids(db, staff_only: bool) -> list[str]:
    """Persons with mixed track gender and/or residual same-cam overlap."""
    staff_sql = "AND pi.is_staff = TRUE" if staff_only else ""
    r = await db.execute(
        text(
            f"""
            WITH mixed AS (
              SELECT person_identity_id
              FROM track_sessions
              WHERE person_identity_id IS NOT NULL AND gender IS NOT NULL
              GROUP BY person_identity_id
              HAVING COUNT(DISTINCT CASE
                WHEN UPPER(gender) IN ('M','MALE') THEN 'M'
                WHEN UPPER(gender) IN ('F','FEMALE') THEN 'F'
              END) > 1
            ),
            ov AS (
              SELECT DISTINCT a.person_identity_id
              FROM track_sessions a
              JOIN track_sessions b
                ON a.person_identity_id = b.person_identity_id
               AND a.camera_id = b.camera_id AND a.id < b.id
               AND a.started_at < COALESCE(b.ended_at, b.last_seen_at)
               AND b.started_at < COALESCE(a.ended_at, a.last_seen_at)
              WHERE a.person_identity_id IS NOT NULL
            )
            SELECT pi.id::text FROM person_identities pi
            WHERE (pi.id IN (SELECT person_identity_id FROM mixed)
                OR pi.id IN (SELECT person_identity_id FROM ov))
            {staff_sql}
            ORDER BY 1
            """
        )
    )
    return [row[0] for row in r.fetchall()]


async def _load_tracks(db, pid: str) -> list[Track]:
    rows = (
        await db.execute(
            text(
                """
                SELECT id::text, camera_id::text, started_at,
                       COALESCE(ended_at, last_seen_at), gender, best_crop_path
                FROM track_sessions
                WHERE person_identity_id::text = :pid
                  AND started_at IS NOT NULL
                  AND COALESCE(ended_at, last_seen_at) IS NOT NULL
                ORDER BY started_at
                """
            ),
            {"pid": pid},
        )
    ).fetchall()
    out = []
    for tid, cam, s, e, g, crop in rows:
        if e < s:
            e = s
        out.append(
            Track(
                id=tid,
                camera_id=cam,
                started_at=s,
                ended_at=e,
                gender=g,
                duration_s=(e - s).total_seconds(),
                best_crop_path=crop,
            )
        )
    return out


async def _load_person_faces(db, pid: str) -> list[np.ndarray]:
    r = await db.execute(
        text(
            "SELECT embedding FROM person_face_embeddings "
            "WHERE person_identity_id::text = :pid AND embedding IS NOT NULL"
        ),
        {"pid": pid},
    )
    return [e for e in (_parse_emb(row[0]) for row in r.fetchall()) if e is not None]


def _download_crop(minio_client, crop_path: str):
    if not crop_path:
        return None
    try:
        from app.modules.storage.minio_client import BUCKET_PREFIX
        key = crop_path
        if key.startswith(f"{BUCKET_PREFIX}/"):
            key = key[len(BUCKET_PREFIX) + 1 :]
        resp = minio_client.get_object(BUCKET_PREFIX, key)
        data = resp.read()
        resp.close()
        resp.release_conn()
        import cv2

        arr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _embed_face(analyzer, img) -> Optional[np.ndarray]:
    if img is None or analyzer is None:
        return None
    try:
        res = analyzer.analyze(img)
        if res is None or res.embedding is None:
            return None
        return np.array(res.embedding, dtype=np.float32)
    except Exception:
        return None


async def _build_plan(
    db,
    pid: str,
    min_sec: float,
    face_thr: float,
    with_faces: bool,
    analyzer,
    minio_client,
) -> Optional[PersonPlan]:
    meta = (
        await db.execute(
            text(
                "SELECT id::text, COALESCE(is_staff,false), gender "
                "FROM person_identities WHERE id::text = :pid"
            ),
            {"pid": pid},
        )
    ).fetchone()
    if not meta:
        return None
    tracks = await _load_tracks(db, pid)
    plan = PersonPlan(pid=pid, is_staff=bool(meta[1]), gender=meta[2], tracks=tracks)
    if len(tracks) < 1:
        return plan

    keep = set(t.id for t in tracks)
    reasons: dict[str, str] = {}

    # 1) temporal independent set
    primary = _primary_cluster(tracks, min_sec)
    for t in tracks:
        if t.id not in primary:
            keep.discard(t.id)
            reasons[t.id] = "same_cam_overlap"

    remaining = [t for t in tracks if t.id in keep]

    # 2) gender majority by total duration among remaining (not track count —
    #    one long F shift should beat many short M false attaches)
    dur_by_g: dict[str, float] = defaultdict(float)
    for t in remaining:
        ng = _norm_gender(t.gender)
        if ng:
            dur_by_g[ng] += t.duration_s
    majority = None
    if dur_by_g:
        majority = max(dur_by_g.items(), key=lambda kv: kv[1])[0]
    if majority:
        for t in remaining:
            ng = _norm_gender(t.gender)
            if ng and ng != majority:
                keep.discard(t.id)
                reasons[t.id] = f"gender_vs_majority_{majority}"

    remaining = [t for t in tracks if t.id in keep]

    # 3) face cluster fit
    if with_faces and analyzer is not None:
        gallery = await _load_person_faces(db, pid)
        if len(gallery) >= 1:
            for t in remaining:
                img = _download_crop(minio_client, t.best_crop_path or "")
                emb = _embed_face(analyzer, img)
                if emb is None:
                    keep.discard(t.id)
                    reasons[t.id] = "no_face_on_crop"
                    continue
                sims = [_face_sim(emb, g) for g in gallery]
                med = float(np.median(sims))
                if med < face_thr:
                    keep.discard(t.id)
                    reasons[t.id] = f"face_median={med:.3f}<{face_thr}"

    plan.keep_ids = [t.id for t in tracks if t.id in keep]
    plan.orphan_ids = [t.id for t in tracks if t.id not in keep]
    plan.reasons = reasons
    plan.action = "orphan" if plan.orphan_ids else "skip"
    return plan


async def _orphan_and_refresh(db, pid: str, orphan_ids: list[str]) -> int:
    if not orphan_ids:
        return 0
    res = await db.execute(
        text(
            "UPDATE track_sessions SET person_identity_id = NULL, updated_at = NOW() "
            "WHERE id::text = ANY(:tids)"
        ),
        {"tids": orphan_ids},
    )
    n = res.rowcount or 0
    r = await db.execute(
        text(
            """
            SELECT COUNT(*), MIN(started_at), MAX(COALESCE(ended_at, last_seen_at))
            FROM track_sessions WHERE person_identity_id::text = :pid
            """
        ),
        {"pid": pid},
    )
    cnt, first, last = r.fetchone()
    g = await db.execute(
        text(
            """
            SELECT gender FROM track_sessions
            WHERE person_identity_id::text = :pid AND gender IS NOT NULL
            GROUP BY gender ORDER BY COUNT(*) DESC LIMIT 1
            """
        ),
        {"pid": pid},
    )
    g_row = g.fetchone()
    gender = g_row[0] if g_row else None
    if cnt:
        await db.execute(
            text(
                """
                UPDATE person_identities
                SET visit_count = :n, first_seen_at = :f, last_seen_at = :l,
                    gender = COALESCE(:g, gender), updated_at = NOW()
                WHERE id::text = :pid
                """
            ),
            {"n": int(cnt), "f": first, "l": last, "g": gender, "pid": pid},
        )
    else:
        await db.execute(
            text(
                "UPDATE person_identities SET visit_count = 0, updated_at = NOW() "
                "WHERE id::text = :pid"
            ),
            {"pid": pid},
        )
    return n


async def run(ids, apply, min_sec, face_thr, with_faces, staff_only):
    settings = get_settings()
    print("=" * 80)
    print(f"  MIXED-IDENTITY TRACK CLEANUP  {'— APPLY' if apply else '— DRY RUN'}")
    print(f"  temporal min_overlap={min_sec}s  face_thr={face_thr}  "
          f"with_faces={with_faces}  staff_only={staff_only}")
    print("=" * 80)

    analyzer = None
    minio_client = None
    if with_faces:
        print("Loading InsightFace + MinIO…")
        from app.modules.reid.insightface_analyzer import get_shared_analyzer
        from app.modules.storage.minio_client import get_client

        analyzer = get_shared_analyzer()
        minio_client = get_client()
        print("Models ready.\n")

    async with AsyncSessionLocal() as db:
        if ids:
            pids = [p.strip() for p in ids if p and p.strip()]
        else:
            print("Discovering mixed / residual-overlap persons…")
            pids = await _discover_ids(db, staff_only=staff_only)
            print(f"Found {len(pids)} candidate person(s)")

        plans: list[PersonPlan] = []
        for pid in pids:
            try:
                UUID(pid)
            except ValueError:
                print(f"  skip invalid {pid}")
                continue
            plan = await _build_plan(
                db, pid, min_sec, face_thr, with_faces, analyzer, minio_client
            )
            if plan is None:
                print(f"  {pid} NOT FOUND")
                continue
            plans.append(plan)
            tag = "STAFF" if plan.is_staff else "VISITOR"
            print(
                f"  [{tag}] {plan.pid} tracks={len(plan.tracks)} "
                f"keep={len(plan.keep_ids)} orphan={len(plan.orphan_ids)} → {plan.action}"
            )
            reason_counts = Counter(plan.reasons.values())
            if reason_counts:
                print(f"       reasons: {dict(reason_counts)}")

        actionable = [p for p in plans if p.action == "orphan"]
        total_orph = sum(len(p.orphan_ids) for p in actionable)
        print("\n" + "=" * 80)
        print(
            f"  WILL MODIFY: persons={len(actionable)}  tracks_orphaned={total_orph}"
            if not apply
            else f"  APPLYING: persons={len(actionable)}  tracks={total_orph}"
        )
        print("=" * 80)

        if not apply:
            print("\nDry run only. Pass --apply to execute.")
            return

        try:
            orphans = 0
            for plan in actionable:
                n = await _orphan_and_refresh(db, plan.pid, plan.orphan_ids)
                orphans += n
                print(f"  orphaned {n} on {plan.pid}")
            await db.commit()
            print(f"\nCOMMITTED. tracks_orphaned={orphans}")
        except Exception as e:
            await db.rollback()
            print(f"FAILED: {e}")
            raise


def main():
    settings = get_settings()
    p = argparse.ArgumentParser(description="Orphan mixed-identity tracks")
    p.add_argument("--ids", nargs="+", default=None)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--staff-only", action="store_true")
    p.add_argument("--with-faces", action="store_true", help="InsightFace fit vs gallery")
    p.add_argument("--min-overlap-seconds", type=float, default=MIN_OVERLAP_DEFAULT)
    p.add_argument(
        "--face-threshold",
        type=float,
        default=float(
            getattr(settings, "FACE_CONTAMINATION_THRESHOLD", FACE_CONTAM_DEFAULT)
        ),
    )
    args = p.parse_args()
    asyncio.run(
        run(
            args.ids,
            args.apply,
            args.min_overlap_seconds,
            args.face_threshold,
            args.with_faces,
            args.staff_only,
        )
    )


if __name__ == "__main__":
    main()
