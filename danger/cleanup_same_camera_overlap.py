#!/usr/bin/env python3
"""
cleanup_same_camera_overlap.py — fix PersonIdentity pollution from same-camera concurrent tracks.

Rule: two track_sessions on the SAME camera that OVERLAP in time cannot be the same
physical person. Cross-camera overlap is allowed (entry + counter simultaneously).

Policy (staff AND visitors):
  Keep the identity; keep a "primary cluster" of mutually non-overlapping tracks
  (greedy longest-duration first); ORPHAN the rest (person_identity_id=NULL).
  Refresh first/last/visit/gender from remaining tracks. MinIO left for sweep.

Usage (dry-run default; ALL cameras / all polluted persons unless --ids):
  PYTHONPATH=/gmr/gmr venv/bin/python danger/cleanup_same_camera_overlap.py
  PYTHONPATH=/gmr/gmr venv/bin/python danger/cleanup_same_camera_overlap.py --verbose
  PYTHONPATH=/gmr/gmr venv/bin/python danger/cleanup_same_camera_overlap.py --apply
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import text

from app.core.db.session import AsyncSessionLocal


@dataclass
class Track:
    id: str
    camera_id: str
    started_at: datetime
    ended_at: datetime
    gender: Optional[str]
    duration_s: float


@dataclass
class PersonPlan:
    pid: str
    is_staff: bool
    gender: Optional[str]
    tracks: list[Track] = field(default_factory=list)
    overlap_pairs: int = 0
    keep_ids: list[str] = field(default_factory=list)
    orphan_ids: list[str] = field(default_factory=list)
    action: str = ""  # "orphan" | "skip"


def _overlap_seconds(a0: datetime, a1: datetime, b0: datetime, b1: datetime) -> float:
    start = max(a0, b0)
    end = min(a1, b1)
    return max(0.0, (end - start).total_seconds())


def _count_overlap_pairs(tracks: list[Track], min_sec: float) -> int:
    n = 0
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            a, b = tracks[i], tracks[j]
            if a.camera_id != b.camera_id:
                continue
            if _overlap_seconds(a.started_at, a.ended_at, b.started_at, b.ended_at) >= min_sec:
                n += 1
    return n


def _primary_cluster(tracks: list[Track], min_sec: float) -> tuple[list[str], list[str]]:
    """Greedy longest-first independent set under same-camera overlap conflicts."""
    ordered = sorted(tracks, key=lambda t: (-t.duration_s, t.started_at))
    keep: list[Track] = []
    for t in ordered:
        conflicts = False
        for k in keep:
            if t.camera_id != k.camera_id:
                continue
            if _overlap_seconds(t.started_at, t.ended_at, k.started_at, k.ended_at) >= min_sec:
                conflicts = True
                break
        if not conflicts:
            keep.append(t)
    keep_ids = {t.id for t in keep}
    orphan_ids = [t.id for t in tracks if t.id not in keep_ids]
    return [t.id for t in keep], orphan_ids


async def _discover_polluted_ids(db, min_sec: float) -> list[str]:
    r = await db.execute(
        text(
            """
            SELECT DISTINCT a.person_identity_id::text
            FROM track_sessions a
            JOIN track_sessions b
              ON a.person_identity_id = b.person_identity_id
             AND a.camera_id = b.camera_id
             AND a.id < b.id
             AND a.started_at < COALESCE(b.ended_at, b.last_seen_at)
             AND b.started_at < COALESCE(a.ended_at, a.last_seen_at)
             AND EXTRACT(epoch FROM (
                   LEAST(COALESCE(a.ended_at, a.last_seen_at), COALESCE(b.ended_at, b.last_seen_at))
                 - GREATEST(a.started_at, b.started_at)
                 )) >= :min_sec
            WHERE a.person_identity_id IS NOT NULL
            ORDER BY 1
            """
        ),
        {"min_sec": min_sec},
    )
    return [row[0] for row in r.fetchall()]


async def _load_person_plan(db, pid: str, min_sec: float) -> Optional[PersonPlan]:
    meta = (
        await db.execute(
            text(
                """
                SELECT id::text, COALESCE(is_staff, false), gender
                FROM person_identities WHERE id::text = :pid
                """
            ),
            {"pid": pid},
        )
    ).fetchone()
    if not meta:
        return None

    rows = (
        await db.execute(
            text(
                """
                SELECT id::text, camera_id::text, started_at,
                       COALESCE(ended_at, last_seen_at) AS ended_at,
                       gender
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

    tracks: list[Track] = []
    for tid, cam, start, end, gender in rows:
        if end < start:
            end = start
        dur = (end - start).total_seconds()
        tracks.append(
            Track(
                id=tid,
                camera_id=cam,
                started_at=start,
                ended_at=end,
                gender=gender,
                duration_s=dur,
            )
        )

    plan = PersonPlan(
        pid=pid,
        is_staff=bool(meta[1]),
        gender=meta[2],
        tracks=tracks,
        overlap_pairs=_count_overlap_pairs(tracks, min_sec),
    )
    if plan.overlap_pairs == 0 or len(tracks) < 2:
        plan.action = "skip"
        return plan

    keep, orphan = _primary_cluster(tracks, min_sec)
    plan.keep_ids = keep
    plan.orphan_ids = orphan
    plan.action = "orphan" if orphan else "skip"
    return plan


async def _orphan_tracks(db, track_ids: list[str]) -> int:
    if not track_ids:
        return 0
    res = await db.execute(
        text(
            """
            UPDATE track_sessions
            SET person_identity_id = NULL, updated_at = NOW()
            WHERE id::text = ANY(:tids)
            """
        ),
        {"tids": track_ids},
    )
    return res.rowcount or 0


async def _refresh_person_stats(db, pid: str) -> None:
    r = await db.execute(
        text(
            """
            SELECT COUNT(*) AS n,
                   MIN(started_at) AS first_seen,
                   MAX(COALESCE(ended_at, last_seen_at)) AS last_seen
            FROM track_sessions
            WHERE person_identity_id::text = :pid
            """
        ),
        {"pid": pid},
    )
    n, first_seen, last_seen = r.fetchone()
    if not n:
        await db.execute(
            text(
                """
                UPDATE person_identities
                SET visit_count = 0, updated_at = NOW()
                WHERE id::text = :pid
                """
            ),
            {"pid": pid},
        )
        return

    g = await db.execute(
        text(
            """
            SELECT gender, COUNT(*) c
            FROM track_sessions
            WHERE person_identity_id::text = :pid AND gender IS NOT NULL
            GROUP BY gender ORDER BY c DESC LIMIT 1
            """
        ),
        {"pid": pid},
    )
    g_row = g.fetchone()
    gender = g_row[0] if g_row else None
    params = {
        "pid": pid,
        "n": int(n),
        "first_seen": first_seen,
        "last_seen": last_seen,
        "gender": gender,
    }
    if gender:
        await db.execute(
            text(
                """
                UPDATE person_identities
                SET visit_count = :n,
                    first_seen_at = :first_seen,
                    last_seen_at = :last_seen,
                    gender = :gender,
                    updated_at = NOW()
                WHERE id::text = :pid
                """
            ),
            params,
        )
    else:
        await db.execute(
            text(
                """
                UPDATE person_identities
                SET visit_count = :n,
                    first_seen_at = :first_seen,
                    last_seen_at = :last_seen,
                    updated_at = NOW()
                WHERE id::text = :pid
                """
            ),
            params,
        )


def _print_plan(plan: PersonPlan, verbose: bool) -> None:
    total_dur = sum(t.duration_s for t in plan.tracks)
    orph_dur = sum(t.duration_s for t in plan.tracks if t.id in set(plan.orphan_ids))
    flag = "STAFF" if plan.is_staff else "VISITOR"
    print(
        f"  {plan.pid}  [{flag}] g={plan.gender} tracks={len(plan.tracks)} "
        f"pairs={plan.overlap_pairs} → {plan.action} "
        f"keep={len(plan.keep_ids)} orphan={len(plan.orphan_ids)} "
        f"({orph_dur/60:.1f}m of {total_dur/60:.1f}m)"
    )
    if not verbose or not plan.orphan_ids:
        return
    for tid in plan.orphan_ids[:8]:
        t = next(x for x in plan.tracks if x.id == tid)
        print(
            f"      orphan {tid[:8]} cam={t.camera_id[:8]} g={t.gender} "
            f"dur={t.duration_s:.0f}s"
        )
    if len(plan.orphan_ids) > 8:
        print(f"      ... +{len(plan.orphan_ids) - 8} more")


def _print_summary(plans: list[PersonPlan], apply: bool) -> None:
    actionable = [p for p in plans if p.action == "orphan"]
    skipped = [p for p in plans if p.action == "skip"]
    staff_n = sum(1 for p in actionable if p.is_staff)
    visitor_n = sum(1 for p in actionable if not p.is_staff)
    orphan_tracks = sum(len(p.orphan_ids) for p in actionable)
    keep_tracks = sum(len(p.keep_ids) for p in actionable)
    mode = "WILL MODIFY" if not apply else "MODIFIED"

    print("\n" + "=" * 80)
    print(f"  {mode} SUMMARY  (all cameras / orphan-only, keep every person_identity)")
    print("=" * 80)
    print(f"  Persons scanned:              {len(plans)}")
    print(f"  Persons that will change:     {len(actionable)}")
    print(f"    staff:                      {staff_n}")
    print(f"    visitors:                   {visitor_n}")
    print(f"  Tracks to orphan:             {orphan_tracks}")
    print(f"  Tracks kept on person:        {keep_tracks}")
    print(f"  Skipped:                      {len(skipped)}")
    print("-" * 80)
    for p in sorted(actionable, key=lambda x: -len(x.orphan_ids))[:40]:
        tag = "STAFF" if p.is_staff else "VISITOR"
        print(
            f"    [{tag}] {p.pid}  orphan {len(p.orphan_ids)} / keep {len(p.keep_ids)}  "
            f"(pairs={p.overlap_pairs})"
        )
    if len(actionable) > 40:
        print(f"    ... +{len(actionable) - 40} more")
    print("=" * 80)


async def run(
    ids: Optional[list[str]],
    apply: bool,
    min_sec: float,
    verbose: bool,
) -> None:
    print("=" * 80)
    print(f"  SAME-CAMERA OVERLAP CLEANUP  {'— APPLY' if apply else '— DRY RUN'}")
    print("  scope: ALL polluted person_identities (both cameras) unless --ids")
    print(f"  min overlap: {min_sec}s")
    print("  POLICY: orphan conflicting tracks only (staff AND visitors) — never delete person")
    print("=" * 80)

    async with AsyncSessionLocal() as db:
        if ids:
            pids = [p.strip() for p in ids if p and p.strip()]
            print(f"\nUsing {len(pids)} id(s) from --ids")
        else:
            print("\nDiscovering polluted persons across ALL cameras...")
            pids = await _discover_polluted_ids(db, min_sec)
            print(f"Found {len(pids)} person(s) with same-cam overlap ≥ {min_sec}s")

        if not pids:
            print("Nothing to do.")
            return

        plans: list[PersonPlan] = []
        for pid in pids:
            try:
                UUID(pid)
            except ValueError:
                print(f"  skip invalid UUID: {pid}")
                continue
            plan = await _load_person_plan(db, pid, min_sec)
            if plan is None:
                print(f"  {pid} NOT FOUND")
                continue
            plans.append(plan)
            if verbose or len(pids) <= 30:
                _print_plan(plan, verbose=verbose)

        if not verbose and len(pids) > 30:
            print(f"\n  (compact list; use --verbose for track details)")

        _print_summary(plans, apply=apply)

        if not apply:
            print("\nDry run only — no DB writes. Re-run with --apply to execute.")
            return

        try:
            changed_persons = 0
            orphaned_tracks = 0
            for plan in plans:
                if plan.action != "orphan":
                    continue
                n = await _orphan_tracks(db, plan.orphan_ids)
                await _refresh_person_stats(db, plan.pid)
                changed_persons += 1
                orphaned_tracks += n
                tag = "STAFF" if plan.is_staff else "VISITOR"
                print(f"  [{tag}] {plan.pid}: orphaned {n} tracks, stats refreshed")

            await db.commit()
            print("\nCOMMITTED.")
            print(f"  persons_changed={changed_persons}  tracks_orphaned={orphaned_tracks}")
            print("  MinIO crops left for periodic sweep.")
        except Exception as e:
            await db.rollback()
            print(f"\nFAILED — rolled back: {e}")
            raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cleanup same-camera concurrent-track pollution across ALL cameras. "
            "Orphans conflicting tracks; never deletes person_identities."
        )
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        default=None,
        help="Optional person UUIDs (default: auto-discover all polluted persons)",
    )
    parser.add_argument(
        "--min-overlap-seconds",
        type=float,
        default=1.0,
        help="Minimum overlapping seconds to count as conflict (default 1.0)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-track orphan details",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Execute changes (default is dry-run with full modification counts)",
    )
    args = parser.parse_args()
    asyncio.run(run(args.ids, args.apply, args.min_overlap_seconds, args.verbose))


if __name__ == "__main__":
    main()
