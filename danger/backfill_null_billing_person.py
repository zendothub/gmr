#!/usr/bin/env python3
"""Backfill null billing_interactions.person_identity_id from track_sessions.

Root cause: billing_interaction rule fired while track.person_identity_id was
still None (identity deferred past dwell threshold). Analytics
COUNT(DISTINCT person_identity_id) ignores NULL → under-count purchases.

Live fix: camera_worker._backfill_null_person_fks on ReID resolve + close.
This script repairs historical rows where track_sessions already has a person.

Default: dry-run. Pass --apply to write.

Usage:
  cd /gmr/gmr && python -m danger.backfill_null_billing_person
  python -m danger.backfill_null_billing_person --since 2026-07-20 --apply
  python -m danger.backfill_null_billing_person --days 3 --apply
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import text

from app.core.db.session import AsyncSessionLocal

IST = ZoneInfo("Asia/Kolkata")


def _parse_since(s: str | None, days: int | None) -> datetime | None:
    if s:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=IST)
        return dt.astimezone(timezone.utc)
    if days is not None:
        return datetime.now(timezone.utc) - timedelta(days=days)
    return None


async def run(*, since: datetime | None, apply: bool) -> None:
    params: dict = {}
    where_extra = ""
    if since is not None:
        where_extra = " AND bi.entered_at >= :since"
        params["since"] = since

    count_sql = f"""
        SELECT COUNT(*) AS n
        FROM billing_interactions bi
        JOIN track_sessions ts ON ts.id = bi.track_session_id
        WHERE bi.person_identity_id IS NULL
          AND ts.person_identity_id IS NOT NULL
          {where_extra}
    """
    sample_sql = f"""
        SELECT
            bi.id::text AS bi_id,
            bi.entered_at AT TIME ZONE 'Asia/Kolkata' AS entered_ist,
            bi.dwell_seconds,
            bi.track_session_id::text AS sid,
            ts.person_identity_id::text AS pid,
            COALESCE(pi.is_staff, false) AS is_staff
        FROM billing_interactions bi
        JOIN track_sessions ts ON ts.id = bi.track_session_id
        LEFT JOIN person_identities pi ON pi.id = ts.person_identity_id
        WHERE bi.person_identity_id IS NULL
          AND ts.person_identity_id IS NOT NULL
          {where_extra}
        ORDER BY bi.entered_at DESC
        LIMIT 30
    """
    impact_sql = f"""
        WITH candidates AS (
            SELECT
                bi.id AS bi_id,
                ts.person_identity_id AS pid,
                COALESCE(pi.is_staff, false) AS is_staff,
                bi.entered_at
            FROM billing_interactions bi
            JOIN track_sessions ts ON ts.id = bi.track_session_id
            LEFT JOIN person_identities pi ON pi.id = ts.person_identity_id
            WHERE bi.person_identity_id IS NULL
              AND ts.person_identity_id IS NOT NULL
              {where_extra}
        ),
        day_ix AS (
            SELECT
                (entered_at AT TIME ZONE 'Asia/Kolkata')::date AS d,
                pid,
                is_staff
            FROM candidates
        )
        SELECT d::text, COUNT(*) AS null_bis,
               COUNT(DISTINCT pid) FILTER (WHERE NOT is_staff) AS guest_pids
        FROM day_ix
        GROUP BY d
        ORDER BY d DESC
        LIMIT 14
    """
    apply_sql = f"""
        UPDATE billing_interactions bi
        SET person_identity_id = ts.person_identity_id,
            updated_at = now()
        FROM track_sessions ts
        WHERE bi.track_session_id = ts.id
          AND bi.person_identity_id IS NULL
          AND ts.person_identity_id IS NOT NULL
          {where_extra}
    """

    async with AsyncSessionLocal() as db:
        n = (await db.execute(text(count_sql), params)).scalar() or 0
        print(f"Candidates (null BI, session has person): {n}")
        if since:
            print(f"Filter since (UTC): {since.isoformat()}")
        print("\n--- by day (guest distinct pids that would gain a countable BI row) ---")
        rows = (await db.execute(text(impact_sql), params)).mappings().all()
        for r in rows:
            print(f"  {r['d']}: null_bis={r['null_bis']} guest_pids={r['guest_pids']}")
        print("\n--- sample (latest 30) ---")
        for r in (await db.execute(text(sample_sql), params)).mappings().all():
            print(
                f"  bi={r['bi_id'][:8]} sid={r['sid'][:8]} pid={r['pid'][:8]} "
                f"staff={r['is_staff']} dwell={r['dwell_seconds']} entered={r['entered_ist']}"
            )
        if not apply:
            print("\nDry-run only. Re-run with --apply to write.")
            return
        res = await db.execute(text(apply_sql), params)
        await db.commit()
        print(f"\nApplied. rowcount={res.rowcount}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--since", type=str, default=None, help="ISO date/datetime (IST if naive)")
    p.add_argument("--days", type=int, default=None, help="Only last N days")
    p.add_argument("--apply", action="store_true", help="Write updates (default dry-run)")
    args = p.parse_args()
    since = _parse_since(args.since, args.days)
    asyncio.run(run(since=since, apply=args.apply))


if __name__ == "__main__":
    main()
