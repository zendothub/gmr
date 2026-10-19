"""Backfill daily_analytics_frozen for historical IST days + repair old-table gaps.

Strategy per day:
  - Full metrics computed from LIVE data via compute_day_metrics (same code as
    the 00:10 freeze job), then:
      * if daily_analytics_summary has a row for that day (a true pre-drift
        day-close snapshot), unique_total comes from it
        (source='backfill_snapshot'); the rest is live-best-effort;
      * else everything is live-recomputed (source='backfill_live').
  - daily_analytics_summary gaps (e.g. 2026-10-03..06) are INSERTed from the
    same recompute; existing rows are never overwritten.

Writes are OFF by default. Dry-run prints the day-by-day table first:
  PYTHONPATH=/gmr/gmr venv/bin/python danger/backfill_daily_frozen.py --start 2026-07-10
  PYTHONPATH=/gmr/gmr venv/bin/python danger/backfill_daily_frozen.py --start 2026-07-10 --apply
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, "/gmr/gmr")

IST = timezone(timedelta(hours=5, minutes=30))

import uuid  # noqa: E402

from loguru import logger  # noqa: E402
from sqlalchemy import select, text  # noqa: E402

from app.core.db.session import AsyncSessionLocal  # noqa: E402
from app.core.db.models.analytics import DailyAnalyticsFrozen, DailyAnalyticsSummary  # noqa: E402
from app.modules.analytics.adjustment import adjustment_factor, adjusted_count, model_version  # noqa: E402
from app.modules.analytics.day_metrics import compute_day_metrics  # noqa: E402


async def backfill_day(db, day: date, *, apply: bool, force: bool) -> dict:
    existing = (await db.execute(
        select(DailyAnalyticsFrozen).where(DailyAnalyticsFrozen.day == day)
    )).scalar_one_or_none()
    if existing is not None and not force:
        return {"day": str(day), "action": "skip-exists"}

    metrics = await compute_day_metrics(db, day, source="backfill_live")
    frozen_row = {
        "day": day,
        "computed_at": datetime.now(IST),
        "source": "backfill_live",
        "is_partial": False,
        "unique_total": metrics.unique_total,
        "unique_customers": metrics.unique_customers,
        "unique_staff": metrics.unique_staff,
        "night_only_persons": metrics.night_only_persons,
        "purchases": metrics.purchases,
        "billing_interactions": metrics.billing_interactions,
        "total_events": metrics.total_events,
        "total_tracks": metrics.total_tracks,
        "avg_dwell_seconds": metrics.avg_dwell_seconds,
        "demographics": metrics.demographics,
        "hourly": metrics.hourly,
        "per_camera": metrics.per_camera,
        "zone_counts": metrics.zone_counts,
        "person_ids": metrics.person_ids,
        "meta": dict(metrics.meta or {}),
    }

    # True day-close person total where a snapshot row exists.
    summary = (await db.execute(
        select(DailyAnalyticsSummary).where(DailyAnalyticsSummary.summary_date == day)
    )).scalar_one_or_none()
    summary_inserted = False
    if summary is not None and (summary.unique_visitors or 0) > 0:
        frozen_row["unique_total"] = int(summary.unique_visitors)
        frozen_row["source"] = "backfill_snapshot"
        frozen_row["meta"]["snapshot_unique_source"] = "daily_analytics_summary"
        frozen_row["meta"]["live_total_at_backfill"] = metrics.unique_total
    elif apply:
        # Repair the old-table gap for this day (never overwrite existing).
        db.add(DailyAnalyticsSummary(
            summary_date=day,
            total_footfall=metrics.unique_customers,
            unique_visitors=metrics.unique_total,
            avg_dwell_seconds=metrics.avg_dwell_seconds,
            total_billing_interactions=metrics.purchases,
            total_events=metrics.total_events,
            hourly_footfall={h: v.get("persons", 0) for h, v in (metrics.hourly or {}).items()},
        ))
        summary_inserted = True

    frozen_row["adjustment_factor"] = float(adjustment_factor(frozen_row["unique_total"]))
    frozen_row["adjusted_unique"] = float(adjusted_count(frozen_row["unique_total"]))
    frozen_row["factor_model_version"] = model_version()

    if not apply:
        return {
            "day": str(day), "action": "dry-run", "source": frozen_row["source"],
            "unique": frozen_row["unique_total"],
            "adjusted": round(frozen_row["adjusted_unique"], 1),
            "customers": frozen_row["unique_customers"],
            "purchases": frozen_row["purchases"],
        }
    if existing is not None and force:
        for k, v in frozen_row.items():
            if k != "day":
                setattr(existing, k, v)
    else:
        db.add(DailyAnalyticsFrozen(id=uuid.uuid4(), **frozen_row))
    await db.commit()
    return {
        "day": str(day), "action": "replaced" if (existing and force) else "inserted",
        "source": frozen_row["source"], "unique": frozen_row["unique_total"],
        "summary_gap_fixed": summary_inserted,
    }


async def main_async(args) -> int:
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else (datetime.now(IST).date() - timedelta(days=1))
    if end < start:
        raise SystemExit("--end must be >= --start")
    mode = "APPLY" if args.apply else "DRY-RUN"
    logger.info(f"Backfill daily_analytics_frozen {start}..{end} [{mode}] force={args.force}")
    results = []
    async with AsyncSessionLocal() as db:
        day = start
        while day <= end:
            try:
                results.append(await backfill_day(db, day, apply=args.apply, force=args.force))
            except Exception as e:
                logger.error(f"{day}: FAILED ({e})")
                if args.apply:
                    await db.rollback()
                results.append({"day": str(day), "action": f"error: {e}"})
            day += timedelta(days=1)
    ins = sum(1 for r in results if r.get("action") in ("inserted", "replaced", "dry-run"))
    skipped = sum(1 for r in results if r.get("action") == "skip-exists")
    print(f"{'day':<12} {'action':<12} {'source':<17} {'unique':>6} {'adj':>7} {'cust':>5} {'purch':>5}")
    for r in results:
        print(f"{r.get('day',''):<12} {str(r.get('action','')):<12} {str(r.get('source','-')):<17} "
              f"{r.get('unique','-'):>6} {r.get('adjusted','-'):>7} {r.get('customers','-'):>5} {r.get('purchases','-'):>5}")
    print(f"\n{mode}: days={len(results)} written_or_previewed={ins} skipped={skipped}")
    if not args.apply:
        print("Dry-run only — nothing was written. Re-run with --apply to write.")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-07-10")
    ap.add_argument("--end", default=None, help="defaults to yesterday (IST)")
    ap.add_argument("--apply", action="store_true", help="actually write rows")
    ap.add_argument("--force", action="store_true", help="overwrite existing frozen rows")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
