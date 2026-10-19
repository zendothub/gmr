"""Refit person-adjustment factor fields on daily_analytics_frozen rows.

Raw columns (unique_total / unique_customers / ...) are NEVER touched.
Only the four factor columns are recomputed with the CURRENT env model:
  adjusted_unique, adjustment_factor            (total scale, internal)
  adjusted_customers, customer_factor           (customer scale, UI display)

Usage:
  # accuracy report for current env model vs store_counts.csv truth
  PYTHONPATH=/gmr/gmr venv/bin/python danger/refit_person_adjust.py --report
  # dry-run rewrite preview
  PYTHONPATH=/gmr/gmr venv/bin/python danger/refit_person_adjust.py --dry-run
  # apply to all frozen rows (same immutability exception path as backfill --force)
  PYTHONPATH=/gmr/gmr venv/bin/python danger/refit_person_adjust.py --apply
  # restrict to a day window
  PYTHONPATH=/gmr/gmr venv/bin/python danger/refit_person_adjust.py --apply --start 2026-10-01
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, "/gmr/gmr")

from loguru import logger  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.core.db.session import AsyncSessionLocal  # noqa: E402
from app.core.db.models.analytics import DailyAnalyticsFrozen  # noqa: E402
from app.modules.analytics.adjustment import (  # noqa: E402
    adjusted_count,
    adjusted_customers,
    adjustment_factor,
    customer_factor,
    model_version,
)

TRUTH = Path("/gmr/gmr/danger/store_counts.csv")
OUTAGE = {"2026-10-03", "2026-10-04", "2026-10-05", "2026-10-06"}


def load_truth() -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(TRUTH) as f:
        for row in csv.DictReader(f):
            if not row.get("date"):
                continue
            out[row["date"]] = {
                "ui": int(row["ui"]) if row.get("ui") else None,
                "gmr": int(row["gmr"]) if row.get("gmr") else None,
            }
    return out


async def run(*, apply: bool, start: str | None, report: bool) -> int:
    settings = get_settings()
    logger.info(f"refit: model={settings.PERSON_ADJUST_MODEL} version={model_version()} apply={apply}")
    async with AsyncSessionLocal() as db:
        q = select(DailyAnalyticsFrozen).order_by(DailyAnalyticsFrozen.day)
        if start:
            q = q.where(DailyAnalyticsFrozen.day >= date.fromisoformat(start))
        rows = (await db.execute(q)).scalars().all()
        logger.info(f"refit: {len(rows)} frozen rows in scope")

        if report:
            truth = load_truth()
            cust_hits, cust_errs, n = 0.0, [], 0
            tot_hits, tot_errs, m = 0.0, [], 0
            for r in rows:
                t = truth.get(r.day.isoformat())
                if not t or t["gmr"] is None or r.day.isoformat() in OUTAGE:
                    continue
                est_c = max(r.unique_total - r.unique_staff, r.unique_customers)
                ac = max(1, round(adjusted_customers(est_c, r.unique_total)))
                if t["gmr"] > 0:
                    cust_errs.append(abs(ac - t["gmr"]) / t["gmr"] * 100)
                    n += 1
                at = max(1, round(adjusted_count(r.unique_total)))
                if (t["gmr"] + 40) > 0:
                    tot_errs.append(abs(at - (t["gmr"] + 40)) / (t["gmr"] + 40) * 100)
                    m += 1
            import statistics

            logger.info(
                f"REPORT model={model_version()}: "
                f"customers-vs-GMR days={n} mean_err={statistics.fmean(cust_errs):.1f}% "
                f"p50={statistics.median(cust_errs):.1f}% | "
                f"total-vs-GMR40 days={m} mean_err={statistics.fmean(tot_errs):.1f}% "
                f"p50={statistics.median(tot_errs):.1f}%"
            )
            if not apply:
                return 0

        if not apply:
            logger.info("DRY-RUN — no changes written. Re-run with --apply.")
            for r in rows[:10]:
                logger.info(
                    f"{r.day} total={r.unique_total} cust={r.unique_customers} staff={r.unique_staff} "
                    f"-> adj_unique={adjusted_count(r.unique_total):.1f} "
                    f"adj_cust={adjusted_customers(max(r.unique_total - r.unique_staff, r.unique_customers), r.unique_total):.1f}"
                )
            return 0

        n = 0
        for r in rows:
            est_c = max(r.unique_total - r.unique_staff, r.unique_customers)
            r.adjustment_factor = float(adjustment_factor(r.unique_total))
            r.adjusted_unique = float(adjusted_count(r.unique_total))
            r.customer_factor = float(customer_factor(est_c))
            r.adjusted_customers = float(adjusted_customers(est_c, r.unique_total))
            r.factor_model_version = model_version()
            n += 1
        await db.commit()
        logger.info(f"Done — refreshed factor fields on {n} frozen row(s) (model {model_version()}).")
        return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually rewrite factor columns")
    ap.add_argument("--report", action="store_true", help="accuracy vs store truth, then exit unless --apply")
    ap.add_argument("--dry-run", action="store_true", help="alias of default no-write mode")
    ap.add_argument("--start", default=None, help="only days >= this IST date")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(run(apply=args.apply, start=args.start, report=args.report)))


if __name__ == "__main__":
    main()
