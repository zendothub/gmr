"""Frozen-day read layer for dashboard / analytics APIs.

Rule: every day strictly before today (IST) is served from
`daily_analytics_frozen`. Today is always computed live via
`compute_day_metrics`. Multi-day ranges merge frozen rows + live today.

Multi-day DISTINCT counts stay correct because each frozen row stores the
person UUID sets (`person_ids.customers/staff/purchasers`); the provider
unions them instead of summing daily counts.

Callers pass `missing="live"` to fall back to live compute for past days
that have no frozen row yet (backfill gap), or `"skip"` to omit them.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.db.models.analytics import DailyAnalyticsFrozen
from app.modules.analytics.adjustment import adjusted_customers
from app.modules.analytics.day_metrics import DayMetrics, IST, compute_day_metrics


def today_ist() -> date:
    return datetime.now(IST).date()


def _row_to_metrics(row: DailyAnalyticsFrozen) -> DayMetrics:
    hourly = row.hourly or {}
    return DayMetrics(
        day=row.day,
        unique_total=row.unique_total,
        unique_customers=row.unique_customers,
        unique_staff=row.unique_staff,
        night_only_persons=row.night_only_persons,
        adjusted_unique=float(row.adjusted_unique or 0),
        adjustment_factor=float(row.adjustment_factor or 1),
        factor_model_version=row.factor_model_version or "none",
        adjusted_customers=float(getattr(row, "adjusted_customers", 0) or 0),
        customer_factor=float(getattr(row, "customer_factor", 1) or 1),
        purchases=row.purchases,
        billing_interactions=row.billing_interactions,
        total_events=row.total_events,
        total_tracks=row.total_tracks,
        avg_dwell_seconds=row.avg_dwell_seconds,
        demographics=row.demographics or {},
        hourly={k: v for k, v in hourly.items()},
        per_camera=row.per_camera or {},
        zone_counts=row.zone_counts or {},
        person_ids=row.person_ids or {},
        meta={**(row.meta or {}), "source": row.source, "frozen": True},
    )


async def get_day_metrics(
    db: AsyncSession,
    day: date,
    *,
    missing: str = "live",
) -> Optional[DayMetrics]:
    """One IST day: frozen row if day < today, else live compute."""
    if day < today_ist():
        try:
            row = (await db.execute(
                select(DailyAnalyticsFrozen).where(DailyAnalyticsFrozen.day == day)
            )).scalar_one_or_none()
        except Exception as e:
            # Table missing (migration 0012 pending) or transient DB error:
            # never break the dashboard — degrade to live compute.
            sig = f"{type(e).__name__} {e!s}"
            if "UndefinedTable" in sig or "UndefinedColumn" in sig:
                logger.warning(
                    f"daily_analytics_frozen unavailable; using live compute for {day}"
                )
                await db.rollback()
                return await compute_day_metrics(db, day, source="live_table_missing")
            raise
        if row is not None:
            return _row_to_metrics(row)
        if missing == "live":
            return await compute_day_metrics(db, day, source="backfill_gap_live")
        return None
    return await compute_day_metrics(db, day, source="live_today")


async def get_days(
    db: AsyncSession,
    start_day: date,
    end_day: date,
    *,
    missing: str = "live",
) -> list[DayMetrics]:
    """Ordered DayMetrics for [start_day, end_day] (skips Nones)."""
    out: list[DayMetrics] = []
    d = start_day
    while d <= end_day:
        m = await get_day_metrics(db, d, missing=missing)
        if m is not None:
            out.append(m)
        d += timedelta(days=1)
    return out


def union_person_sets(days: list[DayMetrics], key: str = "customers") -> set[str]:
    bags: set[str] = set()
    for m in days:
        for pid in (m.person_ids or {}).get(key, []):
            bags.add(str(pid))
    return bags


def sum_hourly(
    days: list[DayMetrics],
    *,
    hour_lo: Optional[int] = None,
    hour_hi: Optional[int] = None,
    today_hour_cap: Optional[int] = None,
    today: Optional[date] = None,
) -> dict[int, dict]:
    """Merge hourly maps across days. Partial first/last days clip by hour bounds."""
    agg: dict[int, dict] = {}
    for m in days:
        is_today = today is not None and m.day == today
        for h in range(24):
            if hour_lo is not None and m.day == days[0].day and h < hour_lo:
                continue
            if hour_hi is not None and m.day == days[-1].day and h > hour_hi:
                continue
            if is_today and today_hour_cap is not None and h > today_hour_cap:
                continue
            src = (m.hourly or {}).get(str(h), {})
            dst = agg.setdefault(h, {"slot": h})
            for k, v in src.items():
                if isinstance(v, bool):
                    continue
                if isinstance(v, (int, float)):
                    dst[k] = (dst.get(k, 0) or 0) + v
                elif v is None:
                    dst.setdefault(k, None)
                else:
                    dst[k] = v
    return agg


def merge_per_camera(days: list[DayMetrics], cam_filter: Optional[set[str]] = None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    seen: dict[str, dict[str, set]] = defaultdict(lambda: {"persons": set(), "purchasers": set()})
    for m in days:
        for cid, v in (m.per_camera or {}).items():
            if cam_filter and cid not in cam_filter:
                continue
            o = out.setdefault(str(cid), {
                "name": v.get("name", str(cid)), "tracks": 0, "dwell_sum": 0.0,
                "persons": 0, "purchasers": 0,
                "male": 0, "female": 0, "unidentified": 0,
            })
            o["name"] = v.get("name", o["name"])
            o["tracks"] += int(v.get("tracks", 0) or 0)
            o["dwell_sum"] += float(v.get("dwell_sum", 0) or 0)
            for k in ("persons", "purchasers", "male", "female", "unidentified"):
                seen[str(cid)][k] |= {(m.day.isoformat(), k, i) for i in range(int(v.get(k, 0) or 0))}
            # NOTE: per-camera person sets are approximate counts merged with day
            # granularity; distinct-by-person union across cameras is not tracked
            # per camera, so day sums are used (matches pre-freeze per-camera sums).
            o["persons"] += int(v.get("persons", 0) or 0)
            o["purchasers"] += int(v.get("purchasers", 0) or 0)
            o["male"] += int(v.get("male", 0) or 0)
            o["female"] += int(v.get("female", 0) or 0)
            o["unidentified"] += int(v.get("unidentified", 0) or 0)
    return out


def sum_scalar(days: list[DayMetrics], attr: str) -> float:
    return float(sum(getattr(m, attr, 0) or 0 for m in days))


def range_adjusted_customers(days: list[DayMetrics], cam_filter: Optional[set[str]] = None) -> float:
    """Adjusted footfall customers for a day list (no camera split on factor).

    Frozen rows carry their own stored `adjusted_customers` (backfilled by the
    refit script with the active model version); live/today rows compute it
    on the fly. Factors differ per day, so values are summed per day.
    """
    if cam_filter:
        return float(sum(m.unique_customers or 0 for m in days))
    total = 0.0
    for m in days:
        if getattr(m.meta, "get", lambda *_: None)("frozen") and m.adjusted_customers:
            total += float(m.adjusted_customers)
        else:
            # Same estimator as compute_day_metrics: total-minus-staff, never
            # below the recomputed customer count (split can drift post-freeze).
            est = max((m.unique_total or 0) - (m.unique_staff or 0), m.unique_customers or 0)
            total += float(adjusted_customers(est, m.unique_total or 0))
    return total
