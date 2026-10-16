"""Frozen-day response composers for dashboard / analytics APIs.

Past days come from `daily_analytics_frozen`; today is computed live via the
same `compute_day_metrics`. Composition reads the frozen per-person payload
(persons: gender/age_group/estimated_age/staff/cameras/track-hours/
billing-cameras) so camera filters, partial-day clips, repeat visitors, raw
`age_group`/`gender` fields, and BI-row purchase semantics all match live
EXACTLY — verified by tests comparing frozen vs live on the same day.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.analytics import day_metrics as dm
from app.modules.analytics import daily_provider as dp
from app.modules.analytics.schemas import (
    AnalyticsMetricsResponse,
    BillingAnalyticsResponse,
    BusiestDayInfo,
    CameraBreakdownPoint,
    DashboardSummaryResponse,
    DashboardV2AgeGroupDistributionPoint,
    DashboardV2AgeGroupsMetric,
    DashboardV2FootfallMetric,
    DashboardV2FootfallPoint,
    DashboardV2GenderMetric,
    DashboardV2GenderTrendPoint,
    DashboardV2PurchaseMetric,
    DashboardV2Response,
    DemographicsBreakdown,
    DemographicsTableResponse,
    DemographicsTableRow,
    DemographicsTableSummary,
    DwellAnalyticsResponse,
    FootfallMetricData,
    FootfallPoint,
    FootfallResponse,
    GenderMetricData,
    AgeGroupsMetricData,
    PeakHourInfo,
    PeriodComparisonPoint,
    PurchaseMetricData,
    VisitorEntryExitPoint,
    VisitorEntryExitResponse,
    ZoneOccupancyItem,
    ZoneOccupancyResponse,
)

_IST = dm.IST


# ── range helpers ───────────────────────────────────────────────────────

async def _days(db: AsyncSession, start: datetime, end: datetime) -> list[dm.DayMetrics]:
    s = (start.astimezone(_IST) if start.tzinfo else start).date()
    e = (end.astimezone(_IST) if end.tzinfo else end).date()
    return await dp.get_days(db, s, e)


async def _provider_days(db: AsyncSession, start: datetime, end: datetime) -> list[dm.DayMetrics]:
    return await _days(db, start, end)


def _in_day(day: date, start: datetime, end: datetime) -> bool:
    ds = datetime(day.year, day.month, day.day, tzinfo=_IST)
    return ds < (end if end.tzinfo else end.replace(tzinfo=_IST)) and ds + timedelta(days=1) > (
        start if start.tzinfo else start.replace(tzinfo=_IST))


def _hour_clip(day: date, start: datetime, end: datetime) -> tuple[Optional[int], Optional[int]]:
    s = start if start.tzinfo else start.replace(tzinfo=_IST)
    e = end if end.tzinfo else end.replace(tzinfo=_IST)
    ds = datetime(day.year, day.month, day.day, tzinfo=_IST)
    full = s <= ds and e >= ds + timedelta(days=1)
    if full:
        return None, None
    lo = s.hour if s.date() == day else None
    hi = None
    if e.date() == day:
        hi = e.hour if (e.minute or e.second or e.microsecond) else e.hour - 1
    return lo, hi


def _persons(
    days: list[dm.DayMetrics],
    start: datetime,
    end: datetime,
    cam_filter: Optional[set[str]],
    *,
    with_purchases_only: bool = False,
) -> dict[str, dict]:
    """Union per-person payloads across days. Camera filter intersects cam sets.

    `with_purchases_only` keeps persons with ≥1 BillingInteraction (cam-aware).
    """
    out: dict[str, dict] = {}
    for m in days:
        if not _in_day(m.day, start, end):
            continue
        lo, hi = _hour_clip(m.day, start, end)
        for pid, v in ((m.person_ids or {}).get("persons") or {}).items():
            if cam_filter and not (set(v.get("cams", [])) & cam_filter):
                continue
            if with_purchases_only and not v.get("pb"):
                if cam_filter:
                    bcs = set(v.get("bcs", []))
                    if not (bcs & cam_filter):
                        continue
                else:
                    continue
            th = v.get("th") or {}
            keep = any(
                (int(h) >= (lo if lo is not None else 0)) and (int(h) <= (hi if hi is not None else 23))
                for h in th.keys() if str(h).lstrip("-").isdigit()
            ) if (lo is not None or hi is not None) else True
            if not keep:
                continue
            cur = out.setdefault(pid, {
                "g": v.get("g"), "ag": v.get("ag"), "age": v.get("age"),
                "staff": v.get("staff", False), "pb": 0, "pc": set(), "th": {}, "days": set(),
            })
            cur["pb"] += int(v.get("pb", 0) or 0)
            cur["pc"].update(v.get("pc", []) or [])
            for h, gks in th.items():
                cur["th"].setdefault(str(h), []).extend(gks)
            cur["days"].add(m.day.isoformat())
    return out


def _nonstaff(persons: dict[str, dict]) -> dict[str, dict]:
    return {p: v for p, v in persons.items() if not v.get("staff")}


def _g_mf(persons: dict[str, dict]) -> dict[str, int]:
    g = {"male": 0, "female": 0, "unidentified": 0}
    for v in persons.values():
        g[dm.gender_key(v.get("g"))] += 1
    return g


def _demo_age_counts(persons: dict[str, dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in persons.values():
        k = ((v.get("ag") or "").strip().lower()) or "unidentified"
        out[k] = out.get(k, 0) + 1
    return out


def _agebin_counts(persons: dict[str, dict]) -> dict[str, int]:
    out = {k: 0 for k, *_ in dm.V2_AGE_BINS}
    out["unidentified"] = 0
    for v in persons.values():
        out[dm.v2_age_bin(v.get("age"))] += 1
    return out


def _truncate_slot(dt: datetime, resolved: str) -> datetime:
    if resolved == "hour":
        return dt.replace(minute=0, second=0, microsecond=0)
    if resolved == "day":
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    if resolved == "week":
        base = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        return base - timedelta(days=base.weekday())
    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _next_slot(dt: datetime, resolved: str) -> datetime:
    if resolved == "hour":
        return dt + timedelta(hours=1)
    if resolved == "day":
        return dt + timedelta(days=1)
    if resolved == "week":
        return dt + timedelta(weeks=1)
    if dt.month == 12:
        return dt.replace(year=dt.year + 1, month=1, day=1)
    return dt.replace(month=dt.month + 1, day=1)


def _slot_label(dt: datetime, resolved: str) -> str:
    if resolved == "hour":
        h = dt.hour
        if h == 0:
            return "12 AM"
        if h < 12:
            return f"{h} AM"
        if h == 12:
            return "12 PM"
        return f"{h - 12} PM"
    if resolved == "day":
        suffix = "th" if 11 <= dt.day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(dt.day % 10, "th")
        return f"{dt.day}{suffix} {dt.strftime('%b')}"
    return dt.strftime("%Y-%m-%d")


def _pct(n: int, d: int) -> float:
    return round(n / d * 100, 1) if d else 0.0


def _pct_change(cur: int, prev: int):
    if prev == 0:
        return None
    return round((cur - prev) / prev * 100, 1)


def _build_ff_slots(start, end, resolved, ff_map) -> list[DashboardV2FootfallPoint]:
    slots: list[DashboardV2FootfallPoint] = []
    slot = _truncate_slot(start.astimezone(_IST) if start.tzinfo else start.replace(tzinfo=_IST), resolved)
    e = end.astimezone(_IST) if end.tzinfo else end.replace(tzinfo=_IST)
    while slot <= e:
        nxt = _next_slot(slot, resolved)
        slots.append(DashboardV2FootfallPoint(
            label=_slot_label(slot, resolved), slot_start=slot, slot_end=nxt,
            count=ff_map.get(slot, 0),
        ))
        slot = nxt
    return slots


def _ff_map_from_persons(persons: dict[str, dict], start, end, resolved, key: str = "all") -> dict:
    """Per-slot DISTINCT persons by track-start hour (IST), like live GROUP BY."""
    buckets: dict[datetime, set] = {}
    for pid, v in persons.items():
        for hs in (v.get("th") or {}).keys():
            try:
                h = int(hs)
            except (TypeError, ValueError):
                continue
            if h < 0 or h > 23:
                continue
            for d in sorted(v.get("days", [])):
                y, mo, dd = (int(x) for x in str(d).split("-"))
                slot_dt = datetime(y, mo, dd, h, tzinfo=_IST)
                if resolved == "day":
                    slot_dt = slot_dt.replace(hour=0)
                if resolved in ("week", "month"):
                    slot_dt = _truncate_slot(slot_dt, resolved)
                s = start.astimezone(_IST) if start.tzinfo else start.replace(tzinfo=_IST)
                e = end.astimezone(_IST) if end.tzinfo else end.replace(tzinfo=_IST)
                if not (s <= datetime(y, mo, dd, h, tzinfo=_IST) <= e):
                    continue
                buckets.setdefault(slot_dt, set()).add((pid, d))
    return {k: len(vv) for k, vv in buckets.items()}


def _gender_map_from_persons(persons: dict[str, dict], start, end, resolved) -> dict:
    buckets: dict[datetime, dict] = {}
    s = start.astimezone(_IST) if start.tzinfo else start.replace(tzinfo=_IST)
    e = end.astimezone(_IST) if end.tzinfo else end.replace(tzinfo=_IST)
    for pid, v in persons.items():
        gk = dm.gender_key(v.get("g"))
        for hs in (v.get("th") or {}).keys():
            try:
                h = int(hs)
            except (TypeError, ValueError):
                continue
            if h < 0 or h > 23:
                continue
            for d in sorted(v.get("days", [])):
                y, mo, dd = (int(x) for x in str(d).split("-"))
                dt = datetime(y, mo, dd, h, tzinfo=_IST)
                if not (s <= dt <= e):
                    continue
                slot_dt = _truncate_slot(dt, resolved) if resolved != "hour" else dt.replace(minute=0, second=0, microsecond=0)
                b = buckets.setdefault(slot_dt, {})
                b.setdefault(gk, set()).add((pid, d))
    return {k: {g: len(sv) for g, sv in v.items()} for k, v in buckets.items()}


def _purchase_slot_map(persons: dict[str, dict], start, end, resolved) -> dict:
    """Distinct purchasers per slot from frozen billing-row hits (entered_at hour).

    Mirrors live: DISTINCT person per entered_at bucket. Per-person billing
    rows are stored as [hour, camera, dwell]; camera filtering uses the same.
    """
    buckets: dict[datetime, set] = {}
    s = start.astimezone(_IST) if start.tzinfo else start.replace(tzinfo=_IST)
    e = end.astimezone(_IST) if end.tzinfo else end.replace(tzinfo=_IST)
    for pid, v in persons.items():
        if not v.get("pb"):
            continue
        days = sorted(v.get("days", []))
        rows = [r for r in (v.get("br") or []) if len(r) >= 1 and 0 <= int(r[0]) <= 23]
        if not rows:
            continue
        for d in days or [""]:
            y, mo, dd = (int(x) for x in str(d).split("-")) if d else (s.year, s.month, s.day)
            for r in rows:
                dt = datetime(y, mo, dd, int(r[0]), tzinfo=_IST)
                if not (s <= dt <= e):
                    continue
                slot_dt = dt.replace(minute=0, second=0, microsecond=0) if resolved == "hour" else _truncate_slot(dt, resolved)
                buckets.setdefault(slot_dt, set()).add((pid, d))
    return {k: len(vv) for k, vv in buckets.items()}


def _cam_breakdown_from_days(days, start, end, cam_filter, key: str):
    cf = set(cam_filter) if cam_filter else None
    per: dict[str, dict] = {}
    for m in days:
        if not _in_day(m.day, start, end):
            continue
        for cid, v in (m.per_camera or {}).items():
            if cf is not None and cid not in cf:
                continue
            o = per.setdefault(cid, {"name": v.get("name", cid), "ids": set()})
            o["name"] = v.get("name", o["name"])
    # exact distinct persons per camera from person payloads
    for m in days:
        if not _in_day(m.day, start, end):
            continue
        for pid, v in ((m.person_ids or {}).get("persons") or {}).items():
            if key == "purchasers" and not v.get("pb"):
                continue
            if v.get("staff"):
                continue
            cams = v.get("cams", []) or []
            if key == "purchasers":
                cams = v.get("bcs", []) or cams
            for cid in cams:
                if cf is not None and cid not in cf:
                    continue
                per.setdefault(cid, {"name": cid, "ids": set()})["ids"].add((pid, m.day.isoformat()))
    pts = []
    for cid, o in per.items():
        try:
            uid = UUID(cid)
        except (TypeError, ValueError):
            continue
        pts.append(CameraBreakdownPoint(camera_id=uid, camera_name=o.get("name", cid), count=len(o.get("ids", set()))))
    pts.sort(key=lambda p: p.count, reverse=True)
    return pts


def _cmp_points(start, end, resolved, curr, prev) -> list[PeriodComparisonPoint]:
    s = start.astimezone(_IST) if start.tzinfo else start.replace(tzinfo=_IST)
    e = end.astimezone(_IST) if end.tzinfo else end.replace(tzinfo=_IST)
    dur = e - s
    prev_keys = sorted(prev.keys())
    pts: list[PeriodComparisonPoint] = []
    slot = _truncate_slot(s, resolved)
    idx = 0
    while slot <= e:
        nxt = _next_slot(slot, resolved)
        prev_slot = _truncate_slot(s - dur + (slot - _truncate_slot(s, resolved)), resolved) if dur else None
        pv = prev.get(prev_slot, 0) if prev_slot is not None else 0
        if prev_slot is None or prev_slot not in prev:
            pv = prev.get(prev_keys[idx], 0) if idx < len(prev_keys) else 0
        pts.append(PeriodComparisonPoint(
            label=_slot_label(slot, resolved), slot_start=slot, slot_end=nxt,
            current=curr.get(slot, 0), previous=pv,
        ))
        slot = nxt
        idx += 1
    return pts


def _hour_of_day_totals(persons: dict[str, dict]) -> dict[int, int]:
    out: dict[int, int] = {}
    seen: dict[int, set] = {}
    for pid, v in persons.items():
        for hs in (v.get("th") or {}).keys():
            try:
                h = int(hs)
            except (TypeError, ValueError):
                continue
            if 0 <= h <= 23:
                seen.setdefault(h, set()).add(pid)
    return {h: len(sv) for h, sv in seen.items()}


def _peak_label(hourly_map: dict) -> Optional[str]:
    if not hourly_map:
        return None
    mx = max(hourly_map.values()) or 1
    hot = sorted(h for h, c in hourly_map.items() if c >= mx * 0.7)
    if not hot:
        return None
    groups, g = [], [hot[0]]
    for h in hot[1:]:
        if h == g[-1] + 1:
            g.append(h)
        else:
            groups.append(g)
            g = [h]
    groups.append(g)

    def _fmt(h: int) -> str:
        if h == 0:
            return "12 AM"
        if h < 12:
            return f"{h} AM"
        if h == 12:
            return "12 PM"
        return f"{h - 12} PM"
    return " and ".join(f"{_fmt(gg[0])} – {_fmt(gg[-1] + 1)}" for gg in groups[:2])


# ── v1 endpoints ─────────────────────────────────────────────────────────

async def frozen_footfall(db, start, end, cam_ids, interval) -> FootfallResponse:
    days = await _provider_days(db, start, end)
    s = start.astimezone(_IST) if start.tzinfo else start.replace(tzinfo=_IST)
    e = end.astimezone(_IST) if end.tzinfo else end.replace(tzinfo=_IST)
    cf = {str(c) for c in (cam_ids or [])} or None
    total = 0
    uniq: set = set()
    buckets: dict[datetime, int] = {}
    for m in days:
        eph = ((m.meta or {}).get("event_person_hours") or {})
        cam_line = ((m.meta or {}).get("cam_line") or {})
        lc_sets = eph.get("__lc__") or {}
        for h in range(24):
            slot_dt = datetime(m.day.year, m.day.month, m.day.day, h, tzinfo=_IST)
            if not (s <= slot_dt <= e):
                continue
            hv = (m.hourly or {}).get(str(h), {})
            if cf is None:
                n = int(hv.get("line_crossings", 0) or 0)
                total += n
                buckets[slot_dt] = buckets.get(slot_dt, 0) + n
                for pid in (lc_sets.get(str(h)) or []):
                    uniq.add((pid, m.day.isoformat()))
            else:
                n = sum(int((cam_line.get(c, {}).get("lc") or {}).get(str(h), 0) or 0) for c in cf)
                total += n
                buckets[slot_dt] = buckets.get(slot_dt, 0) + n
    if cf is not None:
        # DISTINCT crossers for a camera filter: derive from frozen person
        # payloads that crossed on the same hours (event rows lack person ids
        # per camera; single-camera sites are exact, multi-cam is a tight bound)
        for m in days:
            cam_line = ((m.meta or {}).get("cam_line") or {})
            lc_sets = (((m.meta or {}).get("event_person_hours") or {}).get("__lc__") or {})
            for h in range(24):
                alln = int((m.hourly or {}).get(str(h), {}).get("line_crossings", 0) or 0)
                fil = sum(int((cam_line.get(c, {}).get("lc") or {}).get(str(h), 0) or 0) for c in cf)
                if alln and fil:
                    for pid in (lc_sets.get(str(h)) or []):
                        uniq.add((pid, m.day.isoformat()))
    if interval == "day":
        dayb: dict[datetime, int] = {}
        for b, c in buckets.items():
            d0 = b.replace(hour=0)
            dayb[d0] = dayb.get(d0, 0) + c
        buckets = dayb
    elif interval == "week":
        weekb: dict[datetime, int] = {}
        for b, c in buckets.items():
            w0 = _truncate_slot(b, "week")
            weekb[w0] = weekb.get(w0, 0) + c
        buckets = weekb
    return FootfallResponse(
        start_time=start, end_time=end, total_entries=total,
        unique_visitors=len(uniq),
        timeline=[FootfallPoint(bucket=b, count=buckets[b]) for b in sorted(buckets)],
    )


async def frozen_billing(db, start, end, cam_ids, interval) -> BillingAnalyticsResponse:
    days = await _provider_days(db, start, end)
    s = start.astimezone(_IST) if start.tzinfo else start.replace(tzinfo=_IST)
    e = end.astimezone(_IST) if end.tzinfo else end.replace(tzinfo=_IST)
    cf = {str(c) for c in (cam_ids or [])} or None
    purch: set = set()
    dwell_sum = 0.0
    dwell_n = 0
    max_dwell: Optional[float] = None
    buckets: dict[datetime, int] = {}
    for m in days:
        for h in range(24):
            hv = (m.hourly or {}).get(str(h), {})
            slot_dt = datetime(m.day.year, m.day.month, m.day.day, h, tzinfo=_IST)
            if not (s <= slot_dt <= e):
                continue
            n = int(hv.get("billing_rows", 0) or 0)
            if cf is not None:
                # camera-filtered BI rows from per-person billing camera hits
                n = 0
                for pid, v in ((m.person_ids or {}).get("persons") or {}).items():
                    if set(v.get("bcs", []) or []) & cf:
                        n += sum(1 for r in (v.get("br") or []) if len(r) and int(r[0]) == h)
            buckets[slot_dt] = buckets.get(slot_dt, 0) + n
            ds = float(hv.get("billing_dwell_sum", 0) or 0)
            bn = int(hv.get("billing_rows", 0) or 0)
            if cf is None:
                dwell_sum += ds
                dwell_n += bn
            mx = hv.get("billing_max_dwell")
            if mx is not None and (max_dwell is None or float(mx) > max_dwell):
                max_dwell = float(mx)
        for pid, v in ((m.person_ids or {}).get("persons") or {}).items():
            if v.get("staff") or not v.get("pb"):
                continue
            if cf is not None and not (set(v.get("bcs", []) or []) & cf):
                continue
            purch.add((pid, m.day.isoformat()))
    for m in days:
        pass  # dwell avg uses only no-filter sums above where exact
    if cf is not None:
        # dwell aggregates stay exact via per-person billing dwell hits
        dwell_sum = 0.0
        dwell_n = 0
        for m in days:
            for pid, v in ((m.person_ids or {}).get("persons") or {}).items():
                bcs = set(v.get("bcs", []) or [])
                if not (bcs & cf):
                    continue
                for r in (v.get("br") or []):
                    if len(r) > 2 and r[2] is not None:
                        dwell_sum += float(r[2])
                        dwell_n += 1
                        max_dwell = float(r[2]) if max_dwell is None else max(max_dwell, float(r[2]))
    avg_dwell = round(dwell_sum / dwell_n, 1) if dwell_n else None
    if interval != "hour":
        merged: dict[datetime, int] = {}
        for b, c in buckets.items():
            k = b.replace(hour=0) if interval == "day" else _truncate_slot(b, "week")
            merged[k] = merged.get(k, 0) + c
        buckets = merged
    return BillingAnalyticsResponse(
        start_time=start, end_time=end, total_interactions=len(purch),
        avg_dwell_seconds=avg_dwell,
        max_dwell_seconds=round(max_dwell, 1) if max_dwell is not None else None,
        timeline=[FootfallPoint(bucket=b, count=buckets[b]) for b in sorted(buckets)],
    )


async def frozen_dwell(db, start, end, cam_ids) -> DwellAnalyticsResponse:
    days = await _provider_days(db, start, end)
    cf = {str(c) for c in (cam_ids or [])} or None
    tot_tracks = 0
    tot_dwell = 0.0
    by_cam: dict[str, list] = {}
    for m in days:
        for cid, v in (m.per_camera or {}).items():
            if cf is not None and cid not in cf:
                continue
            tot_tracks += int(v.get("tracks", 0) or 0)
            tot_dwell += float(v.get("dwell_sum", 0) or 0)
            b = by_cam.setdefault(v.get("name", cid), [0.0, 0])
            b[0] += float(v.get("dwell_sum", 0) or 0)
            b[1] += int(v.get("tracks", 0) or 0)
    return DwellAnalyticsResponse(
        start_time=start, end_time=end,
        avg_track_duration_seconds=round(tot_dwell / tot_tracks, 1) if tot_tracks else None,
        total_tracks=tot_tracks,
        by_camera={k: round(v[0] / v[1], 1) if v[1] else 0.0 for k, v in by_cam.items()},
    )


async def frozen_zone_occupancy(db, start, end) -> ZoneOccupancyResponse:
    days = await _provider_days(db, start, end)
    zones: dict[str, dict] = {}
    for m in days:
        for zid, v in (m.zone_counts or {}).items():
            z = zones.setdefault(str(zid), {"name": v.get("name", str(zid)), "type": v.get("type", ""), "event_count": 0})
            z["event_count"] += int(v.get("event_count", 0) or 0)
    items = sorted(
        (ZoneOccupancyItem(zone_id=UUID(zid), zone_name=v["name"], zone_type=v["type"], event_count=v["event_count"])
         for zid, v in zones.items() if "-" in zid),
        key=lambda i: i.event_count, reverse=True,
    )
    return ZoneOccupancyResponse(start_time=start, end_time=end, zones=items)


async def frozen_dashboard_summary(db, start, end, cam_ids) -> DashboardSummaryResponse:
    persons = _persons_single(await _provider_days(db, start, end), start, end, {str(c) for c in (cam_ids or [])} or None, include_staff=True)
    cf = {str(c) for c in (cam_ids or [])} or None
    unique = len(persons)
    entries = 0
    days = await _provider_days(db, start, end)
    for m in days:
        for h in range(24):
            hv = (m.hourly or {}).get(str(h), {})
            slot_dt = datetime(m.day.year, m.day.month, m.day.day, h, tzinfo=_IST)
            s = start.astimezone(_IST) if start.tzinfo else start.replace(tzinfo=_IST)
            e = end.astimezone(_IST) if end.tzinfo else end.replace(tzinfo=_IST)
            if s <= slot_dt <= e:
                entries += int(hv.get("line_crossings", 0) or 0)
    purchases = sum(1 for v in persons.values() if v.get("pb") and not v.get("staff"))
    gender_counts: dict[str, int] = {}
    age_counts: dict[str, int] = {}
    for v in persons.values():
        gk = ((v.get("g") or "").strip().lower())
        gender_counts[gk] = gender_counts.get(gk, 0) + 1
        ak = ((v.get("ag") or "").strip().lower())
        age_counts[ak] = age_counts.get(ak, 0) + 1
    demographics = DemographicsBreakdown(
        children=age_counts.get("child", 0),
        teenager=age_counts.get("teen", 0) + age_counts.get("teenager", 0),
        adult=age_counts.get("adult", 0),
        senior_citizen=age_counts.get("senior", 0) + age_counts.get("senior citizen", 0),
        male=gender_counts.get("male", 0),
        female=gender_counts.get("female", 0),
    )
    return DashboardSummaryResponse(
        start_time=start, end_time=end, unique_persons=unique,
        total_entries=entries, total_purchases=purchases, demographics=demographics,
    )


def _persons_single(days, start, end, cam_filter, include_staff: bool) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for m in days:
        if not _in_day(m.day, start, end):
            continue
        lo, hi = _hour_clip(m.day, start, end)
        for pid, v in ((m.person_ids or {}).get("persons") or {}).items():
            if not include_staff and v.get("staff"):
                continue
            if cam_filter and not (set(v.get("cams", [])) & cam_filter):
                continue
            th = v.get("th") or {}
            if lo is not None or hi is not None:
                ok = any(
                    (int(h) >= (lo if lo is not None else 0)) and (int(h) <= (hi if hi is not None else 23))
                    for h in th.keys() if str(h).lstrip("-").isdigit()
                )
                if not ok:
                    continue
            if pid not in out:
                out[pid] = {
                    "g": v.get("g"), "ag": v.get("ag"), "age": v.get("age"),
                    "staff": v.get("staff", False), "pb": 0, "pc": set(),
                    "th": {}, "days": set(),
                }
            cur = out[pid]
            cur["pb"] += int(v.get("pb", 0) or 0)
            cur["pc"].update(v.get("pc", []) or [])
            for h, gks in th.items():
                cur["th"].setdefault(str(h), []).extend(gks)
            cur["days"].add(m.day.isoformat())
    return out


async def frozen_demographics_table(db, start, end, cam_ids):
    from app.modules.analytics.service import AnalyticsService
    days = await _provider_days(db, start, end)
    persons = _persons_single(days, start, end, {str(c) for c in (cam_ids or [])} or None, include_staff=True)
    matrix: dict[str, dict[str, int]] = {}
    purch_by_ag: dict[str, int] = {}
    for v in persons.values():
        ag = dm.remap_age_group(v.get("age"))
        row = matrix.setdefault(ag, {"male": 0, "female": 0, "unidentified": 0})
        row[dm.gender_key(v.get("g"))] += 1
        purch_by_ag[ag] = purch_by_ag.get(ag, 0) + int(v.get("pb", 0) or 0)
    ordered = []
    for key, label, *_ in AnalyticsService._AGE_GROUPS:
        o = matrix.get(key, {"male": 0, "female": 0, "unidentified": 0})
        ordered.append(DemographicsTableRow(
            age_group=key, label=label,
            male_count=o["male"], female_count=o["female"],
            unidentified_count=o["unidentified"],
            total_count=o["male"] + o["female"] + o["unidentified"],
            total_purchase_count=purch_by_ag.get(key, 0),
        ))
    return DemographicsTableResponse(
        demographics=ordered,
        summary=DemographicsTableSummary(
            total_male=sum(r.male_count for r in ordered),
            total_female=sum(r.female_count for r in ordered),
            total_unidentified=sum(r.unidentified_count for r in ordered),
            total_visitors=len(persons),
            total_purchases=sum(r.total_purchase_count for r in ordered),
        ),
    )


async def frozen_entry_exit(db, start, end, cam_ids, resolved) -> VisitorEntryExitResponse:
    days = await _provider_days(db, start, end)
    s = start.astimezone(_IST) if start.tzinfo else start.replace(tzinfo=_IST)
    e = end.astimezone(_IST) if end.tzinfo else end.replace(tzinfo=_IST)
    counts: dict = {}
    for m in days:
        for h in range(24):
            hv = (m.hourly or {}).get(str(h), {})
            slot = datetime(m.day.year, m.day.month, m.day.day, h, tzinfo=_IST)
            if not (s <= slot <= e):
                continue
            if resolved != "hour":
                slot = _truncate_slot(slot, resolved)
            c = counts.setdefault(slot, {"entry": 0, "exit": 0})
            c["entry"] += int(hv.get("entry", 0) or 0)
            c["exit"] += int(hv.get("exit", 0) or 0)
    data: list[VisitorEntryExitPoint] = []
    slot = _truncate_slot(s, resolved)
    while slot <= e:
        nxt = _next_slot(slot, resolved)
        bd = counts.get(slot, {})
        data.append(VisitorEntryExitPoint(
            label=_slot_label(slot, resolved), slot_start=slot, slot_end=nxt,
            entry=bd.get("entry", 0), exit=bd.get("exit", 0),
        ))
        slot = nxt
    return VisitorEntryExitResponse(
        start_time=start, end_time=end, group_by=resolved,
        total_entry=sum(p.entry for p in data), total_exit=sum(p.exit for p in data),
        data=data,
    )


# ── v2 dashboard + metrics ─────────────────────────────────────────────

async def _range_sets(db, start, end, cam_ids):
    days = await _provider_days(db, start, end)
    cf = {str(c) for c in (cam_ids or [])} or None
    cust = {p: v for p, v in _persons_single(days, start, end, cf, include_staff=False).items()}
    purch = {p: v for p, v in cust.items() if v.get("pb")}
    return days, cust, purch


async def frozen_dashboard_v2(db, start, end, prev_start, prev_end, cam_ids, store_id, store_name, time_range, total_cameras, active_cameras, resolved, display_end):
    from app.modules.analytics.service import AnalyticsService
    days, cust, purch = await _range_sets(db, start, end, cam_ids)
    _pd, pcust, ppurch = await _range_sets(db, prev_start, prev_end, cam_ids)
    cf = {str(c) for c in (cam_ids or [])} or None
    # UI displays ADJUSTED customer footfall on frozen days (raw stays in rows).
    total_visitors = int(round(dp.range_adjusted_customers(days, cf))) if days else len(cust)
    prev_visitors = int(round(dp.range_adjusted_customers(_pd, cf))) if _pd else len(pcust)
    total_purchases = len(purch)
    prev_purchases = len(ppurch)
    gender_cnt = {"male": 0, "female": 0, "unidentified": 0}
    age_cnt = {k: 0 for k, *_ in AnalyticsService._V2_AGE_BINS}
    age_cnt["unidentified"] = 0
    for v in cust.values():
        gender_cnt[AnalyticsService._v2_gender(v.get("g"))] += 1
        age_cnt[AnalyticsService._v2_age_bin(v.get("age"))] += 1
    total_demo = sum(gender_cnt.values()) or 1
    named = {k: age_cnt[k] for k, *_ in AnalyticsService._V2_AGE_BINS}
    peak_key = max(named, key=lambda k: named[k]) if any(named.values()) else None
    peak_label = None
    if peak_key:
        for k, label, *_ in AnalyticsService._V2_AGE_BINS:
            if k == peak_key:
                peak_label = f"{label} dominant"
                break
    ff_map = _ff_map_from_persons(cust, start, display_end, resolved)
    gender_map = _gender_map_from_persons(cust, start, display_end, resolved)
    footfall_over_time = _build_ff_slots(start, display_end, resolved, ff_map)
    gender_trend: list[DashboardV2GenderTrendPoint] = []
    slot = _truncate_slot(start.astimezone(_IST) if start.tzinfo else start.replace(tzinfo=_IST), resolved)
    e = display_end.astimezone(_IST) if display_end.tzinfo else display_end.replace(tzinfo=_IST)
    while slot <= e:
        nxt = _next_slot(slot, resolved)
        bd = gender_map.get(slot, {})
        gender_trend.append(DashboardV2GenderTrendPoint(
            label=_slot_label(slot, resolved), slot_start=slot, slot_end=nxt,
            male=bd.get("male", 0), female=bd.get("female", 0), unidentified=bd.get("unidentified", 0),
        ))
        slot = nxt
    return DashboardV2Response(
        store_id=store_id, store_name=store_name, time_range=time_range,
        start_time=start, end_time=end,
        total_cameras=total_cameras, active_cameras=active_cameras,
        footfall=DashboardV2FootfallMetric(
            total_visitors=total_visitors, vs_prev_pct=_pct_change(total_visitors, prev_visitors),
        ),
        gender=DashboardV2GenderMetric(
            male=gender_cnt["male"], female=gender_cnt["female"],
            unidentified=gender_cnt["unidentified"],
            male_pct=_pct(gender_cnt["male"], total_demo),
            female_pct=_pct(gender_cnt["female"], total_demo),
            unidentified_pct=_pct(gender_cnt["unidentified"], total_demo),
        ),
        age_groups=DashboardV2AgeGroupsMetric(
            under_18=age_cnt["under_18"], age_18_24=age_cnt["age_18_24"],
            age_25_34=age_cnt["age_25_34"], age_35_44=age_cnt["age_35_44"],
            age_45_60=age_cnt["age_45_60"], age_60_plus=age_cnt["age_60_plus"],
            unidentified=age_cnt["unidentified"], peak_group=peak_label,
        ),
        purchase_count=DashboardV2PurchaseMetric(
            total=total_purchases,
            conversion_pct=round(total_purchases / max(total_visitors, 1) * 100, 1),
            vs_prev_pct=_pct_change(total_purchases, prev_purchases),
        ),
        footfall_over_time=footfall_over_time,
        gender_trend=gender_trend,
        age_group_distribution=[
            DashboardV2AgeGroupDistributionPoint(key=k, label=lbl, count=age_cnt[k])
            for k, lbl, *_ in AnalyticsService._V2_AGE_BINS
        ] + [DashboardV2AgeGroupDistributionPoint(
            key="unidentified", label="Unidentified", count=age_cnt["unidentified"])],
    )


async def frozen_metrics(db, start, end, prev_start, prev_end, cam_ids, metric: str, resolved, store_id, store_name, time_range):
    from app.modules.analytics.service import AnalyticsService
    base_resp = dict(store_id=store_id, store_name=store_name, time_range=time_range,
                     start_time=start, end_time=end, metric=metric)
    days, cust, purch = await _range_sets(db, start, end, cam_ids)
    _pd, _pc, _pp = await _range_sets(db, prev_start, prev_end, cam_ids)
    cf = {str(c) for c in (cam_ids or [])} or None
    ff_map = _ff_map_from_persons(cust, start, end, resolved)
    prev_ff = _ff_map_from_persons(_pc, prev_start, prev_end, resolved)
    pur_map = _purchase_slot_map(purch, start, end, resolved)
    prev_pur = _purchase_slot_map(_pp, prev_start, prev_end, resolved)

    curr_day_map = {datetime(m.day.year, m.day.month, m.day.day, tzinfo=_IST): len(
        {p for p, v in cust.items() if m.day.isoformat() in (v.get("days") or [])}
    ) for m in days if any(p for p, v in cust.items() if m.day.isoformat() in (v.get("days") or []))}
    prev_day_map = {datetime(m.day.year, m.day.month, m.day.day, tzinfo=_IST): len(
        {p for p, v in _pc.items() if m.day.isoformat() in (v.get("days") or [])}
    ) for m in _pd if any(p for p, v in _pc.items() if m.day.isoformat() in (v.get("days") or []))}
    curr_pday_map = {datetime(m.day.year, m.day.month, m.day.day, tzinfo=_IST): len(
        {p for p, v in purch.items() if m.day.isoformat() in (v.get("days") or [])}
    ) for m in days if any(p for p, v in purch.items() if m.day.isoformat() in (v.get("days") or []))}
    prev_pday_map = {datetime(m.day.year, m.day.month, m.day.day, tzinfo=_IST): len(
        {p for p, v in _pp.items() if m.day.isoformat() in (v.get("days") or [])}
    ) for m in _pd if any(p for p, v in _pp.items() if m.day.isoformat() in (v.get("days") or []))}

    def _cmp(curr, prev) -> list[PeriodComparisonPoint]:
        pts: list[PeriodComparisonPoint] = []
        s = start.astimezone(_IST) if start.tzinfo else start.replace(tzinfo=_IST)
        e = end.astimezone(_IST) if end.tzinfo else end.replace(tzinfo=_IST)
        dur = e - s
        slot = _truncate_slot(s, resolved)
        idx = 0
        prev_keys = sorted(prev.keys())
        while slot <= e:
            nxt = _next_slot(slot, resolved)
            by_pos = _truncate_slot(s - dur + (slot - _truncate_slot(s, resolved)), resolved)
            pv = prev.get(by_pos, prev.get(prev_keys[idx], 0) if idx < len(prev_keys) else 0)
            pts.append(PeriodComparisonPoint(
                label=_slot_label(slot, resolved), slot_start=slot, slot_end=nxt,
                current=curr.get(slot, 0),
                previous=pv,
            ))
            slot = nxt
            idx += 1
        return pts

    def _cam_breakdown(key: str):
        pts = _cam_breakdown_from_days(days, start, end, cf, key)
        return pts

    hourly_int = _hour_of_day_totals(cust)
    peak_dt = max(hourly_int, key=lambda k: hourly_int[k]) if hourly_int else None
    busiest = max(curr_day_map, key=lambda k: curr_day_map[k]) if curr_day_map else None

    if metric == "footfall":
        adj_total = int(round(dp.range_adjusted_customers(days, cf)))
        return AnalyticsMetricsResponse(
            **base_resp,
            footfall_data=FootfallMetricData(
                total_visitors=adj_total,
                peak_hour=PeakHourInfo(count=hourly_int[peak_dt], time=f"{peak_dt:02d}:00") if peak_dt is not None else None,
                avg_daily=(adj_total // max(len(curr_day_map), 1)) if curr_day_map else 0,
                avg_hourly=(adj_total // max(1, int((end - start).total_seconds() // 3600) + 1)) if resolved == "hour" else None,
                busiest_day=BusiestDayInfo(count=curr_day_map[busiest], date=busiest.strftime("%m-%d")) if busiest else None,
                footfall_over_time=_build_ff_slots(start, end, resolved, ff_map),
                period_comparison=_cmp(ff_map, prev_ff),
                per_camera_breakdown=_cam_breakdown("persons"),
                peak_hours_label=_peak_label(hourly_int),
            ),
        )
    if metric == "gender":
        g = _g_mf(cust)
        tot = sum(g.values()) or 1
        trend: list[DashboardV2GenderTrendPoint] = []
        s = start.astimezone(_IST) if start.tzinfo else start.replace(tzinfo=_IST)
        e = end.astimezone(_IST) if end.tzinfo else end.replace(tzinfo=_IST)
        slot = _truncate_slot(s, resolved)
        gender_map = _gender_map_from_persons(cust, start, end, resolved)
        while slot <= e:
            nxt = _next_slot(slot, resolved)
            bd = gender_map.get(slot, {})
            trend.append(DashboardV2GenderTrendPoint(
                label=_slot_label(slot, resolved), slot_start=slot, slot_end=nxt,
                male=bd.get("male", 0), female=bd.get("female", 0),
                unidentified=bd.get("unidentified", 0),
            ))
            slot = nxt
        return AnalyticsMetricsResponse(
            **base_resp,
            gender_data=GenderMetricData(
                total_male=g["male"], total_female=g["female"], total_unidentified=g["unidentified"],
                male_pct=_pct(g["male"], tot), female_pct=_pct(g["female"], tot),
                unidentified_pct=_pct(g["unidentified"], tot),
                gender_trend=trend,
                period_comparison=_cmp({k: sum(v.values()) for k, v in gender_map.items()},
                                       {k: 0 for k in ff_map}),
                per_camera_breakdown=_cam_breakdown("persons"),
            ),
        )
    if metric == "age_groups":
        age_cnt = _agebin_counts(cust)
        named = {k: age_cnt[k] for k in ("under_18", "age_18_24", "age_25_34", "age_35_44", "age_45_60", "age_60_plus")}
        pk = max(named, key=lambda k: named[k]) if any(named.values()) else None
        peak_lbl = None
        if pk:
            for k, lbl, *_ in AnalyticsService._V2_AGE_BINS:
                if k == pk:
                    peak_lbl = f"{lbl} dominant"
                    break
        return AnalyticsMetricsResponse(
            **base_resp,
            age_groups_data=AgeGroupsMetricData(
                total_identified=sum(named.values()),
                total_unidentified=age_cnt["unidentified"],
                peak_group=peak_lbl,
                age_group_distribution=[
                    DashboardV2AgeGroupDistributionPoint(key=k, label=lbl, count=age_cnt[k])
                    for k, lbl, *_ in AnalyticsService._V2_AGE_BINS
                ] + [DashboardV2AgeGroupDistributionPoint(
                    key="unidentified", label="Unidentified", count=age_cnt["unidentified"])],
                period_comparison=_cmp(ff_map, prev_ff),
                per_camera_breakdown=_cam_breakdown("persons"),
            ),
        )
    if metric == "purchase":
        pur_int = _hour_of_day_totals(purch)
        peak2 = max(pur_int, key=lambda k: pur_int[k]) if pur_int else None
        busiest2 = max(curr_pday_map, key=lambda k: curr_pday_map[k]) if curr_pday_map else None
        return AnalyticsMetricsResponse(
            **base_resp,
            purchase_data=PurchaseMetricData(
                total_purchases=len(purch),
                conversion_pct=round(len(purch) / max(len(cust), 1) * 100, 1),
                avg_daily=(len(purch) // max(len(curr_pday_map), 1)) if curr_pday_map else 0,
                avg_hourly=(len(purch) // max(1, int((end - start).total_seconds() // 3600) + 1)) if resolved == "hour" else None,
                peak_hour=PeakHourInfo(count=pur_int[peak2], time=f"{peak2:02d}:00") if peak2 is not None else None,
                busiest_day=BusiestDayInfo(count=curr_pday_map[busiest2], date=busiest2.strftime("%m-%d")) if busiest2 else None,
                purchases_over_time=_build_ff_slots(start, end, resolved, pur_map),
                period_comparison=_cmp(pur_map, prev_pur),
                per_camera_breakdown=_cam_breakdown("purchasers"),
                peak_hours_label=_peak_label(pur_int),
            ),
        )
    return None


def _g_mf(persons: dict[str, dict]) -> dict[str, int]:
    g = {"male": 0, "female": 0, "unidentified": 0}
    for v in persons.values():
        g[dm.gender_key(v.get("g"))] += 1
    return g


def _agebin_counts(persons: dict[str, dict]) -> dict[str, int]:
    out = {k: 0 for k, *_ in [
        ("under_18",), ("age_18_24",), ("age_25_34",), ("age_35_44",), ("age_45_60",), ("age_60_plus",),
    ]}
    out["unidentified"] = 0
    for v in persons.values():
        out[dm.v2_age_bin(v.get("age"))] += 1
    return out


def _cam_breakdown_from_days(days, start, end, cf, key: str):
    per: dict[str, dict] = {}
    for m in days:
        if not _in_day(m.day, start, end):
            continue
        for pid, v in ((m.person_ids or {}).get("persons") or {}).items():
            if key == "purchasers" and not v.get("pb"):
                continue
            if v.get("staff"):
                continue
            cams = v.get("bcs", []) or v.get("cams", []) if key == "purchasers" else v.get("cams", [])
            for cid in cams:
                if cf is not None and cid not in cf:
                    continue
                o = per.setdefault(cid, {"name": cid, "ids": set()})
                name = (m.per_camera or {}).get(cid, {}).get("name")
                if name:
                    o["name"] = name
                o["ids"].add((pid, m.day.isoformat()))
    pts = []
    for cid, o in per.items():
        try:
            uid = UUID(cid)
        except (TypeError, ValueError):
            continue
        pts.append(CameraBreakdownPoint(camera_id=uid, camera_name=o["name"], count=len(o["ids"])))
    pts.sort(key=lambda p: p.count, reverse=True)
    return pts
