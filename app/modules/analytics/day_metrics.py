"""Day-level analytics computation (IST calendar days).

`compute_day_metrics()` is the SINGLE implementation used by:
  - the 00:10 IST freeze job (`freeze_daily_analytics`)
  - the live-today path (`daily_provider`) when the API serves the current day
  - the backfill script (`danger/backfill_daily_frozen.py`)

Sharing one implementation guarantees frozen and live numbers can never
diverge in DEFINITION — only in data (frozen is immutable; live keeps
changing as identities merge/are deleted).

Semantics (mirrors AnalyticsService live queries):
  - Person window = track_sessions.started_at inside the IST day (track-start
    semantics, CONTEXT 2026-09-21), person FK not NULL.
  - v2/dashboard/metrics = non-staff persons; demographics-table + staff
    counts include staff. Gender: M->male, F->female, else unidentified.
  - Purchases = distinct non-staff persons with billing_interactions.entered_at
    inside the IST day (store-aware caller filtering by camera ids).
  - line crossing / entry-exit / zone counts come from events.occurred_at
    inside the IST day (non false-positive).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.analytics.adjustment import (
    adjustment_factor,
    adjusted_count,
    adjusted_customers,
    customer_factor,
    model_version,
)

IST = timezone(timedelta(hours=5, minutes=30))

# Matches AnalyticsService._AGE_GROUPS (demographics table bins)
AGE_GROUPS = [
    ("child", "Child (0–10)", 0, 10),
    ("teenager", "Teenager (11–17)", 11, 17),
    ("young_adult", "Young Adult (18–35)", 18, 35),
    ("middle_adult", "Middle Adult (36–55)", 36, 55),
    ("senior", "Senior (55+)", 55, 999),
]

# Matches AnalyticsService._V2_AGE_BINS (dashboard/UI bins)
V2_AGE_BINS = [
    ("under_18", "Under 18", 0, 17),
    ("age_18_24", "18-24", 18, 24),
    ("age_25_34", "25-34", 25, 34),
    ("age_35_44", "35-44", 35, 44),
    ("age_45_60", "45-60", 45, 60),
    ("age_60_plus", "60+", 61, 999),
]


def remap_age_group(estimated_age: Optional[int]) -> str:
    if estimated_age is None:
        return "unidentified"
    for key, _label, lo, hi in AGE_GROUPS:
        if lo <= estimated_age <= hi:
            return key
    return "unidentified"


def v2_age_bin(estimated_age: Optional[int]) -> str:
    if estimated_age is None:
        return "unidentified"
    for key, _label, lo, hi in V2_AGE_BINS:
        if lo <= estimated_age <= hi:
            return key
    return "unidentified"


def gender_key(raw_gender: Optional[str]) -> str:
    if not raw_gender:
        return "unidentified"
    g = raw_gender.strip().upper()
    if g in ("M", "MALE"):
        return "male"
    if g in ("F", "FEMALE"):
        return "female"
    return "unidentified"


def day_bounds_ist(day: date) -> tuple[datetime, datetime]:
    """Return (start, end-exclusive) aware datetimes for an IST calendar day."""
    start = datetime(day.year, day.month, day.day, tzinfo=IST)
    return start, start + timedelta(days=1)


def _blank_hour() -> dict[str, Any]:
    return {
        "persons": 0, "male": 0, "female": 0, "unidentified": 0,
        "purchasers": 0, "billing_rows": 0, "billing_dwell_sum": 0.0,
        "billing_max_dwell": None, "line_crossings": 0, "line_cross_persons": 0,
        "entry": 0, "exit": 0, "events": 0, "tracks": 0, "dwell_sum": 0.0,
    }


@dataclass
class DayMetrics:
    day: date
    unique_total: int = 0
    unique_customers: int = 0
    unique_staff: int = 0
    night_only_persons: int = 0
    adjusted_unique: float = 0.0
    adjustment_factor: float = 1.0
    factor_model_version: str = "none"
    adjusted_customers: float = 0.0
    customer_factor: float = 1.0
    purchases: int = 0
    billing_interactions: int = 0
    total_events: int = 0
    total_tracks: int = 0
    avg_dwell_seconds: Optional[float] = None
    demographics: dict = field(default_factory=dict)
    hourly: dict = field(default_factory=dict)
    per_camera: dict = field(default_factory=dict)
    zone_counts: dict = field(default_factory=dict)
    person_ids: dict = field(default_factory=dict)   # customers/staff/purchasers uuid lists
    meta: dict = field(default_factory=dict)


async def compute_day_metrics(
    db: AsyncSession,
    day: date,
    *,
    source: str = "live",
) -> DayMetrics:
    """Compute every frozen analytics field for one IST calendar day."""
    start, end = day_bounds_ist(day)
    m = DayMetrics(day=day)

    # ── 1. Persons seen (track-start in day) with demographics ─────────
    # One row per (person, camera pair is NOT split): first hour + cam set +
    # raw age_group/gender columns kept verbatim for exact range composition.
    rows = (await db.execute(text("""
        SELECT ts.person_identity_id::text,
               pi.gender, pi.age_group, pi.estimated_age,
               COALESCE(pi.is_staff, false) AS is_staff,
               MIN(EXTRACT(hour FROM ts.started_at AT TIME ZONE 'Asia/Kolkata'))::int AS first_hour,
               bool_or(EXTRACT(hour FROM ts.started_at AT TIME ZONE 'Asia/Kolkata') >= 8
                       AND EXTRACT(hour FROM ts.started_at AT TIME ZONE 'Asia/Kolkata') < 22)
                 AS has_daytime,
               array_agg(DISTINCT ts.camera_id::text) AS cam_ids
        FROM track_sessions ts
        JOIN person_identities pi ON pi.id = ts.person_identity_id
        WHERE ts.person_identity_id IS NOT NULL
          AND ts.started_at >= :s AND ts.started_at < :e
        GROUP BY ts.person_identity_id, pi.gender, pi.age_group, pi.estimated_age, pi.is_staff
    """), {"s": start, "e": end})).fetchall()

    persons: dict[str, dict] = {}
    for _pid, gender, age_group, age, is_staff, fh, has_daytime, cam_ids in rows:
        persons[_pid] = {
            "g": gender if gender else None,
            "ag": age_group or None,
            "age": int(age) if age is not None else None,
            "staff": bool(is_staff),
            "cams": sorted(c for c in (cam_ids or []) if c),
            "pc": [], "pb": 0, "br": [],
            "fh": int(fh) if fh is not None else None,
            "night": not bool(has_daytime),
        }
    m.person_ids = {
        "customers": [p for p, v in persons.items() if not v["staff"]],
        "staff": [p for p, v in persons.items() if v["staff"]],
        "purchasers": [],
        "persons": persons,
    }
    m.unique_staff = sum(1 for v in persons.values() if v["staff"])
    m.unique_customers = len(persons) - m.unique_staff
    m.unique_total = len(persons)
    m.night_only_persons = sum(1 for v in persons.values() if v["night"])

    # ── 2. Adjustment factors (total-scale internal + customer-scale UI) ──
    m.adjustment_factor = float(adjustment_factor(m.unique_total))
    m.adjusted_unique = float(adjusted_count(m.unique_total))
    m.factor_model_version = model_version()
    est_cust = float(max(m.unique_total - m.unique_staff, m.unique_customers))
    m.customer_factor = float(customer_factor(est_cust))
    m.adjusted_customers = float(adjusted_customers(est_cust, m.unique_total))

    customers = [(p, v) for p, v in persons.items() if not v["staff"]]
    staff = [(p, v) for p, v in persons.items() if v["staff"]]

    # ── 3. Demographics (v1 cross-tab: ALL persons incl staff; matches live) ─
    matrix: dict[str, dict[str, int]] = {}
    for _pid, v in customers + staff:
        ag = remap_age_group(v["age"])
        row = matrix.setdefault(ag, {"male": 0, "female": 0, "unidentified": 0})
        row[gender_key(v["g"])] += 1
    m.demographics = {
        "rows": [
            {
                "age_group": key, "label": label,
                "male_count": matrix.get(key, {}).get("male", 0),
                "female_count": matrix.get(key, {}).get("female", 0),
                "unidentified_count": matrix.get(key, {}).get("unidentified", 0),
                "total_count": sum(matrix.get(key, {}).values()),
            }
            for key, label, _lo, _hi in AGE_GROUPS
        ],
        "v2_cards": _v2_cards([(p, v["g"], v["age"]) for p, v in customers]),
    }

    # ── 4. Billing (distinct non-staff purchasers + raw rows + dwell) ──
    bi_rows = (await db.execute(text("""
        SELECT person_identity_id::text, entered_at, dwell_seconds,
               EXTRACT(hour FROM entered_at AT TIME ZONE 'Asia/Kolkata')::int AS hr,
               camera_id::text
        FROM billing_interactions
        WHERE entered_at >= :s AND entered_at < :e
    """), {"s": start, "e": end})).fetchall()
    cust_pids = {p for p, v in persons.items() if not v["staff"]}
    purchasers: dict[str, set[int]] = {}
    hourly: dict[int, dict] = {h: _blank_hour() for h in range(24)}
    billing_rows = 0
    billing_dwell_sum = 0.0
    billing_max: Optional[float] = None
    for pid, _ea, dwell, hr, _cam in bi_rows:
        billing_rows += 1
        if dwell is not None:
            billing_dwell_sum += float(dwell)
            billing_max = max(billing_max or 0.0, float(dwell))
        if hr is not None and 0 <= hr <= 23:
            hb = hourly[hr]
            hb["billing_rows"] += 1
            if dwell is not None:
                hb["billing_dwell_sum"] += float(dwell)
                mx = hb["billing_max_dwell"]
                hb["billing_max_dwell"] = max(mx or 0.0, float(dwell))
        if pid and pid in cust_pids:
            purchasers.setdefault(pid, set())
            if hr is not None and 0 <= hr <= 23:
                purchasers[pid].add(hr)
    m.purchases = len(purchasers)
    m.billing_interactions = billing_rows
    m.person_ids["purchasers"] = sorted(purchasers.keys())
    purch_by_hour: dict[int, set] = {h: set() for h in range(24)}
    for pid, hours in purchasers.items():
        for h in hours:
            purch_by_hour[h].add(pid)
    for h in range(24):
        hourly[h]["purchasers"] = len(purch_by_hour[h])

    # Demographics-table per-age-group purchase counts (matches live: BI row
    # counts per age group of the purchaser, cam-filtered by caller). Only
    # rows belonging to seen persons count, exactly like live.
    pid_to_ag = {p: remap_age_group(v["age"]) for p, v in persons.items()}
    ag_purchases: dict[str, int] = {}
    for pid, _ea, dwell, hr, cam in bi_rows:
        if pid:
            if pid in persons and hr is not None and 0 <= hr <= 23:
                persons[pid]["br"].append([int(hr), str(cam) if cam else "", float(dwell) if dwell is not None else None])
            if pid in persons:
                persons[pid]["pb"] += 1
                if cam:
                    persons[pid].setdefault("bcs", set()).add(str(cam))
            ag = pid_to_ag.get(pid)
            if ag:
                ag_purchases[ag] = ag_purchases.get(ag, 0) + 1
    m.person_ids["purchasers"] = sorted(
        pid for pid, v in persons.items() if not v["staff"] and v["pb"] > 0
    )
    for pid, v in persons.items():
        v["pc"] = sorted(set(v["pc"]))
        if "bcs" in v:
            v["bcs"] = sorted(v["bcs"])
    for row in m.demographics["rows"]:
        row["total_purchase_count"] = ag_purchases.get(row["age_group"], 0)

    # ── 5. Track/hourly persons + dwell + per-camera ───────────────────
    trk = (await db.execute(text("""
        SELECT ts.person_identity_id::text, pi.gender, pi.is_staff,
               EXTRACT(hour FROM ts.started_at AT TIME ZONE 'Asia/Kolkata')::int AS hr,
               EXTRACT(epoch FROM (ts.last_seen_at - ts.started_at))::float AS dur,
               ts.camera_id::text, c.name AS cam_name
        FROM track_sessions ts
        LEFT JOIN person_identities pi ON pi.id = ts.person_identity_id
        LEFT JOIN cameras c ON c.id = ts.camera_id
        WHERE ts.started_at >= :s AND ts.started_at < :e
    """), {"s": start, "e": end})).fetchall()
    hour_persons: dict[int, set] = {h: set() for h in range(24)}
    hour_gender: dict[int, dict[str, set]] = {
        h: {"male": set(), "female": set(), "unidentified": set()} for h in range(24)
    }
    per_cam: dict[str, dict] = {}
    dwell_sum, tracks = 0.0, 0
    for pid, gender, is_staff, hr, dur, cam, cam_name in trk:
        tracks += 1
        if dur is not None:
            dwell_sum += float(dur)
        key = cam or "unknown"
        c = per_cam.setdefault(key, {
            "name": cam_name or "unknown", "tracks": 0, "dwell_sum": 0.0,
            "persons": set(), "purchasers": set(),
            "male": set(), "female": set(), "unidentified": set(),
        })
        c["tracks"] += 1
        if dur is not None:
            c["dwell_sum"] += float(dur)
        if pid:
            gk = gender_key(gender)
            if hr is not None and 0 <= hr <= 23 and not is_staff:
                hour_persons[hr].add(pid)
                hour_gender[hr][gk].add(pid)
            if pid in persons:
                persons[pid].setdefault("th", {}).setdefault(str(hr if hr is not None else -1), []).append(gk)
        if pid and not is_staff:
            c["persons"].add(pid)
            c[gender_key(gender)].add(pid)
    m.total_tracks = tracks
    m.avg_dwell_seconds = (dwell_sum / tracks) if tracks else None
    for h in range(24):
        hourly[h]["persons"] = len(hour_persons[h])
        hourly[h]["male"] = len(hour_gender[h]["male"])
        hourly[h]["female"] = len(hour_gender[h]["female"])
        hourly[h]["unidentified"] = len(hour_gender[h]["unidentified"])
        hourly[h]["tracks"] = 0  # filled below via per-track hour pass
    # tracks per hour (all sessions starting that hour)
    for _pid, _g, _s, hr, _d, _c, _n in trk:
        if hr is not None and 0 <= hr <= 23:
            hourly[hr]["tracks"] += 1
    # per-camera purchasers
    cam_of_bi = {r[0]: r[3] for r in bi_rows if r[0]}
    for pid in purchasers:
        cam = cam_of_bi.get(pid)
        if cam and cam in per_cam:
            per_cam[cam]["purchasers"].add(pid)
    m.per_camera = {
        cid: {
            "name": v["name"], "tracks": v["tracks"],
            "dwell_sum": round(v["dwell_sum"], 2),
            "persons": len(v["persons"]),
            "purchasers": len(v["purchasers"]),
            "male": len(v["male"]), "female": len(v["female"]),
            "unidentified": len(v["unidentified"]),
        }
        for cid, v in per_cam.items()
    }
    m.hourly = {str(h): v for h, v in hourly.items()}

    # ── 6. Events (line/zone/entry-exit/totals) ────────────────────────
    ev = (await db.execute(text("""
        SELECT event_type, person_identity_id::text,
               EXTRACT(hour FROM occurred_at AT TIME ZONE 'Asia/Kolkata')::int AS hr,
               zone_id::text, z.name, z.zone_type, e.camera_id::text
        FROM events e LEFT JOIN zones z ON z.id = e.zone_id
        WHERE occurred_at >= :s AND occurred_at < :e AND is_false_positive IS NOT TRUE
    """), {"s": start, "e": end})).fetchall()
    zone_counts: dict[str, dict] = {}
    lc_persons: dict[int, set] = {h: set() for h in range(24)}
    ev_person_by_cam: dict[str, dict[int, set]] = {}
    for etype, pid, hr, zid, zname, ztype, ecam in ev:
        if hr is not None and 0 <= hr <= 23:
            hk = str(hr)
            m.hourly[hk]["events"] += 1
            if ecam:
                d = ev_person_by_cam.setdefault(str(ecam), {h: set() for h in range(24)})
                d[hr].add(pid or f"anon:{etype}:{zid or ''}:{hr}")
            if etype == "line_crossing":
                m.hourly[hk]["line_crossings"] += 1
                if pid:
                    lc_persons[hr].add(pid)
                    ev_person_by_cam.setdefault("__lc__", {h: set() for h in range(24)})[hr].add(pid)
            elif etype == "line_crossing_forward":
                m.hourly[hk]["entry"] += 1
            elif etype == "line_crossing_backward":
                m.hourly[hk]["exit"] += 1
        if zid:
            z = zone_counts.setdefault(zid, {
                "name": zname or str(zid), "type": str(ztype), "event_count": 0,
            })
            z["event_count"] += 1
    m.meta["event_person_hours"] = {
        k: {str(h): sorted(s) for h, s in v.items() if s} for k, v in ev_person_by_cam.items()
    }
    for h in range(24):
        m.hourly[str(h)]["line_cross_persons"] = len(lc_persons[h])

    # per-camera line-crossing + entry/exit rows (hour-granular, for filtered v1 endpoints)
    cam_events: dict[str, dict[str, dict[int, int]]] = {}
    for etype, _pid, hr, _zid, _zn, _zt, ecam in ev:
        if hr is None or not (0 <= hr <= 23) or not ecam:
            continue
        c = cam_events.setdefault(str(ecam), {"lc": {}, "entry": {}, "exit": {}})
        if etype == "line_crossing":
            c["lc"][hr] = c["lc"].get(hr, 0) + 1
        elif etype == "line_crossing_forward":
            c["entry"][hr] = c["entry"].get(hr, 0) + 1
        elif etype == "line_crossing_backward":
            c["exit"][hr] = c["exit"].get(hr, 0) + 1
    m.meta["cam_line"] = {c: {k: {str(h): n for h, n in v.items()} for k, v in d.items()} for c, d in cam_events.items()}

    m.total_events = len(ev)
    m.zone_counts = zone_counts
    m.meta.update({"tz": "Asia/Kolkata", "source": source})
    return m


def _v2_cards(customers: list[tuple]) -> dict:
    """v2 summary cards: gender metric + 6 age bins + peak (non-staff)."""
    g = {"male": 0, "female": 0, "unidentified": 0}
    bins = {k: 0 for k, _l, _a, _b in V2_AGE_BINS}
    bins["unidentified"] = 0
    for _pid, gender, age in customers:
        g[gender_key(gender)] += 1
        bins[v2_age_bin(age)] += 1
    peak = max(
        ((k, v) for k, v in bins.items() if k != "unidentified"),
        key=lambda kv: kv[1], default=(None, 0),
    )
    peak_label = None
    for key, label, _a, _b in V2_AGE_BINS:
        if key == peak[0] and peak[1] > 0:
            peak_label = label
    return {"gender": g, "age_bins": bins, "peak_group": peak_label}
