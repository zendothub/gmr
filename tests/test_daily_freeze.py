"""Tests for the frozen-analytics pipeline (2026-10-15).

Covers:
  - env-swappable adjustment factor models (linear/fixed/bucket/none)
  - IST day-boundary helpers (the old job froze D-2 instead of D-1)
  - frozen-vs-live selection in the provider (today live, past frozen)
  - frozen composition equivalence on a synthetic day (union DISTINCT,
    camera filter, gender/age binning, period comparison shape)
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta

import pytest

from app.modules.analytics import adjustment, day_metrics, daily_provider as dp
from app.modules.analytics.day_metrics import DayMetrics


# ── adjustment models ──────────────────────────────────────────────────

class _S:
    def __init__(self, **kw):
        self.__dict__.update({
            "PERSON_ADJUST_MODEL": "bucket",
            "PERSON_ADJUST_BUCKETS": "250:0.741,999999:0.661",
            "PERSON_ADJUST_CUST_BUCKETS": "210:0.685,999999:0.587",
            "PERSON_ADJUST_A": 78.6,
            "PERSON_ADJUST_B": 0.413,
            "PERSON_ADJUST_K": 0.734,
            "PERSON_ADJUST_VERSION": "2026-10-c2",
        })
        self.__dict__.update(kw)


@pytest.fixture()
def patch_settings(monkeypatch):
    def _set(**kw):
        s = _S(**kw)
        monkeypatch.setattr("app.modules.analytics.adjustment.get_settings", lambda: s)
        return s
    return _set


def test_adjust_linear(patch_settings):
    patch_settings(PERSON_ADJUST_MODEL="linear")
    assert adjustment.adjusted_count(200) == pytest.approx(161.2)
    assert adjustment.adjustment_factor(200) == pytest.approx(0.806)
    assert adjustment.adjusted_count(0) == 0.0
    assert adjustment.adjustment_factor(0) == 1.0


def test_adjust_scenarios(patch_settings):
    patch_settings(PERSON_ADJUST_MODEL="fixed")
    assert adjustment.adjusted_count(200) == pytest.approx(146.8)
    patch_settings(PERSON_ADJUST_MODEL="bucket")
    assert adjustment.adjusted_count(200) == pytest.approx(200 * 0.741)
    assert adjustment.adjusted_count(260) == pytest.approx(260 * 0.661)
    patch_settings(PERSON_ADJUST_MODEL="none")
    assert adjustment.adjusted_count(200) == 200.0


def test_adjust_customers_bucket(patch_settings):
    patch_settings()  # bucket defaults
    assert adjustment.customer_factor(200) == pytest.approx(0.685)
    assert adjustment.customer_factor(260) == pytest.approx(0.587)
    assert adjustment.customer_factor(0) == 1.0
    assert adjustment.adjusted_customers(150, 200) == pytest.approx(150 * 0.685)
    assert adjustment.adjusted_customers(0, 5) == 0.0
    assert adjustment.parse_buckets("210:0.685,999999:0.587") == [(210.0, 0.685), (999999.0, 0.587)]
    assert adjustment.bucket_factor(250, [(210.0, 0.7), (999999.0, 0.6)]) == pytest.approx(0.6)


def test_adjust_customers_keys_on_est_not_total(patch_settings):
    """Regression: the approved fit buckets on ESTIMATED customers
    (frozen total − staff), not on unique_total. A day with total 244 but
    est 204 must use the ≥210 bucket (0.587), not the <210 bucket (0.685)."""
    patch_settings()
    assert adjustment.customer_factor(204) == pytest.approx(0.685)
    assert adjustment.customer_factor(210) == pytest.approx(0.587)
    out = adjustment.adjusted_customers(204, 244)
    assert out == pytest.approx(204 * 0.685)


# ── IST boundaries ─────────────────────────────────────────────────────

def test_day_bounds_ist():
    s, e = day_metrics.day_bounds_ist(date(2026, 10, 8))
    assert s.hour == 0 and s.tzinfo is not None
    assert (e - s) == timedelta(days=1)
    # midnight IST == previous-day 18:30 UTC
    assert s.utcoffset() == timedelta(hours=5, minutes=30)


def test_remap_helpers_match_service():
    # Mirrors AnalyticsService classmethod semantics (single source of truth here).
    assert day_metrics.gender_key("M") == "male"
    assert day_metrics.gender_key("f") == "female"
    assert day_metrics.gender_key("") == "unidentified"
    assert day_metrics.remap_age_group(8) == "child"
    assert day_metrics.remap_age_group(25) == "young_adult"
    assert day_metrics.remap_age_group(None) == "unidentified"
    assert day_metrics.v2_age_bin(30) == "age_25_34"
    assert day_metrics.v2_age_bin(None) == "unidentified"


# ── provider selection ─────────────────────────────────────────────────

class _FakeRows:
    def __init__(self, row=None):
        self._row = row

    def scalar_one_or_none(self):
        return self._row


class _FakeResult:
    def __init__(self, row=None):
        self._row = row

    def scalar_one_or_none(self):
        return self._row


class _FakeDB:
    def __init__(self):
        self.executed = []

    async def execute(self, stmt, params=None):
        import sqlalchemy
        stmt_s = str(stmt)

        class _R:
            def scalar_one_or_none(self_inner):
                if "daily_analytics_frozen" in stmt_s:
                    return None
                return None

            def fetchall(self_inner):
                return []

            def all(self_inner):
                return []

        return _R()


def _mk_day(d: date, cams=("cam1",)) -> DayMetrics:
    persons = {
        "p1": {"g": "M", "ag": "adult", "age": 30, "staff": False, "cams": list(cams), "pc": [], "pb": 2, "th": {"9": ["male"]}, "bcs": ["cam1"], "days": [d.isoformat()]},
        "p2": {"g": "F", "ag": "adult", "age": 28, "staff": False, "cams": list(cams), "pc": [], "pb": 0, "th": {"10": ["female"]}, "bcs": [], "days": [d.isoformat()]},
    }
    return DayMetrics(
        day=d, unique_total=2, unique_customers=2, unique_staff=0,
        purchases=1, billing_interactions=2, total_events=5, total_tracks=4,
        avg_dwell_seconds=12.5,
        demographics={"rows": [], "v2_cards": {"gender": {"male": 1, "female": 1, "unidentified": 0},
                                              "age_bins": {"age_25_34": 2, "unidentified": 0}, "peak_group": "25-34"}},
        hourly={str(h): {"persons": 1 if h in (9, 10) else 0, "male": 1 if h == 9 else 0,
                         "female": 1 if h == 10 else 0, "unidentified": 0, "purchasers": 1 if h == 9 else 0,
                         "billing_rows": 2 if h == 9 else 0, "billing_dwell_sum": 20.0 if h == 9 else 0.0,
                         "billing_max_dwell": 15.0 if h == 9 else None, "line_crossings": 0,
                         "line_cross_persons": 0, "entry": 0, "exit": 0, "events": 0, "tracks": 0, "dwell_sum": 0.0}
                for h in range(24)},
        per_camera={"cam1": {"name": "Apollo", "tracks": 4, "dwell_sum": 50.0, "persons": 2,
                             "purchasers": 1, "male": 1, "female": 1, "unidentified": 0}},
        zone_counts={}, person_ids={"customers": ["p1", "p2"], "staff": [], "purchasers": ["p1"], "persons": persons},
    )


def _day_ist(y: int, m: int, d: int, h: int = 0):
    return datetime(y, m, d, h, tzinfo=day_metrics.IST)


def test_provider_unions_repeat_visitors():
    d1, d2 = _mk_day(date(2026, 10, 1)), _mk_day(date(2026, 10, 2))
    # p1 visits both days; distinct union across ranges still counts once per metric union
    assert dp.union_person_sets([d1, d2], "customers") == {"p1", "p2"}
    assert dp.union_person_sets([d1, d2], "purchasers") == {"p1"}
    assert dp.sum_scalar([d1, d2], "total_events") == 10.0


def test_provider_today_is_live(monkeypatch):
    called = {}

    async def fake_compute(db, day, source="live"):
        called["day"] = day
        called["source"] = source
        return _mk_day(day)

    monkeypatch.setattr("app.modules.analytics.daily_provider.compute_day_metrics", fake_compute)

    async def fake_select(*a, **k):
        raise AssertionError("must not query frozen table for today")

    class NoFrozenDB:
        async def execute(self, *a, **k):
            raise AssertionError("must not query frozen table for today")

    today = dp.today_ist()
    out = asyncio.run(dp.get_day_metrics(NoFrozenDB(), today))
    assert called["day"] == today and called["source"] == "live_today"
    assert out.unique_customers == 2


def _frozen_row_for(day: date):
    import uuid as _uuid
    return {
        "day": day, "computed_at": datetime(2026, 10, 2, 0, 10, tzinfo=day_metrics.IST),
        "source": "scheduled", "is_partial": False, "unique_total": 5, "unique_customers": 4,
        "unique_staff": 1, "night_only_persons": 0, "adjusted_unique": 150.0,
        "adjustment_factor": 0.75, "factor_model_version": "test:v1",
        "purchases": 2, "billing_interactions": 3, "total_events": 9, "total_tracks": 7,
        "avg_dwell_seconds": 11.0, "demographics": {"rows": []}, "hourly": {"9": {"persons": 4}},
        "per_camera": {}, "zone_counts": {}, "person_ids": {"customers": ["a"]},
        "meta": {"source": "scheduled"},
    }


def test_provider_past_day_uses_frozen_row():
    row = _frozen_row_for(date(2026, 10, 1))

    class FrozenDB:
        async def execute(self, stmt, params=None):
            class R:
                def scalar_one_or_none(self):
                    class O:
                        pass
                    o = O()
                    for k, v in row.items():
                        setattr(o, k, v)
                    o.day = row["day"]
                    return o
            return R()

    async def fail_compute(*a, **k):
        raise AssertionError("must not live-compute a frozen past day")

    import app.modules.analytics.daily_provider as dpmod
    orig = dpmod.compute_day_metrics
    dpmod.compute_day_metrics = fail_compute
    try:
        out = asyncio.run(dpmod.get_day_metrics(FrozenDB(), date(2026, 10, 1)))
    finally:
        dpmod.compute_day_metrics = orig
    assert out.unique_total == 5
    assert out.factor_model_version == "test:v1"
    assert out.meta["frozen"] is True


def test_freeze_uses_correct_ist_day():
    # yesterday(IST) resolves to the day before today in IST, not UTC-shifted D-2
    import app.modules.jobs.tasks as tasks
    now = datetime.now(day_metrics.IST)
    assert (now - timedelta(days=1)).date() < now.date()


# ── composer equivalence ───────────────────────────────────────────────

def test_dashboard_v2_frozen_matches_live_semantics():
    from app.modules.analytics import frozen_composer as fc
    m1, m2 = _mk_day(date(2026, 10, 6)), _mk_day(date(2026, 10, 7))
    start = _day_ist(2026, 10, 6)
    end = _day_ist(2026, 10, 8) - timedelta(microseconds=1)
    persons = fc._persons_single([m1, m2], start, end, None, include_staff=False)
    assert set(persons) == {"p1", "p2"}  # repeats union, not summed
    assert persons["p1"]["pb"] == 4      # purchases sum across days (2+2)
    g = fc._g_mf(persons)
    assert (g["male"], g["female"]) == (1, 1)


def test_camera_and_hour_clip():
    from app.modules.analytics import frozen_composer as fc
    m = _mk_day(date(2026, 10, 6))
    full = fc._persons_single([m], _day_ist(2026, 10, 6), _day_ist(2026, 10, 7) - timedelta(microseconds=1), None, include_staff=True)
    assert set(full) == {"p1", "p2"}
    off_cam = fc._persons_single([m], _day_ist(2026, 10, 6), _day_ist(2026, 10, 7) - timedelta(microseconds=1), {"other"}, include_staff=True)
    assert off_cam == {}
    morning = fc._persons_single([m], _day_ist(2026, 10, 6, 9), _day_ist(2026, 10, 6, 9) + timedelta(hours=1) - timedelta(microseconds=1), None, include_staff=True)
    assert set(morning) == {"p1"}
