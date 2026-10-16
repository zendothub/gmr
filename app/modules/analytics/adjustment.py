"""Person-count adjustment applied to frozen past-day analytics (UI display only).

Today stays raw/live. Swap the model via PERSON_ADJUST_MODEL (env) to compare
scenarios without code changes. All coefficients are env-overridable.

Two scales:
  CUSTOMER (display): dashboard footfall/customers vs GMR Actual.
    Bucket key = estimated customers (frozen unique_total − frozen_staff);
    estimate scaled by k.
  TOTAL (internal report): unique_total vs GMR+staff-constant.

Models (both scales):
  bucket  first upper-bound that raw falls under wins (PERSON_ADJUST_BUCKETS
          total-scale / PERSON_ADJUST_CUST_BUCKETS customer-scale)
  linear  adjusted = A + B * raw
  fixed   adjusted = raw * K
  none    adjusted = raw

Calibration (2026-10-15, model version 2026-10-c2): Sep–Oct panel, clean
snapshot-anchored days, Oct 3–6 outage excluded:
  customers <210 → ×0.685, ≥210 → ×0.587   (daily 86.3%, net 99.2%)
  total     <250 → ×0.741, ≥250 → ×0.661
Old August data (live-eroded, no snapshots) is excluded from all fits.
"""

from __future__ import annotations

from app.config import get_settings


def model_version() -> str:
    s = get_settings()
    return f"{s.PERSON_ADJUST_VERSION}:{s.PERSON_ADJUST_MODEL}"


def parse_buckets(spec: str) -> list[tuple[float, float]]:
    """Parse 'UPPER:FACTOR,UPPER:FACTOR,...' (ascending upper bounds)."""
    out: list[tuple[float, float]] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        upper, _, k = part.partition(":")
        try:
            out.append((float(upper), float(k)))
        except ValueError:
            continue
    return sorted(out)


def bucket_factor(raw: float, buckets: list[tuple[float, float]], default: float = 1.0) -> float:
    """First upper-bound that raw falls under wins; beyond all → last factor."""
    for upper, k in buckets:
        if raw < upper:
            return k
    return buckets[-1][1] if buckets else default


def _apply_model(raw: float, s, *, buckets_spec: str) -> float:
    model = (s.PERSON_ADJUST_MODEL or "bucket").strip().lower()
    raw = float(raw)
    if raw <= 0:
        return 0.0  # never invent people on an empty day
    if model == "none":
        out = raw
    elif model == "fixed":
        out = raw * float(s.PERSON_ADJUST_K)
    elif model == "linear":
        out = float(s.PERSON_ADJUST_A) + float(s.PERSON_ADJUST_B) * raw
    else:  # bucket (default)
        out = raw * bucket_factor(raw, parse_buckets(buckets_spec))
    return max(0.0, out)


def adjustment_factor(raw: float) -> float:
    """TOTAL-scale multiplier (internal GMR+staff reporting)."""
    if raw <= 0:
        return 1.0
    return adjusted_count(raw) / raw


def adjusted_count(raw: float) -> float:
    """TOTAL-scale adjusted value (internal). Never returns negative."""
    s = get_settings()
    return _apply_model(raw, s, buckets_spec=s.PERSON_ADJUST_BUCKETS)


def customer_factor(est_customers: float) -> float:
    """CUSTOMER-scale multiplier for a day, keyed on the customer estimate.

    The approved c2 fit buckets on ESTIMATED day-close customers
    (frozen_unique - frozen_staff), NOT on unique_total — keying on total
    mis-buckets ~35 staff and cost ~6pts of net accuracy. Keep in sync.
    """
    if est_customers <= 0:
        return 1.0
    s = get_settings()
    model = (s.PERSON_ADJUST_MODEL or "bucket").strip().lower()
    if model == "none":
        return 1.0
    if model == "fixed":
        return float(s.PERSON_ADJUST_K)
    if model == "linear":
        raw = float(est_customers)
        return max(0.0, (float(s.PERSON_ADJUST_A) + float(s.PERSON_ADJUST_B) * raw)) / raw
    return bucket_factor(float(est_customers), parse_buckets(s.PERSON_ADJUST_CUST_BUCKETS))


def adjusted_customers(est_customers: float, raw_total: float | None = None) -> float:
    """Customer count to display in dashboard/analytics UIs.

    est_customers = frozen_unique - frozen_staff (day-close estimate).
    raw_total kept only for signature compat; the bucket key is est.
    Never returns negative; 0 stays 0.
    """
    est = float(est_customers)
    if est <= 0:
        return 0.0
    return max(0.0, est * customer_factor(est))
