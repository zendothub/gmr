"""Time utility functions."""

from datetime import datetime, timezone, timedelta
from typing import Optional


def utc_now() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


def seconds_since(dt: datetime) -> float:
    """Calculate seconds elapsed since a given datetime."""
    now = utc_now()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (now - dt).total_seconds()


def time_score(last_seen: datetime, max_hours: float = 24.0) -> float:
    """
    Calculate a time proximity score (0.0 to 1.0).
    Score decays as time since last_seen increases.

    Args:
        last_seen: Last seen timestamp
        max_hours: Time window for full decay

    Returns:
        Score between 0.0 and 1.0
    """
    elapsed = seconds_since(last_seen)
    max_seconds = max_hours * 3600
    if elapsed >= max_seconds:
        return 0.0
    return max(0.0, 1.0 - (elapsed / max_seconds))


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def is_within_cooldown(last_event_time: Optional[datetime], cooldown_seconds: int) -> bool:
    """Check if we're still within cooldown period."""
    if last_event_time is None:
        return False
    return seconds_since(last_event_time) < cooldown_seconds
