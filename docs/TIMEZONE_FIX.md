# PostgreSQL Timezone Fix: Asia/Calcutta → Asia/Kolkata

## Problem
Dashboard API endpoints were failing with:
```
asyncpg.exceptions.InvalidParameterValueError: time zone "Asia/Calcutta" not recognized
```

## Solution
PostgreSQL requires the canonical IANA timezone name `'Asia/Kolkata'` instead of the deprecated alias `'Asia/Calcutta'`.

## Technical Details
- **PostgreSQL Timezone Function**: `timezone(zone, timestamp)`
- **Old Value**: `'Asia/Calcutta'` (deprecated alias)
- **New Value**: `'Asia/Kolkata'` (canonical IANA name)
- **Timezone**: IST - Indian Standard Time (UTC+5:30)

## Files Changed
- `gmr/app/modules/analytics/service.py` - 13 occurrences updated

## Query Pattern Fixed
```sql
-- Before (failing):
SELECT date_trunc('hour', timezone('Asia/Calcutta', track_sessions.started_at))

-- After (working):
SELECT date_trunc('hour', timezone('Asia/Kolkata', track_sessions.started_at))
```

## Endpoints Fixed
All `/api/v2/analytics/*` endpoints that use timezone-based bucketing:
- Dashboard metrics
- Footfall analytics
- Gender trends
- Age group analytics
- Purchase analytics

## References
- IANA Timezone Database: https://www.iana.org/time-zones
- PostgreSQL Timezone Documentation: https://www.postgresql.org/docs/current/datatype-datetime.html
