# Dashboard V2 Data Extraction Flow

## Endpoint Overview
```
GET http://localhost:8000/api/v2/analytics/dashboard?time_range=today
```

This document explains how data is extracted and processed for the V2 Dashboard endpoint.

---

## 1. Request Flow

### 1.1 Endpoint Definition
**File:** `app/modules/analytics/router.py` (Line 191-237)

```python
@v2_router.get("/dashboard", response_model=DashboardV2Response)
async def dashboard_v2(
    store_id: Optional[UUID] = Query(None),
    time_range: str = Query("today", pattern="^(today|this_week|custom)$"),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
)
```

**Parameters:**
- `time_range=today` → Analyzes data from 00:00 today (IST) to current time
- `store_id` → Optional filter for specific store (omit for all stores)
- Authentication required via `get_current_user` dependency

---

## 2. Service Method: `get_dashboard_v2()`

**File:** `app/modules/analytics/service.py` (Line 812-1137)

### 2.1 Time Range Resolution
**Method:** `_v2_resolve_range()` (Line 770-810)

For `time_range="today"`:
```python
now = datetime.now(IST)  # Current time in IST (Asia/Kolkata, UTC+5:30)
start = now.replace(hour=0, minute=0, second=0, microsecond=0)  # 00:00 today IST
end = now  # Current time
duration = end - start
prev_start = start - duration  # Yesterday same time
prev_end = start  # Midnight today
```

**Result:** Returns (start, end, prev_start, prev_end) all in IST timezone

---

## 3. Data Extraction Steps

### 3.1 Camera Filter Resolution
**Method:** `_resolve_camera_ids()` (Line 66-86)

```python
if store_id:
    # Get all camera IDs linked to the store
    SELECT camera.id FROM cameras WHERE camera.store_id = {store_id}
else:
    # No filter → include all cameras
    cam_ids = None
```

---

### 3.2 Store Name Lookup
**Query:** 
```sql
SELECT store.name FROM stores WHERE store.id = {store_id}
```
Returns: `"GMR - T2"` or `"All Stores"` if no store_id provided

---

### 3.3 Camera Statistics
**Queries:**
```sql
-- Total cameras (active in system)
SELECT COUNT(camera.id) 
FROM cameras 
WHERE camera.is_active = true
AND (camera.store_id = {store_id} OR {store_id} IS NULL)

-- Active cameras (currently streaming)
SELECT COUNT(camera.id) 
FROM cameras 
WHERE camera.is_active = true 
AND camera.status = 'active'
AND (camera.store_id = {store_id} OR {store_id} IS NULL)
```

**Result:** `total_cameras: 8, active_cameras: 6`

---

### 3.4 Footfall (Unique Visitors)
**Database Tables:** `track_sessions`, `person_identities`

**Current Period Query:**
```sql
SELECT COUNT(DISTINCT track_session.person_identity_id)
FROM track_sessions
WHERE track_session.started_at >= '2026-06-30 00:00:00+05:30'  -- Today 00:00 IST
  AND track_session.started_at <= '2026-06-30 15:02:00+05:30'  -- Current time IST
  AND track_session.person_identity_id IS NOT NULL
  AND (track_session.camera_id IN ({cam_ids}) OR {cam_ids} IS NULL)
```

#### **Detailed Query Breakdown:**

**1. `SELECT COUNT(DISTINCT track_session.person_identity_id)`**
   - **COUNT**: Counts the number of rows returned
   - **DISTINCT**: Ensures each unique `person_identity_id` is counted only once
   - **Why DISTINCT is critical**: 
     - A single person can have multiple track sessions (e.g., seen by different cameras, or re-entering)
     - Example: Person with ID `abc-123` might have 5 track sessions (Camera 1, Camera 2, re-entry, etc.)
     - WITHOUT DISTINCT: Would count 5 (incorrect - overcounts the same person)
     - WITH DISTINCT: Counts 1 (correct - one unique visitor)

**2. `FROM track_sessions`**
   - The main table storing all tracking sessions
   - Each row represents one continuous observation of a person by one camera
   - Contains: `id`, `person_identity_id`, `camera_id`, `started_at`, `last_seen_at`

**3. `WHERE track_session.started_at >= '2026-06-30 00:00:00+05:30'`**
   - **Time range start**: Filters sessions that started on or after midnight today (IST)
   - **Timezone**: `+05:30` indicates IST (India Standard Time = UTC+5:30)
   - Ensures we only count visitors from today

**4. `AND track_session.started_at <= '2026-06-30 15:02:00+05:30'`**
   - **Time range end**: Filters sessions that started before or at current time (3:02 PM IST)
   - Combined with condition #3, creates a time window: [00:00 today → 15:02 today]
   - Excludes future sessions (which shouldn't exist anyway)

**5. `AND track_session.person_identity_id IS NOT NULL`**
   - **Filters out unidentified tracks**: Some track sessions may not be linked to a person identity
   - Reasons for NULL values:
     - Re-identification process hasn't run yet
     - Person couldn't be identified (no clear face/body features)
     - System errors or incomplete data
   - Only counts visitors who have been successfully identified

**6. `AND (track_session.camera_id IN ({cam_ids}) OR {cam_ids} IS NULL)`**
   - **Store/Camera filtering logic**:
   
   **Case A: When `cam_ids` is NOT NULL** (store filter is applied):
   ```sql
   track_session.camera_id IN (camera1_uuid, camera2_uuid, camera3_uuid)
   ```
   - Only counts sessions from specific cameras belonging to the selected store
   - Example: "GMR - T2" store has cameras [C1, C2, C3] → only count visitors seen by these cameras
   
   **Case B: When `cam_ids` IS NULL** (no store filter, "All Stores"):
   ```sql
   {cam_ids} IS NULL → TRUE
   ```
   - The OR condition evaluates to TRUE
   - All track sessions are included regardless of camera
   - Gives total visitors across all stores/cameras

#### **Real-World Example:**

**Scenario:** GMR Airport has 8 cameras across 2 terminals. Today (00:00 to 15:02):

```
Track Sessions Data:
+-------+------------------+------------+---------------------+
| ID    | person_id        | camera_id  | started_at          |
+-------+------------------+------------+---------------------+
| ts001 | person_abc       | cam1_T1    | 2026-06-30 08:00    |  ← Person ABC at Terminal 1
| ts002 | person_abc       | cam2_T1    | 2026-06-30 08:15    |  ← Same person, different camera
| ts003 | person_abc       | cam5_T2    | 2026-06-30 09:00    |  ← Same person at Terminal 2!
| ts004 | person_xyz       | cam1_T1    | 2026-06-30 10:00    |  ← Person XYZ at Terminal 1
| ts005 | NULL             | cam2_T1    | 2026-06-30 11:00    |  ← Unidentified person (excluded)
| ts006 | person_xyz       | cam3_T1    | 2026-06-30 12:00    |  ← Person XYZ again
| ts007 | person_klm       | cam6_T2    | 2026-06-30 14:00    |  ← Person KLM at Terminal 2
+-------+------------------+------------+---------------------+
```

**Query Result WITHOUT Store Filter** (All Stores):
```sql
COUNT(DISTINCT person_identity_id) WHERE person_identity_id IS NOT NULL
= COUNT(DISTINCT [person_abc, person_abc, person_abc, person_xyz, person_xyz, person_klm])
= COUNT([person_abc, person_xyz, person_klm])
= 3 unique visitors
```

**Query Result WITH Store Filter** (Terminal 1 only, cameras: cam1_T1, cam2_T1, cam3_T1):
```sql
COUNT(DISTINCT person_identity_id) 
WHERE person_identity_id IS NOT NULL 
AND camera_id IN (cam1_T1, cam2_T1, cam3_T1)
= COUNT(DISTINCT [person_abc, person_abc, person_xyz, person_xyz])
= COUNT([person_abc, person_xyz])
= 2 unique visitors at Terminal 1
```

**Key Insight**: Person ABC visited both terminals but is counted only once in "All Stores" view, and counted only once in each terminal's individual view.

**Previous Period Query:** Same query with `prev_start` and `prev_end`

**Calculation:**
```python
total_visitors = 156  # Unique persons seen today
prev_visitors = 142   # Unique persons seen yesterday (same duration)
vs_prev_pct = round((156 - 142) / 142 * 100, 1) = +9.9%
```

---

### 3.5 Demographics (Gender & Age Groups)
**Database Tables:** `track_sessions`, `person_identities`

**Step 1: Get Distinct Person IDs in Time Range**
```sql
SELECT DISTINCT track_session.person_identity_id
FROM track_sessions
WHERE track_session.started_at >= start
  AND track_session.started_at <= end
  AND track_session.person_identity_id IS NOT NULL
  AND (track_session.camera_id IN ({cam_ids}) OR {cam_ids} IS NULL)
```

**Step 2: Join with PersonIdentity to Get Demographics**
```sql
SELECT person_identity.id, 
       person_identity.gender, 
       person_identity.estimated_age
FROM person_identities
WHERE person_identity.id IN ({distinct_person_ids})
```

**Step 3: Aggregate by Gender**
```python
gender_cnt = {"male": 0, "female": 0, "unidentified": 0}

for person_id, raw_gender, estimated_age in demo_rows:
    if person_id in seen:
        continue  # Count each person only once
    seen.add(person_id)
    
    # Normalize gender: M/MALE → "male", F/FEMALE → "female", else → "unidentified"
    gender = _v2_gender(raw_gender)
    gender_cnt[gender] += 1
```

**Result:**
```python
{
    "male": 89,           # 57.1% (89/156 * 100)
    "female": 54,         # 34.6% (54/156 * 100)
    "unidentified": 13    # 8.3% (13/156 * 100)
}
```

**Step 4: Aggregate by Age Groups**
Age bins defined in `_V2_AGE_BINS` (Line 732-739):
```python
[
    ("under_18",   "Under 18",  0,   17),
    ("age_18_24",  "18-24",    18,   24),
    ("age_25_34",  "25-34",    25,   34),
    ("age_35_44",  "35-44",    35,   44),
    ("age_45_60",  "45-60",    45,   60),
    ("age_60_plus","60+",      61,  999),
]
```

**Mapping Logic:**
```python
for person_id, raw_gender, estimated_age in demo_rows:
    if person_id in seen:
        continue
    seen.add(person_id)
    
    # Map estimated_age → age group key
    age_bin = _v2_age_bin(estimated_age)  # e.g., 28 → "age_25_34"
    age_cnt[age_bin] += 1
```

**Result:**
```python
{
    "under_18": 8,
    "age_18_24": 23,
    "age_25_34": 45,    # Peak group (highest count)
    "age_35_44": 38,
    "age_45_60": 27,
    "age_60_plus": 12,
    "unidentified": 3
}
```

**Peak Group Calculation:**
```python
peak_key = "age_25_34"  # Group with highest count
peak_label = "25-34 dominant"
```

---

### 3.6 Purchase Count (Billing Interactions)
**Database Table:** `billing_interactions`

**Current Period Query:**
```sql
SELECT COUNT(billing_interaction.id)
FROM billing_interactions
WHERE billing_interaction.entered_at >= '2026-06-30 00:00:00+05:30'
  AND billing_interaction.entered_at <= '2026-06-30 15:02:00+05:30'
  AND (billing_interaction.camera_id IN ({cam_ids}) OR {cam_ids} IS NULL)
```

**Previous Period Query:** Same with yesterday's timeframe

**Calculations:**
```python
total_purchases = 42      # Purchases today
prev_purchases = 38       # Purchases yesterday (same duration)
conversion_pct = round(42 / max(156, 1) * 100, 1) = 26.9%
vs_prev_pct = round((42 - 38) / 38 * 100, 1) = +10.5%
```

---

### 3.7 Footfall Over Time (Hourly Chart)
**Database Table:** `track_sessions`

**Granularity Resolution:**
```python
range_days = (end - start).total_seconds() / 86400  # 0.625 days
resolved = _resolve_group_by("auto", 0.625)  # Returns "hour" (≤2 days)
```

**Query: Count Unique Persons per Hour Bucket**
```sql
-- Step 1: Get distinct person per hour bucket (subquery)
SELECT DISTINCT
    date_trunc('hour', timezone('Asia/Kolkata', track_session.started_at)) as bucket,
    track_session.person_identity_id
FROM track_sessions
WHERE track_session.started_at >= '2026-06-30 00:00:00+05:30'
  AND track_session.started_at <= '2026-06-30 15:02:00+05:30'
  AND track_session.person_identity_id IS NOT NULL
  AND (track_session.camera_id IN ({cam_ids}) OR {cam_ids} IS NULL)

-- Step 2: Count per bucket
SELECT bucket, COUNT(person_identity_id) as cnt
FROM distinct_persons_subquery
GROUP BY bucket
ORDER BY bucket
```

**Result Map:**
```python
ff_map = {
    datetime(2026,6,30,0,0,tzinfo=IST): 3,   # 00:00 IST → 3 unique visitors
    datetime(2026,6,30,1,0,tzinfo=IST): 2,   # 01:00 IST → 2 unique visitors
    datetime(2026,6,30,6,0,tzinfo=IST): 8,   # 06:00 IST → 8 unique visitors
    datetime(2026,6,30,9,0,tzinfo=IST): 15,  # 09:00 IST → 15 unique visitors
    datetime(2026,6,30,12,0,tzinfo=IST): 28, # 12:00 IST → 28 unique visitors (peak)
    datetime(2026,6,30,15,0,tzinfo=IST): 12, # 15:00 IST → 12 unique visitors (partial hour)
    # ... other hours
}
```

**Gap Filling (Line 1019-1027):**
```python
# Generate ALL hour slots from 00:00 to 23:59 today
footfall_over_time = []
slot = truncate_slot(start, "hour")  # 00:00 today IST
display_end = end.replace(hour=23, minute=59, second=59)  # 23:59 today IST

while slot <= display_end:
    next_slot = slot + timedelta(hours=1)
    footfall_over_time.append({
        "label": slot.strftime("%H:%M"),  # "00:00", "01:00", ...
        "slot_start": slot,
        "slot_end": next_slot,
        "count": ff_map.get(slot, 0)  # 0 for future/empty hours
    })
    slot = next_slot
```

**Result Array (24 hourly points):**
```json
[
  {"label": "00:00", "slot_start": "2026-06-30T00:00:00+05:30", "slot_end": "2026-06-30T01:00:00+05:30", "count": 3},
  {"label": "01:00", "slot_start": "2026-06-30T01:00:00+05:30", "slot_end": "2026-06-30T02:00:00+05:30", "count": 2},
  ...
  {"label": "12:00", "slot_start": "2026-06-30T12:00:00+05:30", "slot_end": "2026-06-30T13:00:00+05:30", "count": 28},
  ...
  {"label": "15:00", "slot_start": "2026-06-30T15:00:00+05:30", "slot_end": "2026-06-30T16:00:00+05:30", "count": 12},
  {"label": "16:00", "slot_start": "2026-06-30T16:00:00+05:30", "slot_end": "2026-06-30T17:00:00+05:30", "count": 0},
  ...
  {"label": "23:00", "slot_start": "2026-06-30T23:00:00+05:30", "slot_end": "2026-07-01T00:00:00+05:30", "count": 0}
]
```

---

### 3.8 Gender Trend (Hourly Gender Breakdown)
**Database Tables:** `track_sessions`, `person_identities`

**Query: Count Unique Persons by Gender per Hour**
```sql
-- Step 1: Get distinct person per hour bucket with gender (subquery)
SELECT DISTINCT
    date_trunc('hour', timezone('Asia/Kolkata', track_session.started_at)) as bucket,
    track_session.person_identity_id,
    person_identity.gender
FROM track_sessions
JOIN person_identities ON person_identities.id = track_session.person_identity_id
WHERE track_session.started_at >= '2026-06-30 00:00:00+05:30'
  AND track_session.started_at <= '2026-06-30 15:02:00+05:30'
  AND track_session.person_identity_id IS NOT NULL
  AND (track_session.camera_id IN ({cam_ids}) OR {cam_ids} IS NULL)

-- Step 2: Count per bucket per gender
SELECT bucket, gender, COUNT(person_identity_id) as cnt
FROM distinct_gender_subquery
GROUP BY bucket, gender
ORDER BY bucket
```

**Result Map:**
```python
gt_map = {
    datetime(2026,6,30,9,0,tzinfo=IST): {"male": 9, "female": 5, "unidentified": 1},
    datetime(2026,6,30,12,0,tzinfo=IST): {"male": 16, "female": 10, "unidentified": 2},
    # ... other hours
}
```

**Gap Filling (similar to footfall):**
```json
[
  {"label": "00:00", "slot_start": "...", "slot_end": "...", "male": 2, "female": 1, "unidentified": 0},
  {"label": "01:00", "slot_start": "...", "slot_end": "...", "male": 1, "female": 1, "unidentified": 0},
  ...
  {"label": "12:00", "slot_start": "...", "slot_end": "...", "male": 16, "female": 10, "unidentified": 2},
  ...
  {"label": "23:00", "slot_start": "...", "slot_end": "...", "male": 0, "female": 0, "unidentified": 0}
]
```

---

## 4. Response Structure

**File:** `app/modules/analytics/schemas.py` (Line 322-354)

```json
{
  "store_id": null,
  "store_name": "All Stores",
  "time_range": "today",
  "start_time": "2026-06-30T00:00:00+05:30",
  "end_time": "2026-06-30T15:02:00+05:30",
  
  "total_cameras": 8,
  "active_cameras": 6,
  
  "footfall": {
    "total_visitors": 156,
    "vs_prev_pct": 9.9
  },
  
  "gender": {
    "male": 89,
    "female": 54,
    "unidentified": 13,
    "male_pct": 57.1,
    "female_pct": 34.6,
    "unidentified_pct": 8.3
  },
  
  "age_groups": {
    "under_18": 8,
    "age_18_24": 23,
    "age_25_34": 45,
    "age_35_44": 38,
    "age_45_60": 27,
    "age_60_plus": 12,
    "unidentified": 3,
    "peak_group": "25-34 dominant"
  },
  
  "purchase_count": {
    "total": 42,
    "conversion_pct": 26.9,
    "vs_prev_pct": 10.5
  },
  
  "footfall_over_time": [
    // 24 hourly data points (00:00 to 23:00)
  ],
  
  "gender_trend": [
    // 24 hourly stacked bar data points
  ],
  
  "age_group_distribution": [
    {"key": "under_18", "label": "Under 18", "count": 8},
    {"key": "age_18_24", "label": "18-24", "count": 23},
    {"key": "age_25_34", "label": "25-34", "count": 45},
    {"key": "age_35_44", "label": "35-44", "count": 38},
    {"key": "age_45_60", "label": "45-60", "count": 27},
    {"key": "age_60_plus", "label": "60+", "count": 12},
    {"key": "unidentified", "label": "Unidentified", "count": 3}
  ]
}
```

---

## 5. Database Tables Used

### 5.1 `cameras`
- **Purpose:** Filter by store, count total/active cameras
- **Key Columns:** `id`, `store_id`, `is_active`, `status`, `name`

### 5.2 `stores`
- **Purpose:** Lookup store name
- **Key Columns:** `id`, `name`

### 5.3 `track_sessions`
- **Purpose:** Source of visitor tracking data
- **Key Columns:** 
  - `id` (Primary Key)
  - `person_identity_id` (Foreign Key → person_identities)
  - `camera_id` (Foreign Key → cameras)
  - `started_at` (Timestamp when person appeared)
  - `last_seen_at` (Timestamp when person disappeared)

### 5.4 `person_identities`
- **Purpose:** Demographics (gender, age) for each unique person
- **Key Columns:**
  - `id` (Primary Key, linked from track_sessions.person_identity_id)
  - `gender` (Value: 'M', 'F', NULL)
  - `estimated_age` (Integer: 0-999)

### 5.5 `billing_interactions`
- **Purpose:** Purchase/checkout transactions
- **Key Columns:**
  - `id` (Primary Key)
  - `person_identity_id` (Foreign Key → person_identities)
  - `camera_id` (Foreign Key → cameras)
  - `entered_at` (Timestamp when person entered billing zone)
  - `dwell_seconds` (Time spent at billing counter)

---

## 6. Key Concepts

### 6.1 Unique Visitor Counting
- Uses **`person_identity_id`** from track_sessions
- Re-identification across cameras links multiple track sessions to same person
- Each person is counted only once, even if seen by multiple cameras

### 6.2 Time Bucketing (Hourly)
- PostgreSQL `date_trunc('hour', ...)` rounds timestamps to hour boundary
- Timezone conversion: `timezone('Asia/Kolkata', timestamp)` ensures IST bucketing
- Gap filling ensures all 24 hours are present (future hours show 0)

### 6.3 Comparison with Previous Period
- For "today" from 00:00 to 15:02, previous period is yesterday 00:00 to 15:02
- Percentage change: `(current - previous) / previous * 100`
- Returns `null` if previous period had 0 data (avoid division by zero)

### 6.4 Demographics Deduplication
- Multiple track sessions can link to same `person_identity_id`
- Uses `seen` set to count each person only once in demographics
- Demographics come from `person_identities` table (estimated via AI models)

---

## 7. Performance Considerations

1. **Indexes Required:**
   ```sql
   CREATE INDEX idx_track_sessions_started_at ON track_sessions(started_at);
   CREATE INDEX idx_track_sessions_person_identity_id ON track_sessions(person_identity_id);
   CREATE INDEX idx_track_sessions_camera_id ON track_sessions(camera_id);
   CREATE INDEX idx_billing_entered_at ON billing_interactions(entered_at);
   CREATE INDEX idx_cameras_store_id ON cameras(store_id);
   ```

2. **Query Optimization:**
   - Uses subqueries with `DISTINCT` to ensure unique person counting
   - `date_trunc` leverages index on `started_at`
   - Camera filter via `IN` clause allows index usage

3. **Caching Opportunities:**
   - Store name lookup can be cached
   - Camera counts change infrequently
   - Historical time slots (completed hours) can be cached

---

## 8. Data Flow Summary

```
HTTP Request (time_range=today)
    ↓
Router (dashboard_v2) - Authentication & Validation
    ↓
Service (get_dashboard_v2)
    ↓
├─→ Time Range Resolution: today → 00:00 IST to now
├─→ Camera Filter: store_id → cam_ids[]
├─→ Store Lookup: store_id → "GMR - T2"
├─→ Camera Stats: COUNT queries → total_cameras, active_cameras
├─→ Footfall: COUNT DISTINCT person_identity_id → total_visitors + % change
├─→ Demographics: JOIN person_identities → gender & age breakdowns
├─→ Purchases: COUNT billing_interactions → total + conversion % + % change
├─→ Footfall Timeline: DISTINCT persons per hour → 24 hourly points (gap-filled)
├─→ Gender Timeline: DISTINCT persons by gender per hour → 24 stacked bar points
    ↓
Schemas (DashboardV2Response) - JSON Serialization
    ↓
HTTP Response (JSON)
```

---

## 9. Testing the Endpoint

```bash
# All stores, today
curl -X GET "http://localhost:8000/api/v2/analytics/dashboard?time_range=today" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Specific store, today
curl -X GET "http://localhost:8000/api/v2/analytics/dashboard?store_id=123e4567-e89b-12d3-a456-426614174000&time_range=today" \
  -H "Authorization: Bearer YOUR_TOKEN"

# This week
curl -X GET "http://localhost:8000/api/v2/analytics/dashboard?time_range=this_week" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Custom range
curl -X GET "http://localhost:8000/api/v2/analytics/dashboard?time_range=custom&start_time=2026-06-25T00:00:00&end_time=2026-06-30T23:59:59" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

---

## 10. Related Endpoints

- `/api/v2/analytics/metrics?metric=footfall` - Detailed footfall metrics with peak hours
- `/api/v2/analytics/metrics?metric=gender` - Detailed gender analytics
- `/api/v2/analytics/metrics?metric=age_groups` - Detailed age group analytics
- `/api/v2/analytics/metrics?metric=purchase` - Detailed purchase analytics

---

**Document Version:** 1.0  
**Last Updated:** 2026-06-30  
**Author:** System Documentation
