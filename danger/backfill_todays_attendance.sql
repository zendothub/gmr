-- Backfill today's attendance records that were missed due to UTC vs IST timezone bug.
-- 
-- The bug: detected_at was UTC but shift times are IST. _is_within_shift compared
-- UTC 06:15 against IST 09:00 → always False → no record created.
--
-- Run this inside the postgres container:
--   docker exec -i coal_mine_postgres psql -U coal_mine_user -d coal_mine_db < danger/backfill_todays_attendance.sql

-- Define IST offset
-- NOTE: Adjust '2026-09-10' below to the correct date if needed
WITH target_date AS (
    SELECT '2026-09-10'::date AS dt
),
day_bounds AS (
    SELECT
        dt,
        (dt AT TIME ZONE 'Asia/Kolkata') AT TIME ZONE 'UTC' AS day_start_utc,
        ((dt + interval '1 day') AT TIME ZONE 'Asia/Kolkata') AT TIME ZONE 'UTC' AS day_end_utc
    FROM target_date
),
active_employees AS (
    SELECT e.id, e.emp_id, e.name, e.person_identity_id, e.shift_slot_id,
           s.label AS shift_label, s.start_time, s.end_time, s.crosses_midnight
    FROM employees e
    JOIN shift_slots s ON s.id = e.shift_slot_id
    WHERE e.is_active = true AND e.person_identity_id IS NOT NULL
),
-- Get track sessions for these employees within the IST day
employee_sessions AS (
    SELECT ae.*, ts.started_at AS first_seen_utc,
           COALESCE(ts.last_seen_at, ts.started_at) AS last_seen_utc,
           ts.camera_id
    FROM active_employees ae
    JOIN track_sessions ts ON ts.person_identity_id = ae.person_identity_id
    CROSS JOIN day_bounds db
    WHERE ts.started_at >= db.day_start_utc
      AND ts.started_at < db.day_end_utc
),
-- Aggregate per employee: first and last seen
employee_agg AS (
    SELECT
        id, emp_id, name, person_identity_id, shift_slot_id,
        shift_label, start_time, end_time, crosses_midnight,
        MIN(first_seen_utc) AS first_seen_utc,
        MAX(last_seen_utc) AS last_seen_utc,
        (array_agg(camera_id ORDER BY first_seen_utc ASC))[1] AS check_in_camera_id,
        (array_agg(camera_id ORDER BY last_seen_utc DESC))[1] AS check_out_camera_id
    FROM employee_sessions
    GROUP BY id, emp_id, name, person_identity_id, shift_slot_id,
             shift_label, start_time, end_time, crosses_midnight
),
-- Convert to IST for shift window check
employee_ist AS (
    SELECT *,
           first_seen_utc AT TIME ZONE 'Asia/Kolkata' AS first_seen_ist,
           last_seen_utc AT TIME ZONE 'Asia/Kolkata' AS last_seen_ist
    FROM employee_agg
    CROSS JOIN target_date
),
-- Filter: within shift window
-- For day shifts: shift_start <= first_seen_ist_time <= shift_end
-- For night shifts: handles crossing midnight
within_shift AS (
    SELECT e.*,
           EXTRACT(EPOCH FROM (e.last_seen_utc - e.first_seen_utc)) / 3600.0 AS total_hours,
           -- Determine status: late if first_seen_ist > shift_start + 15 min
           CASE
               WHEN (e.first_seen_ist::time) > (e.start_time + interval '15 minutes') THEN 'late'
               ELSE 'present'
           END AS status
    FROM employee_ist e
    WHERE
        -- Day shift: within same day
        (e.crosses_midnight = false
         AND (e.first_seen_ist::time) >= e.start_time
         AND (e.first_seen_ist::time) <= e.end_time)
        OR
        -- Night shift: spans midnight (e.g. 19:00 - 04:00)
        (e.crosses_midnight = true
         AND (
             -- Evening portion (19:00 - 23:59)
             ((e.first_seen_ist::time) >= e.start_time)
             OR
             -- Early morning portion (00:00 - 04:00)
             ((e.first_seen_ist::time) <= e.end_time)
         ))
),
-- Exclude employees who already have an attendance record for this date
new_records AS (
    SELECT ws.*
    FROM within_shift ws
    LEFT JOIN attendance_records ar
        ON ar.employee_id = ws.id
        AND ar.attendance_date = ws.dt
    WHERE ar.id IS NULL
)
-- Insert the missing attendance records
INSERT INTO attendance_records (
    id, employee_id, attendance_date, shift_slot_id,
    first_seen_at, last_seen_at, total_hours, status,
    check_in_camera_id, check_out_camera_id,
    created_at, updated_at
)
SELECT
    gen_random_uuid(),
    id,
    dt,
    shift_slot_id,
    first_seen_utc,
    last_seen_utc,
    ROUND(total_hours::numeric, 4),
    status::attendance_status,
    check_in_camera_id,
    check_out_camera_id,
    NOW(),
    NOW()
FROM new_records;

-- Show what was inserted
SELECT
    e.emp_id,
    e.name,
    nr.status,
    nr.first_seen_utc AT TIME ZONE 'Asia/Kolkata' AS first_seen_ist,
    nr.last_seen_utc AT TIME ZONE 'Asia/Kolkata' AS last_seen_ist,
    ROUND(nr.total_hours::numeric, 2) AS total_hours
FROM new_records nr
JOIN employees e ON e.id = nr.id
ORDER BY e.emp_id;