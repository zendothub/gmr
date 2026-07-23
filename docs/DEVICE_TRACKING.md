# Device Tracking — Live Viewer Monitoring

> Tracks how many unique devices are currently logged in and watching live camera feeds, even when multiple users share the same credentials.

---

## Problem

When the same credentials are shared across multiple users/devices, the system has no way to distinguish them. This feature fingerprints each device (browser + IP) and tracks:

1. **How many devices are currently logged in** (active sessions)
2. **How many devices are watching live camera feeds** (stream viewers)
3. **Which cameras each device is watching** and for how long

---

## Architecture

```
┌──────────────┐     ┌─────────────────────┐     ┌──────────────────────────┐
│  User Login  │────▶│  DeviceSession       │     │  device_sessions table   │
│  (auth)      │     │  created on login    │     │  (user_id, device_hash,  │
└──────────────┘     │  with device hash    │     │   UA, IP, last_active)   │
                     └─────────────────────┘     └──────────────────────────┘
                                                           │
┌──────────────┐     ┌─────────────────────┐              │
│  Every API   │────▶│  DeviceTracker       │──────────────┘
│  request     │     │  Middleware          │  updates last_active_at
└──────────────┘     │  (extracts UA+IP)    │
                     └─────────────────────┘

┌──────────────┐     ┌─────────────────────┐     ┌──────────────────────────┐
│  Start       │────▶│  StreamViewerSession │     │  stream_viewer_sessions  │
│  Streaming   │     │  created with        │     │  (camera_id, device_hash,│
│  (HLS/WS)    │     │  camera_id + device  │     │   started_at, ended_at)  │
└──────────────┘     └─────────────────────┘     └──────────────────────────┘

┌──────────────┐     ┌─────────────────────┐
│  Stop        │────▶│  ended_at = now()   │
│  Streaming   │     │  (or marked by      │
│  / Reap      │     │   cleanup job)      │
└──────────────┘     └─────────────────────┘

┌──────────────┐     ┌─────────────────────┐
│  Scheduled   │────▶│  Deactivate idle     │
│  Job (5 min) │     │  device sessions     │
│              │     │  Mark stale viewers  │
└──────────────┘     └─────────────────────┘
```

### Device Fingerprint

Each device is identified by a SHA-256 hash of:
```
{user_agent}|{ip_address}
```

This means:
- Same browser on same network → same device hash
- Different browser OR different network → different device hash
- Same credentials, different device → tracked separately

---

## Database Tables

### `device_sessions`

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `user_id` | UUID → users.id | Which user account |
| `device_hash` | VARCHAR(128) | SHA-256 of UA+IP |
| `user_agent` | TEXT | Raw browser user-agent string |
| `ip_address` | VARCHAR(45) | Client IP address |
| `login_at` | TIMESTAMPTZ | When the session was created |
| `last_active_at` | TIMESTAMPTZ | Last API request from this device |
| `expires_at` | TIMESTAMPTZ | Token expiry time |
| `is_active` | BOOLEAN | Whether session is still active |

### `stream_viewer_sessions`

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `user_id` | UUID → users.id | Which user (nullable for anonymous) |
| `camera_id` | UUID → cameras.id | Which camera feed |
| `device_hash` | VARCHAR(128) | SHA-256 of UA+IP |
| `ip_address` | VARCHAR(45) | Client IP |
| `user_agent` | TEXT | Raw browser user-agent |
| `started_at` | TIMESTAMPTZ | When viewing started |
| `ended_at` | TIMESTAMPTZ | When viewing ended (NULL = still watching) |
| `last_heartbeat_at` | TIMESTAMPTZ | Last stream activity ping |

---

## API Endpoints

### `GET /api/analytics/live-viewers`

Real-time snapshot of devices watching live feeds.

**Query Parameters:**
| Param | Type | Description |
|-------|------|-------------|
| `store_id` | UUID (optional) | Filter by store |

**Response:**
```json
{
  "total_devices_connected": 5,
  "total_devices_watching_feeds": 3,
  "cameras": [
    {
      "camera_id": "uuid-here",
      "camera_name": "Entry Camera",
      "active_viewers": 2,
      "viewers": [
        {
          "device_hash": "abc123...",
          "device_label": "Chrome 126 / Windows",
          "ip_address": "192.168.1.100",
          "camera_id": "uuid-here",
          "camera_name": "Entry Camera",
          "viewing_since": "2026-07-23T10:15:00Z",
          "duration_minutes": 12.5
        }
      ]
    }
  ]
}
```

### `GET /api/auth/sessions`

List all active device sessions for the current user.

**Response:**
```json
{
  "sessions": [
    {
      "id": "uuid-here",
      "device_hash": "abc123...",
      "device_label": "Chrome 126 / Windows",
      "ip_address": "192.168.1.100",
      "login_at": "2026-07-23T09:00:00Z",
      "last_active_at": "2026-07-23T10:30:00Z",
      "is_active": true
    }
  ],
  "total_active": 1
}
```

---

## Configuration

In `app/config.py`:

```python
# How long a device session stays active without any API calls
SESSION_IDLE_TIMEOUT_SECONDS: int = 1800  # 30 minutes
```

---

## Cleanup

A scheduled job runs every **5 minutes** (`cleanup_stale_sessions`):

1. **Device sessions:** Deactivates sessions where:
   - `expires_at` has passed, OR
   - `last_active_at` is older than `SESSION_IDLE_TIMEOUT_SECONDS`

2. **Stream viewers:** Marks `ended_at` for viewers where:
   - `ended_at` is NULL (still watching), AND
   - `last_heartbeat_at` is older than 2 hours (stream was reaped but viewer row never cleaned up)

---

## Files

| File | Purpose |
|------|---------|
| `app/config.py` | `SESSION_IDLE_TIMEOUT_SECONDS` setting |
| `app/core/db/models/device_session.py` | DeviceSession SQLAlchemy model |
| `app/core/db/models/stream_viewer.py` | StreamViewerSession SQLAlchemy model |
| `app/utils/device_fingerprint.py` | SHA-256 device hashing + label extraction |
| `app/utils/encryption.py` | JWT `jti` (token ID) for session tracking |
| `app/modules/auth/service.py` | Creates DeviceSession on login |
| `app/modules/auth/schemas.py` | Session list response schemas |
| `app/modules/auth/router.py` | `GET /api/auth/sessions` endpoint |
| `app/middleware/device_tracker.py` | Middleware — updates `last_active_at` on every request |
| `app/modules/streaming/service.py` | Creates StreamViewerSession on stream start |
| `app/modules/streaming/manager.py` | Marks viewers ended when streams are reaped |
| `app/modules/streaming/router.py` | Passes `Request` to service for IP/UA |
| `app/modules/analytics/schemas.py` | `LiveViewersResponse` schema |
| `app/modules/analytics/router.py` | `GET /api/analytics/live-viewers` endpoint |
| `app/modules/analytics/service.py` | `get_live_viewers()` query logic |
| `app/modules/jobs/scheduler.py` | Registers `cleanup_stale_sessions` job |
| `app/modules/jobs/tasks.py` | `cleanup_stale_sessions()` implementation |
| `alembic/versions/0006_*.py` | Database migration |

---

## Deployment Steps

1. **Run migration:**
   ```bash
   alembic upgrade head
   ```

2. **Register middleware in `app/main.py`:**
   ```python
   from app.middleware.device_tracker import DeviceTrackerMiddleware
   app.add_middleware(DeviceTrackerMiddleware)
   ```

3. **Restart the server:**
   ```bash
   sudo systemctl restart retail-ai
   ```

4. **Verify:**
   ```bash
   curl -H "Authorization: Bearer <token>" \
     http://localhost:8000/api/analytics/live-viewers