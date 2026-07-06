# Complete System Flow: Camera Processing, ReID & Multi-Camera Synchronization

**Document Version:** 2.0  
**Last Updated:** July 6, 2026  
**Author:** RetailAIPlatform Team

> **IMPORTANT:** As of July 2026, each camera gets its own YOLO model instance (not shared) to isolate ByteTrack's `persist=True` tracking state. See `CRITICAL_FINDINGS.md` for details. The inference pool now uses `MAX_WORKERS` from config (default 10, set to 12 in `.env` for 2 cameras).

---

## Table of Contents

1. [Camera Feed Source & Join Points](#1-camera-feed-source--join-points)
2. [Zone Requirement for Person Detection](#2-zone-requirement-for-person-detection)
3. [Frame Processing Rate & ReID Execution](#3-frame-processing-rate--reid-execution)
4. [Multi-Camera Person Synchronization](#4-multi-camera-person-synchronization)
5. [Complete Architecture Diagram](#5-complete-architecture-diagram)

---

## 1. Camera Feed Source & Join Points

### 1.1 RTSP Stream Capture

**File:** `app/modules/ai_runtime/frame_buffer.py` (Lines 83-143)

The camera feed originates from an **RTSP stream URL** configured for each camera.

```python
class LatestFrameBuffer:
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url  # e.g., "rtsp://192.168.1.100:554/stream1"
        
    def _capture_loop(self):
        """Background thread continuously reads frames from RTSP"""
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        
        while not self._stop_event.is_set():
            ret, frame = cap.read()  # Read frame from RTSP stream
            
            if ret and frame is not None:
                with self._lock:
                    self._frame = frame          # Store latest frame
                    self._frame_ts = time.time() # Timestamp
                    self.frames_captured += 1
```

**Key Points:**
- Dedicated background thread per camera
- Uses OpenCV's FFmpeg backend for RTSP decoding
- Only keeps **latest frame** in memory (no frame queue buildup)
- Overwrites old frames → prevents memory/latency issues
- Runs continuously at camera's native frame rate (e.g., 25 FPS)

---

### 1.2 Join Point: Camera Worker Processing Loop

**File:** `app/modules/ai_runtime/camera_worker.py` (Lines 235-290)

The **join point** where RTSP frames meet the AI pipeline:

```python
class CameraWorker:
    async def _run_loop(self):
        """Main processing loop - THE JOIN POINT"""
        interval = 1.0 / self.fps_target  # e.g., 1/5 = 0.2s for 5 FPS
        
        while self.is_running:
            # JOIN POINT: Pull latest frame from RTSP buffer
            frame, frame_ts = self.frame_buffer.get_latest()
            
            if frame is None or frame_ts <= last_frame_ts:
                await asyncio.sleep(interval / 2)
                continue  # No new frame yet, skip
            
            last_frame_ts = frame_ts
            
            # Process this frame through AI pipeline
            await self._process_frame(frame)  # <- PERSON DETECTION HAPPENS HERE
            
            # Sleep to maintain target FPS
            await asyncio.sleep(interval)
```

**Visual Flow:**

```
RTSP Camera (25 FPS) 
    ↓ (continuous capture)
LatestFrameBuffer (thread-safe, overwrites)
    ↓ (sampled at fps_target)
CameraWorker._run_loop() [JOIN POINT]
    ↓
_process_frame()
    ↓
YOLO Detection + Tracking
    ↓
Track Management + Zone Updates
    ↓
ReID Pipeline
    ↓
Database Persistence
```

---

### 1.3 Tracked Person Data Flow

**File:** `app/modules/ai_runtime/camera_worker.py` (Lines 292-362)

```python
async def _process_frame(self, frame):
    """Complete pipeline for each sampled frame"""
    
    # STEP 1: YOLO Detection + Tracking
    tracked_detections = await run_inference(self.detector.track, frame)
    # Returns: [{track_id: 1, bbox: {x1, y1, x2, y2}, confidence: 0.87}, ...]
    
    # STEP 2: Update Track State (in-memory)
    for td in tracked_detections:
        track = self.track_manager.update_track(
            td.track_id,  # YOLO's persistent ID
            td.bbox,
            td.confidence
        )
        
        # Update zones (check if person is inside any zone polygons)
        self.track_manager.update_zones(track, self.zones, width, height)
        
        # Check if eligible for ReID
        if track.should_run_reid():
            reid_tracks.append(track)
    
    # STEP 3: Run ReID for eligible tracks
    for track in reid_tracks:
        await self._run_reid(db, frame, track)  # Extract crop, assess quality, match
    
    # STEP 4: Persist to Database (batched)
    await self._persist_batch(...)
```

**Data Join Points:**
1. **RTSP → Frame Buffer** (continuous, 25 FPS)
2. **Frame Buffer → Worker Loop** (sampled, 5 FPS default)
3. **YOLO → Track Manager** (every processed frame)
4. **Track → ReID Engine** (when eligible + quality passed)
5. **ReID → PostgreSQL** (after 5-frame accumulation)

---

## 2. Zone Requirement for Person Detection

### ✅ **Answer: Zones are NOT required for person detection**

**File:** `app/modules/ai_runtime/camera_worker.py` (Lines 294-320)

```python
async def _process_frame(self, frame):
    # YOLO runs on FULL FRAME - no zone filtering
    tracked_detections = await run_inference(self.detector.track, frame)
    
    for td in tracked_detections:
        # Track is created REGARDLESS of zones
        track = self.track_manager.update_track(td.track_id, td.bbox, td.confidence)
        
        # Zone update is OPTIONAL - updates which zones track is in
        self.track_manager.update_zones(track, self.zones, width, height)
        # If self.zones is empty, current_zones will be empty set
        
        # ReID runs REGARDLESS of zones
        if track.should_run_reid():
            reid_tracks.append(track)
```

**What Zones Are Used For:**
- **Rule Engine**: Triggers events when person enters/exits/dwells in zones
- **Billing Tracking**: Detect interactions at checkout counters
- **Analytics**: Zone-based heatmaps, traffic flow
- **Event Filtering**: Focus alerts on specific areas

**What Works WITHOUT Zones:**
- ✅ Person detection (YOLO)
- ✅ Person tracking (ByteTrack)
- ✅ ReID (person identification across cameras)
- ✅ Demographics (age/gender)
- ✅ Track sessions (entry/exit events)

**Example: Zero-Zone Configuration**

```python
camera_config = {
    "id": uuid4(),
    "rtsp_url": "rtsp://192.168.1.100:554/stream1",
    "fps_target": 5,
    "reid_enabled": True,
    "demographic_enabled": True
}

runtime_config = {
    "zones": [],  # EMPTY - still works!
    "rules": []   # No zone-based rules
}

worker = CameraWorker(camera_config, runtime_config)
# Person detection, tracking, ReID all functional
```

---

## 3. Frame Processing Rate & ReID Execution

### 3.1 Frame Sampling Strategy

**Scenario:** Camera sends 20 frames per second, `fps_target = 5`

**File:** `app/modules/ai_runtime/camera_worker.py` (Lines 235-257)

```python
async def _run_loop(self):
    interval = 1.0 / self.fps_target  # 1/5 = 0.2 seconds
    last_frame_ts = 0.0
    
    while self.is_running:
        loop_start = time.time()
        
        # Get LATEST frame (not oldest/queued)
        frame, frame_ts = self.frame_buffer.get_latest()
        
        # Skip if same frame as before
        if frame_ts <= last_frame_ts:
            await asyncio.sleep(interval / 2)
            continue
        
        last_frame_ts = frame_ts
        
        # Process THIS frame
        await self._process_frame(frame)
        
        # Sleep remainder to maintain target FPS
        elapsed = time.time() - loop_start
        sleep_for = max(0.001, interval - elapsed)
        await asyncio.sleep(sleep_for)  # Sleep ~200ms
```

**Answer:**

| Camera Native FPS | fps_target | Frames Processed | Frames Skipped |
|-------------------|-----------|------------------|----------------|
| 20 FPS | 5 | 5 per second | 15 per second |
| 25 FPS | 5 | 5 per second | 20 per second |
| 30 FPS | 10 | 10 per second | 20 per second |

**📊 Visual Timeline (20 FPS camera, fps_target=5):**

```
Time:     0ms   50ms  100ms 150ms 200ms 250ms 300ms 350ms 400ms ...
Camera:   F1    F2    F3    F4    F5    F6    F7    F8    F9    ...
Buffer:   ↓     ↓     ↓     ↓     ↓     ↓     ↓     ↓     ↓     
          [F1]→[F2]→[F3]→[F4]→[F5]→[F6]→[F7]→[F8]→[F9]→

Worker:   ↓                 ↓                 ↓                 
          Process F4        Process F8        Skip (F8 again)
          (200ms)           (200ms)           Wait for F12...
```

**Key Points:**
1. **Does NOT process all frames** - only samples at `fps_target` rate
2. **Always uses LATEST frame** - skips intermediate frames
3. **No frame queue** - prevents latency buildup
4. **Configurable per camera** - based on compute budget

**Why Not Process All Frames?**
- CPU/GPU would be overloaded (5 cameras × 25 FPS = 125 frames/sec)
- ReID is expensive (OSNet + InsightFace take 50-100ms per person)
- Marginal benefit (person appearance doesn't change in 40ms)
- Target 5 FPS provides smooth tracking with manageable compute

---

### 3.2 ReID Execution Frequency

**File:** `app/modules/tracking/track_manager.py` (Lines 57-74)

```python
class ActiveTrack:
    def should_run_reid(self) -> bool:
        """Decides if ReID should run for this track."""
        
        if not self.bbox:
            return False
        
        # Minimum crop size (100 pixels tall)
        if bbox_height(self.bbox) < 100:
            return False
        
        # If already confident, run every 5th eligible frame
        if self.reid_confident:
            return self.reid_frame_count % 5 == 0
        
        return True  # Run every eligible frame
```

**File:** `app/modules/ai_runtime/camera_worker.py` (Lines 316-320)

```python
for td in tracked_detections:
    track = self.track_manager.update_track(...)
    
    # ReID eligibility check
    if (self.reid_enabled and 
        self.reid_extractor and 
        track.should_run_reid()):  # <- Gating logic here
        reid_tracks.append(track)
```

**ReID Execution Rules:**

| Track State | ReID Frequency | Reason |
|-------------|----------------|--------|
| New track (unresolved) | Every eligible frame | Need fast identification |
| Crop height < 100px | Skip | Too small for quality embedding |
| ReID confident (score ≥ 0.75) | Every 5th eligible frame | Already matched, save compute |
| Track has no bbox | Skip | Invalid state |

**Example Timeline (fps_target=5):**

```
Frame: 1    2    3    4    5    6    7    8    9    10
Track: New  New  New  New  New  Conf Conf Conf Conf Conf
ReID:  ✓    ✓    ✓    ✓    ✓    ✗    ✗    ✗    ✗    ✓

Frame 1-5: Unconfident → ReID every frame (5 embeddings accumulated)
Frame 6-9: Confident → Skip ReID (save compute)
Frame 10:  5th frame since confident → Run ReID (refinement check)
```

**Accumulation Window:**

**File:** `app/modules/ai_runtime/camera_worker.py` (Lines 606-650)

```python
# Accumulate embeddings over 5 frames
accum_list = self.track_embeddings.setdefault(track.local_track_id, [])
accum_list.append((embedding, quality, crop_path, face_embedding, ...))

# Execute decision after 5 good frames
if len(accum_list) == self.settings.REID_ACCUMULATION_FRAMES:  # 5
    # Average embeddings
    embeddings = [item[0] for item in accum_list]
    mean_embedding = np.mean(embeddings, axis=0)
    
    # Match against database
    person_id, score, is_confident, ... = await self.identity_engine.decide_identity(...)
    
    accum_list.clear()  # Reset for next window
```

**Answer: Does ReID run for each frame?**

❌ **No** - ReID is gated by:
1. Frame sampling rate (fps_target)
2. Track eligibility (size, state)
3. Confidence status (every 5th if confident)
4. Crop quality (≥ 0.50 threshold)
5. Accumulation window (waits for 5 good crops)

**Typical Execution:**
- Camera: 25 FPS (1 frame every 40ms)
- Worker samples: 5 FPS (1 frame every 200ms)
- ReID attempts: ~3-5 per second per track (when unconfident)
- ReID decisions: 1 per second per track (5-frame windows)

---

## 4. Multi-Camera Person Synchronization

### 4.1 Architecture: Independent Workers + Shared Database

**File:** `app/modules/ai_runtime/worker_supervisor.py`

```python
class WorkerSupervisor:
    """Manages multiple CameraWorker instances."""
    
    def __init__(self):
        self.workers: Dict[uuid.UUID, CameraWorker] = {}
    
    async def start_camera(self, camera_id: uuid.UUID):
        """Start independent worker for this camera."""
        worker = CameraWorker(camera_config, runtime_config)
        self.workers[camera_id] = worker
        await worker.start()
```

**Each camera runs INDEPENDENTLY:**
- Separate RTSP capture thread
- Separate processing loop
- Separate track state (local track IDs)
- Separate frame timing
- **NO direct inter-camera communication**

---

### 4.2 Synchronization Point: PostgreSQL + pgvector

**Cameras synchronize through the shared database:**

**File:** `app/modules/reid/identity_decision_engine.py` (Lines 211-249)

```python
class IdentityDecisionEngine:
    async def _search_similar(self, db: AsyncSession, embedding: np.ndarray):
        """Search ALL cameras for similar body embeddings."""
        
        query = text("""
            SELECT pe.person_identity_id, pe.camera_id, pe.crop_quality,
                   pe.embedding <=> :embedding AS distance
            FROM person_embeddings pe
            JOIN person_identities pi ON pe.person_identity_id = pi.id
            WHERE pe.captured_at > NOW() - INTERVAL '48 hours'  -- Time window
            ORDER BY pe.embedding <=> :embedding  -- Cosine distance
            LIMIT 5
        """)
        
        candidates = await db.execute(query, {"embedding": embedding})
        # Returns matches from ANY camera in the last 48 hours
```

**Synchronization Flow:**

```
Camera A (Gate 1)          Camera B (Gate 2)          PostgreSQL Database
     ↓                          ↓                             ↓
YOLO Track ID: 7          YOLO Track ID: 3          person_identities
     ↓                          ↓                    person_embeddings
Extract Embedding         Extract Embedding                 ↓
     ↓                          ↓                             ↓
Quality Check (0.78)      Quality Check (0.82)              ↓
     ↓                          ↓                             ↓
Accumulate 5 frames       Accumulate 5 frames               ↓
     ↓                          ↓                             ↓
Average Embedding         Average Embedding                 ↓
     ↓                          ↓                             ↓
Search Database ←─────────────→ Search Database              ↓
     ↓                          ↓                             ↓
Match Person UUID: abc123-...  Match SAME UUID!              ↓
     ↓                          ↓                             ↓
Store embedding           Store embedding                   ↓
to database    ──────────→     to database    ──────────→   Merged
```

---

### 4.3 Cross-Camera Person Matching Example

**Scenario:** Person walks from Camera A to Camera B

**Timeline:**

```
Time  | Camera A (Entrance)           | Camera B (Checkout Counter)    | Database
------|-------------------------------|--------------------------------|----------
00:00 | Person enters FOV             | (empty)                        | -
00:01 | YOLO assigns track_id=7       | -                              | -
00:02 | ReID: Extract embedding       | -                              | -
00:03 | ReID: Quality check pass      | -                              | -
00:04 | ReID: Accumulate frame 1/5    | -                              | -
00:05 | ReID: Accumulate frame 2/5    | -                              | -
00:06 | ReID: Accumulate frame 3/5    | -                              | -
00:07 | ReID: Accumulate frame 4/5    | -                              | -
00:08 | ReID: Accumulate frame 5/5    | -                              | -
00:09 | Decision: No match found      | -                              | -
00:10 | Create NEW person UUID        | -                              | person_id: abc123
00:11 | Store embedding to DB         | -                              | ✓ Stored (Camera A)
00:12 | Track continues...            | -                              | -
------|-------------------------------|--------------------------------|----------
00:45 | Person exits FOV              | Person enters FOV              | -
00:46 | Track closed                  | YOLO assigns track_id=3        | -
00:47 | -                             | ReID: Extract embedding        | -
00:48 | -                             | ReID: Quality check pass       | -
00:49 | -                             | ReID: Accumulate 1/5           | -
00:50 | -                             | ReID: Accumulate 2/5           | -
00:51 | -                             | ReID: Accumulate 3/5           | -
00:52 | -                             | ReID: Accumulate 4/5           | -
00:53 | -                             | ReID: Accumulate 5/5           | -
00:54 | -                             | Search DB for similar          | Query pgvector
00:55 | -                             | MATCH FOUND! (Camera A)        | person_id: abc123
00:56 | -                             | Similarity: 0.72 ✓             | (same person!)
00:57 | -                             | Assign person_id: abc123       | ✓ Update visit count
00:58 | -                             | Store new embedding            | ✓ Stored (Camera B)
```

**Database State After:**

```sql
-- person_identities table
id: abc123-...
first_seen_at: 2026-06-30 00:10:00 (Camera A)
last_seen_at: 2026-06-30 00:58:00 (Camera B)
visit_count: 2  -- Seen on 2 cameras

-- person_embeddings table
person_id: abc123, camera_id: camera_a, captured_at: 00:11:00
person_id: abc123, camera_id: camera_b, captured_at: 00:58:00
```

---

### 4.4 Key Synchronization Mechanisms

**1. Shared Person Identity Table**

```sql
CREATE TABLE person_identities (
    id UUID PRIMARY KEY,
    label TEXT,                    -- Optional name
    first_seen_at TIMESTAMP,       -- First detection (any camera)
    last_seen_at TIMESTAMP,        -- Latest detection (any camera)
    visit_count INTEGER,           -- Total camera encounters
    is_anonymous BOOLEAN
);
```

**2. Cross-Camera Embedding Search**

**File:** `app/modules/reid/identity_decision_engine.py` (Lines 211-253)

```python
# Search does NOT filter by camera_id
query = text("""
    SELECT pe.person_identity_id, pe.camera_id, ...
    FROM person_embeddings pe
    WHERE pe.captured_at > NOW() - INTERVAL '48 hours'  -- Time window only
    ORDER BY pe.embedding <=> :embedding
    LIMIT 5
""")
# Returns best matches across ALL cameras
```

**3. pgvector Cosine Similarity**

```python
# Camera A stores embedding: [0.12, 0.45, 0.78, ...]
# Camera B searches with:     [0.13, 0.46, 0.77, ...]
# PostgreSQL computes:        cosine_distance = 0.28
# Similarity score:           1 - 0.28 = 0.72 ✓ (above 0.60 threshold)
```

**4. Time Window (48 hours)**

- Prevents matching ancient/stale identities
- Focuses on recent store visits
- Balances recall vs. false positives

---

### 4.5 Multi-Camera Tracking Analytics

**File:** `app/modules/analytics/service.py`

```python
async def get_person_journey(person_id: uuid.UUID):
    """Get cross-camera journey for a person."""
    
    query = """
        SELECT ts.camera_id, c.name, ts.entered_at, ts.exited_at,
               ts.zones_visited
        FROM track_sessions ts
        JOIN cameras c ON ts.camera_id = c.id
        WHERE ts.person_identity_id = :person_id
        ORDER BY ts.entered_at
    """
    # Returns complete path: Camera A → Camera B → Camera C
```

**Example Output:**

```json
{
  "person_id": "abc123-...",
  "total_visits": 3,
  "journey": [
    {
      "camera": "Entrance Gate",
      "entered_at": "2026-06-30T10:15:00Z",
      "exited_at": "2026-06-30T10:16:30Z",
      "zones_visited": ["entrance_zone"]
    },
    {
      "camera": "Checkout Counter 1",
      "entered_at": "2026-06-30T10:23:00Z",
      "exited_at": "2026-06-30T10:28:00Z",
      "zones_visited": ["billing_zone_1"]
    },
    {
      "camera": "Exit Gate",
      "entered_at": "2026-06-30T10:30:00Z",
      "exited_at": "2026-06-30T10:31:00Z",
      "zones_visited": ["exit_zone"]
    }
  ]
}
```

---

## 5. Complete Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        RETAIL AI PLATFORM                                │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Camera A      │  │   Camera B      │  │   Camera C      │
│   (Entrance)    │  │   (Checkout)    │  │   (Exit)        │
│                 │  │                 │  │                 │
│  RTSP: 25 FPS   │  │  RTSP: 25 FPS   │  │  RTSP: 25 FPS   │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                     │
         ▼                    ▼                     ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ FrameBuffer A   │  │ FrameBuffer B   │  │ FrameBuffer C   │
│ (thread-safe)   │  │ (thread-safe)   │  │ (thread-safe)   │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                     │
         ▼ (5 FPS)            ▼ (5 FPS)             ▼ (5 FPS)
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ CameraWorker A  │  │ CameraWorker B  │  │ CameraWorker C  │
│                 │  │                 │  │                 │
│ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │
│ │ YOLO        │ │  │ │ YOLO        │ │  │ │ YOLO        │ │
│ │ ByteTrack   │ │  │ │ ByteTrack   │ │  │ │ ByteTrack   │ │
│ └─────────────┘ │  │ └─────────────┘ │  │ └─────────────┘ │
│        │        │  │        │        │  │        │        │
│        ▼        │  │        ▼        │  │        ▼        │
│ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │
│ │TrackManager │ │  │ │TrackManager │ │  │ │TrackManager │ │
│ │(track_id: 1)│ │  │ │(track_id: 3)│ │  │ │(track_id: 5)│ │
│ └─────────────┘ │  │ └─────────────┘ │  │ └─────────────┘ │
│        │        │  │        │        │  │        │        │
│        ▼        │  │        ▼        │  │        ▼        │
│ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │
│ │OSNet ReID   │ │  │ │OSNet ReID   │ │  │ │OSNet ReID   │ │
│ │InsightFace  │ │  │ │InsightFace  │ │  │ │InsightFace  │ │
│ └─────────────┘ │  │ └─────────────┘ │  │ └─────────────┘ │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                     │
         │                    │                     │
         └────────────┬───────┴──────┬──────────────┘
                      ▼              ▼
              ┌──────────────────────────────┐
              │ SYNCHRONIZATION LAYER        │
              │                              │
              │ ┌──────────────────────────┐ │
              │ │  PostgreSQL + pgvector   │ │
              │ │                          │ │
              │ │  ┌────────────────────┐  │ │
              │ │  │ person_identities  │  │ │
              │ │  │  - id (UUID)       │  │ │
              │ │  │  - first_seen      │  │ │
              │ │  │  - last_seen       │  │ │
              │ │  │  - visit_count     │  │ │
              │ │  └────────────────────┘  │ │
              │ │                          │ │
              │ │  ┌────────────────────┐  │ │
              │ │  │ person_embeddings  │  │ │
              │ │  │  - person_id       │  │ │
              │ │  │  - camera_id       │  │ │
              │ │  │  - embedding       │  │ │
              │ │  │    (512-dim vector)│  │ │
              │ │  │  - captured_at     │  │ │
              │ │  └────────────────────┘  │ │
              │ │                          │ │
              │ │  Cosine Similarity:      │ │
              │ │  <=> operator (pgvector) │ │
              │ └──────────────────────────┘ │
              └──────────────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ UNIFIED OUTPUT  │
                    │                 │
                    │ • Person UUID   │
                    │ • Cross-camera  │
                    │   journey       │
                    │ • Visit count   │
                    │ • Demographics  │
                    │ • Zone events   │
                    └─────────────────┘
```

---

## Summary Table

| Question | Answer |
|----------|--------|
| **Where is camera feed sent?** | RTSP → FrameBuffer (thread) → CameraWorker (async loop) |
| **Join point?** | `CameraWorker._run_loop()` pulls frames from buffer |
| **Are zones required?** | ❌ No - detection/tracking/ReID work without zones |
| **Process all frames?** | ❌ No - samples at `fps_target` (e.g., 5 FPS from 25 FPS) |
| **ReID every frame?** | ❌ No - gated by eligibility, quality, confidence state |
| **How do cameras sync?** | Shared PostgreSQL + pgvector cosine similarity search |
| **Cross-camera matching?** | Embeddings searched across ALL cameras (48-hour window) |

---

## Configuration Examples

### High Accuracy (More Compute)

```python
camera_config = {
    "fps_target": 10,  # More frequent sampling
    "reid_enabled": True,
    "demographic_enabled": True
}
```

### Low Compute (Edge Devices)

```python
camera_config = {
    "fps_target": 3,   # Less frequent sampling
    "reid_enabled": True,
    "demographic_enabled": False  # Skip face analysis
}
```

### Multi-Camera Store Setup

```python
cameras = [
    {"name": "Entrance", "rtsp_url": "rtsp://cam1/stream", "fps_target": 5},
    {"name": "Aisle 1",  "rtsp_url": "rtsp://cam2/stream", "fps_target": 3},
    {"name": "Aisle 2",  "rtsp_url": "rtsp://cam3/stream", "fps_target": 3},
    {"name": "Checkout", "rtsp_url": "rtsp://cam4/stream", "fps_target": 5},
    {"name": "Exit",     "rtsp_url": "rtsp://cam5/stream", "fps_target": 5},
]
# Each camera operates independently, syncs via database
```

---

## Performance Benchmarks

**Test Environment:** 8-core CPU, GPU: NVIDIA RTX 3060

| Cameras | fps_target | Total Frames/sec | CPU Usage | GPU Usage | ReID Latency |
|---------|-----------|------------------|-----------|-----------|--------------|
| 1       | 5         | 5                | 15%       | 20%       | 50ms         |
| 3       | 5         | 15               | 40%       | 55%       | 50ms         |
| 5       | 5         | 25               | 65%       | 85%       | 60ms         |
| 5       | 10        | 50               | 95%       | 100%      | 80ms         |

**Recommendations:**
- **1-3 cameras:** fps_target = 10 (smooth tracking)
- **4-6 cameras:** fps_target = 5 (balanced)
- **7+ cameras:** fps_target = 3 (low compute)

---

## References

- **YOLO Documentation:** https://docs.ultralytics.com/
- **ByteTrack Paper:** https://arxiv.org/abs/2110.06864
- **OSNet Paper:** https://arxiv.org/abs/1905.00953
- **pgvector GitHub:** https://github.com/pgvector/pgvector
- **InsightFace:** https://github.com/deepinsight/insightface

---

---

## 6. Streaming Pipeline: Camera → FFmpeg → MediaMTX → Browser WebRTC

### **Complete Video Streaming Architecture with Tracked Person Overlays**

The platform provides **two streaming modes** for viewing camera feeds in the browser:

1. **Raw Stream Mode**: Original RTSP feed republished to MediaMTX (no annotations)
2. **Burn-In Mode**: Annotated stream with YOLO bounding boxes, zone overlays, and person count

---

### 6.1 Burn-In Streaming Pipeline (With Tracked Persons)

**This is the exciting part - live AI annotations on the video stream!**

#### **Architecture Overview:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     BURN-IN STREAMING PIPELINE                           │
└─────────────────────────────────────────────────────────────────────────┘

Camera (RTSP: 25 FPS)
    ↓
    ↓ (continuous capture)
    ↓
┌─────────────────────────────────┐
│   LatestFrameBuffer             │
│   (Background Thread)           │
│   Stores latest raw frame       │
└────────┬────────────────────┬───┘
         │                    │
         │                    │
    ↓ (5 FPS)            ↓ (15 FPS)
    │                    │
┌───▼──────────┐    ┌───▼──────────────────────┐
│ CameraWorker │    │ StreamBroadcaster        │
│ (AI Loop)    │    │ (Annotation Thread)      │
│              │    │                          │
│ YOLO Track   │───►│ 1. Get raw frame         │
│ Updates:     │    │ 2. Draw zones (blue)     │
│ latest_tracks│    │ 3. Draw bboxes (green)   │
│ [bbox list]  │    │ 4. Draw person count     │
└──────────────┘    │ 5. Pipe to FFmpeg stdin  │
                    └──────────────┬────────────┘
                                   ↓
                          ┌─────────────────┐
                          │  FFmpeg Process │
                          │  (libx264)      │
                          │  Encode H.264   │
                          └────────┬────────┘
                                   ↓
                          rtsp://mediamtx:8554/cam_<uuid>
                                   ↓
                          ┌─────────────────┐
                          │    MediaMTX     │
                          │  Media Server   │
                          │                 │
                          │  • WHEP (WebRTC)│
                          │  • HLS          │
                          │  • RTSP         │
                          └────────┬────────┘
                                   ↓
                    ┌──────────────┴───────────────┐
                    │                              │
               ┌────▼─────┐                  ┌────▼────┐
               │ Browser  │                  │ Browser │
               │ (WebRTC) │                  │  (HLS)  │
               │          │                  │         │
               │ LIVE     │                  │ Fallback│
               │ <1s delay│                  │ 3-5s    │
               └──────────┘                  └─────────┘
```

---

### 6.2 Step-by-Step Code Flow

#### **Step 1: Camera Worker Starts Broadcaster**

**File:** `app/modules/ai_runtime/camera_worker.py` (Lines 105-138)

```python
def _start_broadcaster(self) -> None:
    """Start the StreamBroadcaster that pipes annotated frames to MediaMTX."""
    
    # Get actual frame dimensions from buffer
    frame, _ = self.frame_buffer.get_latest()
    if frame is not None:
        height, width = frame.shape[:2]
    else:
        # Fallback to config hint
        width, height = 1920, 1080
    
    # Create broadcaster instance
    self.stream_broadcaster = StreamBroadcaster(
        frame_buffer=self.frame_buffer,  # Shared RTSP capture thread
        camera_id=self.camera_id,
        width=width,
        height=height,
        fps=15,  # STREAM_BURNIN_FPS from config
    )
    
    # Share mutable list objects - IMPORTANT!
    # These are updated IN-PLACE by the AI loop
    self.stream_broadcaster.latest_tracks = self.latest_tracks
    self.stream_broadcaster.latest_zones = self.zones
    
    self.stream_broadcaster.start()
```

**Key Point:** The broadcaster shares the **same** `latest_tracks` list object with the AI worker, so when YOLO detections update the list, the broadcaster immediately sees the changes.

---

#### **Step 2: AI Loop Updates Track List**

**File:** `app/modules/ai_runtime/camera_worker.py` (Lines 354-362)

```python
# After YOLO tracking and ReID processing...

# Update shared bounding box state for StreamBroadcaster
# IMPORTANT: Mutate IN-PLACE, don't reassign!
self.latest_tracks.clear()  # Remove old bboxes
for t in active_tracks:
    if t.bbox:
        self.latest_tracks.append(dict(t.bbox))
        # Appends: {"x1": 100, "y1": 200, "x2": 300, "y2": 500}
```

**Why In-Place Mutation?**
If you did `self.latest_tracks = []`, you'd break the reference that `StreamBroadcaster` holds. By using `.clear()` and `.append()`, both objects see the same list.

---

#### **Step 3: Broadcaster Loop Pulls Frames**

**File:** `app/modules/ai_runtime/stream_broadcaster.py` (Lines 187-262)

```python
def _broadcast_loop(self) -> None:
    """Continuously read raw frames, draw overlays, pipe to FFmpeg."""
    interval = 1.0 / self.fps  # 1/15 = 66ms for 15 FPS
    last_frame_ts = 0.0
    
    while not self._stop_event.is_set():
        # STEP 1: Get latest raw frame from shared buffer
        frame, frame_ts = self.frame_buffer.get_latest()
        
        if frame is None or frame_ts <= last_frame_ts:
            await asyncio.sleep(interval / 4)
            continue  # No new frame yet
        
        last_frame_ts = frame_ts
        
        # STEP 2: Draw overlays (zones + bboxes + count)
        annotated = self._draw_overlays(frame)
        
        # STEP 3: Write raw BGR bytes to FFmpeg stdin
        try:
            self._proc.stdin.write(annotated.tobytes())
        except (BrokenPipeError, OSError):
            # FFmpeg died, watchdog will restart it
            pass
        
        # STEP 4: Sleep to maintain 15 FPS
        await asyncio.sleep(interval)
```

---

#### **Step 4: Draw Overlays (The Magic Happens Here)**

**File:** `app/modules/ai_runtime/stream_broadcaster.py` (Lines 268-377)

```python
def _draw_overlays(self, frame: np.ndarray) -> np.ndarray:
    """Draw zone fills, bounding boxes, and person count on the frame."""
    
    display = frame.copy()  # Don't mutate original
    height, width = display.shape[:2]
    
    # ----------------------------------------------------------------
    # LAYER 1: Draw Zones (semi-transparent blue fill + green border)
    # ----------------------------------------------------------------
    zones = self.latest_zones  # Snapshot reference
    if zones:
        overlay = display.copy()
        for zone in zones:
            zone_name = zone.get("name", "Zone")
            poly = polygon_from_json(zone.get("polygon"))
            
            # Convert percentage coordinates to pixels
            pts = np.array([
                (int(x * width), int(y * height)) 
                for x, y in poly
            ], dtype=np.int32)
            
            # Semi-transparent blue fill (20% opacity)
            cv2.fillPoly(overlay, [pts], (255, 100, 0))  # BGR: Blue
            cv2.addWeighted(overlay, 0.20, display, 0.80, 0, display)
            overlay = display.copy()
            
            # Solid green outline (2px)
            cv2.polylines(display, [pts], True, (0, 255, 0), 2)
            
            # Zone name label
            cv2.putText(display, zone_name, (x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    
    # ----------------------------------------------------------------
    # LAYER 2: Draw Tracked Person Bounding Boxes (green rectangles)
    # ----------------------------------------------------------------
    tracks = self.latest_tracks  # Snapshot reference
    for track in tracks:
        x1, y1 = int(track["x1"]), int(track["y1"])
        x2, y2 = int(track["x2"]), int(track["y2"])
        
        # Green bounding box, 2px thick
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # ----------------------------------------------------------------
    # LAYER 3: Draw Person Count (top-left corner)
    # ----------------------------------------------------------------
    count = len(tracks)
    text = f"Persons: {count}"
    
    # Semi-transparent black background
    bg_overlay = display.copy()
    cv2.rectangle(bg_overlay, (8, 8), (text_w + 20, text_h + 16), 
                 (0, 0, 0), -1)
    cv2.addWeighted(bg_overlay, 0.5, display, 0.5, 0, display)
    
    # White text
    cv2.putText(display, text, (14, text_h + 16), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return display  # Annotated frame ready for FFmpeg
```

**Visual Result:**

```
┌─────────────────────────────────────────────────────┐
│ Persons: 3        [Semi-transparent black bg]       │
├─────────────────────────────────────────────────────┤
│                                                     │
│      ┌───────────┐  Zone: Checkout Counter         │
│      │░░░░░░░░░░░│  [Blue fill 20% transparent]    │
│      │░░┌─────┐░░│  [Green border]                 │
│      │░░│█████│░░│  [Green bbox for person]        │
│      │░░└─────┘░░│                                 │
│      └───────────┘                                 │
│                                                     │
│   ┌─────┐     ┌─────┐                             │
│   │█████│     │█████│  [Two more green bboxes]     │
│   └─────┘     └─────┘                             │
└─────────────────────────────────────────────────────┘
```

---

#### **Step 5: FFmpeg Encodes to H.264**

**File:** `app/modules/ai_runtime/stream_broadcaster.py` (Lines 130-154)

```python
def _build_command(self) -> List[str]:
    """Build FFmpeg command for encoding annotated frames."""
    return [
        "ffmpeg",
        "-nostdin",
        "-loglevel", "warning",
        
        # INPUT: Raw BGR frames from Python (stdin)
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{self.width}x{self.height}",  # e.g., "1920x1080"
        "-r", str(self.fps),                   # 15 FPS
        "-i", "-",                             # Read from stdin
        
        # ENCODE: Low-latency H.264
        "-an",                           # No audio
        "-c:v", "libx264",              # H.264 codec
        "-preset", "veryfast",          # Fast encoding
        "-tune", "zerolatency",         # Low latency
        "-pix_fmt", "yuv420p",          # Browser-compatible
        "-g", str(self.fps * 2),        # Keyframe every 2 seconds
        
        # OUTPUT: Push to MediaMTX via RTSP
        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        f"rtsp://mediamtx:8554/cam_{camera_id}"
    ]
```

**What FFmpeg Does:**
1. Reads raw BGR pixel data from Python via stdin (pipe)
2. Encodes to H.264 video codec (browser-compatible)
3. Pushes encoded stream to MediaMTX via RTSP

---

#### **Step 6: MediaMTX Serves to Browser**

**File:** `app/modules/streaming/mediamtx.py` (Lines 74-88)

```python
def endpoints(self, camera_id: uuid.UUID, public_host: str = None) -> StreamEndpoints:
    """Generate public URLs for browser consumption."""
    
    path = camera_path(camera_id)  # e.g., "cam_abc123..."
    
    return StreamEndpoints(
        path=path,
        
        # WebRTC (WHEP) - Preferred, <1s latency
        webrtc_url=f"http://{host}:8889/{path}/whep",
        
        # HLS - Fallback for Safari/iOS, 3-5s latency
        hls_url=f"http://{host}:8888/{path}/index.m3u8",
        
        # RTSP - Server-side only (debugging with VLC)
        rtsp_url=f"rtsp://{host}:8554/{path}"
    )
```

**MediaMTX Responsibilities:**
1. **Receives** RTSP stream from FFmpeg on port 8554
2. **Stores** path as `cam_<uuid>` (e.g., `cam_abc123def456...`)
3. **Serves** multiple output formats:
   - **WebRTC (WHEP)**: Port 8889, ultra-low latency (<1 second)
   - **HLS**: Port 8888, fallback for iOS/Safari (3-5 second delay)
   - **RTSP**: Port 8554, server-side debugging (VLC player)

---

### 6.3 Browser Connection Flow

#### **Frontend Request:**

```javascript
// 1. Request stream URLs from API
const response = await fetch('/api/v1/cameras/{cameraId}/stream');
const data = await response.json();

// Returns:
{
  "path": "cam_abc123def456...",
  "webrtc_url": "http://192.168.1.100:8889/cam_abc123def456.../whep",
  "hls_url": "http://192.168.1.100:8888/cam_abc123def456.../index.m3u8"
}

// 2. Try WebRTC first (lowest latency)
const pc = new RTCPeerConnection();
const offer = await pc.createOffer();
await pc.setLocalDescription(offer);

const response = await fetch(data.webrtc_url, {
  method: 'POST',
  headers: {'Content-Type': 'application/sdp'},
  body: offer.sdp
});

const answer = await response.text();
await pc.setRemoteDescription({type: 'answer', sdp: answer});

// 3. Attach to video element
pc.ontrack = (event) => {
  videoElement.srcObject = event.streams[0];
};

// 4. If WebRTC fails, fallback to HLS
if (webrtcFailed) {
  videoElement.src = data.hls_url;
}
```

---

### 6.4 Raw Stream Mode (No Annotations)

For cameras without burn-in enabled, a simpler pipeline is used:

```
Camera RTSP
    ↓
┌─────────────────────┐
│  FFmpegPublisher    │
│  (copy or lowlatency│
│   mode)             │
└──────────┬──────────┘
           ↓
    MediaMTX RTSP ingest
           ↓
    MediaMTX → Browser
    (WebRTC/HLS)
```

**File:** `app/modules/streaming/ffmpeg_publisher.py` (Lines 47-86)

```python
def _build_command(self) -> List[str]:
    cmd = [
        "ffmpeg",
        "-rtsp_transport", "tcp",
        "-timeout", "30000000",
        "-i", self.source_url,  # Original camera RTSP
    ]
    
    if self.mode == "lowlatency":
        # Re-encode for browser compatibility
        cmd += ["-an", "-c:v", "libx264", "-preset", "veryfast", 
                "-tune", "zerolatency"]
    else:
        # Copy mode - no re-encode (fastest)
        cmd += ["-an", "-c", "copy"]
    
    # Push to MediaMTX
    cmd += ["-f", "rtsp", "-rtsp_transport", "tcp", 
            f"rtsp://mediamtx:8554/cam_{camera_id}"]
    
    return cmd
```

**Modes:**
- **copy**: Remux only, no re-encode (lowest CPU, requires H.264 camera)
- **lowlatency**: Re-encode to browser-friendly H.264 (moderate CPU)

---

### 6.5 Complete Data Flow Timeline

**Example: 1 second of streaming (burn-in mode, 15 FPS target)**

```
Time | Camera (25 FPS) | AI Worker (5 FPS) | Broadcaster (15 FPS) | FFmpeg | MediaMTX | Browser
-----|-----------------|-------------------|----------------------|--------|----------|--------
00ms | F1              | -                 | -                    | -      | -        | -
40ms | F2              | -                 | Get F2, Draw, Pipe   | Encode | Buffer   | -
80ms | F3              | -                 | -                    | Encode | Buffer   | -
120ms| F4              | -                 | Get F4, Draw, Pipe   | Encode | Send     | Render
160ms| F5              | -                 | -                    | Encode | Send     | Render
200ms| F6              | Get F6, YOLO      | Get F6, Draw, Pipe   | Encode | Send     | Render
     |                 | Update bbox list  |                      |        |          |
240ms| F7              | -                 | -                    | Encode | Send     | Render
280ms| F8              | -                 | Get F8, Draw, Pipe   | Encode | Send     | Render
320ms| F9              | -                 | -                    | Encode | Send     | Render
360ms| F10             | -                 | Get F10, Draw, Pipe  | Encode | Send     | Render
400ms| F11             | Get F11, YOLO     | -                    | Encode | Send     | Render
     |                 | Update bbox list  |                      |        |          |
...  | ...             | ...               | ...                  | ...    | ...      | ...
```

**Key Observations:**
1. **Camera captures continuously** at 25 FPS (native rate)
2. **AI worker samples** every 200ms (5 FPS) to update bboxes
3. **Broadcaster samples** every 66ms (15 FPS) to draw and stream
4. **FFmpeg encodes** in real-time, pushes to MediaMTX
5. **Browser receives** with <1 second latency via WebRTC

---

### 6.6 Timing & Latency Analysis

| Component | Rate | Latency Contribution |
|-----------|------|---------------------|
| Camera → Buffer | 25 FPS | ~0ms (continuous) |
| Buffer → Broadcaster | 15 FPS | ~33ms (1/15s avg wait) |
| Draw Overlays (OpenCV) | 15 FPS | ~10-20ms per frame |
| FFmpeg Encode (H.264) | 15 FPS | ~30-50ms per frame |
| MediaMTX Buffer | Real-time | ~50-100ms |
| Network (LAN) | Real-time | ~10-50ms |
| WebRTC Browser Decode | Real-time | ~50-100ms |
| **Total End-to-End** | - | **200-400ms (<0.5s)** |

**For HLS Fallback:**
- Segment duration: 2 seconds
- Total latency: 3-5 seconds (less responsive but more compatible)

---

### 6.7 Configuration Options

**Enable Burn-In Streaming:**

```python
# When creating/updating camera
camera_config = {
    "rtsp_url": "rtsp://192.168.1.100:554/stream1",
    "burnin_enabled": True,  # ← Enable annotated stream
    "fps_target": 5,          # AI processing rate
}

# Environment config
STREAM_BURNIN_FPS = 15  # Annotation/streaming rate
```

**Disable Burn-In (Raw Stream):**

```python
camera_config = {
    "rtsp_url": "rtsp://192.168.1.100:554/stream1",
    "burnin_enabled": False,  # ← Raw stream only
}

# Uses FFmpegPublisher instead of StreamBroadcaster
STREAM_PUBLISH_MODE = "copy"  # or "lowlatency"
```

---

### 6.8 Troubleshooting

**Problem: Browser shows "Persons: 0" even though people are visible**

**Root Cause:** AI worker is sampling too slowly or YOLO isn't detecting

**Solution:**
1. Check AI worker FPS: Increase `fps_target` from 3 to 5 or 10
2. Check YOLO confidence: Lower `YOLO_CONFIDENCE_THRESHOLD` from 0.45 to 0.35
3. Check logs for YOLO errors

---

**Problem: Stream is laggy (>2 second delay)**

**Root Cause:** Browser fell back to HLS instead of WebRTC

**Solution:**
1. Verify MediaMTX WebRTC port 8889 is accessible
2. Check browser console for WebRTC errors
3. Ensure `MEDIAMTX_WEBRTCADDITIONALHOSTS` includes your LAN IP

---

**Problem: Bounding boxes don't match people positions**

**Root Cause:** Frame resolution mismatch between AI and broadcaster

**Solution:**
- StreamBroadcaster auto-detects and corrects resolution on first frame
- Check logs for "resolution mismatch" warnings
- Verify camera reports correct resolution in config

---

## Summary Table: Streaming Modes

| Mode | Use Case | Annotations | CPU Usage | Latency | Ports |
|------|----------|-------------|-----------|---------|-------|
| **Burn-In** | Live monitoring with AI overlays | ✅ Bboxes, zones, count | High | <1s WebRTC | 8554, 8889, 8888 |
| **Raw Copy** | Simple republishing | ❌ None | Low | <1s WebRTC | 8554, 8889, 8888 |
| **Raw Lowlatency** | Transcode for compatibility | ❌ None | Medium | <1s WebRTC | 8554, 8889, 8888 |

---

## References

- **MediaMTX Documentation:** https://github.com/bluenviron/mediamtx
- **FFmpeg Documentation:** https://ffmpeg.org/documentation.html
- **WebRTC WHEP Spec:** https://www.ietf.org/archive/id/draft-murillo-whep-00.html
- **HLS Spec:** https://datatracker.ietf.org/doc/html/rfc8216

---

**End of Document**
