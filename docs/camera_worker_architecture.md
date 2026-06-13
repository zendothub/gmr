# Camera Worker Architecture & Event Lifecycle

This document explains the internal mechanics of the Retail AI Platform's Camera Worker. The worker processes real-time RTSP video streams to detect, track, identify (via face/body), and generate analytical events for retail environments.

## 🌊 Pipeline Data Flow

The following diagram illustrates the frame-by-frame data flow within a single camera worker.

```text
+----------------+
| RTSP Stream    |
+-------+--------+
        |
        v
+-------+--------+
| YOLOv11 Detect |
+-------+--------+
        | Bounding Boxes
        v
+-------+--------+
| Camera Filter  | ---> [Discard if Ignored]
+-------+--------+
        | Valid Track
        v
+-------+--------+      +---------------------+
| Track Manager  | ---> | Zone Event Detector | ---> [Generate Zone Events]
+-------+--------+      +---------------------+
        |
        | Track Quality >= 5 High-Q Frames
        v
+-------+--------+
| InsightFace &  |
| OSNet Extractor|
+-------+--------+
        | Face/Body Embeddings
        v
+-------+--------+
| Identity Engine|
+-------+--------+
        | Search pgvector
        v
    Match Found?
   /            \
 Yes             No
 /                \
v                  v
Update Person ID   Register New Person ID
        |                  |
        +---------+--------+
                  |
                  v
          +---------------+
          | PostgreSQL DB |
          +---------------+
```

---

## ⚙️ How the Worker Works

### 1. Frame Ingestion & YOLO Tracking
The worker runs asynchronously at a target FPS (e.g., 5 FPS). It retrieves the absolute latest frame from the RTSP stream using a dedicated background thread buffer. The frame is passed into `YOLOv11`, coupled with ByteTrack, to generate temporally consistent bounding boxes and `local_track_ids`.

### 2. Track Management & ROI Filtering
Detected boxes are first filtered through the `CameraViewEngine`. If a bounding box center falls inside an "ignore" polygon, it is discarded. Valid tracks update their state in the `TrackManager`, which keeps track of the bounding box history, average confidence, and stability score over time.

### 3. ReID & Demographics (InsightFace + OSNet)
When a track is stable and has a bounding box large enough, the system crops the person from the frame. 
- If the crop passes a quality threshold (`REID_CROP_QUALITY_THRESHOLD`), the crop is saved.
- **OSNet** generates a 512-d body embedding.
- **InsightFace (buffalo_l)** runs on the crop to extract age, gender, and a 512-d face embedding (if a face is visible).
- After accumulating 5 high-quality frames, the **Identity Decision Engine** is invoked. It searches PostgreSQL (`pgvector`) using the face embedding first (highest accuracy, clothing independent). If no face match is found, it falls back to the mean body embedding.

### 4. Database Storage & Crops
To ensure the UI always has visual context:
- `TrackSession` records store a `best_crop_path`.
- When a track first starts, an immediate crop is extracted and saved.
- As the track progresses, if sharper crops are obtained, `best_crop_path` is continually updated.

---

## 📡 Event Generation Lifecycle

The system generates core lifecycle events automatically. These events are saved to the `events` table and form the foundation of the analytics dashboard.

### 1. `person_entered_view`
- **When:** Frame 1 (The exact moment a new `local_track_id` is created).
- **How:** Fired immediately with `person_identity_id = null`. The `snapshot_path` is set to the initial crop, and `metadata_json` contains the `local_track_id`. 
- **Refinement:** Once ReID finishes processing (usually 5 frames later), an `UPDATE` query runs against this exact event row to populate the `person_identity_id` with the matched identity.

### 2. `new_person_registered`
- **When:** During ReID resolution (Frame 5+).
- **How:** Fired by the `IdentityDecisionEngine` ONLY if the embeddings do not match anyone in the database. A new anonymous `PersonIdentity` is created, and this event signifies their first-ever appearance in the system.

### 3. `zone_enter` & `zone_exit`
- **When:** Continuously evaluated on every frame.
- **How:** The `ZoneEventDetector` checks if the bottom-center point of a person's bounding box intersects with any defined polygon zones. If it enters, `zone_enter` is queued. If it leaves, `zone_exit` is queued (including the total dwell time in seconds).

### 4. `person_left_view`
- **When:** When a track is lost or times out (Track Session closes).
- **How:** The `TrackManager` flags a track as stale if it hasn't been seen for X seconds. The system updates the `ended_at` timestamp on the `TrackSession` and fires the `person_left_view` event.
