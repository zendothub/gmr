import os
# Set OpenCV FFMPEG timeout option (10,000,000 microseconds = 10 seconds)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;10000000"
# Silence FFmpeg standard error output logs in OpenCV
os.environ["OPENCV_FFMPEG_LOG_LEVEL"] = "-8"


import cv2
import sqlite3
import json
import numpy as np
import torch
import time
import threading
from ultralytics import YOLO
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from torchreid.reid.utils import FeatureExtractor

DB_FILE = "zones.db"
CAMERA_ID = "rtsp://admin:admin@192.168.0.84:1935"

# Global state
polygon_points = []
frame_width = 0
frame_height = 0

# Qdrant client with local persistent storage
qdrant_client = QdrantClient(path="qdrant_db")
if not qdrant_client.collection_exists("people"):
    qdrant_client.create_collection(
        collection_name="people",
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )

# ReID global state
next_global_id = 1
resolved_tracks = {}
track_embeddings = {}
THRESHOLD = 0.60
temporary_ids = set()

def get_next_global_id():
    """
    Check the local Qdrant database on startup to find the highest existing
    integer ID, ensuring we resume counting correctly across runs.
    """
    try:
        points, _ = qdrant_client.scroll(
            collection_name="people",
            limit=10000,
            with_payload=False,
            with_vectors=False
        )
        if points:
            max_id = max(p.id for p in points if isinstance(p.id, int))
            return max_id + 1
    except Exception as e:
        print(f"Error checking next global ID: {e}")
    return 1

class ThreadedCamera:
    """
    RTSP stream reader that runs in a background thread.
    This guarantees that the main loop always processes the absolute latest frame,
    completely preventing RTSP buffer build-up (lag/latency).
    """
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()

    def _update(self):
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.ret = ret
                        self.frame = frame
                else:
                    time.sleep(0.005)
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            # Return a copy to prevent race conditions during rotation/drawing
            return self.ret, self.frame.copy()

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS zones (
            camera_id TEXT PRIMARY KEY,
            polygon TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_polygon(camera_id, points):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO zones (camera_id, polygon)
        VALUES (?, ?)
    """, (camera_id, json.dumps(points)))
    conn.commit()
    conn.close()

def load_polygon(camera_id):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT polygon FROM zones WHERE camera_id = ?", (camera_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return []

def delete_polygon(camera_id):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM zones WHERE camera_id = ?", (camera_id,))
    conn.commit()
    conn.close()

def mouse_callback(event, x, y, flags, param):
    global polygon_points, frame_width, frame_height
    if event == cv2.EVENT_LBUTTONDOWN:
        # Only allow clicking up to 4 points
        if len(polygon_points) < 4:
            if frame_width > 0 and frame_height > 0:
                norm_x = x / frame_width
                norm_y = y / frame_height
                polygon_points.append((norm_x, norm_y))
                print(f"Point {len(polygon_points)} added: ({x}, {y}) -> Normalized: ({norm_x:.3f}, {norm_y:.3f})")
                
                # Save to database immediately when the 4th point is clicked
                if len(polygon_points) == 4:
                    save_polygon(CAMERA_ID, polygon_points)
                    print("Polygon definition complete. Saved to database.")

def query_reid(embedding):
    try:
        response = qdrant_client.query_points(
            collection_name="people",
            query=embedding.tolist(),
            limit=1
        )
        if response.points:
            return response.points[0].id, response.points[0].score
    except Exception as e:
        print(f"Qdrant query error: {e}")
    return None, None

def register_reid(global_id, embedding):
    try:
        qdrant_client.upsert(
            collection_name="people",
            points=[
                PointStruct(
                    id=global_id,
                    vector=embedding.tolist()
                )
            ]
        )
    except Exception as e:
        print(f"Qdrant upsert error: {e}")

def delete_reid(global_id):
    try:
        qdrant_client.delete(
            collection_name="people",
            points_selector=[global_id]
        )
    except Exception as e:
        print(f"Qdrant delete error: {e}")

def main():
    global polygon_points, frame_width, frame_height, next_global_id, resolved_tracks, qdrant_client, track_embeddings, temporary_ids
    
    # Initialize DB and load existing polygon
    init_db()
    polygon_points = load_polygon(CAMERA_ID)
    if len(polygon_points) == 4:
        print("Loaded existing polygon from database.")
    else:
        print("No polygon found in database. Click 4 points on the window to define the zone.")
    
    # Initialize next global ID based on existing Qdrant db points
    next_global_id = get_next_global_id()
    print(f"Next global ID starts at: {next_global_id}")

    # Detect hardware acceleration device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Inference device: {device.upper()}")
    
    # Load YOLO11 model
    model = YOLO("yolo11n.pt")
    model.to(device)
    
    # Load torchreid OSNet Feature Extractor
    # Note: torchreid internally handles resize and transforms.
    reid_model = FeatureExtractor(
        model_name='osnet_x1_0',
        device=device
    )

    cap = None
    try:
        # Initialize threaded camera reader
        print(f"Connecting to RTSP stream: {CAMERA_ID}")
        cap = ThreadedCamera(CAMERA_ID)

        # Wait up to 10 seconds to retrieve the initial frame
        print("Waiting for initial frame from stream (up to 10 seconds)...")
        start_wait = time.time()
        initial_frame = None
        ret = False
        while time.time() - start_wait < 10.0:
            ret, initial_frame = cap.read()
            if ret:
                break
            time.sleep(0.1)
            
        if not ret:
            print("\n[ERROR] Could not retrieve initial frame from stream. Timeout reached.")
            print("-" * 60)
            print("Troubleshooting network/camera connection:")
            print("  1. Verify the camera is powered on and LAN cable is securely plugged in.")
            print("  2. Check if the camera IP changed (run 'arp -a' to look for MAC matches).")
            print(f"  3. Verify RTSP port 1935 is open by running: nc -zv 192.168.0.84 1935")
            print("  4. Check if you can ping the device: ping 192.168.0.84")
            print("-" * 60)
            return

        # Set up OpenCV GUI window and register mouse callback
        cv2.namedWindow('Camera Feed')
        cv2.setMouseCallback('Camera Feed', mouse_callback)

        print("Camera feed started. Press 'q' or 'ESC' to exit. Press 'c' to clear/cancel polygon.")

        while True:
            loop_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                print("Warning: Empty frame received. Retrying...")
                time.sleep(0.01)
                continue

            # Rotate the frame 90 degrees counter-clockwise
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Update frame dimensions for normalization
            height, width, _ = frame.shape
            frame_width = width
            frame_height = height

            # Run YOLO tracking with ByteTrack
            results = model.track(frame, classes=[0], verbose=False, device=device, persist=True, tracker="bytetrack.yaml")

            # Draw the polygon zone
            is_polygon_complete = (len(polygon_points) == 4)
            pts = None
            if is_polygon_complete:
                pts = np.array([(int(x * width), int(y * height)) for x, y in polygon_points], dtype=np.int32)
                # Semi-transparent overlay for the zone
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], (0, 255, 0))  # Green fill
                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                # Bounding line
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            else:
                # Draw individual points & connecting lines while drawing
                for i, p in enumerate(polygon_points):
                    px, py = int(p[0] * width), int(p[1] * height)
                    cv2.circle(frame, (px, py), 6, (0, 0, 255), -1)  # Red dots for definition
                    cv2.putText(frame, str(i + 1), (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                if len(polygon_points) > 1:
                    pts_drawing = np.array([(int(x * width), int(y * height)) for x, y in polygon_points], dtype=np.int32)
                    cv2.polylines(frame, [pts_drawing], False, (0, 0, 255), 2)

            # Process detections & resolve identities via ReID
            boxes = results[0].boxes
            track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else []

            for i, box in enumerate(boxes):
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                conf = float(box.conf[0].cpu().numpy())
                
                # Get track ID from ByteTrack
                track_id = track_ids[i] if i < len(track_ids) else None
                
                global_id = None
                is_confident = False
                if track_id is not None:
                    # If already resolved this track_id, get cached details
                    if track_id in resolved_tracks:
                        global_id, score, is_confident = resolved_tracks[track_id]
                    
                    # If we don't have an ID, or if we have an ID but it is NOT confident, run ReID pipeline
                    if global_id is None or not is_confident:
                        # Reject tiny crops to ensure ReID accuracy (height < 100)
                        h = y2 - y1
                        if h >= 100:
                            # Crop the person bounding box from the frame
                            crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                            if crop.size > 0:
                                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                                
                                # Generate ReID feature embedding
                                with torch.no_grad():
                                    features = reid_model(crop_rgb)
                                    embedding = features[0].cpu().numpy() # shape is (512,)
                                    norm = np.linalg.norm(embedding)
                                    if norm > 0:
                                        embedding = embedding / norm  # L2 normalize
                                
                                # Accumulate embedding for the track
                                if track_id not in track_embeddings:
                                    track_embeddings[track_id] = []
                                track_embeddings[track_id].append(embedding)
                                
                                # Average and resolve on the 5th frame of this accumulation cycle
                                if len(track_embeddings[track_id]) == 5:
                                    mean_embedding = np.mean(track_embeddings[track_id], axis=0)
                                    mean_norm = np.linalg.norm(mean_embedding)
                                    if mean_norm > 0:
                                        mean_embedding = mean_embedding / mean_norm
                                    
                                    # Query vector DB to find matching identity using tuned threshold
                                    matched_id, new_score = query_reid(mean_embedding)
                                    
                                    # Confidence limit: 0.75 maps to highly confident/certain
                                    CONFIDENCE_LIMIT = 0.75
                                    
                                    if global_id is None:
                                        # Initial resolution at frame 5
                                        if matched_id is not None and new_score is not None and new_score >= THRESHOLD:
                                            global_id = matched_id
                                            is_confident = (new_score >= CONFIDENCE_LIMIT)
                                            resolved_tracks[track_id] = (global_id, new_score, is_confident)
                                            print(f"[Initial ReID] Track {track_id} -> ID {global_id} (score: {new_score:.2f}, confident={is_confident})")
                                        else:
                                            # Register new temporary visitor ID
                                            global_id = next_global_id
                                            next_global_id += 1
                                            register_reid(global_id, mean_embedding)
                                            temporary_ids.add(global_id)
                                            is_confident = False
                                            resolved_tracks[track_id] = (global_id, 0.0, is_confident)
                                            if new_score is not None:
                                                print(f"[Initial ReID] Track {track_id} -> ID {global_id} (New visitor, best score: {new_score:.2f} < {THRESHOLD})")
                                            else:
                                                print(f"[Initial ReID] Track {track_id} -> ID {global_id} (New visitor, first database entry)")
                                    else:
                                        # Periodic refinement (frames 15, 25, 35...)
                                        prev_score = resolved_tracks[track_id][1]
                                        
                                        # If we find a match to an existing database ID with higher similarity
                                        if matched_id is not None and new_score is not None and new_score > prev_score and new_score >= THRESHOLD:
                                            old_global_id = global_id
                                            if matched_id != old_global_id:
                                                # If the old ID was a temporary ID, delete it to keep database clean
                                                if old_global_id in temporary_ids:
                                                    delete_reid(old_global_id)
                                                    temporary_ids.discard(old_global_id)
                                                    print(f"[ReID Cleanup] Deleted temporary ID {old_global_id} from database.")
                                                
                                                global_id = matched_id
                                                is_confident = (new_score >= CONFIDENCE_LIMIT)
                                                resolved_tracks[track_id] = (global_id, new_score, is_confident)
                                                print(f"[ReID Refined] Track {track_id} switched identity: ID {old_global_id} -> ID {global_id} (New score: {new_score:.2f}, confident={is_confident})")
                                            else:
                                                # Same ID, but score upgraded
                                                is_confident = (new_score >= CONFIDENCE_LIMIT)
                                                resolved_tracks[track_id] = (global_id, new_score, is_confident)
                                                # Refine the embedding vector in Qdrant
                                                register_reid(global_id, mean_embedding)
                                                print(f"[ReID Refined] Track {track_id} score upgraded for ID {global_id}: {prev_score:.2f} -> {new_score:.2f} (confident={is_confident})")
                                        else:
                                            # If this is a temporary ID, refine its signature if score is better
                                            if global_id in temporary_ids and new_score is not None and new_score > prev_score:
                                                register_reid(global_id, mean_embedding)
                                                resolved_tracks[track_id] = (global_id, new_score, False)
                                                print(f"[ReID Refined] Temporary ID {global_id} signature refined (score: {prev_score:.2f} -> {new_score:.2f})")
                                    
                                    # Clear current window's embeddings to start accumulating the next 5 frames
                                    track_embeddings[track_id].clear()

                # Compute foot point (bottom center of bounding box)
                foot_x = int((x1 + x2) / 2)
                foot_y = int(y2)
                
                is_inside = False
                if is_polygon_complete and pts is not None:
                    # Check if the foot point is inside the polygon
                    dist = cv2.pointPolygonTest(pts, (foot_x, foot_y), False)
                    if dist >= 0:
                        is_inside = True
                
                # Color coding: Red if inside the zone, Green if outside
                color = (0, 0, 255) if is_inside else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (foot_x, foot_y), 4, (0, 255, 255), -1)
                
                # Label box with ReID ID & ByteTrack ID
                if global_id is not None:
                    if is_confident:
                        label = f"ID: {global_id} (Track: {track_id})"
                    else:
                        accum_count = len(track_embeddings.get(track_id, []))
                        label = f"ID: {global_id}* ({accum_count}/5) (Track: {track_id})"
                elif track_id is not None:
                    # Show progress to 5 frames
                    accum_count = len(track_embeddings.get(track_id, []))
                    label = f"ID: TBD ({accum_count}/5) (Track: {track_id})"
                else:
                    label = f"Detecting: {conf:.2f}"
                    
                if is_inside:
                    label += " [IN ZONE]"
                    
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw minimal instructions when zone is incomplete
            if not is_polygon_complete:
                remaining = 4 - len(polygon_points)
                cv2.putText(frame, f"Click {remaining} more point(s) to define zone", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow('Camera Feed', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('c'):
                # Cancel creation OR delete completed polygon
                polygon_points.clear()
                delete_polygon(CAMERA_ID)
                print("Polygon cleared and deleted from database.")

            # Enforce target 10 FPS (100ms per loop iteration)
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0.001, 0.1 - elapsed)
            time.sleep(sleep_time)
    finally:
        if cap is not None:
            cap.release()
        qdrant_client.close()
        cv2.destroyAllWindows()
        del qdrant_client

if __name__ == "__main__":
    main()
