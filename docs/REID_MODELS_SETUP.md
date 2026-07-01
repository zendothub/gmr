# ReID Models Setup Guide

## Problem Diagnosis

**Issue:** All `track_sessions` have `NULL` `person_identity_id` and `person_identities` table is empty.

**Root Cause:** ReID (Re-Identification) models are missing from the `models/` directory.

### Current Status:
```bash
models/
├── .gitkeep
└── yolov8n.pt     ✅ YOLO detector (present)
```

### Required Models (Missing):
```bash
models/
├── osnet_x1_0.pth          ❌ OSNet for body ReID embeddings
├── yolo11n-pose.pt         ❌ YOLO-Pose for keypoint detection
└── buffalo_l/              ❌ InsightFace for face analysis & demographics
```

---

## What Each Model Does

### 1. **OSNet (osnet_x1_0.pth)**
- **Purpose:** Extracts 512-dimensional body embeddings for person re-identification
- **Function:** Matches people across different cameras based on appearance
- **Without it:** Cannot identify unique persons → `person_identity_id` stays NULL

### 2. **YOLO-Pose (yolo11n-pose.pt)**
- **Purpose:** Detects body keypoints (shoulders, hips, knees, etc.)
- **Function:** Assesses crop quality for ReID (torso visibility, occlusion)
- **Without it:** Cannot determine if crop is good enough for ReID

### 3. **InsightFace (buffalo_l)**
- **Purpose:** Face detection, face embeddings, demographics (age/gender)
- **Function:** 
  - Extracts face embeddings for face-based matching
  - Estimates age and gender for analytics
  - Prevents false matches via face contradiction detection
- **Without it:** No demographic data, no face-based ReID

---

## Solution: Download Missing Models

### Step 1: Download OSNet Model

```bash
cd /Users/zendot/Desktop/zendot/GMR/retail-ai-platform/models

# Option A: Download from official Torchreid repo
wget https://github.com/KaiyangZhou/deep-person-reid/releases/download/v1.0.0/osnet_x1_0_imagenet.pth.tar -O osnet_x1_0.pth

# Option B: If above fails, use Google Drive mirror
# (You'll need to manually download from the Torchreid model zoo)
```

**Verification:**
```bash
ls -lh osnet_x1_0.pth
# Should be ~9-10 MB
```

---

### Step 2: Download YOLO-Pose Model

```bash
cd /Users/zendot/Desktop/zendot/GMR/retail-ai-platform/models

# Download YOLO11n-Pose from Ultralytics
pip install ultralytics  # if not already installed

# The model will auto-download on first use, or download manually:
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11n-pose.pt
```

**Verification:**
```bash
ls -lh yolo11n-pose.pt
# Should be ~6-7 MB
```

---

### Step 3: Install InsightFace and Download buffalo_l

```bash
# Install InsightFace
pip install insightface onnxruntime

# The buffalo_l model will auto-download on first use to:
# ~/.insightface/models/buffalo_l/

# To pre-download, run this Python snippet:
python3 << 'EOF'
from insightface.app import FaceAnalysis
import os

print("Downloading InsightFace buffalo_l model...")
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))
print("✅ InsightFace buffalo_l model downloaded successfully!")
print(f"Model location: {os.path.expanduser('~/.insightface/models/buffalo_l/')}")
EOF
```

**Verification:**
```bash
ls -la ~/.insightface/models/buffalo_l/
# Should contain: det_10g.onnx, genderage.onnx, w600k_r50.onnx, etc.
```

---

### Step 4: Update Config (if needed)

Check `app/config.py` points to correct paths:

```python
# AI Models (Lines 36-55)
YOLO_MODEL_PATH: str = "models/yolo11n.pt"             # ✅ or yolov8n.pt
OSNET_MODEL_PATH: str = "models/osnet_x1_0.pth"        # ✅ Must exist
YOLO_POSE_MODEL_PATH: str = "models/yolo11n-pose.pt"  # ✅ Must exist
INSIGHTFACE_MODEL: str = "buffalo_l"                   # ✅ Auto-downloads
```

---

### Step 5: Restart Application

```bash
# If running as systemd service
sudo systemctl restart retail-ai

# Or if running manually:
pkill -f "uvicorn app.main:app"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Verification Steps

### 1. Check Application Logs

```bash
# Look for successful model loading
tail -f nohup.out  # or wherever your logs are

# Expected log messages:
# ✅ "OSNet FeatureExtractor loaded: models/osnet_x1_0.pth"
# ✅ "InsightFace FaceAnalysis prepared successfully (model=buffalo_l)"
# ✅ "YOLO-Pose model loaded: models/yolo11n-pose.pt"
```

### 2. Check Database After New Tracks

```bash
# Wait for new camera activity, then check:
psql "postgresql://retail_user:retail_pass@localhost:5432/retail_ai_db" << 'EOF'

-- Check if new track_sessions have person_identity_id
SELECT COUNT(*) as total, 
       COUNT(person_identity_id) as with_person_id
FROM track_sessions 
WHERE started_at > NOW() - INTERVAL '10 minutes';

-- Check if person_identities table is being populated
SELECT COUNT(*) as total_persons FROM person_identities;

-- Check latest person with demographics
SELECT id, gender, estimated_age, created_at 
FROM person_identities 
ORDER BY created_at DESC 
LIMIT 5;

EOF
```

### 3. Test ReID Functionality

```bash
# New tracks should now have:
# ✅ person_identity_id populated
# ✅ person_identities records created
# ✅ Demographics (age, gender) populated
# ✅ Face and body embeddings stored

psql "postgresql://retail_user:retail_pass@localhost:5432/retail_ai_db" -c "
SELECT 
    ts.id as track_id,
    ts.person_identity_id,
    pi.gender,
    pi.estimated_age,
    ts.started_at
FROM track_sessions ts
LEFT JOIN person_identities pi ON pi.id = ts.person_identity_id
ORDER BY ts.started_at DESC
LIMIT 10;
"
```

---

## Expected Behavior After Fix

### Before (Current State):
```sql
person_identity_id | gender | estimated_age
--------------------+--------+---------------
                   |        |              -- ALL NULL
                   |        |              
                   |        |              
```

### After (With ReID Models):
```sql
person_identity_id                   | gender | estimated_age
-------------------------------------+--------+---------------
abc123-uuid-here                     | M      | 32
abc123-uuid-here                     | M      | 32  -- Same person!
def456-uuid-here                     | F      | 28
ghi789-uuid-here | M      | 45
```

---

## Quick Download Script

Run this to download all models at once:

```bash
#!/bin/bash

cd /Users/zendot/Desktop/zendot/GMR/retail-ai-platform/models

echo "📥 Downloading OSNet model..."
wget -q https://github.com/KaiyangZhou/deep-person-reid/releases/download/v1.0.0/osnet_x1_0_imagenet.pth.tar -O osnet_x1_0.pth
echo "✅ OSNet downloaded ($(ls -lh osnet_x1_0.pth | awk '{print $5}'))"

echo "📥 Downloading YOLO-Pose model..."
wget -q https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11n-pose.pt
echo "✅ YOLO-Pose downloaded ($(ls -lh yolo11n-pose.pt | awk '{print $5}'))"

echo "📥 Downloading InsightFace buffalo_l..."
python3 << 'EOF'
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))
print("✅ InsightFace buffalo_l downloaded")
EOF

echo ""
echo "🎉 All ReID models downloaded successfully!"
echo ""
echo "📁 Model files:"
ls -lh osnet_x1_0.pth yolo11n-pose.pt 2>/dev/null
echo ""
echo "📁 InsightFace location:"
ls -la ~/.insightface/models/buffalo_l/ 2>/dev/null | head -5
echo ""
echo "🔄 Now restart your application:"
echo "   sudo systemctl restart retail-ai"
```

---

## Troubleshooting

### Issue: "OSNet model file not found"
```bash
# Solution: Verify path in config.py matches downloaded file
ls -la models/osnet_x1_0.pth
# Update OSNET_MODEL_PATH in app/config.py or .env if needed
```

### Issue: "InsightFace download fails"
```bash
# Solution: Check internet connection and try manual download
pip install --upgrade insightface onnxruntime
# Or download models manually from InsightFace GitHub
```

### Issue: "CUDA/GPU errors"
```bash
# Solution: Force CPU mode
# In app/modules/reid/osnet_extractor.py and insightface_analyzer.py
# Models will auto-fallback to CPU if CUDA unavailable
```

### Issue: "Still no person_identity_id after restart"
```bash
# Check logs for errors:
journalctl -u retail-ai -f  # if systemd service
# or
tail -f nohup.out

# Look for:
# ❌ "Failed to initialize OSNet extractor"
# ❌ "Failed to initialize InsightFace analyzer"
```

---

## Performance Notes

- **OSNet inference:** ~50-100ms per crop (CPU), ~10-20ms (GPU)
- **InsightFace:** ~100-200ms per crop (CPU), ~20-40ms (GPU)
- **YOLO-Pose:** ~30-50ms per frame (CPU), ~5-10ms (GPU)

With all models loaded, expect:
- **First track:** ~200-400ms for initial ReID
- **Re-identification:** ~50-150ms (matching against existing persons)
- **Memory usage:** +2-3 GB for all models loaded

---

## Summary

**Problem:** ReID not working → No `person_identity_id` → Analytics show 0 visitors

**Solution:** Download 3 missing models:
1. OSNet (body ReID)
2. YOLO-Pose (quality assessment)
3. InsightFace (face analysis + demographics)

**Result:** Unique visitor tracking + demographics + cross-camera re-identification working!

---

**Last Updated:** 2026-06-30  
**Status:** Models missing - needs setup
