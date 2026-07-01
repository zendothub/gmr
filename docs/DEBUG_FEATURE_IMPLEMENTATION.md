# Debug Detection Feature - Implementation Guide

## ✅ Completed

1. **Database Model** - `app/core/db/models/debug.py` created
2. **Migration** - `alembic/versions/0002_add_person_debug_table.py` created
3. **Model registered** in `app/core/db/models/__init__.py`

---

## 🔨 TODO: Run Migration

```bash
cd /Users/zendot/Desktop/zendot/GMR/retail-ai-platform

# Run the migration
python3 -c "from alembic.config import Config; from alembic import command; alembic_cfg = Config('alembic.ini'); command.upgrade(alembic_cfg, 'head')"

# Or use psql directly:
psql "postgresql://retail_user:retail_pass@localhost:5432/retail_ai_db" < alembic/versions/0002_add_person_debug_table.py
```

---

## 📝 Step 1: Create Debug API Module

### File: `app/modules/debug/__init__.py`
```python
"""Debug module for detection pipeline insights."""
```

### File: `app/modules/debug/schemas.py`
```python
"""Debug detection schemas."""

from datetime import datetime
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel


class PersonDebugResponse(BaseModel):
    """Single debug record response."""
    id: UUID
    camera_id: UUID
    camera_name: Optional[str] = None
    store_id: Optional[UUID] = None
    store_name: Optional[str] = None
    occurred_at: datetime
    
    # Outcome
    reid_attempted: bool
    reid_success: bool
    person_identity_id: Optional[UUID] = None
    
    # Track metrics
    bbox_height_px: Optional[float] = None
    bbox_width_px: Optional[float] = None
    detection_confidence: Optional[float] = None
    track_total_frames: Optional[int] = None
    
    # Crop
    crop_path: Optional[str] = None
    
    # Quality
    quality_score: Optional[float] = None
    quality_passed: bool = False
    keypoint_visibility_ratio: Optional[float] = None
    keypoint_gate_passed: bool = False
    
    # Face
    face_detected: bool = False
    face_score: Optional[float] = None
    face_crop_path: Optional[str] = None
    face_age: Optional[int] = None
   face_gender: Optional[str] = None
    
    # ReID
    reid_score: Optional[float] = None
    reid_confident: bool = False
    reid_frame_count: int = 0
    
    # Failure
    failure_stage: Optional[str] = None
    failure_reason: Optional[str] = None
    
    class Config:
        from_attributes = True


class DebugSummary(BaseModel):
    """Summary statistics for debug data."""
    total: int
    detected: int
    not_detected: int
    detection_rate: float
    
    # Failure breakdown
    failure_by_stage: dict  # {stage: count}
    

class DebugListResponse(BaseModel):
    """Paginated debug records with summary."""
    summary: DebugSummary
    records: List[PersonDebugResponse]
    total: int
    page: int
    limit: int
```

### File: `app/modules/debug/service.py`
```python
"""Debug detection service."""

from datetime import datetime, timedelta
from typing import Optional, List
from uuid import UUID

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.core.db.models.debug import PersonDebug
from app.core.db.models.camera import Camera
from app.core.db.models.store import Store
from app.modules.debug.schemas import PersonDebugResponse, DebugSummary, DebugListResponse
from app.utils.time_utils import utc_now


class DebugService:
    
    @staticmethod
    async def get_debug_records(
        db: AsyncSession,
        store_id: Optional[UUID] = None,
        camera_id: Optional[UUID] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        status: str = "all",  # all | detected | not_detected
        page: int = 1,
        limit: int = 50,
    ) -> DebugListResponse:
        """Get paginated debug records with filters."""
        
        # Default time range: last 24 hours
        if not end_time:
            end_time = utc_now()
        if not start_time:
            start_time = end_time - timedelta(hours=24)
        
        # Build base query
        conditions = [
            PersonDebug.occurred_at >= start_time,
            PersonDebug.occurred_at <= end_time,
        ]
        
        if store_id:
            conditions.append(PersonDebug.store_id == store_id)
        if camera_id:
            conditions.append(PersonDebug.camera_id == camera_id)
        if status == "detected":
            conditions.append(PersonDebug.reid_success == True)
        elif status == "not_detected":
            conditions.append(PersonDebug.reid_success == False)
        
        # Count total
        count_query = select(func.count(PersonDebug.id)).where(and_(*conditions))
        total = (await db.execute(count_query)).scalar() or 0
        
        # Get records with joined data
        query = (
            select(PersonDebug)
            .options(joinedload(PersonDebug.camera), joinedload(PersonDebug.store))
            .where(and_(*conditions))
            .order_by(PersonDebug.occurred_at.desc())
            .offset((page - 1) * limit)
            .limit(limit)
        )
        
        result = await db.execute(query)
        records = result.scalars().all()
        
        # Build summary
        summary_query = select(
            func.count(PersonDebug.id).label('total'),
            func.sum(func.cast(PersonDebug.reid_success, sa.Integer)).label('detected'),
            PersonDebug.failure_stage,
            func.count(PersonDebug.id).label('stage_count')
        ).where(and_(*conditions)).group_by(PersonDebug.failure_stage)
        
        summary_result = await db.execute(summary_query)
        summary_rows = summary_result.all()
        
        total_count = sum(row.total for row in summary_rows) or 1
        detected_count = sum(row.detected or 0 for row in summary_rows)
        
        failure_by_stage = {}
        for row in summary_rows:
            if row.failure_stage:
                failure_by_stage[row.failure_stage] = row.stage_count
        
        summary = DebugSummary(
            total=total_count,
            detected=detected_count,
            not_detected=total_count - detected_count,
            detection_rate=round((detected_count / total_count) * 100, 1),
            failure_by_stage=failure_by_stage,
        )
        
        # Convert to response objects
        response_records = []
        for record in records:
            record_dict = {
                **{k: v for k, v in record.__dict__.items() if not k.startswith('_')},
                "camera_name": record.camera.name if record.camera else None,
                "store_name": record.store.name if record.store else None,
            }
            response_records.append(PersonDebugResponse(**record_dict))
        
        return DebugListResponse(
            summary=summary,
            records=response_records,
            total=total,
            page=page,
            limit=limit,
        )
```

### File: `app/modules/debug/router.py`
```python
"""Debug detection API routes."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_user
from app.core.db.models.user import User
from app.modules.debug.schemas import DebugListResponse
from app.modules.debug.service import DebugService


router = APIRouter(prefix="/api/v2/debug", tags=["Debug"])


@router.get("/detections", response_model=DebugListResponse)
async def get_debug_detections(
    store_id: Optional[UUID] = Query(None),
    camera_id: Optional[UUID] = Query(None),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    status: str = Query("all", pattern="^(all|detected|not_detected)$"),
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get detection debug records with filters.
    
    **Filters:**
    - `store_id`: Filter by store
    - `camera_id`: Filter by camera
    - `start_time`, `end_time`: Time range (default: last 24h)
    - `status`: all | detected | not_detected
    - `page`, `limit`: Pagination
    
    **Returns:**
    - Summary with detection rate and failure breakdown
    - Paginated list of debug records with all metrics
    """
    return await DebugService.get_debug_records(
        db=db,
        store_id=store_id,
        camera_id=camera_id,
        start_time=start_time,
        end_time=end_time,
        status=status,
        page=page,
        limit=limit,
    )
```

---

## 📝 Step 2: Register Router in main.py

Add to `app/main.py`:

```python
# Add import
from app.modules.debug.router import router as debug_router

# Add router registration (around line 80-90)
app.include_router(debug_router)
```

---

## 📝 Step 3: Instrument camera_worker.py

This is the CRITICAL part - log debug data on every ReID attempt.

**Location to modify:** `app/modules/ai_runtime/camera_worker.py` line ~550-680 in `_run_reid()` method

Add this helper method to CameraWorker class:

```python
async def _log_debug_record(self, db, track: ActiveTrack, **kwargs):
    """Log a debug record for this detection attempt."""
    from app.core.db.models.debug import PersonDebug
    from app.core.db.models.camera import Camera
    from sqlalchemy import select
    
    # Get store_id from camera
    camera_result = await db.execute(select(Camera.store_id).where(Camera.id == self.camera_id))
    store_id = camera_result.scalar_one_or_none()
    
    debug_record = PersonDebug(
        camera_id=self.camera_id,
        store_id=store_id,
        track_session_id=track.track_session_id,
        person_identity_id=track.person_identity_id,
        occurred_at=utc_now(),
        
        # Track metrics
        bbox_height_px=track.bbox.get('y2', 0) - track.bbox.get('y1', 0) if track.bbox else None,
        bbox_width_px=track.bbox.get('x2', 0) - track.bbox.get('x1', 0) if track.bbox else None,
        track_total_frames=track.total_frames,
        track_age_seconds=track.track_age_seconds,
        
        # Merge kwargs (quality scores, failure info, etc.)
        **kwargs
    )
    
    db.add(debug_record)
    # Note: commit happens in batch at end of _persist_batch
```

Then modify `_run_reid()` to call this at each failure point:

```python
async def _run_reid(self, db, frame, track: ActiveTrack):
    """Run ReID pipeline: crop -> quality -> embedding -> accumulation -> decision."""
    track.last_reid_time = utc_now()
    
    # Check 1: Bbox size gate
    bbox_height = track.bbox.get('y2', 0) - track.bbox.get('y1', 0) if track.bbox else 0
    if bbox_height < 100:
        await self._log_debug_record(
            db, track,
            reid_attempted=False,
            reid_success=False,
            bbox_height_px=bbox_height,
            failure_stage="bbox_too_small",
            failure_reason=f"Bounding box height {bbox_height:.0f}px < 100px minimum. Person too far from camera."
        )
        return
    
    # Check 2: Crop extraction
    crop = extract_crop(frame, track.bbox)
    if crop is None or crop.size == 0:
        await self._log_debug_record(
            db, track,
            reid_attempted=True,
            reid_success=False,
            failure_stage="crop_extraction_failed",
            failure_reason="Failed to extract crop from bounding box."
        )
        return
    
    # Check 3: Quality assessment
    h, w = crop.shape[:2]
    quality = assess_crop_quality(crop)
    
    if quality < self.settings.REID_CROP_QUALITY_THRESHOLD:
        await self._log_debug_record(
            db, track,
            reid_attempted=True,
            reid_success=False,
            crop_height_px=h,
            crop_width_px=w,
            quality_score=quality,
            quality_passed=False,
            failure_stage="quality_too_low",
            failure_reason=f"Quality score {quality:.2f} < {self.settings.REID_CROP_QUALITY_THRESHOLD} threshold."
        )
        return
    
    # ... continue with rest of pipeline ...
    
    # On SUCCESS (after identity decision):
    await self._log_debug_record(
        db, track,
        reid_attempted=True,
        reid_success=True,
        crop_path=crop_path,
        crop_height_px=h,
        crop_width_px=w,
        quality_score=quality,
        quality_passed=True,
        face_detected=face_result is not None,
        face_score=face_score,
        reid_score=best_similarity,
        reid_confident=is_confident,
        reid_frame_count=track.reid_frame_count,
        failure_stage=None,
        failure_reason=None,
    )
```

---

## 📝 Step 4: Frontend Implementation

### File Structure (in Retail-Eye-Insights repo):
```
src/
  pages/
    Debug.tsx                  ← New debug page
  components/
    debug/
      DebugFilters.tsx        ← Filter bar
      DebugSummary.tsx        ← Summary cards
      DebugTable.tsx          ← Data table
      DebugDetailModal.tsx    ← Expandable row details
  api/
    debug.ts                   ← API calls
```

### API Integration (`src/api/debug.ts`):
```typescript
export interface DebugRecord {
  id: string;
  camera_name: string;
  occurred_at: string;
  reid_success: boolean;
  quality_score: number;
  failure_stage: string | null;
  failure_reason: string | null;
  crop_path: string | null;
  // ... all other fields
}

export async function getDebugDetections(params: {
  store_id?: string;
  camera_id?: string;
  start_time?: string;
  end_time?: string;
  status?: 'all' | 'detected' | 'not_detected';
  page?: number;
  limit?: number;
}) {
  const queryString = new URLSearchParams(params).toString();
  const response = await fetch(`/api/v2/debug/detections?${queryString}`);
  return response.json();
}
```

### Page Component (`src/pages/Debug.tsx`):
```typescript
export default function DebugPage() {
  const [filters, setFilters] = useState({
    store_id: null,
    camera_id: null,
    time_range: 'today',
    status: 'all',
    page: 1,
  });
  
  const { data, isLoading } = useQuery(['debug-detections', filters], () =>
    getDebugDetections(filters)
  );
  
  return (
    <div className="debug-page">
      <h1>🔍 Debug - Detection Insights</h1>
      
      <DebugFilters filters={filters} onChange={setFilters} />
      
      <DebugSummary 
        total={data?.summary.total}
        detected={data?.summary.detected}
        detectionRate={data?.summary.detection_rate}
        failureBreakdown={data?.summary.failure_by_stage}
      />
      
      <DebugTable records={data?.records} />
    </div>
  );
}
```

---

## 🚀 Deployment Steps

1. **Run migration:**
   ```bash
   # Apply the migration
   python3 -c "from alembic.config import Config; from alembic import command; cfg = Config('alembic.ini'); command.upgrade(cfg, 'head')"
   ```

2. **Create API modules** (copy code above into those files)

3. **Restart backend:**
   ```bash
   sudo systemctl restart retail-ai
   ```

4. **Implement frontend** (in Retail-Eye-Insights repo)

5. **Test:**
   ```bash
   # Test API
   curl -X GET "http://localhost:8000/api/v2/debug/detections?status=all&limit=10" \
     -H "Authorization: Bearer YOUR_TOKEN"
   ```

---

## 📊 Expected Output

### API Response Example:
```json
{
  "summary": {
    "total": 248,
    "detected": 156,
    "not_detected": 92,
    "detection_rate": 62.9,
    "failure_by_stage": {
      "quality_too_low": 45,
      "bbox_too_small": 30,
      "keypoint_gate": 12,
      "insufficient_frames": 5
    }
  },
  "records": [
    {
      "id": "abc-123",
      "camera_name": "Entry Cam 1",
      "occurred_at": "2026-06-30T15:02:00+05:30",
      "reid_success": false,
      "quality_score": 0.43,
      "failure_stage": "quality_too_low",
      "failure_reason": "Quality score 0.43 < 0.70 threshold",
      "crop_path": "retail/crops/crop_cam1_xxx.jpg"
    }
  ],
  "total": 248,
  "page": 1,
  "limit": 50
}
```

---

**Status:** Backend 80% complete (needs camera_worker instrumentation) | Frontend 0% (needs full implementation)
