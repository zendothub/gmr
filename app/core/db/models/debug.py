"""Debug models for tracking detection pipeline failures."""

from sqlalchemy import Column, String, Boolean, Float, Integer, Text, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
import uuid

from app.core.db.base import Base, UUIDPrimaryKeyMixin, TimestampMixin


class PersonDebug(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """
    Debug records for person detection pipeline.
    
    Tracks every detection attempt through the full pipeline:
    YOLO detect → Crop → Quality check → OSNet → Face → Accumulation → ReID decision
    
    Records all metrics at each stage and captures failure reasons.
    """
    __tablename__ = "person_debug"

    # Context
    camera_id = Column(UUID(as_uuid=True), ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True, index=True)
    store_id = Column(UUID(as_uuid=True), ForeignKey("stores.id", ondelete="SET NULL"), nullable=True, index=True)
    track_session_id = Column(UUID(as_uuid=True), ForeignKey("track_sessions.id", ondelete="CASCADE"), nullable=True, index=True)
    person_identity_id = Column(UUID(as_uuid=True), ForeignKey("person_identities.id", ondelete="SET NULL"), nullable=True, index=True)
    occurred_at = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # Detection outcome
    reid_attempted = Column(Boolean, default=False, nullable=False)
    reid_success = Column(Boolean, default=False, nullable=False, index=True)
    
    # Track metrics
    bbox_height_px = Column(Float, nullable=True)
    bbox_width_px = Column(Float, nullable=True)
    detection_confidence = Column(Float, nullable=True)
    track_total_frames = Column(Integer, nullable=True)
    track_age_seconds = Column(Float, nullable=True)
    
    # Crop
    crop_path = Column(Text, nullable=True)           # body crop MinIO path (legacy)
    body_crop_path = Column(Text, nullable=True)       # full body crop MinIO path
    crop_height_px = Column(Integer, nullable=True)
    crop_width_px = Column(Integer, nullable=True)
    
    # Quality scores
    quality_score = Column(Float, nullable=True)
    quality_passed = Column(Boolean, default=False)
    keypoint_visibility_ratio = Column(Float, nullable=True)
    keypoint_gate_passed = Column(Boolean, default=False)
    sharpness_score = Column(Float, nullable=True)
    size_score = Column(Float, nullable=True)
    aspect_ratio = Column(Float, nullable=True)
    brightness_mean = Column(Float, nullable=True)
    
    # Face
    face_detected = Column(Boolean, default=False)
    face_score = Column(Float, nullable=True)
    face_crop_path = Column(Text, nullable=True)
    face_age = Column(Integer, nullable=True)
    face_gender = Column(String(10), nullable=True)
    
    # ReID result
    reid_score = Column(Float, nullable=True)
    reid_confident = Column(Boolean, default=False)
    reid_frame_count = Column(Integer, default=0)
    
    # Failure info
    failure_stage = Column(String(100), nullable=True, index=True)
    failure_reason = Column(Text, nullable=True)
    
    # Additional metadata
    metadata_json = Column(JSONB, nullable=True)

    # Relationships — passive_deletes=True prevents SQLAlchemy from setting
    # FKs to NULL before the DB-level ON DELETE CASCADE fires.  Without this,
    # deleting a Camera fails because camera_id is NOT NULL.
    camera = relationship("Camera", backref="debug_records", passive_deletes=True)
    store = relationship("Store", backref="debug_records", passive_deletes=True)
    track_session = relationship("TrackSession", backref="debug_records", passive_deletes=True)
    person_identity = relationship("PersonIdentity", backref="debug_records", passive_deletes=True)

    def __repr__(self):
        status = "✅" if self.reid_success else "❌"
        return f"<PersonDebug {status} camera={self.camera_id} stage={self.failure_stage}>"
