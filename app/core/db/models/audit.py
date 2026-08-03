"""Audit / debug event models for identity merges and fragmented billing visits."""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, Integer, Float, ForeignKey, DateTime, Text, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class IdentityMergeEvent(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """One row per loser→winner merge (dedup job). Loser id is snapshot only (row deleted)."""

    __tablename__ = "identity_merge_events"

    merged_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    source: Mapped[str] = mapped_column(String(40), nullable=False, default="dedup", index=True)
    # Parent dedup job cycle (same id/at for all merges in one run)
    job_run_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )
    job_run_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )

    winner_person_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("person_identities.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    # Loser is deleted after merge — store UUID only (no FK).
    loser_person_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )

    face_similarity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    winner_face_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    loser_face_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    winner_first_seen_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    loser_first_seen_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    loser_visit_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    loser_track_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    winner_visit_count_before: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    winner_face_crop_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    loser_face_crop_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    metadata_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)


class FragmentedTrackEvent(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Audit when visit-repair stitches null sessions or inserts a billing row from fragments."""

    __tablename__ = "fragmented_track_events"

    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    # billing_insert | null_stitch | null_bi_fill
    event_type: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    # Parent dedup job that invoked visit-repair (groups BI inserts from one cycle)
    job_run_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )
    job_run_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )

    person_identity_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("person_identities.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    camera_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("cameras.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    zone_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("zones.id", ondelete="SET NULL"),
        nullable=True,
    )
    billing_interaction_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("billing_interactions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    primary_track_session_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("track_sessions.id", ondelete="SET NULL"),
        nullable=True,
    )

    fragment_session_ids: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    fragment_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    sum_dwell_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dwell_threshold: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stitch_gap_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stitch_reason: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    body_median: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    face_max: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    metadata_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
