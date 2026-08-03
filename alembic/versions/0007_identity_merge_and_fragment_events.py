"""identity_merge_events + fragmented_track_events audit tables

Revision ID: 0007
Revises: 0006
Create Date: 2026-08-03 13:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0007"
down_revision: Union[str, None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "identity_merge_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("merged_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("source", sa.String(length=40), nullable=False, server_default="dedup"),
        sa.Column("winner_person_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("loser_person_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("face_similarity", sa.Float(), nullable=True),
        sa.Column("winner_face_score", sa.Float(), nullable=True),
        sa.Column("loser_face_score", sa.Float(), nullable=True),
        sa.Column("winner_first_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("loser_first_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("loser_visit_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("loser_track_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("winner_visit_count_before", sa.Integer(), nullable=True),
        sa.Column("winner_face_crop_path", sa.Text(), nullable=True),
        sa.Column("loser_face_crop_path", sa.Text(), nullable=True),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["winner_person_id"], ["person_identities.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_identity_merge_events_merged_at", "identity_merge_events", ["merged_at"])
    op.create_index("ix_identity_merge_events_source", "identity_merge_events", ["source"])
    op.create_index("ix_identity_merge_events_winner_person_id", "identity_merge_events", ["winner_person_id"])
    op.create_index("ix_identity_merge_events_loser_person_id", "identity_merge_events", ["loser_person_id"])

    op.create_table(
        "fragmented_track_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("event_type", sa.String(length=40), nullable=False),
        sa.Column("person_identity_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("camera_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("zone_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("billing_interaction_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("primary_track_session_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("fragment_session_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("fragment_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("sum_dwell_seconds", sa.Float(), nullable=True),
        sa.Column("dwell_threshold", sa.Float(), nullable=True),
        sa.Column("stitch_gap_seconds", sa.Float(), nullable=True),
        sa.Column("stitch_reason", sa.String(length=40), nullable=True),
        sa.Column("body_median", sa.Float(), nullable=True),
        sa.Column("face_max", sa.Float(), nullable=True),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["person_identity_id"], ["person_identities.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["camera_id"], ["cameras.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["zone_id"], ["zones.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["billing_interaction_id"], ["billing_interactions.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["primary_track_session_id"], ["track_sessions.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_fragmented_track_events_occurred_at", "fragmented_track_events", ["occurred_at"])
    op.create_index("ix_fragmented_track_events_event_type", "fragmented_track_events", ["event_type"])
    op.create_index("ix_fragmented_track_events_person_identity_id", "fragmented_track_events", ["person_identity_id"])
    op.create_index("ix_fragmented_track_events_camera_id", "fragmented_track_events", ["camera_id"])
    op.create_index(
        "ix_fragmented_track_events_billing_interaction_id",
        "fragmented_track_events",
        ["billing_interaction_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_fragmented_track_events_billing_interaction_id", table_name="fragmented_track_events")
    op.drop_index("ix_fragmented_track_events_camera_id", table_name="fragmented_track_events")
    op.drop_index("ix_fragmented_track_events_person_identity_id", table_name="fragmented_track_events")
    op.drop_index("ix_fragmented_track_events_event_type", table_name="fragmented_track_events")
    op.drop_index("ix_fragmented_track_events_occurred_at", table_name="fragmented_track_events")
    op.drop_table("fragmented_track_events")

    op.drop_index("ix_identity_merge_events_loser_person_id", table_name="identity_merge_events")
    op.drop_index("ix_identity_merge_events_winner_person_id", table_name="identity_merge_events")
    op.drop_index("ix_identity_merge_events_source", table_name="identity_merge_events")
    op.drop_index("ix_identity_merge_events_merged_at", table_name="identity_merge_events")
    op.drop_table("identity_merge_events")
