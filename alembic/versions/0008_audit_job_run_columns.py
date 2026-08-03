"""add job_run_id/job_run_at to audit tables

Revision ID: 0008
Revises: 0007
Create Date: 2026-08-03 16:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0008"
down_revision: Union[str, None] = "0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "identity_merge_events",
        sa.Column("job_run_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "identity_merge_events",
        sa.Column("job_run_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_identity_merge_events_job_run_id",
        "identity_merge_events",
        ["job_run_id"],
    )
    op.create_index(
        "ix_identity_merge_events_job_run_at",
        "identity_merge_events",
        ["job_run_at"],
    )

    op.add_column(
        "fragmented_track_events",
        sa.Column("job_run_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "fragmented_track_events",
        sa.Column("job_run_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_fragmented_track_events_job_run_id",
        "fragmented_track_events",
        ["job_run_id"],
    )
    op.create_index(
        "ix_fragmented_track_events_job_run_at",
        "fragmented_track_events",
        ["job_run_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_fragmented_track_events_job_run_at", table_name="fragmented_track_events")
    op.drop_index("ix_fragmented_track_events_job_run_id", table_name="fragmented_track_events")
    op.drop_column("fragmented_track_events", "job_run_at")
    op.drop_column("fragmented_track_events", "job_run_id")

    op.drop_index("ix_identity_merge_events_job_run_at", table_name="identity_merge_events")
    op.drop_index("ix_identity_merge_events_job_run_id", table_name="identity_merge_events")
    op.drop_column("identity_merge_events", "job_run_at")
    op.drop_column("identity_merge_events", "job_run_id")
