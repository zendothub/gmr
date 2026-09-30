"""track_sessions started_at index for seen-in-window person counting

Person-based analytics (footfall, gender, age, demographics) switched from
PersonIdentity.last_seen_at to "track session started in window" (EXISTS /
date_trunc on track_sessions.started_at). This index supports the per-slot
bucketing and range scans on started_at.

Revision ID: 0009
Revises: 0008
Create Date: 2026-09-21 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op

revision: str = "0009"
down_revision: Union[str, None] = "0008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_track_sessions_started_at",
        "track_sessions",
        ["started_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_track_sessions_started_at", table_name="track_sessions")
