"""daily_analytics_frozen — immutable IST-day analytics snapshots

Dashboard / analytics / camera-feed past days read this table only.
Today stays live. Written at 00:10 IST for the previous IST day.

Revision ID: 0012
Revises: 0011
Create Date: 2026-10-15 18:30:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0012"
down_revision: Union[str, None] = "0011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "daily_analytics_frozen",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source", sa.String(32), nullable=False, server_default="scheduled"),
        sa.Column("is_partial", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("unique_total", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("unique_customers", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("unique_staff", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("night_only_persons", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("adjusted_unique", sa.Float(), nullable=False, server_default="0"),
        sa.Column("adjustment_factor", sa.Float(), nullable=False, server_default="1"),
        sa.Column("factor_model_version", sa.String(64), nullable=False, server_default="none"),
        sa.Column("purchases", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("billing_interactions", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_events", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_tracks", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("avg_dwell_seconds", sa.Float(), nullable=True),
        sa.Column("demographics", postgresql.JSONB(), nullable=True),
        sa.Column("hourly", postgresql.JSONB(), nullable=True),
        sa.Column("per_camera", postgresql.JSONB(), nullable=True),
        sa.Column("zone_counts", postgresql.JSONB(), nullable=True),
        sa.Column("person_ids", postgresql.JSONB(), nullable=True),
        sa.Column("meta", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_daily_analytics_frozen_day", "daily_analytics_frozen", ["day"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_daily_analytics_frozen_day", table_name="daily_analytics_frozen")
    op.drop_table("daily_analytics_frozen")
