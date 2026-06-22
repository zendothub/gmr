"""Add feature_requests table for admin dashboard feature communication.

Revision ID: 0010_feature_requests
Revises: 0009_camera_fks
Create Date: 2026-06-16
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0010_feature_requests"
down_revision: Union[str, None] = "0009_camera_fks"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _enum_exists(enum_name: str) -> bool:
    """Check if an enum type exists"""
    conn = op.get_bind()
    result = conn.execute(
        sa.text(
            "SELECT EXISTS ("
            "SELECT 1 FROM pg_type WHERE typname = :enum_name"
            ")"
        ),
        {"enum_name": enum_name},
    )
    return result.scalar()


def upgrade() -> None:
    # Create enum type if it doesn't exist using raw SQL
    if not _enum_exists("feature_status_enum"):
        op.execute(
            "CREATE TYPE feature_status_enum AS ENUM ('queued', 'in_progress', 'live')"
        )
    
    op.create_table(
        "feature_requests",
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column(
            "status",
            sa.Enum("queued", "in_progress", "live", name="feature_status_enum", create_type=False),
            nullable=False,
        ),
        sa.Column("forecast_message", sa.Text(), nullable=True),
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("feature_requests")
    # Drop the enum type so a re-run won't fail
    op.execute("DROP TYPE IF EXISTS feature_status_enum")