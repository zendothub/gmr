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
    
    # Create table without using sa.Enum to avoid auto-creation
    op.execute("""
        CREATE TABLE feature_requests (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            title VARCHAR(500) NOT NULL,
            description TEXT NOT NULL,
            status feature_status_enum NOT NULL,
            forecast_message TEXT,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
        )
    """)


def downgrade() -> None:
    op.drop_table("feature_requests")
    # Drop the enum type so a re-run won't fail
    op.execute("DROP TYPE IF EXISTS feature_status_enum")