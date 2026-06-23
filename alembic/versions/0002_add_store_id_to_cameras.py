"""Add store_id FK to cameras table.

Revision ID: 0002_add_store_id
Revises: 0001_initial
Create Date: 2026-06-23
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0002_add_store_id"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "cameras",
        sa.Column(
            "store_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("stores.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    # Index for store-wise camera lookups
    op.create_index("ix_cameras_store_id", "cameras", ["store_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_cameras_store_id", table_name="cameras")
    op.drop_column("cameras", "store_id")