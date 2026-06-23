"""Add footfall_count, purchase_count, camera_count to stores table.

Revision ID: 0004_add_store_counts
Revises: 0003_add_store_lookups
Create Date: 2026-06-23
"""

import sqlalchemy as sa
from alembic import op

revision = "0004_add_store_counts"
down_revision = "0003_add_store_lookups"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for col in ("footfall_count", "purchase_count", "camera_count"):
        op.add_column(
            "stores",
            sa.Column(col, sa.Integer(), nullable=False, server_default="0"),
        )


def downgrade() -> None:
    for col in ("footfall_count", "purchase_count", "camera_count"):
        op.drop_column("stores", col)
