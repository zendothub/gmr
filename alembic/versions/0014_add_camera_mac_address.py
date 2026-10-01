"""Add mac_address to cameras for IP-change auto-recovery.

Revision ID: 0014
Revises: 0013
Create Date: 2026-09-28
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "cameras",
        sa.Column("mac_address", sa.String(length=17), nullable=True),
    )
    op.create_index(
        "ix_cameras_mac_address", "cameras", ["mac_address"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_cameras_mac_address", table_name="cameras")
    op.drop_column("cameras", "mac_address")
