"""Add onvif_id to cameras - stable identity fallback for MAC randomization.

Revision ID: 0015
Revises: 0014
Create Date: 2026-09-30
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "cameras",
        sa.Column("onvif_id", sa.String(length=255), nullable=True),
    )
    op.create_index(
        "ix_cameras_onvif_id", "cameras", ["onvif_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_cameras_onvif_id", table_name="cameras")
    op.drop_column("cameras", "onvif_id")
