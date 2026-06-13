"""simplify area to name-only, camera store_id server-side, slim zone fields

Revision ID: 0004_simplify_acz
Revises: 0003_areas_camera_zones
Create Date: 2026-06-13

Changes (per Apollo single-store flow):
- areas: keep only id/name/timestamps -> drop area_type, description, store_id, is_active.
- cameras: make store_id nullable (auto-assigned server-side, never sent by client).
- zones: drop point_indices, line_config, color (operator only draws the polygon).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0004_simplify_acz"
down_revision: Union[str, None] = "0003_areas_camera_zones"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # areas: collapse to name-only
    op.drop_index("ix_areas_store_id", table_name="areas")
    op.drop_constraint("areas_store_id_fkey", "areas", type_="foreignkey")
    op.drop_column("areas", "store_id")
    op.drop_column("areas", "area_type")
    op.drop_column("areas", "description")
    op.drop_column("areas", "is_active")

    # cameras: store_id assigned server-side
    op.alter_column("cameras", "store_id", existing_type=sa.dialects.postgresql.UUID(), nullable=True)

    # zones: only polygon + meta needed
    op.drop_column("zones", "point_indices")
    op.drop_column("zones", "line_config")
    op.drop_column("zones", "color")


def downgrade() -> None:
    import sqlalchemy.dialects.postgresql as pg

    # zones
    op.add_column("zones", sa.Column("color", sa.String(length=20), nullable=True, server_default="#FF0000"))
    op.add_column("zones", sa.Column("line_config", pg.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("zones", sa.Column("point_indices", pg.JSONB(astext_type=sa.Text()), nullable=True))

    # cameras
    op.alter_column("cameras", "store_id", existing_type=pg.UUID(), nullable=False)

    # areas
    op.add_column("areas", sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()))
    op.add_column("areas", sa.Column("description", sa.String(length=500), nullable=True))
    op.add_column("areas", sa.Column("area_type", sa.String(length=50), nullable=False, server_default="general"))
    op.add_column("areas", sa.Column("store_id", pg.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key("areas_store_id_fkey", "areas", "stores", ["store_id"], ["id"], ondelete="SET NULL")
    op.create_index("ix_areas_store_id", "areas", ["store_id"])
