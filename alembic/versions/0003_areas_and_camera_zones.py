"""areas independent + camera->many zones + medicine pickup zone type

Revision ID: 0003_areas_camera_zones
Revises: d02b913dd571
Create Date: 2026-06-13

Changes:
- New independent `areas` table (Entry/Exit/Billing/Medicine Pickup, ...).
- `cameras.area_id` FK -> areas (area picked from dropdown when adding a camera).
- Drop the old single `cameras.zone_id` relationship.
- `zones.camera_id` FK -> cameras (a camera now owns MANY zones).
- `zones.point_indices` JSONB (selected polygon point indices).
- New `medicine_pickup_zone` value on the zone_type enum.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "0003_areas_camera_zones"
down_revision: Union[str, None] = "d02b913dd571"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1) areas table -------------------------------------------------------
    op.create_table(
        "areas",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("area_type", sa.String(length=50), nullable=False, server_default="general"),
        sa.Column("description", sa.String(length=500), nullable=True),
        sa.Column("store_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["store_id"], ["stores.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_areas_store_id", "areas", ["store_id"])

    # 2) cameras: drop old single zone_id, add area_id --------------------
    op.drop_constraint("cameras_zone_id_fkey", "cameras", type_="foreignkey")
    op.drop_column("cameras", "zone_id")

    op.add_column("cameras", sa.Column("area_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key(
        "cameras_area_id_fkey", "cameras", "areas", ["area_id"], ["id"], ondelete="SET NULL"
    )
    op.create_index("ix_cameras_area_id", "cameras", ["area_id"])

    # 3) zones: add camera_id + point_indices -----------------------------
    op.add_column("zones", sa.Column("camera_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column("zones", sa.Column("point_indices", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.create_foreign_key(
        "zones_camera_id_fkey", "zones", "cameras", ["camera_id"], ["id"], ondelete="CASCADE"
    )
    op.create_index("ix_zones_camera_id", "zones", ["camera_id"])

    # 4) new enum value for the pharmacy medicine pickup zone -------------
    # PG 16 supports ADD VALUE inside a transaction (value just can't be used same txn).
    op.execute("ALTER TYPE zone_type_enum ADD VALUE IF NOT EXISTS 'medicine_pickup_zone'")


def downgrade() -> None:
    # zones
    op.drop_index("ix_zones_camera_id", table_name="zones")
    op.drop_constraint("zones_camera_id_fkey", "zones", type_="foreignkey")
    op.drop_column("zones", "point_indices")
    op.drop_column("zones", "camera_id")

    # cameras: restore single zone_id
    op.drop_index("ix_cameras_area_id", table_name="cameras")
    op.drop_constraint("cameras_area_id_fkey", "cameras", type_="foreignkey")
    op.drop_column("cameras", "area_id")

    op.add_column("cameras", sa.Column("zone_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key(
        "cameras_zone_id_fkey", "cameras", "zones", ["zone_id"], ["id"], ondelete="SET NULL"
    )

    # areas
    op.drop_index("ix_areas_store_id", table_name="areas")
    op.drop_table("areas")

    # NOTE: the 'medicine_pickup_zone' enum value is intentionally NOT removed -
    # PostgreSQL cannot drop a single enum value without recreating the type.
