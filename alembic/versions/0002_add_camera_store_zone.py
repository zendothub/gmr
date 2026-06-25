"""Add zone_id column to cameras table — links camera to a store zone (position).

Cameras can optionally be linked to a store zone / position within the store
(e.g. "Entry", "Checkout", "Aisle 3") via the store_zones lookup table.
This is separate from the detection polygon zones (camera.py → Zone).

Revision ID: 0002_add_camera_store_zone
Revises: 0001_initial
Create Date: 2026-06-25
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "0002_add_camera_store_zone"
down_revision: Union[str, None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "cameras",
        sa.Column(
            "zone_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("store_zones.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("cameras", "zone_id")