"""drop camera_views (cameras are static, no ROI/view concept)

Revision ID: 0005_drop_views
Revises: 0004_simplify_acz
Create Date: 2026-06-13

Cameras are statically mounted, so per-camera "views"/ROIs are not needed.
Detection runs on the full frame and is filtered by zones only.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlalchemy.dialects.postgresql as pg


# revision identifiers, used by Alembic.
revision: str = "0005_drop_views"
down_revision: Union[str, None] = "0004_simplify_acz"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_table("camera_views")
    # The enum type is now unused; drop it so a future re-create is clean.
    op.execute("DROP TYPE IF EXISTS view_type_enum")


def downgrade() -> None:
    view_type_enum = sa.Enum(
        "full_frame",
        "entry_gate_view",
        "billing_counter_view",
        "queue_view",
        "product_shelf_view",
        "ignore_area",
        name="view_type_enum",
    )
    view_type_enum.create(op.get_bind(), checkfirst=True)
    op.create_table(
        "camera_views",
        sa.Column("id", pg.UUID(as_uuid=True), primary_key=True),
        sa.Column("camera_id", pg.UUID(as_uuid=True), sa.ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("view_type", view_type_enum, nullable=False, server_default="full_frame"),
        sa.Column("polygon", pg.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("is_default", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
