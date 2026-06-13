"""add cameras.stream_path (MediaMTX feed path)

Revision ID: 0008_stream_path
Revises: 0007_drop_stores
Create Date: 2026-06-13

The backend republishes each camera's RTSP into a MediaMTX path; the browser
pulls the feed back (WebRTC/HLS) from that path. The path is stored on the
camera so the frontend can render the feed directly.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0008_stream_path"
down_revision: Union[str, None] = "0007_drop_stores"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("cameras", sa.Column("stream_path", sa.String(length=255), nullable=True))
    # Backfill existing rows with the deterministic path: cam_<id-without-dashes>.
    op.execute(
        sa.text("UPDATE cameras SET stream_path = 'cam_' || replace(id::text, '-', '')")
    )


def downgrade() -> None:
    op.drop_column("cameras", "stream_path")
