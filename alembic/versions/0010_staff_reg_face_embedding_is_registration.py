"""person_face_embeddings.is_registration — pinned staff registration faces

Staff are registered with a photo (POST /api/staff/register) instead of the
attendance-based auto classifier. Registration faces are the trusted anchor for
that identity, so they are pinned: excluded from the per-person face cap
(MAX_FACE_EMBEDDINGS_PER_PERSON) and never removed by contamination cleanup or
dedup absorb.

Revision ID: 0010_staff_reg
Revises: 0009
Create Date: 2026-10-05 12:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0010_staff_reg"
down_revision: Union[str, None] = "0009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "person_face_embeddings",
        sa.Column(
            "is_registration",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )


def downgrade() -> None:
    op.drop_column("person_face_embeddings", "is_registration")
