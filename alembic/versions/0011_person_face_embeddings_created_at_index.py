"""person_face_embeddings created_at index for incremental dedup probing

The dedup pair-discovery query (2026-10-08) only uses embeddings created within
DEDUP_PROBE_WINDOW_MINUTES as the OUTER probe side; the inner side is the
IVFFlat ANN index. This btree on created_at turns the outer side from a
filtered seq scan into a range scan of ~tens of rows per run.

Revision ID: 0011
Revises: 0010_staff_reg
Create Date: 2026-10-08 18:30:00.000000

"""
from typing import Sequence, Union

from alembic import op

revision: str = "0011"
down_revision: Union[str, None] = "0010_staff_reg"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_person_face_embeddings_created_at",
        "person_face_embeddings",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_person_face_embeddings_created_at",
        table_name="person_face_embeddings",
    )
