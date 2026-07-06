"""add_ivfflat_index_on_person_face_embeddings

Revision ID: 0005
Revises: 38ae744f3444
Create Date: 2026-07-06 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op

revision: str = "0005"
down_revision: Union[str, None] = "38ae744f3444"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_person_face_embeddings_embedding
        ON person_face_embeddings
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 50)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_person_face_embeddings_embedding")
