"""Fix all camera_id foreign keys: add proper ON DELETE behavior.

Previously several FK constraints had no ON DELETE clause (NO ACTION),
which caused NOT NULL violations when a camera was deleted because
PostgreSQL tried to SET NULL on non-nullable columns.

- NOT NULL + CASCADE: track_sessions, events, billing_interactions
- nullable + SET NULL:  person_embeddings, person_face_embeddings, storage_objects
- nullable + CASCADE (already correct): zones

Revision ID: 0009_camera_fks
Revises: 8b8b559916d9
Create Date: 2026-06-14
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0009_camera_fks"
down_revision: Union[str, None] = "8b8b559916d9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # NOT NULL → CASCADE
    for table in ("track_sessions", "events", "billing_interactions"):
        op.execute(f"""
            ALTER TABLE {table}
            DROP CONSTRAINT IF EXISTS {table}_camera_id_fkey,
            ADD CONSTRAINT {table}_camera_id_fkey
                FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE CASCADE
        """)
    # nullable → SET NULL
    for table in ("person_embeddings", "person_face_embeddings", "storage_objects"):
        op.execute(f"""
            ALTER TABLE {table}
            DROP CONSTRAINT IF EXISTS {table}_camera_id_fkey,
            ADD CONSTRAINT {table}_camera_id_fkey
                FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE SET NULL
        """)


def downgrade() -> None:
    for table in ("track_sessions", "events", "billing_interactions"):
        op.execute(f"""
            ALTER TABLE {table}
            DROP CONSTRAINT IF EXISTS {table}_camera_id_fkey,
            ADD CONSTRAINT {table}_camera_id_fkey
                FOREIGN KEY (camera_id) REFERENCES cameras(id)
        """)
    for table in ("person_embeddings", "person_face_embeddings", "storage_objects"):
        op.execute(f"""
            ALTER TABLE {table}
            DROP CONSTRAINT IF EXISTS {table}_camera_id_fkey,
            ADD CONSTRAINT {table}_camera_id_fkey
                FOREIGN KEY (camera_id) REFERENCES cameras(id)
        """)