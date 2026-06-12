"""Initial schema - all tables, uuid-ossp and pgvector extensions.

Revision ID: 0001_initial
Revises:
Create Date: 2026-06-12
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Required PostgreSQL extensions
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create all tables from the SQLAlchemy metadata.
    # All models are imported in alembic/env.py so metadata is complete.
    from app.core.db.base import Base
    import app.core.db.models  # noqa: F401 - ensure all models registered

    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)

    # IVFFlat index for fast cosine-distance ReID search on person_embeddings
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_person_embeddings_embedding
        ON person_embeddings
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
        """
    )


def downgrade() -> None:
    from app.core.db.base import Base
    import app.core.db.models  # noqa: F401

    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)