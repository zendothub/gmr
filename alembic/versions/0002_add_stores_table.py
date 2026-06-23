"""Add stores table.

Revision ID: 0002_add_stores
Revises: 0001_initial
Create Date: 2026-06-23
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0002_add_stores"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE TYPE store_status AS ENUM ('active', 'inactive')")

    op.create_table(
        "stores",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("category", sa.String(100), nullable=False),
        sa.Column(
            "status",
            sa.Enum("active", "inactive", name="store_status", create_type=False),
            nullable=False,
            server_default="active",
        ),
        sa.Column("terminal", sa.String(100), nullable=True),
        sa.Column("level", sa.String(100), nullable=True),
        sa.Column("zone_gate", sa.String(100), nullable=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_index("ix_stores_name", "stores", ["name"])


def downgrade() -> None:
    op.drop_index("ix_stores_name", table_name="stores")
    op.drop_table("stores")
    op.execute("DROP TYPE IF EXISTS store_status")
