"""Add store lookup tables (store_categories, store_levels, store_zones).

Revision ID: 0003_add_store_lookups
Revises: 0002_add_stores
Create Date: 2026-06-23
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from alembic import op

revision = "0003_add_store_lookups"
down_revision = "0002_add_stores"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for table in ("store_categories", "store_levels", "store_zones"):
        cols = [
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
            sa.Column("name", sa.String(100), nullable=False),
            sa.Column("description", sa.Text, nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        ]
        if table == "store_zones":
            cols.insert(2, sa.Column("terminal", sa.String(100), nullable=True))
        op.create_table(table, *cols)
        op.create_index(f"ix_{table}_name", table, ["name"], unique=True)


def downgrade() -> None:
    for table in ("store_zones", "store_levels", "store_categories"):
        op.drop_index(f"ix_{table}_name", table_name=table)
        op.drop_table(table)
