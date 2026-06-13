"""drop the stores concept entirely

Revision ID: 0007_drop_stores
Revises: 0006_user_slim
Create Date: 2026-06-13

Single-pharmacy (Apollo) deployment: the multi-store concept is removed.
- cameras.store_id / users.store_id FK columns are dropped
- daily_analytics_summary becomes a single global row per day (store_id dropped,
  summary_date made unique)
- the stores table is dropped
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "0007_drop_stores"
down_revision: Union[str, None] = "0006_user_slim"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _drop_fk_if_exists(table: str, column: str) -> None:
    """Drop any FK constraint on table.column (constraint name is auto-generated)."""
    op.execute(
        sa.text(
            f"""
            DO $$
            DECLARE c text;
            BEGIN
                FOR c IN
                    SELECT conname FROM pg_constraint con
                    JOIN pg_class rel ON rel.oid = con.conrelid
                    JOIN pg_attribute att ON att.attrelid = con.conrelid
                        AND att.attnum = ANY(con.conkey)
                    WHERE rel.relname = '{table}'
                      AND att.attname = '{column}'
                      AND con.contype = 'f'
                LOOP
                    EXECUTE format('ALTER TABLE {table} DROP CONSTRAINT %I', c);
                END LOOP;
            END $$;
            """
        )
    )


def upgrade() -> None:
    # cameras.store_id
    _drop_fk_if_exists("cameras", "store_id")
    op.drop_column("cameras", "store_id")

    # users.store_id
    _drop_fk_if_exists("users", "store_id")
    op.drop_column("users", "store_id")

    # daily_analytics_summary: drop store_id, make summary_date unique
    _drop_fk_if_exists("daily_analytics_summary", "store_id")
    op.drop_column("daily_analytics_summary", "store_id")
    op.create_unique_constraint(
        "uq_daily_analytics_summary_date", "daily_analytics_summary", ["summary_date"]
    )

    # finally drop the stores table
    op.drop_table("stores")


def downgrade() -> None:
    # recreate stores table
    op.create_table(
        "stores",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("address", sa.String(length=500), nullable=True),
        sa.Column("city", sa.String(length=100), nullable=True),
        sa.Column("state", sa.String(length=100), nullable=True),
        sa.Column("country", sa.String(length=100), nullable=True),
        sa.Column("timezone", sa.String(length=50), nullable=False, server_default="Asia/Kolkata"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.drop_constraint(
        "uq_daily_analytics_summary_date", "daily_analytics_summary", type_="unique"
    )
    op.add_column(
        "daily_analytics_summary",
        sa.Column("store_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "daily_analytics_summary_store_id_fkey",
        "daily_analytics_summary", "stores", ["store_id"], ["id"],
    )

    op.add_column("users", sa.Column("store_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key("users_store_id_fkey", "users", "stores", ["store_id"], ["id"])

    op.add_column("cameras", sa.Column("store_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key("cameras_store_id_fkey", "cameras", "stores", ["store_id"], ["id"])
