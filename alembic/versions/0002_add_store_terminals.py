"""Add store_terminals lookup table.

Revision ID: 0002_add_store_terminals
Revises: 0001_initial
Create Date: 2026-06-23
"""

from alembic import op
import sqlalchemy as sa


revision = "0002_add_store_terminals"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "store_terminals",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_store_terminals")),
        sa.UniqueConstraint("name", name=op.f("uq_store_terminals_name")),
    )
    op.create_index(
        op.f("ix_store_terminals_name"), "store_terminals", ["name"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_store_terminals_name"), table_name="store_terminals")
    op.drop_table("store_terminals")