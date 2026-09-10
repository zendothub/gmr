"""Add employee gender and employee-specific weekend/weekly-off configuration.

Revision ID: 0010
Revises: 0009
Create Date: 2026-09-09
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ---------------------------------------------------------------------------
    # 1. employee_gender ENUM
    # ---------------------------------------------------------------------------
    employee_gender_enum = sa.Enum("MALE", "FEMALE", "OTHER", name="employee_gender")
    employee_gender_enum.create(op.get_bind(), checkfirst=True)

    # ---------------------------------------------------------------------------
    # 2. employees.gender — nullable, existing employees keep gender unset
    # ---------------------------------------------------------------------------
    op.add_column(
        "employees",
        sa.Column("gender", employee_gender_enum, nullable=True),
    )

    # ---------------------------------------------------------------------------
    # 3. employees.weekends — employee-specific weekly-off days.
    #    Backfilled to ['SATURDAY', 'SUNDAY'] for existing rows so absence
    #    reporting for pre-existing employees doesn't regress.
    # ---------------------------------------------------------------------------
    op.add_column(
        "employees",
        sa.Column(
            "weekends",
            postgresql.ARRAY(sa.String(9)),
            nullable=False,
            server_default=sa.text("ARRAY['SATURDAY','SUNDAY']::varchar[]"),
        ),
    )


def downgrade() -> None:
    op.drop_column("employees", "weekends")
    op.drop_column("employees", "gender")
    sa.Enum(name="employee_gender").drop(op.get_bind(), checkfirst=True)
