"""Demographics and frame rotation columns.

Revision ID: 0003_demographics_rotation
Revises: 0002_perf_indexes
Create Date: 2026-06-13
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0003_demographics_rotation"
down_revision = "0002_perf_indexes"
branch_labels = None
depends_on = None


def _column_exists(table: str, column: str) -> bool:
    conn = op.get_bind()
    result = conn.execute(
        sa.text(
            """
            SELECT 1 FROM information_schema.columns
            WHERE table_name = :table AND column_name = :column
            """
        ),
        {"table": table, "column": column},
    )
    return result.fetchone() is not None


def upgrade() -> None:
    # 1. Person Identities Table Columns
    if not _column_exists("person_identities", "gender"):
        op.add_column(
            "person_identities", sa.Column("gender", sa.String(length=10), nullable=True)
        )
    if not _column_exists("person_identities", "age_group"):
        op.add_column(
            "person_identities", sa.Column("age_group", sa.String(length=20), nullable=True)
        )
    if not _column_exists("person_identities", "estimated_age"):
        op.add_column(
            "person_identities", sa.Column("estimated_age", sa.Integer(), nullable=True)
        )
    if not _column_exists("person_identities", "best_face_score"):
        op.add_column(
            "person_identities", sa.Column("best_face_score", sa.Float(), nullable=True)
        )
    if not _column_exists("person_identities", "face_crop_path"):
        op.add_column(
            "person_identities",
            sa.Column("face_crop_path", sa.String(length=500), nullable=True),
        )

    # 2. Track Sessions Table Columns
    if not _column_exists("track_sessions", "gender"):
        op.add_column(
            "track_sessions", sa.Column("gender", sa.String(length=10), nullable=True)
        )
    if not _column_exists("track_sessions", "age_group"):
        op.add_column(
            "track_sessions", sa.Column("age_group", sa.String(length=20), nullable=True)
        )

    # 3. Cameras Table Columns
    if not _column_exists("cameras", "frame_rotation"):
        op.add_column(
            "cameras", sa.Column("frame_rotation", sa.Integer(), nullable=True)
        )


def downgrade() -> None:
    # 1. Cameras Table Columns
    if _column_exists("cameras", "frame_rotation"):
        op.drop_column("cameras", "frame_rotation")

    # 2. Track Sessions Table Columns
    if _column_exists("track_sessions", "age_group"):
        op.drop_column("track_sessions", "age_group")
    if _column_exists("track_sessions", "gender"):
        op.drop_column("track_sessions", "gender")

    # 3. Person Identities Table Columns
    if _column_exists("person_identities", "face_crop_path"):
        op.drop_column("person_identities", "face_crop_path")
    if _column_exists("person_identities", "best_face_score"):
        op.drop_column("person_identities", "best_face_score")
    if _column_exists("person_identities", "estimated_age"):
        op.drop_column("person_identities", "estimated_age")
    if _column_exists("person_identities", "age_group"):
        op.drop_column("person_identities", "age_group")
    if _column_exists("person_identities", "gender"):
        op.drop_column("person_identities", "gender")