"""merge_stream_path_and_main_chain

Revision ID: 40a9c0e6c353
Revises: 0008_stream_path, 0012_update_event_severity_enum
Create Date: 2026-06-17 17:03:39.729830

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = '40a9c0e6c353'
down_revision: Union[str, None] = ('0008_stream_path', '0012_update_event_severity_enum')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
