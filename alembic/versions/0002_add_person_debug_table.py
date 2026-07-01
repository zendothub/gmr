"""add_person_debug_table

Revision ID: 0002
Revises: 0001_initial
Create Date: 2026-06-30 16:14:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0002'
down_revision = '0001_initial'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create person_debug table
    op.create_table(
        'person_debug',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        
        # Context
        sa.Column('camera_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('store_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('track_session_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('person_identity_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('occurred_at', sa.DateTime(timezone=True), nullable=False),
        
        # Detection outcome
        sa.Column('reid_attempted', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('reid_success', sa.Boolean(), nullable=False, server_default='false'),
        
        # Track metrics
        sa.Column('bbox_height_px', sa.Float(), nullable=True),
        sa.Column('bbox_width_px', sa.Float(), nullable=True),
        sa.Column('detection_confidence', sa.Float(), nullable=True),
        sa.Column('track_total_frames', sa.Integer(), nullable=True),
        sa.Column('track_age_seconds', sa.Float(), nullable=True),
        
        # Crop
        sa.Column('crop_path', sa.Text(), nullable=True),
        sa.Column('crop_height_px', sa.Integer(), nullable=True),
        sa.Column('crop_width_px', sa.Integer(), nullable=True),
        
        # Quality scores
        sa.Column('quality_score', sa.Float(), nullable=True),
        sa.Column('quality_passed', sa.Boolean(), server_default='false'),
        sa.Column('keypoint_visibility_ratio', sa.Float(), nullable=True),
        sa.Column('keypoint_gate_passed', sa.Boolean(), server_default='false'),
        sa.Column('sharpness_score', sa.Float(), nullable=True),
        sa.Column('size_score', sa.Float(), nullable=True),
        sa.Column('aspect_ratio', sa.Float(), nullable=True),
        sa.Column('brightness_mean', sa.Float(), nullable=True),
        
        # Face
        sa.Column('face_detected', sa.Boolean(), server_default='false'),
        sa.Column('face_score', sa.Float(), nullable=True),
        sa.Column('face_crop_path', sa.Text(), nullable=True),
        sa.Column('face_age', sa.Integer(), nullable=True),
        sa.Column('face_gender', sa.String(length=10), nullable=True),
        
        # ReID result
        sa.Column('reid_score', sa.Float(), nullable=True),
        sa.Column('reid_confident', sa.Boolean(), server_default='false'),
        sa.Column('reid_frame_count', sa.Integer(), server_default='0'),
        
        # Failure info
        sa.Column('failure_stage', sa.String(length=100), nullable=True),
        sa.Column('failure_reason', sa.Text(), nullable=True),
        
        # Metadata
        sa.Column('metadata_json', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        
        # Constraints
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['camera_id'], ['cameras.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['store_id'], ['stores.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['track_session_id'], ['track_sessions.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['person_identity_id'], ['person_identities.id'], ondelete='SET NULL'),
    )
    
    # Create indexes for efficient queries
    op.create_index('ix_person_debug_camera_id', 'person_debug', ['camera_id'])
    op.create_index('ix_person_debug_store_id', 'person_debug', ['store_id'])
    op.create_index('ix_person_debug_track_session_id', 'person_debug', ['track_session_id'])
    op.create_index('ix_person_debug_person_identity_id', 'person_debug', ['person_identity_id'])
    op.create_index('ix_person_debug_occurred_at', 'person_debug', ['occurred_at'])
    op.create_index('ix_person_debug_reid_success', 'person_debug', ['reid_success'])
    op.create_index('ix_person_debug_failure_stage', 'person_debug', ['failure_stage'])


def downgrade() -> None:
    # Drop indexes
    op.drop_index('ix_person_debug_failure_stage', table_name='person_debug')
    op.drop_index('ix_person_debug_reid_success', table_name='person_debug')
    op.drop_index('ix_person_debug_occurred_at', table_name='person_debug')
    op.drop_index('ix_person_debug_person_identity_id', table_name='person_debug')
    op.drop_index('ix_person_debug_track_session_id', table_name='person_debug')
    op.drop_index('ix_person_debug_store_id', table_name='person_debug')
    op.drop_index('ix_person_debug_camera_id', table_name='person_debug')
    
    # Drop table
    op.drop_table('person_debug')
