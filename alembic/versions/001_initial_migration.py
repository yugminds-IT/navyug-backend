"""Initial migration - Complete schema with colleges, 4 user roles, refresh tokens, sessions

Revision ID: 001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create colleges table first (without foreign key to users - will add later)
    # Note: created_by FK will be added after users table is created
    op.create_table(
        'colleges',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('college_code', sa.String(length=50), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('address', sa.Text(), nullable=True),
        sa.Column('contact_email', sa.String(length=255), nullable=True),
        sa.Column('contact_phone', sa.String(length=50), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('created_by', sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('college_code')
    )
    op.create_index('idx_college_code', 'colleges', ['college_code'])
    op.create_index('idx_college_active', 'colleges', ['is_active'])
    
    # Create users table with all new fields
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('login_id', sa.String(length=100), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=True),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=255), nullable=True),
        sa.Column('role', sa.Enum('master_admin', 'management', 'faculty', 'student', name='userrole'), nullable=False),
        sa.Column('college_id', sa.Integer(), nullable=True),
        sa.Column('managed_college_id', sa.Integer(), nullable=True),
        sa.Column('college_student_id', sa.String(length=100), nullable=True),
        sa.Column('college_name', sa.String(length=255), nullable=True),
        sa.Column('phone', sa.String(length=50), nullable=True),
        sa.Column('address', sa.Text(), nullable=True),
        sa.Column('username', sa.String(length=100), nullable=True),
        sa.Column('faculty_id', sa.String(length=100), nullable=True),
        sa.Column('department_name', sa.String(length=255), nullable=True),
        sa.Column('roll_number', sa.String(length=100), nullable=True),
        sa.Column('branch', sa.String(length=100), nullable=True),
        sa.Column('year', sa.String(length=50), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('is_verified', sa.Boolean(), nullable=False),
        sa.Column('is_locked', sa.Boolean(), nullable=False),
        sa.Column('failed_login_attempts', sa.Integer(), nullable=False),
        sa.Column('locked_until', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('last_login', sa.DateTime(), nullable=True),
        sa.Column('password_changed_at', sa.DateTime(), nullable=True),
        sa.Column('created_by', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['college_id'], ['colleges.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['managed_college_id'], ['colleges.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['created_by'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('login_id')
    )
    op.create_index('idx_user_email', 'users', ['email'])
    op.create_index('idx_user_login_id', 'users', ['login_id'])
    op.create_index('idx_user_college', 'users', ['college_id'])
    op.create_index('idx_user_managed_college', 'users', ['managed_college_id'])
    op.create_index('idx_user_role', 'users', ['role'])
    
    # Create api_keys table with access_key field
    op.create_table(
        'api_keys',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('access_key', sa.String(length=255), nullable=False),
        sa.Column('key_hash', sa.String(length=255), nullable=False),
        sa.Column('key_prefix', sa.String(length=20), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('last_used_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('key_hash'),
        sa.UniqueConstraint('access_key')
    )
    op.create_index('idx_api_key_hash', 'api_keys', ['key_hash'])
    op.create_index('idx_api_key_access', 'api_keys', ['access_key'])
    
    # Create refresh_tokens table
    op.create_table(
        'refresh_tokens',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('token_hash', sa.String(length=255), nullable=False),
        sa.Column('device_info', sa.String(length=255), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('is_revoked', sa.Boolean(), nullable=False),
        sa.Column('revoked_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('last_used_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('token_hash')
    )
    op.create_index('idx_refresh_token_hash', 'refresh_tokens', ['token_hash'])
    op.create_index('idx_refresh_expires', 'refresh_tokens', ['expires_at'])
    op.create_index('idx_refresh_user', 'refresh_tokens', ['user_id'])
    
    # Create sessions table
    op.create_table(
        'sessions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('session_key', sa.String(length=255), nullable=False),
        sa.Column('session_hash', sa.String(length=255), nullable=False),
        sa.Column('device_info', sa.String(length=255), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.String(length=500), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('last_activity', sa.DateTime(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('session_key'),
        sa.UniqueConstraint('session_hash')
    )
    op.create_index('idx_session_key', 'sessions', ['session_key'])
    op.create_index('idx_session_hash', 'sessions', ['session_hash'])
    op.create_index('idx_session_expires', 'sessions', ['expires_at'])
    op.create_index('idx_session_user', 'sessions', ['user_id'])
    
    # Add foreign key constraint for colleges.created_by (after users table exists)
    op.create_foreign_key(
        'fk_colleges_created_by',
        'colleges', 'users',
        ['created_by'], ['id'],
        ondelete='SET NULL'
    )
    
    # Create persons table
    op.create_table(
        'persons',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('person_id', sa.String(length=100), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('s3_image_path', sa.String(length=500), nullable=True),
        sa.Column('local_image_path', sa.String(length=500), nullable=True),
        sa.Column('embedding', sa.Text(), nullable=True),
        sa.Column('num_images', sa.Integer(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('extra_metadata', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('person_id')
    )
    op.create_index('idx_person_id', 'persons', ['person_id'])
    op.create_index('idx_person_active', 'persons', ['is_active'])
    
    # Create video_jobs table
    op.create_table(
        'video_jobs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('video_id', sa.String(length=36), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('s3_video_path', sa.String(length=500), nullable=True),
        sa.Column('local_video_path', sa.String(length=500), nullable=True),
        sa.Column('file_size_mb', sa.Float(), nullable=True),
        sa.Column('status', sa.Enum('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED', 'CANCELLED', name='jobstatus'), nullable=False),
        sa.Column('progress_percentage', sa.Float(), nullable=False),
        sa.Column('progress_data', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('result', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('uploaded_at', sa.DateTime(), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('failed_at', sa.DateTime(), nullable=True),
        sa.Column('processing_time_seconds', sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('video_id')
    )
    op.create_index('idx_video_status', 'video_jobs', ['status'])
    op.create_index('idx_video_user', 'video_jobs', ['user_id'])
    op.create_index('idx_video_uploaded', 'video_jobs', ['uploaded_at'])
    
    # Create matched_faces table
    op.create_table(
        'matched_faces',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('video_job_id', sa.Integer(), nullable=False),
        sa.Column('person_id', sa.String(length=100), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('total_appearances', sa.Integer(), nullable=False),
        sa.Column('frames_seen', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('best_face_quality', sa.Float(), nullable=True),
        sa.Column('extra_metadata', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['person_id'], ['persons.person_id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['video_job_id'], ['video_jobs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_matched_video', 'matched_faces', ['video_job_id'])
    op.create_index('idx_matched_person', 'matched_faces', ['person_id'])
    op.create_index('idx_matched_confidence', 'matched_faces', ['confidence'])
    
    # Create audit_logs table
    op.create_table(
        'audit_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('action', sa.String(length=100), nullable=False),
        sa.Column('resource_type', sa.String(length=50), nullable=True),
        sa.Column('resource_id', sa.String(length=100), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.String(length=500), nullable=True),
        sa.Column('details', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_audit_user', 'audit_logs', ['user_id'])
    op.create_index('idx_audit_action', 'audit_logs', ['action'])
    op.create_index('idx_audit_created', 'audit_logs', ['created_at'])


def downgrade() -> None:
    op.drop_index('idx_audit_created', table_name='audit_logs')
    op.drop_index('idx_audit_action', table_name='audit_logs')
    op.drop_index('idx_audit_user', table_name='audit_logs')
    op.drop_table('audit_logs')
    op.drop_index('idx_matched_confidence', table_name='matched_faces')
    op.drop_index('idx_matched_person', table_name='matched_faces')
    op.drop_index('idx_matched_video', table_name='matched_faces')
    op.drop_table('matched_faces')
    op.drop_index('idx_video_uploaded', table_name='video_jobs')
    op.drop_index('idx_video_user', table_name='video_jobs')
    op.drop_index('idx_video_status', table_name='video_jobs')
    op.drop_table('video_jobs')
    op.drop_index('idx_person_active', table_name='persons')
    op.drop_index('idx_person_id', table_name='persons')
    op.drop_table('persons')
    # Drop foreign key constraint before dropping users table
    op.drop_constraint('fk_colleges_created_by', 'colleges', type_='foreignkey')
    op.drop_index('idx_session_user', table_name='sessions')
    op.drop_index('idx_session_expires', table_name='sessions')
    op.drop_index('idx_session_hash', table_name='sessions')
    op.drop_index('idx_session_key', table_name='sessions')
    op.drop_table('sessions')
    op.drop_index('idx_refresh_user', table_name='refresh_tokens')
    op.drop_index('idx_refresh_expires', table_name='refresh_tokens')
    op.drop_index('idx_refresh_token_hash', table_name='refresh_tokens')
    op.drop_table('refresh_tokens')
    op.drop_index('idx_api_key_access', table_name='api_keys')
    op.drop_index('idx_api_key_hash', table_name='api_keys')
    op.drop_table('api_keys')
    op.drop_index('idx_user_role', table_name='users')
    op.drop_index('idx_user_managed_college', table_name='users')
    op.drop_index('idx_user_college', table_name='users')
    op.drop_index('idx_user_login_id', table_name='users')
    op.drop_index('idx_user_email', table_name='users')
    op.drop_table('users')
    op.drop_index('idx_college_active', table_name='colleges')
    op.drop_index('idx_college_code', table_name='colleges')
    op.drop_table('colleges')
    sa.Enum(name='jobstatus').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='userrole').drop(op.get_bind(), checkfirst=True)
