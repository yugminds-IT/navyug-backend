"""add_campus_safety_tables

Revision ID: 4afc191e7414
Revises: 002
Create Date: 2026-01-28 18:12:58.195541

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import ENUM
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision = '4afc191e7414'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Helper function to create enum if it doesn't exist
    def create_enum_if_not_exists(enum_name, values):
        conn = op.get_bind()
        # Check if enum exists
        result = conn.execute(sa.text(
            "SELECT EXISTS (SELECT 1 FROM pg_type WHERE typname = :name)"
        ), {"name": enum_name})
        exists = result.scalar()
        if not exists:
            # Create enum using raw SQL
            values_str = ', '.join([f"'{v}'" for v in values])
            conn.execute(sa.text(f"CREATE TYPE {enum_name} AS ENUM ({values_str})"))
    
    # Create enums if they don't exist
    create_enum_if_not_exists('userstatus', ['active', 'inactive', 'suspended'])
    create_enum_if_not_exists('incidenttype', ['ragging', 'fighting', 'misbehaviour', 'money', 'faculty', 'others'])
    create_enum_if_not_exists('reportstatus', ['pending', 'investigating', 'resolved', 'rejected'])
    create_enum_if_not_exists('reportreportertype', ['anonymous', 'identified'])
    create_enum_if_not_exists('actiontype', ['warning', 'suspension_1d', 'suspension_3d', 'suspension_7d', 'counseling', 'expulsion'])
    create_enum_if_not_exists('studentstatus', ['active', 'warning', 'suspended'])
    create_enum_if_not_exists('facultystatus', ['active', 'inactive'])
    create_enum_if_not_exists('collegestatus', ['active', 'inactive'])
    create_enum_if_not_exists('loglevel', ['info', 'warning', 'error', 'success'])
    create_enum_if_not_exists('logcategory', ['authentication', 'report', 'user_management', 'system', 'database', 'ai_detection'])
    create_enum_if_not_exists('mediatype', ['image', 'video'])
    
    # Create enum objects for use in table definitions (using PostgreSQL-specific ENUM with create_type=False)
    userstatus_enum = ENUM('active', 'inactive', 'suspended', name='userstatus', create_type=False)
    incidenttype_enum = ENUM('ragging', 'fighting', 'misbehaviour', 'money', 'faculty', 'others', name='incidenttype', create_type=False)
    reportstatus_enum = ENUM('pending', 'investigating', 'resolved', 'rejected', name='reportstatus', create_type=False)
    reportreportertype_enum = ENUM('anonymous', 'identified', name='reportreportertype', create_type=False)
    actiontype_enum = ENUM('warning', 'suspension_1d', 'suspension_3d', 'suspension_7d', 'counseling', 'expulsion', name='actiontype', create_type=False)
    studentstatus_enum = ENUM('active', 'warning', 'suspended', name='studentstatus', create_type=False)
    facultystatus_enum = ENUM('active', 'inactive', name='facultystatus', create_type=False)
    collegestatus_enum = ENUM('active', 'inactive', name='collegestatus', create_type=False)
    loglevel_enum = ENUM('info', 'warning', 'error', 'success', name='loglevel', create_type=False)
    logcategory_enum = ENUM('authentication', 'report', 'user_management', 'system', 'database', 'ai_detection', name='logcategory', create_type=False)
    mediatype_enum = ENUM('image', 'video', name='mediatype', create_type=False)
    
    # Update colleges table
    op.add_column('colleges', sa.Column('website', sa.String(length=255), nullable=True))
    op.add_column('colleges', sa.Column('status', collegestatus_enum, nullable=True))
    op.add_column('colleges', sa.Column('total_students', sa.Integer(), nullable=True, server_default='0'))
    op.add_column('colleges', sa.Column('total_faculty', sa.Integer(), nullable=True, server_default='0'))
    op.add_column('colleges', sa.Column('total_reports', sa.Integer(), nullable=True, server_default='0'))
    op.create_index('idx_college_status', 'colleges', ['status'])
    
    # Set default status for existing colleges (cast to enum type)
    op.execute("UPDATE colleges SET status = 'active'::collegestatus WHERE status IS NULL")
    op.alter_column('colleges', 'status', nullable=False)
    # Set default using ALTER TABLE directly
    op.execute("ALTER TABLE colleges ALTER COLUMN status SET DEFAULT 'active'::collegestatus")
    
    # Update users table
    op.add_column('users', sa.Column('status', userstatus_enum, nullable=True))
    op.add_column('users', sa.Column('avatar_url', sa.String(length=512), nullable=True))
    op.add_column('users', sa.Column('incidents', sa.Integer(), nullable=True, server_default='0'))
    op.add_column('users', sa.Column('last_incident_at', sa.DateTime(), nullable=True))
    op.add_column('users', sa.Column('face_registered', sa.Boolean(), nullable=True, server_default='false'))
    op.add_column('users', sa.Column('face_image_url', sa.String(length=512), nullable=True))
    op.add_column('users', sa.Column('reports_submitted', sa.Integer(), nullable=True, server_default='0'))
    op.add_column('users', sa.Column('reports_resolved', sa.Integer(), nullable=True, server_default='0'))
    op.add_column('users', sa.Column('points', sa.Integer(), nullable=True, server_default='0'))
    op.add_column('users', sa.Column('last_login_at', sa.DateTime(), nullable=True))
    
    # Set default status for existing users (cast to enum type)
    op.execute("UPDATE users SET status = CASE WHEN is_active THEN 'active'::userstatus ELSE 'inactive'::userstatus END WHERE status IS NULL")
    op.alter_column('users', 'status', nullable=False)
    # Set default using ALTER TABLE directly
    op.execute("ALTER TABLE users ALTER COLUMN status SET DEFAULT 'active'::userstatus")
    
    # Create college_departments table
    op.create_table(
        'college_departments',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('college_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['college_id'], ['colleges.id'], ondelete='CASCADE')
    )
    op.create_index('idx_dept_college', 'college_departments', ['college_id'])
    op.create_index('idx_dept_college_name', 'college_departments', ['college_id', 'name'], unique=True)
    
    # Create reports table
    op.create_table(
        'reports',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('report_id', sa.String(length=50), nullable=False),
        sa.Column('incident_type', incidenttype_enum, nullable=False),
        sa.Column('location', sa.String(length=255), nullable=False),
        sa.Column('occurred_at', sa.DateTime(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('witnesses', sa.String(length=500), nullable=True),
        sa.Column('reporter_type', reportreportertype_enum, nullable=False),
        sa.Column('reporter_id', sa.Integer(), nullable=True),
        sa.Column('college_id', sa.Integer(), nullable=False),
        sa.Column('status', reportstatus_enum, nullable=False, server_default='pending'),
        sa.Column('has_video', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('has_photo', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('ai_processed', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['reporter_id'], ['users.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['college_id'], ['colleges.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('report_id')
    )
    try:
        op.create_index('idx_report_id', 'reports', ['report_id'])
        op.create_index('idx_report_college', 'reports', ['college_id'])
        op.create_index('idx_report_reporter', 'reports', ['reporter_id'])
        op.create_index('idx_report_status', 'reports', ['status'])
        op.create_index('idx_report_incident_type', 'reports', ['incident_type'])
        op.create_index('idx_report_college_status', 'reports', ['college_id', 'status'])
        op.create_index('idx_report_created', 'reports', ['created_at'])
    except Exception:
        pass  # Indexes may already exist
    
    # Create report_media table
    op.create_table(
        'report_media',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('report_id', sa.Integer(), nullable=False),
        sa.Column('media_type', mediatype_enum, nullable=False),
        sa.Column('file_url', sa.String(length=512), nullable=False),
        sa.Column('file_size_bytes', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['report_id'], ['reports.id'], ondelete='CASCADE')
    )
    op.create_index('idx_media_report', 'report_media', ['report_id'])
    
    # Create detected_faces table
    op.create_table(
        'detected_faces',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('report_id', sa.Integer(), nullable=False),
        sa.Column('student_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('department', sa.String(length=255), nullable=True),
        sa.Column('year', sa.String(length=50), nullable=True),
        sa.Column('confidence', sa.Integer(), nullable=False),
        sa.Column('previous_incidents', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('bounding_box', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('reference_image_url', sa.String(length=512), nullable=True),
        sa.Column('detected_image_url', sa.String(length=512), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['report_id'], ['reports.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['student_id'], ['users.id'], ondelete='CASCADE')
    )
    op.create_index('idx_detected_report', 'detected_faces', ['report_id'])
    op.create_index('idx_detected_student', 'detected_faces', ['student_id'])
    
    # Create disciplinary_actions table
    op.create_table(
        'disciplinary_actions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('report_id', sa.Integer(), nullable=True),
        sa.Column('student_id', sa.Integer(), nullable=False),
        sa.Column('action_type', actiontype_enum, nullable=False),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_by', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['report_id'], ['reports.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['student_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['created_by'], ['users.id'], ondelete='CASCADE')
    )
    op.create_index('idx_action_student', 'disciplinary_actions', ['student_id'])
    op.create_index('idx_action_report', 'disciplinary_actions', ['report_id'])
    op.create_index('idx_action_created', 'disciplinary_actions', ['created_at'])
    
    # Create otp_store table
    op.create_table(
        'otp_store',
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('otp', sa.String(length=10), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('email')
    )
    op.create_index('idx_otp_expires', 'otp_store', ['expires_at'])
    
    # Create system_logs table
    op.create_table(
        'system_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('level', loglevel_enum, nullable=False),
        sa.Column('category', logcategory_enum, nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('ip', sa.String(length=45), nullable=True),
        sa.Column('extra_metadata', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL')
    )
    op.create_index('idx_log_level', 'system_logs', ['level'])
    op.create_index('idx_log_category', 'system_logs', ['category'])
    op.create_index('idx_log_user', 'system_logs', ['user_id'])
    op.create_index('idx_log_created', 'system_logs', ['created_at'])


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table('system_logs')
    op.drop_table('otp_store')
    op.drop_table('disciplinary_actions')
    op.drop_table('detected_faces')
    op.drop_table('report_media')
    op.drop_table('reports')
    op.drop_table('college_departments')
    
    # Remove columns from users table
    op.drop_column('users', 'last_login_at')
    op.drop_column('users', 'points')
    op.drop_column('users', 'reports_resolved')
    op.drop_column('users', 'reports_submitted')
    op.drop_column('users', 'face_image_url')
    op.drop_column('users', 'face_registered')
    op.drop_column('users', 'last_incident_at')
    op.drop_column('users', 'incidents')
    op.drop_column('users', 'avatar_url')
    op.drop_column('users', 'status')
    
    # Remove columns from colleges table
    op.drop_index('idx_college_status', table_name='colleges')
    op.drop_column('colleges', 'total_reports')
    op.drop_column('colleges', 'total_faculty')
    op.drop_column('colleges', 'total_students')
    op.drop_column('colleges', 'status')
    op.drop_column('colleges', 'website')
    
    # Drop enums
    sa.Enum(name='mediatype').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='logcategory').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='loglevel').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='collegestatus').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='facultystatus').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='studentstatus').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='actiontype').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='reportreportertype').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='reportstatus').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='incidenttype').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='userstatus').drop(op.get_bind(), checkfirst=True)
