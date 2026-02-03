"""add_s3_link_and_student_media

Revision ID: 003_s3_link
Revises: 592c61c2dbb2
Create Date: 2026-01-29

Links S3 with DB: store s3_bucket + s3_key so we can get data from S3 using DB.
Adds student_media table for passport + face-gallery (360° photos).
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ENUM


revision = '003_s3_link'
down_revision = '592c61c2dbb2'
branch_labels = None
depends_on = None


def _column_exists(conn, table: str, column: str) -> bool:
    r = conn.execute(sa.text(
        "SELECT 1 FROM information_schema.columns WHERE table_schema = 'public' AND table_name = :t AND column_name = :c"
    ), {"t": table, "c": column})
    return r.fetchone() is not None


def _table_exists(conn, table: str) -> bool:
    r = conn.execute(sa.text("SELECT to_regclass(:name)"), {"name": f"public.{table}"})
    row = r.fetchone()
    return row is not None and row[0] is not None


def upgrade() -> None:
    conn = op.get_bind()

    # users: add S3 link columns for face/passport (idempotent)
    if not _column_exists(conn, "users", "s3_bucket"):
        op.add_column('users', sa.Column('s3_bucket', sa.String(length=255), nullable=True))
    if not _column_exists(conn, "users", "s3_face_key"):
        op.add_column('users', sa.Column('s3_face_key', sa.String(length=512), nullable=True))

    # Create studentmediatype enum (PostgreSQL raw SQL for compatibility)
    r = conn.execute(sa.text("SELECT 1 FROM pg_type WHERE typname = 'studentmediatype'"))
    if r.fetchone() is None:
        conn.execute(sa.text("CREATE TYPE studentmediatype AS ENUM ('passport', 'face_gallery')"))

    studentmediatype_enum = ENUM('passport', 'face_gallery', name='studentmediatype', create_type=False)

    # student_media table: create only if not exists (idempotent)
    if not _table_exists(conn, "student_media"):
        op.create_table(
            'student_media',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('user_id', sa.Integer(), nullable=False),
            sa.Column('media_type', studentmediatype_enum, nullable=False),
            sa.Column('s3_bucket', sa.String(length=255), nullable=True),
            sa.Column('s3_key', sa.String(length=512), nullable=True),
            sa.Column('file_url', sa.String(length=512), nullable=True),
            sa.Column('filename', sa.String(length=255), nullable=True),
            sa.Column('display_order', sa.Integer(), nullable=False, server_default='0'),
            sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
            sa.PrimaryKeyConstraint('id'),
            sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        )
        op.create_index('idx_student_media_user', 'student_media', ['user_id'])
        op.create_index('idx_student_media_type', 'student_media', ['media_type'])

    # report_media: add S3 link columns (idempotent)
    if not _column_exists(conn, "report_media", "s3_bucket"):
        op.add_column('report_media', sa.Column('s3_bucket', sa.String(length=255), nullable=True))
    if not _column_exists(conn, "report_media", "s3_key"):
        op.add_column('report_media', sa.Column('s3_key', sa.String(length=512), nullable=True))

    # detected_faces: add S3 link columns (idempotent)
    if not _column_exists(conn, "detected_faces", "s3_bucket"):
        op.add_column('detected_faces', sa.Column('s3_bucket', sa.String(length=255), nullable=True))
    if not _column_exists(conn, "detected_faces", "s3_detected_key"):
        op.add_column('detected_faces', sa.Column('s3_detected_key', sa.String(length=512), nullable=True))


def downgrade() -> None:
    op.drop_column('detected_faces', 's3_detected_key')
    op.drop_column('detected_faces', 's3_bucket')
    op.drop_column('report_media', 's3_key')
    op.drop_column('report_media', 's3_bucket')
    op.drop_index('idx_student_media_type', table_name='student_media')
    op.drop_index('idx_student_media_user', table_name='student_media')
    op.drop_table('student_media')
    op.execute("DROP TYPE IF EXISTS studentmediatype")
    op.drop_column('users', 's3_face_key')
    op.drop_column('users', 's3_bucket')
