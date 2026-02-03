"""fix student_media.media_type to varchar

Revision ID: 008_fix_student_media_media_type
Revises: 007
Create Date: 2026-02-03

This migration changes the student_media.media_type column from the
PostgreSQL enum type `studentmediatype` to VARCHAR so it matches the
SQLAlchemy model, which uses a string-backed EnumValue.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision = "008_fix_student_media_media_type"
down_revision = "007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Convert student_media.media_type from enum to varchar."""
    conn = op.get_bind()

    # Only run if table and column exist
    result = conn.execute(
        text(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'student_media'
              AND column_name = 'media_type'
            """
        )
    ).fetchone()

    if not result:
        return

    # Change type to varchar using cast; works even if column is already varchar
    conn.execute(
        text(
            """
            ALTER TABLE student_media
            ALTER COLUMN media_type TYPE varchar
            USING media_type::text
            """
        )
    )


def downgrade() -> None:
    """Best-effort downgrade: convert media_type back to enum."""
    conn = op.get_bind()

    # Ensure enum type exists
    result = conn.execute(
        text("SELECT 1 FROM pg_type WHERE typname = 'studentmediatype'")
    ).fetchone()
    if not result:
        conn.execute(
            text(
                "CREATE TYPE studentmediatype AS ENUM ('passport', 'face_gallery')"
            )
        )

    # Only run if table/column exist
    result = conn.execute(
        text(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'student_media'
              AND column_name = 'media_type'
            """
        )
    ).fetchone()

    if not result:
        return

    # Convert varchar back to enum where values are valid
    conn.execute(
        text(
            """
            ALTER TABLE student_media
            ALTER COLUMN media_type TYPE studentmediatype
            USING media_type::studentmediatype
            """
        )
    )

