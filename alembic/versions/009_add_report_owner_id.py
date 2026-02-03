"""add owner_id to reports for student visibility

Revision ID: 009_add_report_owner_id
Revises: 008_fix_student_media_media_type
Create Date: 2026-02-03

Adds an internal owner_id to reports so students can always see
their own submitted incidents (even when reporter_type=anonymous),
without exposing this linkage to other roles.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision = "009_add_report_owner_id"
down_revision = "008_fix_student_media_media_type"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    # Only add column if it doesn't exist
    result = conn.execute(
        text(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'reports'
              AND column_name = 'owner_id'
            """
        )
    ).fetchone()

    if not result:
        op.add_column("reports", sa.Column("owner_id", sa.Integer(), nullable=True))
        op.create_foreign_key(
            "fk_reports_owner_id_users",
            "reports",
            "users",
            ["owner_id"],
            ["id"],
            ondelete="SET NULL",
        )
        op.create_index("idx_report_owner", "reports", ["owner_id"])

        # Backfill owner_id for existing non-anonymous reports
        conn.execute(
            text(
                """
                UPDATE reports
                SET owner_id = reporter_id
                WHERE owner_id IS NULL AND reporter_id IS NOT NULL
                """
            )
        )


def downgrade() -> None:
    # Best-effort: drop index, FK, and column if present
    conn = op.get_bind()
    result = conn.execute(
        text(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'reports'
              AND column_name = 'owner_id'
            """
        )
    ).fetchone()

    if result:
        op.drop_index("idx_report_owner", table_name="reports")
        op.drop_constraint("fk_reports_owner_id_users", "reports", type_="foreignkey")
        op.drop_column("reports", "owner_id")

