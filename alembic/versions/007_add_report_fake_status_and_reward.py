"""add report fake status and reporter reward points

Revision ID: 007
Revises: 006
Create Date: 2026-01-30

- Add 'fake' to ReportStatus: management can mark report as fake when student uploads fake video.
- Add reporter_reward_points on reports: when status=resolved, management can set stars/points; visible to management and reporter.
"""
from alembic import op
import sqlalchemy as sa


revision = '007'
down_revision = '006'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add 'fake' to reportstatus enum (PostgreSQL)
    conn = op.get_bind()
    result = conn.execute(sa.text(
        "SELECT EXISTS (SELECT 1 FROM pg_enum e JOIN pg_type t ON e.enumtypid = t.oid WHERE t.typname = 'reportstatus' AND e.enumlabel = 'fake')"
    ))
    if not result.scalar():
        op.execute(sa.text("ALTER TYPE reportstatus ADD VALUE 'fake'"))
    # Add reporter_reward_points column to reports
    op.add_column(
        'reports',
        sa.Column('reporter_reward_points', sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('reports', 'reporter_reward_points')
    # Note: PostgreSQL does not support removing an enum value easily; 'fake' remains in reportstatus.
