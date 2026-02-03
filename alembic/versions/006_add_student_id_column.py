"""add student_id column (generated unique ID, separate from roll_number)

Revision ID: 006
Revises: 005
Create Date: 2026-01-30

"""
from alembic import op
import sqlalchemy as sa


revision = '006'
down_revision = '005'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'users',
        sa.Column('student_id', sa.String(length=50), nullable=True),
    )
    op.create_index('idx_user_student_id', 'users', ['student_id'], unique=True)
    # Backfill: set student_id for existing students (unique per row); sync college_student_id
    conn = op.get_bind()
    r = conn.execute(sa.text(
        "SELECT id, college_id FROM users WHERE role = 'student' AND student_id IS NULL"
    ))
    for row in r:
        uid, cid = row[0], row[1]
        sid = f"STU-{cid or 0}-{uid}"
        conn.execute(sa.text("UPDATE users SET student_id = :sid, college_student_id = :sid WHERE id = :id"), {"sid": sid, "id": uid})


def downgrade() -> None:
    op.drop_index('idx_user_student_id', table_name='users')
    op.drop_column('users', 'student_id')
