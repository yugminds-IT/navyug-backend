"""add user department_id

Revision ID: 005
Revises: 003_s3_link
Create Date: 2026-01-30

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '005'
down_revision = '003_s3_link'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'users',
        sa.Column('department_id', sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        'fk_users_department_id_college_departments',
        'users', 'college_departments',
        ['department_id'], ['id'],
        ondelete='SET NULL',
    )
    op.create_index('idx_user_department', 'users', ['department_id'], unique=False)


def downgrade() -> None:
    op.drop_index('idx_user_department', table_name='users')
    op.drop_constraint('fk_users_department_id_college_departments', 'users', type_='foreignkey')
    op.drop_column('users', 'department_id')
