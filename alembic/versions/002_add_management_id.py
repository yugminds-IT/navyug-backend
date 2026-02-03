"""Add management_id to users table

Revision ID: 002
Revises: 001
Create Date: 2025-01-28 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add management_id column to users table
    op.add_column('users', sa.Column('management_id', sa.Integer(), nullable=True))
    
    # Add foreign key constraint
    op.create_foreign_key(
        'fk_users_management_id',
        'users', 'users',
        ['management_id'], ['id'],
        ondelete='SET NULL'
    )
    
    # Create index for better query performance
    op.create_index('idx_user_management', 'users', ['management_id'])


def downgrade() -> None:
    # Drop index
    op.drop_index('idx_user_management', table_name='users')
    
    # Drop foreign key constraint
    op.drop_constraint('fk_users_management_id', 'users', type_='foreignkey')
    
    # Drop column
    op.drop_column('users', 'management_id')
