"""fix_userrole_enum_lowercase

Revision ID: 592c61c2dbb2
Revises: 4afc191e7414
Create Date: 2026-01-28 19:58:58.717279

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision = '592c61c2dbb2'
down_revision = '4afc191e7414'
branch_labels = None
depends_on = None


def _ensure_lowercase_enum_values(engine, missing: list) -> None:
    """
    Add lowercase userrole enum values using a separate autocommit connection
    from the same engine (same credentials). PostgreSQL requires new enum values
    to be committed before they can be used in the same transaction.
    """
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as ac_conn:
        for value in missing:
            ac_conn.execute(text(f"ALTER TYPE userrole ADD VALUE '{value}'"))


def upgrade() -> None:
    """
    Fix userrole enum to use lowercase values.
    The initial migration created the enum with uppercase values,
    but the Python code uses lowercase enum values.
    """
    conn = op.get_bind()
    engine = conn.engine

    # Step 0: If lowercase enum values are missing (e.g. fresh DB), add them via autocommit
    enum_values = ['master_admin', 'management', 'faculty', 'student']
    missing_values = []
    for value in enum_values:
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT 1 FROM pg_enum
                WHERE enumlabel = :v
                AND enumtypid = (SELECT oid FROM pg_type WHERE typname = 'userrole')
            )
        """), {"v": value})
        if not result.scalar():
            missing_values.append(value)
    if missing_values:
        _ensure_lowercase_enum_values(engine, missing_values)

    # Step 1: Verify that lowercase enum values exist now
    missing_values = []
    for value in enum_values:
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT 1 FROM pg_enum
                WHERE enumlabel = :v
                AND enumtypid = (SELECT oid FROM pg_type WHERE typname = 'userrole')
            )
        """), {"v": value})
        if not result.scalar():
            missing_values.append(value)
    if missing_values:
        raise Exception(
            f"Missing enum values: {', '.join(missing_values)}. "
            "Could not add them automatically. Try running fix_enum_manual.sql, then run this migration again."
        )
    
    # Step 2: Check if migration already completed
    # Query information_schema to check column existence
    result = conn.execute(text("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'users' AND column_name IN ('role', 'role_temp')
    """))
    existing_columns = {row[0] for row in result.fetchall()}
    
    if 'role_temp' in existing_columns and 'role' in existing_columns:
        # Check if role is already using lowercase values
        result = conn.execute(text("""
            SELECT COUNT(*) FROM users 
            WHERE role::text IN ('master_admin', 'management', 'faculty', 'student')
        """))
        lowercase_count = result.scalar()
        result = conn.execute(text("SELECT COUNT(*) FROM users"))
        total_count = result.scalar()
        
        if lowercase_count == total_count and total_count > 0:
            # Already migrated, just clean up
            op.drop_column('users', 'role_temp')
            return
    
    # Step 3: Use a temporary text column to migrate data safely
    if 'role_temp' not in existing_columns:
        op.add_column('users', sa.Column('role_temp', sa.String(50), nullable=True))
    
    # Copy converted values to temp column (as text strings)
    conn.execute(text("""
        UPDATE users 
        SET role_temp = CASE 
            WHEN role::text = 'MASTER_ADMIN' THEN 'master_admin'
            WHEN role::text = 'MANAGEMENT' THEN 'management'
            WHEN role::text = 'FACULTY' THEN 'faculty'
            WHEN role::text = 'STUDENT' THEN 'student'
            WHEN role::text IN ('master_admin', 'management', 'faculty', 'student') THEN role::text
            ELSE role::text
        END
    """))
    
    # Step 4: Drop old enum column if it exists
    if 'role' in existing_columns:
        op.drop_column('users', 'role')
    
    # Step 5: Add role column back using the existing enum type (nullable first)
    op.execute(text("""
        ALTER TABLE users 
        ADD COLUMN role userrole
    """))
    
    # Step 6: Copy data from temp column to new enum column
    # The enum values are already committed (from manual script), so this should work
    conn.execute(text("""
        UPDATE users 
        SET role = role_temp::userrole
    """))
    
    # Step 7: Make role NOT NULL now that all rows have values
    op.alter_column('users', 'role', nullable=False)
    
    # Step 8: Drop temp column
    op.drop_column('users', 'role_temp')


def downgrade() -> None:
    """
    Revert to uppercase enum values.
    Note: PostgreSQL doesn't support removing enum values easily,
    so we'll just update the data back to uppercase.
    """
    conn = op.get_bind()
    
    # Update existing rows back to uppercase values using ALTER TABLE
    conn.execute(text("""
        ALTER TABLE users 
        ALTER COLUMN role TYPE userrole 
        USING CASE 
            WHEN role::text = 'master_admin' THEN 'MASTER_ADMIN'::userrole
            WHEN role::text = 'management' THEN 'MANAGEMENT'::userrole
            WHEN role::text = 'faculty' THEN 'FACULTY'::userrole
            WHEN role::text = 'student' THEN 'STUDENT'::userrole
            ELSE role::userrole
        END
    """))
