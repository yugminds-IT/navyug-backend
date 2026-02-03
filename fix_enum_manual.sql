-- Manual SQL script to fix userrole enum before running migration
-- Run this script FIRST if the migration fails with "unsafe use of new enum value"
-- 
-- This script adds lowercase enum values to the userrole enum type.
-- Each ALTER TYPE statement commits automatically, which is required by PostgreSQL.
--
-- Usage: psql -d your_database -f fix_enum_manual.sql
-- Or run each statement individually in your database client

-- Check and add lowercase enum values
-- Note: These will fail if values already exist, which is fine - just continue

DO $$
BEGIN
    -- Add master_admin if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM pg_enum 
        WHERE enumlabel = 'master_admin' 
        AND enumtypid = (SELECT oid FROM pg_type WHERE typname = 'userrole')
    ) THEN
        ALTER TYPE userrole ADD VALUE 'master_admin';
    END IF;
    
    -- Add management if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM pg_enum 
        WHERE enumlabel = 'management' 
        AND enumtypid = (SELECT oid FROM pg_type WHERE typname = 'userrole')
    ) THEN
        ALTER TYPE userrole ADD VALUE 'management';
    END IF;
    
    -- Add faculty if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM pg_enum 
        WHERE enumlabel = 'faculty' 
        AND enumtypid = (SELECT oid FROM pg_type WHERE typname = 'userrole')
    ) THEN
        ALTER TYPE userrole ADD VALUE 'faculty';
    END IF;
    
    -- Add student if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM pg_enum 
        WHERE enumlabel = 'student' 
        AND enumtypid = (SELECT oid FROM pg_type WHERE typname = 'userrole')
    ) THEN
        ALTER TYPE userrole ADD VALUE 'student';
    END IF;
END $$;

-- After running this script, you can run: python3 -m alembic upgrade head
