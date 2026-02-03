#!/usr/bin/env python3
"""
Script to create a master admin user.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.connection import Database
from database.models import UserRole
from services.auth_service import AuthService
import config

def create_master_admin():
    """Create a master admin user."""
    # Initialize database
    config.db = Database(
        database_url=config.DATABASE_URL,
        pool_size=config.DB_POOL_SIZE,
        max_overflow=config.DB_MAX_OVERFLOW
    )
    
    print("Creating Master Admin user...")
    print("=" * 50)
    
    # Get user input
    login_id = input("Login ID: ").strip()
    password = input("Password: ").strip()
    email = input("Email (optional): ").strip() or None
    full_name = input("Full name (optional): ").strip() or None
    
    if not login_id or not password:
        print("Error: Login ID and password are required")
        sys.exit(1)
    
    try:
        with config.db.get_session() as db:
            user = AuthService.create_user(
                db=db,
                login_id=login_id,
                password=password,
                role=UserRole.MASTER_ADMIN,
                email=email,
                full_name=full_name
            )
            print(f"\n✓ Master Admin user created successfully!")
            print(f"  Login ID: {user.login_id}")
            print(f"  Email: {user.email or 'N/A'}")
            print(f"  Role: {user.role.value}")
            print(f"\nYou can now login at: POST /auth/login/master-admin")
    except ValueError as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    create_master_admin()
