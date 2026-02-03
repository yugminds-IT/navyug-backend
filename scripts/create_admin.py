#!/usr/bin/env python3
"""
Script to create an admin user.
"""
import sys
from pathlib import Path

# Add parent directory to the system path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.connection import Database
from database.models import UserRole
from services.auth_service import AuthService
import config

def create_admin():
    """Create an admin user."""
    # Initialize database
    config.db = Database(
        database_url=config.DATABASE_URL,
        pool_size=config.DB_POOL_SIZE,
        max_overflow=config.DB_MAX_OVERFLOW
    )
    
    print("Creating admin user...")
    print("=" * 50)
    
    # Get user input
    username = input("Username: ").strip()
    email = input("Email: ").strip()
    password = input("Password: ").strip()
    full_name = input("Full name (optional): ").strip() or None
    
    if not username or not email or not password:
        print("Error: Username, email, and password are required")
        sys.exit(1)
    
    try:
        with config.db.get_session() as db:
            user = AuthService.create_user(
                db=db,
                username=username,
                email=email,
                password=password,
                full_name=full_name,
                role=UserRole.ADMIN
            )
            print(f"\n✓ Admin user created successfully!")
            print(f"  Username: {user.username}")
            print(f"  Email: {user.email}")
            print(f"  Role: {user.role.value}")
    except ValueError as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    create_admin()
