#!/usr/bin/env python3
"""
Helper script to run the enum fix SQL script.
This script extracts the DATABASE_URL from your config and runs the SQL script.
"""
import os
import sys
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import config
except ImportError:
    print("Error: Could not import config. Make sure you're in the project root.")
    sys.exit(1)

def parse_database_url(url):
    """Parse PostgreSQL connection URL into components."""
    # Format: postgresql://user:password@host:port/database
    if url.startswith('postgresql://'):
        url = url.replace('postgresql://', '')
    
    # Split into parts
    if '@' in url:
        auth_part, rest = url.split('@', 1)
        if ':' in auth_part:
            user, password = auth_part.split(':', 1)
        else:
            user = auth_part
            password = ''
        
        if '/' in rest:
            host_part, database = rest.rsplit('/', 1)
            if ':' in host_part:
                host, port = host_part.split(':', 1)
            else:
                host = host_part
                port = '5432'
        else:
            host = rest
            port = '5432'
            database = ''
    else:
        # Fallback: assume it's just the database name
        user = os.getenv('PGUSER', 'postgres')
        password = os.getenv('PGPASSWORD', '')
        host = os.getenv('PGHOST', 'localhost')
        port = os.getenv('PGPORT', '5432')
        database = url
    
    return {
        'user': user,
        'password': password,
        'host': host,
        'port': port,
        'database': database
    }

def main():
    """Main function to run the enum fix."""
    database_url = config.DATABASE_URL
    print(f"Database URL: {database_url.split('@')[1] if '@' in database_url else database_url}")
    
    # Parse database URL
    db_info = parse_database_url(database_url)
    
    print("\nDatabase connection info:")
    print(f"  Host: {db_info['host']}")
    print(f"  Port: {db_info['port']}")
    print(f"  Database: {db_info['database']}")
    print(f"  User: {db_info['user']}")
    
    # Get SQL script path
    script_path = Path(__file__).parent / 'fix_enum_manual.sql'
    
    if not script_path.exists():
        print(f"\nError: SQL script not found at {script_path}")
        sys.exit(1)
    
    print(f"\nRunning SQL script: {script_path}")
    print("-" * 60)
    
    # Build psql command
    # Set password via environment variable to avoid prompt
    env = os.environ.copy()
    if db_info['password']:
        env['PGPASSWORD'] = db_info['password']
    
    # Build connection string for psql
    if db_info['password']:
        conn_string = f"postgresql://{db_info['user']}:{db_info['password']}@{db_info['host']}:{db_info['port']}/{db_info['database']}"
    else:
        conn_string = f"postgresql://{db_info['user']}@{db_info['host']}:{db_info['port']}/{db_info['database']}"
    
    # Run psql
    try:
        result = subprocess.run(
            ['psql', conn_string, '-f', str(script_path)],
            env=env,
            check=True,
            capture_output=False
        )
        print("\n" + "=" * 60)
        print("✅ SQL script executed successfully!")
        print("\nNext step: Run the migration:")
        print("  python3 -m alembic upgrade head")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running psql: {e}")
        print("\nAlternative: Run the SQL manually using one of these methods:")
        print(f"\n1. Using psql command line:")
        print(f"   psql '{conn_string}' -f {script_path}")
        print(f"\n2. Using connection parameters:")
        print(f"   psql -h {db_info['host']} -p {db_info['port']} -U {db_info['user']} -d {db_info['database']} -f {script_path}")
        print(f"\n3. Copy and paste the SQL from {script_path} into your database client")
        sys.exit(1)
    except FileNotFoundError:
        print("\n❌ Error: psql command not found.")
        print("Please install PostgreSQL client tools or run the SQL manually.")
        print(f"\nSQL script location: {script_path}")
        print("\nYou can copy the SQL and run it in your database client (pgAdmin, DBeaver, etc.)")
        sys.exit(1)

if __name__ == '__main__':
    main()
