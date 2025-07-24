"""
Check database schema and available columns.
"""

import mysql.connector
from database_utils import DatabaseManager


def check_database_schema():
    """Check what tables and columns are available."""
    db_manager = DatabaseManager()
    connection = db_manager.get_connection()
    
    try:
        cursor = connection.cursor()
        
        # Show all tables
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        print("=== Available Tables ===")
        for table in tables:
            print(f"- {table[0]}")
        
        # Check if kt1_users table exists and its structure
        cursor.execute("SHOW TABLES LIKE 'kt1_users'")
        kt1_table = cursor.fetchone()
        
        if kt1_table:
            print(f"\n=== kt1_users Table Structure ===")
            cursor.execute("DESCRIBE kt1_users")
            columns = cursor.fetchall()
            for col in columns:
                print(f"- {col[0]} ({col[1]})")
            
            # Check sample data
            print(f"\n=== Sample Data from kt1_users ===")
            cursor.execute("SELECT * FROM kt1_users LIMIT 3")
            samples = cursor.fetchall()
            for i, sample in enumerate(samples):
                print(f"Row {i+1}: {sample}")
        else:
            print("\n❌ kt1_users table not found!")
            
            # Look for other user-related tables
            print("\n=== Looking for user-related tables ===")
            cursor.execute("SHOW TABLES LIKE '%user%'")
            user_tables = cursor.fetchall()
            for table in user_tables:
                print(f"Found: {table[0]}")
                cursor.execute(f"DESCRIBE {table[0]}")
                columns = cursor.fetchall()
                for col in columns:
                    print(f"  - {col[0]} ({col[1]})")
    
    finally:
        connection.close()


if __name__ == "__main__":
    check_database_schema()
