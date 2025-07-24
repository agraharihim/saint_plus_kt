#!/usr/bin/env python3
"""
Utility script to create the necessary database tables for EdNet KT data storage.
"""

import argparse
import mysql.connector
import os

def main():
    parser = argparse.ArgumentParser(description="Create EdNet database tables")
    parser.add_argument("--user", required=True, help="MySQL username")
    parser.add_argument("--password", required=True, help="MySQL password")
    parser.add_argument("--host", default="localhost", help="MySQL host")
    parser.add_argument("--port", type=int, default=3306, help="MySQL port")
    
    args = parser.parse_args()
    
    # Read SQL file
    sql_file = os.path.join(os.path.dirname(__file__), "create_tables.sql")
    with open(sql_file, 'r') as f:
        sql_content = f.read()
    
    # Split SQL statements
    statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
    
    # Connect to MySQL
    conn = mysql.connector.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        charset="utf8mb4"
    )
    
    cur = conn.cursor()
    
    try:
        for statement in statements:
            if statement:
                cur.execute(statement)
        conn.commit()
        print("✅ Database tables created successfully!")
        print("Tables created: kt1_users, kt2_users, kt1_load_history, kt2_load_history")
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    main()
