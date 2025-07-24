"""
Script to dump all data from kt1_users table to a local parquet file.
This creates a backup and enables faster data access for analysis.
"""

import mysql.connector
import pandas as pd
import json
from typing import List, Dict, Any
import os
from datetime import datetime


class Config:
    """Database configuration."""
    DB_HOST = "localhost"
    DB_PORT = 3306
    DB_USER = "root"
    DB_PASSWORD = "Colipavar#92"
    DB_NAME = "ednet"


def get_database_connection():
    """Get database connection."""
    try:
        return mysql.connector.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            database=Config.DB_NAME,
            charset="utf8mb4"
        )
    except mysql.connector.Error as e:
        raise ConnectionError(f"Failed to connect to MySQL: {e}")


def dump_kt1_users_to_parquet(output_file: str = "kt1_users_dump.parquet", batch_size: int = 10000):
    """
    Dump all kt1_users data to a parquet file.
    
    Args:
        output_file: Name of the output parquet file
        batch_size: Number of records to process at once
    """
    print(f"=== Dumping kt1_users to {output_file} ===")
    
    connection = get_database_connection()
    
    try:
        cursor = connection.cursor(dictionary=True)
        
        # Get total count
        print("Getting total record count...")
        cursor.execute("SELECT COUNT(*) as total FROM kt1_users")
        total_records = cursor.fetchone()['total']
        print(f"Total records to dump: {total_records:,}")
        
        # Prepare data collection
        all_data = []
        processed = 0
        
        # Process in batches
        print(f"Processing in batches of {batch_size:,}...")
        
        offset = 0
        batch_num = 1
        
        while offset < total_records:
            print(f"Processing batch {batch_num} (records {offset + 1:,} to {min(offset + batch_size, total_records):,})")
            
            # Fetch batch
            query = """
                SELECT user_id, split_type, num_user_id, data 
                FROM kt1_users 
                ORDER BY num_user_id
                LIMIT %s OFFSET %s
            """
            
            cursor.execute(query, (batch_size, offset))
            batch_data = cursor.fetchall()
            
            if not batch_data:
                break
            
            # Process batch data
            for row in batch_data:
                try:
                    # Parse JSON data
                    user_data = json.loads(row['data'])
                    
                    # Create flattened record
                    record = {
                        'user_id': row['user_id'],
                        'split_type': row['split_type'],
                        'num_user_id': row['num_user_id'],
                        'num_interactions': len(user_data),
                        'data_json': row['data']  # Keep original JSON for complete data
                    }
                    
                    # Add some basic statistics from the interaction data
                    if user_data:
                        # Extract basic stats
                        question_ids = [interaction.get('question_id', '') for interaction in user_data]
                        responses = [interaction.get('user_answer', '') for interaction in user_data]
                        elapsed_times = []
                        timestamps = []
                        
                        for interaction in user_data:
                            try:
                                elapsed_times.append(int(interaction.get('elapsed_ms', '0')))
                                timestamps.append(int(interaction.get('ts_ms', '0')))
                            except (ValueError, TypeError):
                                elapsed_times.append(0)
                                timestamps.append(0)
                        
                        record.update({
                            'first_question': question_ids[0] if question_ids else '',
                            'last_question': question_ids[-1] if question_ids else '',
                            'unique_questions': len(set(question_ids)),
                            'avg_elapsed_time': sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0,
                            'total_time_spent': sum(elapsed_times),
                            'first_timestamp': min(timestamps) if timestamps else 0,
                            'last_timestamp': max(timestamps) if timestamps else 0,
                            'session_duration': max(timestamps) - min(timestamps) if timestamps else 0
                        })
                    else:
                        record.update({
                            'first_question': '',
                            'last_question': '',
                            'unique_questions': 0,
                            'avg_elapsed_time': 0,
                            'total_time_spent': 0,
                            'first_timestamp': 0,
                            'last_timestamp': 0,
                            'session_duration': 0
                        })
                    
                    all_data.append(record)
                    processed += 1
                    
                except Exception as e:
                    print(f"Warning: Error processing user {row['user_id']}: {e}")
                    continue
            
            print(f"Batch {batch_num} completed. Total processed: {processed:,}")
            
            offset += batch_size
            batch_num += 1
        
        # Convert to DataFrame and save as parquet
        print(f"Converting {len(all_data):,} records to DataFrame...")
        df = pd.DataFrame(all_data)
        
        # Add metadata
        df.attrs['dump_timestamp'] = datetime.now().isoformat()
        df.attrs['total_records'] = len(all_data)
        df.attrs['source_table'] = 'kt1_users'
        
        # Save to parquet
        print(f"Saving to {output_file}...")
        df.to_parquet(output_file, compression='snappy', index=False)
        
        # Print summary statistics
        print(f"\n=== Dump Summary ===")
        print(f"Output file: {output_file}")
        print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
        print(f"Total records: {len(df):,}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nSplit distribution:")
        print(df['split_type'].value_counts())
        print(f"\nInteraction statistics:")
        print(f"  Min interactions: {df['num_interactions'].min()}")
        print(f"  Max interactions: {df['num_interactions'].max()}")
        print(f"  Mean interactions: {df['num_interactions'].mean():.2f}")
        print(f"  Total interactions: {df['num_interactions'].sum():,}")
        
        print(f"\n✅ Successfully dumped kt1_users to {output_file}")
        
        return output_file
        
    finally:
        connection.close()


def load_parquet_sample(parquet_file: str, n_rows: int = 5):
    """Load and display a sample from the parquet file."""
    print(f"\n=== Sample from {parquet_file} ===")
    
    df = pd.read_parquet(parquet_file)
    print(f"Shape: {df.shape}")
    print("\nFirst {n_rows} rows:")
    print(df.head(n_rows)[['user_id', 'split_type', 'num_interactions', 'unique_questions', 'avg_elapsed_time']])
    
    return df


if __name__ == "__main__":
    # Dump the data
    output_file = dump_kt1_users_to_parquet()
    
    # Show a sample
    load_parquet_sample(output_file)
