"""
Minimal database utilities for SAINT+ Knowledge Tracing.
"""

import json
import mysql.connector
from typing import List, Dict, Optional, Any
from question_metadata import QuestionMetadata


class Config:
    """Database configuration."""
    DB_HOST = "localhost"
    DB_PORT = 3306
    DB_USER = "root"
    DB_PASSWORD = "Colipavar#92"
    DB_NAME = "ednet"
    USERS_PER_BATCH = 1000


class DatabaseManager:
    """Basic database manager for training."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.question_metadata = QuestionMetadata()
    
    def get_connection(self) -> mysql.connector.MySQLConnection:
        """Get database connection."""
        try:
            return mysql.connector.connect(
                host=self.config.DB_HOST,
                port=self.config.DB_PORT,
                user=self.config.DB_USER,
                password=self.config.DB_PASSWORD,
                database=self.config.DB_NAME,
                charset="utf8mb4"
            )
        except mysql.connector.Error as e:
            raise ConnectionError(f"Failed to connect to MySQL: {e}")
    
    def get_user_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Get random batch of users for training."""
        import random
        
        connection = self.get_connection()
        
        try:
            cursor = connection.cursor(dictionary=True)
            
            # Generate random user IDs in the format u1, u2, ..., u840473
            random_user_ids = []
            for _ in range(batch_size * 2):  # Generate more than needed in case some don't exist
                user_num = random.randint(1, 840473)
                random_user_ids.append(f'u{user_num}')
            
            # Remove duplicates while preserving order
            seen = set()
            unique_user_ids = []
            for user_id in random_user_ids:
                if user_id not in seen:
                    seen.add(user_id)
                    unique_user_ids.append(user_id)
            
            # Query for users that actually exist in the database
            if not unique_user_ids:
                return []
            
            # Create placeholders for the IN clause
            placeholders = ','.join(['%s'] * len(unique_user_ids))
            query = f"""
                SELECT user_id, data 
                FROM kt1_users 
                WHERE user_id IN ({placeholders})
                LIMIT %s
            """
            
            cursor.execute(query, unique_user_ids + [batch_size])
            users = cursor.fetchall()
            
            # Convert to training sequences
            sequences = []
            for user in users:
                sequence = self._convert_user_data_to_sequence(
                    json.loads(user['data']), 
                    user['user_id']
                )
                if sequence:
                    sequences.append(sequence)
            
            return sequences
            
        finally:
            connection.close()
    
    def get_all_users(self, batch_size: int = 2000) -> List[Dict[str, Any]]:
        """Get all available users for comprehensive training."""
        connection = self.get_connection()
        
        try:
            cursor = connection.cursor(dictionary=True)
            
            # First, get total count of users
            cursor.execute("SELECT COUNT(*) as total FROM kt1_users")
            total_users = cursor.fetchone()['total']
            print(f"Total users in database: {total_users:,}")
            
            # Fetch all users in batches to avoid memory issues
            all_sequences = []
            offset = 0
            batch_num = 1
            
            while offset < total_users:
                print(f"Processing batch {batch_num} (users {offset + 1:,} to {min(offset + batch_size, total_users):,})")
                
                query = """
                    SELECT user_id, data 
                    FROM kt1_users 
                    ORDER BY num_user_id
                    LIMIT %s OFFSET %s
                """
                
                cursor.execute(query, (batch_size, offset))
                batch_users = cursor.fetchall()
                
                if not batch_users:
                    break
                
                # Convert batch to training sequences
                batch_sequences = []
                for user in batch_users:
                    sequence = self._convert_user_data_to_sequence(
                        json.loads(user['data']), 
                        user['user_id']
                    )
                    if sequence:
                        batch_sequences.append(sequence)
                
                all_sequences.extend(batch_sequences)
                print(f"Batch {batch_num}: {len(batch_sequences):,} valid sequences")
                
                offset += batch_size
                batch_num += 1
            
            print(f"Total valid training sequences: {len(all_sequences):,}")
            return all_sequences
            
        finally:
            connection.close()
    
    def get_users_by_split(self, split_type: str) -> List[Dict[str, Any]]:
        """Get all users for a specific split (train/validation/test)."""
        connection = self.get_connection()
        
        try:
            cursor = connection.cursor(dictionary=True)
            
            # Query users by split type
            query = """
                SELECT user_id, data 
                FROM kt1_users 
                WHERE split_type = %s
            """
            
            cursor.execute(query, (split_type,))
            users = cursor.fetchall()
            
            print(f"Found {len(users)} users for split '{split_type}'")
            
            # Convert to training sequences
            sequences = []
            for user in users:
                sequence = self._convert_user_data_to_sequence(
                    json.loads(user['data']), 
                    user['user_id']
                )
                if sequence:
                    sequences.append(sequence)
            
            print(f"Successfully converted {len(sequences)} user sequences for split '{split_type}'")
            return sequences
            
        finally:
            connection.close()
    
    # def get_random_users_for_analysis(self, num_users: int) -> List[Dict[str, Any]]:
    #     """Get random users for analysis purposes (ignores split_type)."""
    #     import random
        
    #     connection = self.get_connection()
    #     sequences = []
        
    #     try:
    #         cursor = connection.cursor(dictionary=True)
            
    #         # Process in smaller batches to avoid MySQL memory issues
    #         batch_size = 500  # Smaller batches
    #         users_collected = 0
    #         attempts = 0
    #         max_attempts = num_users * 3  # Limit total attempts
            
    #         print(f"Collecting {num_users} users in batches of {batch_size}...")
            
    #         while users_collected < num_users and attempts < max_attempts:
    #             # Generate random user IDs for this batch
    #             random_user_ids = []
    #             for _ in range(batch_size):
    #                 user_num = random.randint(1, 840473)
    #                 user_id = f'u{user_num}'
    #                 if user_id not in random_user_ids:
    #                     random_user_ids.append(user_id)
                
    #             if not random_user_ids:
    #                 continue
                
    #             # Query this batch without ORDER BY RAND() to avoid sort memory issues
    #             placeholders = ','.join(['%s'] * len(random_user_ids))
    #             query = f"""
    #                 SELECT user_id, data 
    #                 FROM kt1_users 
    #                 WHERE user_id IN ({placeholders})
    #                 LIMIT %s
    #             """
                
    #             cursor.execute(query, random_user_ids + [min(batch_size, num_users - users_collected)])
    #             batch_users = cursor.fetchall()
                
    #             print(f"Batch {attempts + 1}: Found {len(batch_users)} users")
                
    #             # Convert to sequences
    #             for user in batch_users:
    #                 if users_collected >= num_users:
    #                     break
                    
    #                 try:
    #                     sequence = self._convert_user_data_to_sequence(
    #                         json.loads(user['data']), 
    #                         user['user_id']
    #                     )
    #                     if sequence:
    #                         sequences.append(sequence)
    #                         users_collected += 1
    #                 except Exception as e:
    #                     print(f"Error processing user {user['user_id']}: {e}")
    #                     continue
                
    #             attempts += 1
                
    #             if len(batch_users) == 0:
    #                 # No users found in this batch, continue trying
    #                 continue
            
    #         print(f"Successfully collected {len(sequences)} user sequences")
    #         return sequences
            
    #     finally:
    #         connection.close()
    
    def _convert_user_data_to_sequence(self, user_data: List[Dict], user_id: str) -> Optional[Dict[str, Any]]:
        """Convert user interaction data to training sequence."""
        if not user_data or len(user_data) < 2:
            return None
        
        questions = []
        responses = []
        elapsed_times = []
        timestamps = []
        
        for interaction in user_data:
            question_id = interaction.get('question_id', '')
            user_answer = interaction.get('user_answer', '')
            elapsed_ms = interaction.get('elapsed_ms', '0')  # Time spent on this question
            timestamp = interaction.get('ts_ms', '0')  # Timestamp when question was answered
            
            # Convert strings to integers
            try:
                elapsed_ms = int(elapsed_ms)
                timestamp = int(timestamp)
            except (ValueError, TypeError):
                continue  # Skip invalid data
            
            # Convert question_id to numeric
            if question_id.startswith('q'):
                try:
                    numeric_question_id = int(question_id[1:])
                except ValueError:
                    continue
            else:
                continue
            
            # Convert answer to correctness
            correctness = self.question_metadata.check_answer_correctness(question_id, user_answer)
            if correctness is None:
                continue
            
            questions.append(numeric_question_id)
            responses.append(1 if correctness else 0)
            elapsed_times.append(elapsed_ms)  # Time spent on question
            timestamps.append(timestamp)  # When question was answered
        
        if len(questions) < 2:
            return None
        
        # Calculate lag times (time between consecutive questions)
        lag_times = [0]  # First question has no lag
        for i in range(1, len(timestamps)):
            lag_time = timestamps[i] - timestamps[i-1]
            lag_times.append(max(0, lag_time))  # Ensure non-negative lag times
        
        return {
            'user_id': user_id,
            'questions': questions,
            'responses': responses,
            'elapsed_times': elapsed_times,  # Time spent on each question
            'lag_times': lag_times  # Time between consecutive questions
        }


# Simple helper functions
def test_database_connection() -> bool:
    """Test database connection."""
    try:
        config = Config()
        conn = mysql.connector.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            database=config.DB_NAME
        )
        conn.close()
        print("✅ Database connection successful")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False
