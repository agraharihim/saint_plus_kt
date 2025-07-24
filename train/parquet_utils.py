"""
Utility functions for working with the kt1_users parquet dump.
This provides convenient functions to load and analyze the dumped data.
"""

import pandas as pd
import json
from typing import List, Dict, Any, Optional
import os


class KT1UsersParquet:
    """Utility class for working with kt1_users parquet data."""
    
    def __init__(self, parquet_file: str = "kt1_users_dump.parquet"):
        """Initialize with parquet file path."""
        self.parquet_file = parquet_file
        self._df = None
        
        if not os.path.exists(parquet_file):
            raise FileNotFoundError(f"Parquet file not found: {parquet_file}")
    
    @property
    def df(self) -> pd.DataFrame:
        """Lazy load the DataFrame."""
        if self._df is None:
            print(f"Loading {self.parquet_file}...")
            self._df = pd.read_parquet(self.parquet_file)
            print(f"Loaded {len(self._df):,} records")
        return self._df
    
    def get_basic_stats(self):
        """Get basic statistics about the dataset."""
        df = self.df
        
        print(f"=== KT1 Users Dataset Statistics ===")
        print(f"Total users: {len(df):,}")
        print(f"File size: {os.path.getsize(self.parquet_file) / 1024 / 1024:.2f} MB")
        print(f"\nSplit distribution:")
        print(df['split_type'].value_counts())
        
        print(f"\nInteraction statistics:")
        print(f"  Min interactions per user: {df['num_interactions'].min()}")
        print(f"  Max interactions per user: {df['num_interactions'].max()}")
        print(f"  Mean interactions per user: {df['num_interactions'].mean():.2f}")
        print(f"  Median interactions per user: {df['num_interactions'].median():.2f}")
        print(f"  Total interactions: {df['num_interactions'].sum():,}")
        
        print(f"\nTime statistics:")
        print(f"  Mean avg elapsed time: {df['avg_elapsed_time'].mean():.2f} ms")
        print(f"  Mean session duration: {df['session_duration'].mean() / 1000 / 60:.2f} minutes")
        
        print(f"\nQuestion statistics:")
        print(f"  Mean unique questions per user: {df['unique_questions'].mean():.2f}")
        print(f"  Total unique questions across all users: {df['unique_questions'].sum():,}")
    
    def get_users_by_split(self, split_type: str) -> pd.DataFrame:
        """Get users by split type."""
        return self.df[self.df['split_type'] == split_type].copy()
    
    def get_user_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed data for a specific user."""
        user_row = self.df[self.df['user_id'] == user_id]
        
        if user_row.empty:
            return None
        
        user_row = user_row.iloc[0]
        
        # Parse the JSON data
        interaction_data = json.loads(user_row['data_json'])
        
        return {
            'user_id': user_row['user_id'],
            'split_type': user_row['split_type'],
            'num_interactions': user_row['num_interactions'],
            'unique_questions': user_row['unique_questions'],
            'avg_elapsed_time': user_row['avg_elapsed_time'],
            'total_time_spent': user_row['total_time_spent'],
            'session_duration': user_row['session_duration'],
            'interactions': interaction_data
        }
    
    def get_sample_users(self, n: int = 5, split_type: Optional[str] = None) -> List[str]:
        """Get sample user IDs."""
        df = self.df
        if split_type:
            df = df[df['split_type'] == split_type]
        
        return df.sample(n)['user_id'].tolist()
    
    def analyze_interaction_lengths(self, split_type: Optional[str] = None):
        """Analyze the distribution of interaction lengths."""
        df = self.df
        if split_type:
            df = df[df['split_type'] == split_type]
            title_suffix = f" ({split_type})"
        else:
            title_suffix = ""
        
        interactions = df['num_interactions']
        
        print(f"=== Interaction Length Analysis{title_suffix} ===")
        print(f"Records: {len(df):,}")
        print(f"Min: {interactions.min()}")
        print(f"Max: {interactions.max()}")
        print(f"Mean: {interactions.mean():.2f}")
        print(f"Median: {interactions.median():.2f}")
        print(f"Std: {interactions.std():.2f}")
        
        print(f"\nPercentiles:")
        for p in [50, 75, 90, 95, 99]:
            print(f"  {p}%: {interactions.quantile(p/100):.0f}")
    
    def get_user_sequences_for_training(self, user_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Convert user IDs to training sequences format.
        Similar to the database_utils format but using parquet data.
        """
        sequences = []
        
        for user_id in user_ids:
            user_data = self.get_user_data(user_id)
            if not user_data or not user_data['interactions']:
                continue
            
            # Convert to training sequence format
            questions = []
            responses = []
            elapsed_times = []
            timestamps = []
            
            for interaction in user_data['interactions']:
                question_id = interaction.get('question_id', '')
                user_answer = interaction.get('user_answer', '')
                elapsed_ms = interaction.get('elapsed_ms', '0')
                timestamp = interaction.get('ts_ms', '0')
                
                # Convert strings to integers
                try:
                    elapsed_ms = int(elapsed_ms)
                    timestamp = int(timestamp)
                except (ValueError, TypeError):
                    continue
                
                # Convert question_id to numeric
                if question_id.startswith('q'):
                    try:
                        numeric_question_id = int(question_id[1:])
                    except ValueError:
                        continue
                else:
                    continue
                
                # For now, assume correctness is 1 if user_answer exists, 0 otherwise
                # In real implementation, you'd use question_metadata
                correctness = 1 if user_answer else 0
                
                questions.append(numeric_question_id)
                responses.append(correctness)
                elapsed_times.append(elapsed_ms)
                timestamps.append(timestamp)
            
            if len(questions) < 2:
                continue
            
            # Calculate lag times
            lag_times = [0]  # First question has no lag
            for i in range(1, len(timestamps)):
                lag_time = timestamps[i] - timestamps[i-1]
                lag_times.append(max(0, lag_time))
            
            sequences.append({
                'user_id': user_id,
                'questions': questions,
                'responses': responses,
                'elapsed_times': elapsed_times,
                'lag_times': lag_times
            })
        
        return sequences


def quick_analysis():
    """Quick analysis of the parquet data."""
    try:
        kt1 = KT1UsersParquet()
        kt1.get_basic_stats()
        
        print(f"\n" + "="*50)
        kt1.analyze_interaction_lengths()
        
        print(f"\n" + "="*50)
        kt1.analyze_interaction_lengths("train")
        
        print(f"\n" + "="*50)
        kt1.analyze_interaction_lengths("validation")
        
        # Sample user analysis
        print(f"\n" + "="*50)
        print("=== Sample User Analysis ===")
        sample_users = kt1.get_sample_users(3, "train")
        
        for user_id in sample_users:
            user_data = kt1.get_user_data(user_id)
            if user_data:
                print(f"\nUser {user_id}:")
                print(f"  Interactions: {user_data['num_interactions']}")
                print(f"  Unique questions: {user_data['unique_questions']}")
                print(f"  Avg elapsed time: {user_data['avg_elapsed_time']:.2f} ms")
                print(f"  Session duration: {user_data['session_duration'] / 1000 / 60:.2f} minutes")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run dump_kt1_users.py first to create the parquet file.")


if __name__ == "__main__":
    quick_analysis()
