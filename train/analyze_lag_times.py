"""
Analyze lag time statistics across users.
This script reads random 1000 users and calculates lag time statistics.
"""

import numpy as np
import json
import mysql.connector
from typing import List, Dict, Any
import statistics
from database_utils import DatabaseManager


def analyze_lag_times():
    """Analyze lag time statistics across random 5000 users."""
    print("=== Lag Time Analysis ===")
    
    # Load questions metadata first
    print("📚 Loading questions from questions.csv...")
    try:
        import pandas as pd
        questions_df = pd.read_csv('questions.csv')  # File is in current directory
        print(f"✅ Loaded {len(questions_df)} questions")
    except Exception as e:
        print(f"❌ Error loading questions: {e}")
        return
    
    # Get database manager
    db_manager = DatabaseManager()
    
    # Get random 5000 users for analysis
    print("Loading 5000 random users...")
    user_sequences = db_manager.get_random_users_for_analysis(5000)
    print(f"Loaded {len(user_sequences)} user sequences")
    
    if not user_sequences:
        print("No user data available!")
        return
    
    # Collect all lag times across all users
    all_lag_times = []
    user_lag_stats = []
    
    for sequence in user_sequences:
        user_id = sequence['user_id']
        lag_times = sequence['lag_times']
        
        # Remove the first lag time (which is always 0) and filter out zeros
        valid_lag_times = [lt for lt in lag_times[1:] if lt > 0]
        
        if valid_lag_times:
            # Add to overall collection
            all_lag_times.extend(valid_lag_times)
            
            # Calculate per-user statistics
            user_stats = {
                'user_id': user_id,
                'num_interactions': len(lag_times),
                'valid_lag_times': len(valid_lag_times),
                'min_lag': min(valid_lag_times),
                'max_lag': max(valid_lag_times),
                'mean_lag': statistics.mean(valid_lag_times),
                'median_lag': statistics.median(valid_lag_times)
            }
            user_lag_stats.append(user_stats)
    
    if not all_lag_times:
        print("No valid lag times found!")
        return
    
    # Calculate overall statistics
    print(f"\n=== Overall Lag Time Statistics (across {len(user_lag_stats)} users) ===")
    print(f"Total valid lag times: {len(all_lag_times):,}")
    print(f"Min lag time: {min(all_lag_times):,} ms ({min(all_lag_times)/1000:.2f} seconds)")
    print(f"Max lag time: {max(all_lag_times):,} ms ({max(all_lag_times)/1000:.2f} seconds)")
    print(f"Median lag time: {statistics.median(all_lag_times):,.2f} ms ({statistics.median(all_lag_times)/1000:.2f} seconds)")
    print(f"Standard deviation: {statistics.stdev(all_lag_times):,.2f} ms")
    
    # Calculate detailed percentiles (p90-p99)
    percentiles = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
    print(f"\n=== Detailed High Percentiles (p90-p99) ===")
    for p in percentiles:
        value = np.percentile(all_lag_times, p)
        print(f"p{p}: {value:,.2f} ms ({value/1000:.2f} seconds)")
    
    # Also show some lower percentiles for context
    lower_percentiles = [10, 25, 50, 75]
    print(f"\n=== Lower Percentiles for Context ===")
    for p in lower_percentiles:
        value = np.percentile(all_lag_times, p)
        print(f"p{p}: {value:,.2f} ms ({value/1000:.2f} seconds)")
    
    # Per-user statistics summary
    user_means = [stats['mean_lag'] for stats in user_lag_stats]
    user_medians = [stats['median_lag'] for stats in user_lag_stats]
    
    print(f"\n=== Per-User Statistics Summary ===")
    print(f"Average of user mean lag times: {statistics.mean(user_means):,.2f} ms")
    print(f"Average of user median lag times: {statistics.mean(user_medians):,.2f} ms")
    print(f"Min user mean lag time: {min(user_means):,.2f} ms")
    print(f"Max user mean lag time: {max(user_means):,.2f} ms")
    
    # Show some example users
    print(f"\n=== Sample User Statistics (first 5 users) ===")
    for i, stats in enumerate(user_lag_stats[:5]):
        print(f"User {stats['user_id']}: {stats['valid_lag_times']} interactions, "
              f"mean: {stats['mean_lag']:,.2f}ms, median: {stats['median_lag']:,.2f}ms")
    
    # Lag time distribution analysis
    print(f"\n=== Lag Time Distribution ===")
    very_short = len([lt for lt in all_lag_times if lt < 1000])  # < 1 second
    short = len([lt for lt in all_lag_times if 1000 <= lt < 5000])  # 1-5 seconds
    medium = len([lt for lt in all_lag_times if 5000 <= lt < 30000])  # 5-30 seconds
    long_lag = len([lt for lt in all_lag_times if 30000 <= lt < 300000])  # 30 seconds - 5 minutes
    very_long = len([lt for lt in all_lag_times if lt >= 300000])  # > 5 minutes
    
    total = len(all_lag_times)
    print(f"Very short (< 1s): {very_short:,} ({very_short/total*100:.1f}%)")
    print(f"Short (1-5s): {short:,} ({short/total*100:.1f}%)")
    print(f"Medium (5-30s): {medium:,} ({medium/total*100:.1f}%)")
    print(f"Long (30s-5m): {long_lag:,} ({long_lag/total*100:.1f}%)")
    print(f"Very long (> 5m): {very_long:,} ({very_long/total*100:.1f}%)")
    
    # Check current MAX_TIME setting
    max_time_ms = 300000  # From config
    above_max = len([lt for lt in all_lag_times if lt > max_time_ms])
    print(f"\n=== MAX_TIME Analysis ===")
    print(f"Current MAX_TIME setting: {max_time_ms:,} ms ({max_time_ms/1000} seconds)")
    print(f"Lag times above MAX_TIME: {above_max:,} ({above_max/total*100:.2f}%)")
    print(f"These will be clamped to 1.0 in normalization")


if __name__ == "__main__":
    analyze_lag_times()
