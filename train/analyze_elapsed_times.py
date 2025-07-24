"""
Analyze elapsed time statistics across users.
This script reads random users and calculates elapsed time statistics.
"""

import numpy as np
import json
import mysql.connector
from typing import List, Dict, Any
import statistics
from database_utils import DatabaseManager


def analyze_elapsed_times():
    """Analyze elapsed time statistics across random users."""
    print("=== Elapsed Time Analysis ===")
    
    # Get database manager
    db_manager = DatabaseManager()
    
    # Get random users for analysis
    print("Loading 5000 random users...")
    user_sequences = db_manager.get_random_users_for_analysis(5000)
    print(f"Loaded {len(user_sequences)} user sequences")
    
    if not user_sequences:
        print("No user data available!")
        return
    
    # Collect all elapsed times across all users
    all_elapsed_times = []
    user_elapsed_stats = []
    
    for sequence in user_sequences:
        user_id = sequence['user_id']
        elapsed_times = sequence['elapsed_times']
        
        # Filter out zero elapsed times
        valid_elapsed_times = [et for et in elapsed_times if et > 0]
        
        if valid_elapsed_times:
            # Add to overall collection
            all_elapsed_times.extend(valid_elapsed_times)
            
            # Calculate per-user statistics
            user_stats = {
                'user_id': user_id,
                'num_interactions': len(elapsed_times),
                'valid_elapsed_times': len(valid_elapsed_times),
                'min_elapsed': min(valid_elapsed_times),
                'max_elapsed': max(valid_elapsed_times),
                'mean_elapsed': statistics.mean(valid_elapsed_times),
                'median_elapsed': statistics.median(valid_elapsed_times)
            }
            user_elapsed_stats.append(user_stats)
    
    if not all_elapsed_times:
        print("No valid elapsed times found!")
        return
    
    # Calculate overall statistics
    print(f"\n=== Overall Elapsed Time Statistics (across {len(user_elapsed_stats)} users) ===")
    print(f"Total valid elapsed times: {len(all_elapsed_times):,}")
    print(f"Min elapsed time: {min(all_elapsed_times):,} ms ({min(all_elapsed_times)/1000:.2f} seconds)")
    print(f"Max elapsed time: {max(all_elapsed_times):,} ms ({max(all_elapsed_times)/1000:.2f} seconds)")
    print(f"Median elapsed time: {statistics.median(all_elapsed_times):,.2f} ms ({statistics.median(all_elapsed_times)/1000:.2f} seconds)")
    print(f"Standard deviation: {statistics.stdev(all_elapsed_times):,.2f} ms")
    
    # Calculate detailed percentiles (p90-p99)
    percentiles = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
    print(f"\n=== Detailed High Percentiles (p90-p99) ===")
    for p in percentiles:
        value = np.percentile(all_elapsed_times, p)
        print(f"p{p}: {value:,.2f} ms ({value/1000:.2f} seconds)")
    
    # Also show some lower percentiles for context
    lower_percentiles = [10, 25, 50, 75]
    print(f"\n=== Lower Percentiles for Context ===")
    for p in lower_percentiles:
        value = np.percentile(all_elapsed_times, p)
        print(f"p{p}: {value:,.2f} ms ({value/1000:.2f} seconds)")
    
    # Per-user statistics summary
    user_means = [stats['mean_elapsed'] for stats in user_elapsed_stats]
    user_medians = [stats['median_elapsed'] for stats in user_elapsed_stats]
    
    print(f"\n=== Per-User Statistics Summary ===")
    print(f"Average of user mean elapsed times: {statistics.mean(user_means):,.2f} ms")
    print(f"Average of user median elapsed times: {statistics.mean(user_medians):,.2f} ms")
    print(f"Min user mean elapsed time: {min(user_means):,.2f} ms")
    print(f"Max user mean elapsed time: {max(user_means):,.2f} ms")
    
    # Show some example users
    print(f"\n=== Sample User Statistics (first 5 users) ===")
    for i, stats in enumerate(user_elapsed_stats[:5]):
        print(f"User {stats['user_id']}: {stats['valid_elapsed_times']} interactions, "
              f"mean: {stats['mean_elapsed']:,.2f}ms, median: {stats['median_elapsed']:,.2f}ms")
    
    # Elapsed time distribution analysis
    print(f"\n=== Elapsed Time Distribution ===")
    very_fast = len([et for et in all_elapsed_times if et < 5000])  # < 5 seconds
    fast = len([et for et in all_elapsed_times if 5000 <= et < 15000])  # 5-15 seconds
    normal = len([et for et in all_elapsed_times if 15000 <= et < 60000])  # 15-60 seconds
    slow = len([et for et in all_elapsed_times if 60000 <= et < 300000])  # 1-5 minutes
    very_slow = len([et for et in all_elapsed_times if et >= 300000])  # > 5 minutes
    
    total = len(all_elapsed_times)
    print(f"Very fast (< 5s): {very_fast:,} ({very_fast/total*100:.1f}%)")
    print(f"Fast (5-15s): {fast:,} ({fast/total*100:.1f}%)")
    print(f"Normal (15-60s): {normal:,} ({normal/total*100:.1f}%)")
    print(f"Slow (1-5m): {slow:,} ({slow/total*100:.1f}%)")
    print(f"Very slow (> 5m): {very_slow:,} ({very_slow/total*100:.1f}%)")
    
    # Check current MAX_TIME setting
    max_time_ms = 300000  # From config
    above_max = len([et for et in all_elapsed_times if et > max_time_ms])
    print(f"\n=== MAX_TIME Analysis for Elapsed Times ===")
    print(f"Current MAX_TIME setting: {max_time_ms:,} ms ({max_time_ms/1000} seconds)")
    print(f"Elapsed times above MAX_TIME: {above_max:,} ({above_max/total*100:.2f}%)")
    print(f"These will be clamped to 1.0 in normalization")
    
    # Question difficulty analysis based on elapsed time
    print(f"\n=== Question Difficulty Insights ===")
    print(f"Questions with very fast answers (< 5s): {very_fast/total*100:.1f}% - likely easy or guessed")
    print(f"Questions with normal thinking time (15-60s): {normal/total*100:.1f}% - appropriate difficulty")
    print(f"Questions with long thinking time (> 1m): {(slow + very_slow)/total*100:.1f}% - challenging or complex")


def compare_elapsed_vs_lag_times():
    """Compare elapsed times vs lag times to understand user behavior patterns."""
    print("\n" + "="*60)
    print("=== Elapsed Time vs Lag Time Comparison ===")
    
    # Get database manager
    db_manager = DatabaseManager()
    
    # Get random users
    print("Loading 50 users for comparison...")
    user_sequences = db_manager.get_user_batch(50)
    
    all_elapsed = []
    all_lag = []
    
    for sequence in user_sequences:
        elapsed_times = sequence['elapsed_times']
        lag_times = sequence['lag_times'][1:]  # Skip first lag time (always 0)
        
        # Filter valid times
        valid_elapsed = [et for et in elapsed_times if et > 0]
        valid_lag = [lt for lt in lag_times if lt > 0]
        
        all_elapsed.extend(valid_elapsed)
        all_lag.extend(valid_lag)
    
    if all_elapsed and all_lag:
        print(f"\nElapsed Time Summary:")
        print(f"  Mean: {statistics.mean(all_elapsed):,.0f} ms ({statistics.mean(all_elapsed)/1000:.1f}s)")
        print(f"  Median: {statistics.median(all_elapsed):,.0f} ms ({statistics.median(all_elapsed)/1000:.1f}s)")
        
        print(f"\nLag Time Summary:")
        print(f"  Mean: {statistics.mean(all_lag):,.0f} ms ({statistics.mean(all_lag)/1000:.1f}s)")
        print(f"  Median: {statistics.median(all_lag):,.0f} ms ({statistics.median(all_lag)/1000:.1f}s)")
        
        # Ratio analysis
        mean_ratio = statistics.mean(all_lag) / statistics.mean(all_elapsed)
        median_ratio = statistics.median(all_lag) / statistics.median(all_elapsed)
        
        print(f"\nTime Ratios:")
        print(f"  Mean lag/elapsed ratio: {mean_ratio:.2f}")
        print(f"  Median lag/elapsed ratio: {median_ratio:.2f}")
        
        if mean_ratio > 2:
            print("  → Users spend more time between questions than on questions (study breaks/navigation)")
        elif mean_ratio < 0.5:
            print("  → Users answer questions quickly with minimal breaks (focused sessions)")
        else:
            print("  → Balanced time spent on questions vs between questions")


if __name__ == "__main__":
    analyze_elapsed_times()
    compare_elapsed_vs_lag_times()
