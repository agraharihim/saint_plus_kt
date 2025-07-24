"""
Analyze the number of questions answered by users.
This script examines sequence lengths to validate MAX_SEQ parameter.
"""

import numpy as np
import statistics
from database_utils import DatabaseManager
from datetime import datetime
import pandas as pd


def analyze_question_counts(num_users: int = 100000, output_file: str = "question_count_analysis.txt"):
    """Analyze the number of questions answered by users."""
    
    # Start timing the analysis
    start_time = datetime.now()
    
    # Prepare output content
    report_lines = []
    
    def add_to_report(text):
        """Add text to report and print to console."""
        print(text)
        report_lines.append(text)
    
    add_to_report("=" * 80)
    add_to_report("QUESTION COUNT ANALYSIS REPORT")
    add_to_report("SAINT+ Knowledge Tracing Model - Sequence Length Statistics")
    add_to_report("=" * 80)
    add_to_report(f"Analysis Date: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    add_to_report(f"Target Users: {num_users:,}")
    add_to_report("")
    
    # Load questions metadata
    add_to_report("📚 Loading questions metadata...")
    try:
        questions_df = pd.read_csv('questions.csv')
        add_to_report(f"✅ Loaded {len(questions_df):,} total questions in dataset")
    except Exception as e:
        add_to_report(f"❌ Error loading questions: {e}")
        return
    
    # Get database manager
    db_manager = DatabaseManager()
    
    # Get random users
    add_to_report(f"🔄 Loading {num_users:,} random users...")
    user_sequences = db_manager.get_random_users_for_analysis(num_users)
    add_to_report(f"✅ Successfully loaded {len(user_sequences):,} user sequences")
    
    if not user_sequences:
        add_to_report("❌ No user data available!")
        return
    
    # Collect question count statistics
    question_counts = []
    user_stats = []
    
    add_to_report("\n🔍 Processing user sequences...")
    
    for i, sequence in enumerate(user_sequences):
        if i % 10000 == 0:
            add_to_report(f"   Processed {i:,}/{len(user_sequences):,} users...")
        
        user_id = sequence['user_id']
        questions = sequence['questions']
        responses = sequence['responses']
        
        # Count valid questions (non-zero)
        valid_questions = [q for q in questions if q > 0]
        question_count = len(valid_questions)
        
        if question_count > 0:
            question_counts.append(question_count)
            
            # Calculate response accuracy
            valid_responses = responses[:question_count]
            correct_responses = sum(valid_responses)
            accuracy = correct_responses / question_count if question_count > 0 else 0
            
            user_stats.append({
                'user_id': user_id,
                'question_count': question_count,
                'correct_responses': correct_responses,
                'accuracy': accuracy
            })
    
    add_to_report(f"✅ Processing complete!")
    
    # ======================================================================
    # QUESTION COUNT ANALYSIS
    # ======================================================================
    
    add_to_report("\n" + "=" * 60)
    add_to_report("SEQUENCE LENGTH STATISTICS")
    add_to_report("=" * 60)
    
    if question_counts:
        add_to_report(f"\n📊 Dataset Overview:")
        add_to_report(f"Total users analyzed: {len(user_stats):,}")
        add_to_report(f"Total questions answered: {sum(question_counts):,}")
        add_to_report(f"Min questions per user: {min(question_counts):,}")
        add_to_report(f"Max questions per user: {max(question_counts):,}")
        add_to_report(f"Mean questions per user: {statistics.mean(question_counts):,.2f}")
        add_to_report(f"Median questions per user: {statistics.median(question_counts):,.2f}")
        add_to_report(f"Standard deviation: {statistics.stdev(question_counts):,.2f}")
        
        # Calculate percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99, 99.5, 99.9]
        add_to_report(f"\n📈 Question Count Percentiles:")
        for p in percentiles:
            value = np.percentile(question_counts, p)
            add_to_report(f"p{p:4.1f}: {value:8.1f} questions")
        
        # Distribution analysis based on common sequence length thresholds
        add_to_report(f"\n📏 Sequence Length Distribution:")
        very_short = sum(1 for c in question_counts if c <= 10)
        short = sum(1 for c in question_counts if 10 < c <= 50)
        medium = sum(1 for c in question_counts if 50 < c <= 100)
        long_seq = sum(1 for c in question_counts if 100 < c <= 200)
        very_long = sum(1 for c in question_counts if 200 < c <= 500)
        extremely_long = sum(1 for c in question_counts if c > 500)
        
        total = len(question_counts)
        add_to_report(f"Very short (≤ 10):       {very_short:8,} ({very_short/total*100:5.1f}%)")
        add_to_report(f"Short (11-50):           {short:8,} ({short/total*100:5.1f}%)")
        add_to_report(f"Medium (51-100):         {medium:8,} ({medium/total*100:5.1f}%)")
        add_to_report(f"Long (101-200):          {long_seq:8,} ({long_seq/total*100:5.1f}%)")
        add_to_report(f"Very long (201-500):     {very_long:8,} ({very_long/total*100:5.1f}%)")
        add_to_report(f"Extremely long (> 500):  {extremely_long:8,} ({extremely_long/total*100:5.1f}%)")
        
        # MAX_SEQ analysis
        current_max_seq = 100  # From Config class
        add_to_report(f"\n⚙️  MAX_SEQ Analysis:")
        add_to_report(f"Current MAX_SEQ setting: {current_max_seq}")
        within_max_seq = sum(1 for c in question_counts if c <= current_max_seq)
        above_max_seq = sum(1 for c in question_counts if c > current_max_seq)
        
        add_to_report(f"Users within MAX_SEQ: {within_max_seq:,} ({within_max_seq/total*100:.2f}%)")
        add_to_report(f"Users above MAX_SEQ: {above_max_seq:,} ({above_max_seq/total*100:.2f}%)")
        add_to_report(f"Coverage: {within_max_seq/total*100:.2f}% of users fit within MAX_SEQ")
        
        # Recommend optimal MAX_SEQ values
        p95_count = np.percentile(question_counts, 95)
        p99_count = np.percentile(question_counts, 99)
        add_to_report(f"\n📋 MAX_SEQ Recommendations:")
        add_to_report(f"For 95% coverage: MAX_SEQ = {int(p95_count)}")
        add_to_report(f"For 99% coverage: MAX_SEQ = {int(p99_count)}")
        
        # Performance impact analysis
        current_coverage = within_max_seq / total * 100
        if current_coverage < 90:
            add_to_report(f"⚠️  Current MAX_SEQ covers only {current_coverage:.1f}% - consider increasing")
        elif current_coverage > 99:
            add_to_report(f"✅ Current MAX_SEQ covers {current_coverage:.1f}% - well optimized")
        else:
            add_to_report(f"✅ Current MAX_SEQ covers {current_coverage:.1f}% - good balance")
    
    # ======================================================================
    # ACCURACY ANALYSIS BY SEQUENCE LENGTH
    # ======================================================================
    
    add_to_report("\n" + "=" * 60)
    add_to_report("ACCURACY BY SEQUENCE LENGTH")
    add_to_report("=" * 60)
    
    if user_stats:
        # Group users by sequence length ranges
        length_groups = {
            'Very short (≤10)': [u for u in user_stats if u['question_count'] <= 10],
            'Short (11-50)': [u for u in user_stats if 10 < u['question_count'] <= 50],
            'Medium (51-100)': [u for u in user_stats if 50 < u['question_count'] <= 100],
            'Long (101-200)': [u for u in user_stats if 100 < u['question_count'] <= 200],
            'Very long (201+)': [u for u in user_stats if u['question_count'] > 200]
        }
        
        add_to_report(f"\n📊 Accuracy Statistics by Sequence Length:")
        for group_name, users in length_groups.items():
            if users:
                accuracies = [u['accuracy'] for u in users]
                counts = [u['question_count'] for u in users]
                add_to_report(f"\n{group_name}:")
                add_to_report(f"  Users: {len(users):,}")
                add_to_report(f"  Avg questions: {statistics.mean(counts):,.1f}")
                add_to_report(f"  Avg accuracy: {statistics.mean(accuracies)*100:.1f}%")
                add_to_report(f"  Median accuracy: {statistics.median(accuracies)*100:.1f}%")
        
        # Overall accuracy
        all_accuracies = [u['accuracy'] for u in user_stats]
        add_to_report(f"\n🎯 Overall Statistics:")
        add_to_report(f"Overall mean accuracy: {statistics.mean(all_accuracies)*100:.2f}%")
        add_to_report(f"Overall median accuracy: {statistics.median(all_accuracies)*100:.2f}%")
    
    # ======================================================================
    # SAMPLE USER ANALYSIS
    # ======================================================================
    
    add_to_report("\n" + "=" * 60)
    add_to_report("SAMPLE USER ANALYSIS")
    add_to_report("=" * 60)
    
    if user_stats:
        # Show sample users from different ranges
        add_to_report(f"\n👥 Sample Users (first 10):")
        for i, user in enumerate(user_stats[:10]):
            add_to_report(f"User {user['user_id']}: {user['question_count']} questions, "
                         f"{user['correct_responses']} correct, {user['accuracy']*100:.1f}% accuracy")
        
        # Show some high-engagement users
        high_engagement = sorted(user_stats, key=lambda x: x['question_count'], reverse=True)[:5]
        add_to_report(f"\n🏆 Top 5 Most Active Users:")
        for user in high_engagement:
            add_to_report(f"User {user['user_id']}: {user['question_count']} questions, "
                         f"{user['accuracy']*100:.1f}% accuracy")
    
    # ======================================================================
    # COMPUTATIONAL IMPACT ANALYSIS
    # ======================================================================
    
    add_to_report("\n" + "=" * 60)
    add_to_report("COMPUTATIONAL IMPACT ANALYSIS")
    add_to_report("=" * 60)
    
    if question_counts:
        # Memory usage analysis
        current_max_seq = 100
        avg_seq_len = statistics.mean(question_counts)
        median_seq_len = statistics.median(question_counts)
        
        add_to_report(f"\n💾 Memory Usage Analysis:")
        add_to_report(f"Current MAX_SEQ: {current_max_seq}")
        add_to_report(f"Average actual sequence length: {avg_seq_len:.1f}")
        add_to_report(f"Median actual sequence length: {median_seq_len:.1f}")
        add_to_report(f"Memory utilization: {avg_seq_len/current_max_seq*100:.1f}% of allocated space")
        
        # Padding analysis
        total_padding = sum(max(0, current_max_seq - count) for count in question_counts)
        total_allocated = len(question_counts) * current_max_seq
        padding_ratio = total_padding / total_allocated * 100
        
        add_to_report(f"\n📏 Padding Analysis:")
        add_to_report(f"Total padding tokens: {total_padding:,}")
        add_to_report(f"Total allocated tokens: {total_allocated:,}")
        add_to_report(f"Padding ratio: {padding_ratio:.1f}%")
        
        if padding_ratio > 50:
            add_to_report(f"⚠️  High padding ratio - consider optimizing MAX_SEQ")
        else:
            add_to_report(f"✅ Reasonable padding ratio")
    
    # ======================================================================
    # SUMMARY AND RECOMMENDATIONS
    # ======================================================================
    
    end_time = datetime.now()
    analysis_duration = end_time - start_time
    
    add_to_report("\n" + "=" * 60)
    add_to_report("SUMMARY AND RECOMMENDATIONS")
    add_to_report("=" * 60)
    
    add_to_report(f"\n📈 Analysis Summary:")
    add_to_report(f"Users analyzed: {len(user_sequences):,}")
    add_to_report(f"Total questions answered: {sum(question_counts):,}")
    add_to_report(f"Average questions per user: {statistics.mean(question_counts):,.1f}")
    add_to_report(f"Analysis duration: {analysis_duration}")
    
    if question_counts:
        current_coverage = sum(1 for c in question_counts if c <= 100) / len(question_counts) * 100
        add_to_report(f"\n⚙️  Current MAX_SEQ=100 Analysis:")
        add_to_report(f"Coverage: {current_coverage:.2f}% of users")
        
        if current_coverage >= 95:
            add_to_report(f"✅ Recommendation: Keep MAX_SEQ=100 (excellent coverage)")
        elif current_coverage >= 90:
            add_to_report(f"✅ Recommendation: MAX_SEQ=100 is adequate")
            p95_val = int(np.percentile(question_counts, 95))
            add_to_report(f"   Consider MAX_SEQ={p95_val} for 95% coverage if needed")
        else:
            p90_val = int(np.percentile(question_counts, 90))
            p95_val = int(np.percentile(question_counts, 95))
            add_to_report(f"⚠️  Recommendation: Consider increasing MAX_SEQ")
            add_to_report(f"   For 90% coverage: MAX_SEQ={p90_val}")
            add_to_report(f"   For 95% coverage: MAX_SEQ={p95_val}")
    
    add_to_report("\n" + "=" * 80)
    add_to_report("END OF QUESTION COUNT ANALYSIS")
    add_to_report("=" * 80)
    
    # Save report to file
    add_to_report(f"\n💾 Saving report to: {output_file}")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        add_to_report(f"✅ Report successfully saved!")
    except Exception as e:
        add_to_report(f"❌ Error saving report: {e}")
    
    return output_file


if __name__ == "__main__":
    # Run question count analysis for 100,000 users
    output_filename = f"question_count_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    analyze_question_counts(100000, output_filename)
