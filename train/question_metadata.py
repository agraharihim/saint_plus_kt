"""
Question metadata utilities for SAINT+ Knowledge Tracing.
Basic functionality for loading question metadata.
"""

import pandas as pd
from typing import Dict, Optional
import os


class QuestionMetadata:
    """Basic question metadata manager."""
    
    def __init__(self, csv_path: str = "questions.csv"):
        self.csv_path = csv_path
        self.question_lookup = {}
        self.load_questions()
    
    def load_questions(self):
        """Load questions from CSV file."""
        try:
            if not os.path.exists(self.csv_path):
                print(f"⚠️  Questions file not found: {self.csv_path}")
                return
            
            print(f"📚 Loading questions from {self.csv_path}...")
            questions_df = pd.read_csv(self.csv_path)
            
            # Create lookup dictionary
            for _, row in questions_df.iterrows():
                question_id = row['question_id']
                self.question_lookup[question_id] = {
                    'correct_answer': row['correct_answer'],
                }
            
            print(f"✅ Loaded {len(self.question_lookup):,} questions")
            
        except Exception as e:
            print(f"❌ Error loading questions: {e}")
            self.question_lookup = {}
    
    def get_correct_answer(self, question_id: str) -> Optional[str]:
        """Get correct answer for a question."""
        return self.question_lookup.get(question_id, {}).get('correct_answer')
    
    def check_answer_correctness(self, question_id: str, student_answer: str) -> Optional[bool]:
        """Check if student answer is correct."""
        correct_answer = self.get_correct_answer(question_id)
        if correct_answer is None:
            return None
        return str(student_answer).lower().strip() == str(correct_answer).lower().strip()


if __name__ == "__main__":
    print("=== Question Metadata Test ===")
    qm = QuestionMetadata()
    print(f"Loaded {len(qm.question_lookup)} questions")
