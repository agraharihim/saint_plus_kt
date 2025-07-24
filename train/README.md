# EdNet Knowledge Tracing - Minimal Training

Simple SAINT+ training pipeline for EdNet knowledge tracing.

## Files

- **`saint_plus_kt.py`** - Main SAINT+ model and training
- **`database_utils.py`** - Basic MySQL database utilities  
- **`question_metadata.py`** - Simple question metadata loader

## Quick Start

### 1. Run Training
```bash
cd train
python saint_plus_kt.py
```

### 2. Test Database (Optional)
```python
from database_utils import test_database_connection
test_database_connection()
```

### 3. Test Question Metadata (Optional)
```python
from question_metadata import QuestionMetadata
qm = QuestionMetadata()
print(f"Loaded {len(qm.question_lookup)} questions")
```

## What This Does

1. **Loads data** from MySQL database (kt1_users table)
2. **Converts** user interactions to training sequences  
3. **Trains** SAINT+ transformer model for knowledge tracing
4. **Saves** model checkpoints after each epoch

## Model Architecture

- **SAINT+ Transformer**: 4 blocks, 8 attention heads
- **Parameters**: ~3.5M trainable parameters
- **Input**: Question IDs, response correctness, timing
- **Output**: Probability of answering next question correctly

## Output

Training produces:
- `saint_plus_epoch_1.pth`, `saint_plus_epoch_2.pth`, etc.
- Console output with loss and AUC metrics

## Requirements

- PyTorch
- MySQL database with EdNet data
- questions.csv file for question metadata

That's it! This is the bare minimum needed for SAINT+ knowledge tracing training.
