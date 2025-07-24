# EdNet Python Project

This project loads EdNet KT datasets into MySQL with **one row per user**, storing each user's activity data as a JSON array sorted by timestamp.

## Environment Setup

- **Python Version**: 3.13.5
- **Environment Type**: Virtual Environment
- **Environment Path**: `.venv/`

## Database Schema

The script creates the following structure:
- **kt1_users/kt2_users**: Main tables with `user_id` (primary key) and `data` (JSON column)
- **kt1_load_history/kt2_load_history**: Track processed files for resumable loading

Each user's data is stored as a JSON array with records sorted by timestamp.

## Setup Instructions

### 1. Create Database Tables

First, set up the database structure:

```bash
/Users/himanshu/development/ednet/.venv/bin/python setup_database.py \
  --user your_mysql_username \
  --password your_mysql_password \
  --host localhost \
  --port 3306
```

### 2. Load Data

Run the main loader script:

```bash
/Users/himanshu/development/ednet/.venv/bin/python sample.py /path/to/kt1/data \
  --dataset kt1 \
  --user your_mysql_username \
  --password your_mysql_password \
  --host localhost \
  --port 3306 \
  --database ednet
```

## Key Features

- ✅ **One row per user**: Each user's complete activity stored in single record
- ✅ **JSON storage**: Data stored as JSON array for flexible querying
- ✅ **Timestamp ordering**: Records sorted by timestamp within each user
- ✅ **Resumable loading**: Skip already processed files
- ✅ **Memory efficient**: No longer chunks (reads full file per user)
- ✅ **Idempotent**: Safe to re-run, updates existing user data

## Example Queries

```sql
-- Get record count per user
SELECT user_id, JSON_LENGTH(data) as record_count 
FROM kt1_users ORDER BY record_count DESC;

-- Get all timestamps for a user
SELECT JSON_EXTRACT(data, '$[*].ts_ms') as timestamps 
FROM kt1_users WHERE user_id = 'u123';

-- Find users with specific question_id
SELECT user_id FROM kt1_users 
WHERE JSON_SEARCH(data, 'one', 'target_question_id', NULL, '$[*].question_id') IS NOT NULL;
```

## Project Structure

```
ednet/
├── .venv/                  # Virtual environment
├── sample.py              # Main data loader (JSON format)
├── setup_database.py      # Database table creation utility
├── create_tables.sql      # SQL schema definitions
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Data Format

Each user's JSON data contains an array of records like:
```json
[
  {
    "ts_ms": 1234567890,
    "solving_id": "abc123",
    "question_id": "q456",
    "user_answer": "A",
    "elapsed_ms": 15000
  },
  ...
]
```

Records are automatically sorted by timestamp for each user.
