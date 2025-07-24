-- SQL script to create tables for EdNet KT user data storage
-- Each table stores one row per user with data as JSON

-- Create database if it doesn't exist
CREATE DATABASE IF NOT EXISTS ednet;
USE ednet;

-- KT1 Users table
CREATE TABLE IF NOT EXISTS kt1_users (
    user_id VARCHAR(50) PRIMARY KEY,
    data JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- KT2 Users table
CREATE TABLE IF NOT EXISTS kt2_users (
    user_id VARCHAR(50) PRIMARY KEY,
    data JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- KT1 Load history table
CREATE TABLE IF NOT EXISTS kt1_load_history (
    filename VARCHAR(255) PRIMARY KEY,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- KT2 Load history table
CREATE TABLE IF NOT EXISTS kt2_load_history (
    filename VARCHAR(255) PRIMARY KEY,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_kt1_users_created ON kt1_users(created_at);
CREATE INDEX IF NOT EXISTS idx_kt2_users_created ON kt2_users(created_at);

-- Example queries to extract data from JSON:

-- Get all timestamps for a specific user in KT1
-- SELECT JSON_EXTRACT(data, '$[*].ts_ms') as timestamps FROM kt1_users WHERE user_id = 'u123';

-- Get specific fields for a user in KT1
-- SELECT JSON_EXTRACT(data, '$[*].solving_id') as solving_ids FROM kt1_users WHERE user_id = 'u123';

-- Count records per user
-- SELECT user_id, JSON_LENGTH(data) as record_count FROM kt1_users ORDER BY record_count DESC;

-- Get users who have specific question_id
-- SELECT user_id FROM kt1_users WHERE JSON_SEARCH(data, 'one', 'specific_question_id', NULL, '$[*].question_id') IS NOT NULL;
