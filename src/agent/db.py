import sqlite3
from pathlib import Path

DB_PATH="food_log.db"

conn=sqlite3.connect(DB_PATH,check_same_thread=False)

conn.execute("""
CREATE TABLE IF NOT EXISTS food_log(
    log_id TEXT PRIMARY KEY,
    user_id TEXT,
    meal_time TEXT,
    food_name TEXT,
    quantity TEXT,
    grams FLOAT,
    calories FLOAT,
    protein FLOAT,
    carbs FLOAT,
    fat FLOAT,
    fiber FLOAT,
    source_text TEXT
             )
""")

conn.execute("""
CREATE TABLE IF NOT EXISTS feelings_log(
    log_id TEXT PRIMARY KEY,
    user_id TEXT,
    date TEXT,
    feeling_type TEXT,
    intensity INTEGER,
    notes TEXT
)
""")

conn.execute("""
CREATE TABLE IF NOT EXISTS glucose_log(
    log_id TEXT PRIMARY KEY,
    user_id TEXT,
    date TEXT,
    glucose_level FLOAT,
    context TEXT,
    notes TEXT
)
""")

conn.execute("""
CREATE TABLE IF NOT EXISTS sleep_log_entries(
    log_id TEXT PRIMARY KEY,
    user_id TEXT,
    date TEXT,
    duration_hours FLOAT,
    quality_score INTEGER,
    notes TEXT
)
""")

conn.execute("""
CREATE TABLE IF NOT EXISTS exercise_log_entries(
    log_id TEXT PRIMARY KEY,
    user_id TEXT,
    exercise_type TEXT,
    duration_minutes FLOAT,
    intensity TEXT,
    calories FLOAT,
    date TEXT,
    notes TEXT
)
""")

conn.execute("""
CREATE TABLE IF NOT EXISTS water_log(
    log_id TEXT PRIMARY KEY,
    user_id TEXT,
    quantity TEXT,
    date TEXT,
    notes TEXT
)
""")

conn.commit()