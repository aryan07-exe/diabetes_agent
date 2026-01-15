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
    timestamp TEXT,
    feeling_type TEXT,
    intensity INTEGER,
    notes TEXT
)
""")
conn.commit()