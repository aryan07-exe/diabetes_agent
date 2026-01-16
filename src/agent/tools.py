from langchain.tools import tool
from uuid import uuid4
from datetime import datetime
from agent.db import conn



def save_food(user_id:str, foods:list, source_text:str):
    """
    Save extracted food items with macros into the long-term food_log database.
    """
    for food in foods:
        conn.execute("""
            INSERT INTO food_log VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                     
        """,(
            str(uuid4()),
            user_id,
            food.get("meal_time"),
            food["food_name"],
            food["quantity"],
            food.get("grams"),
            food["calories"],
            food["protein"],
            food["carbs"],
            food["fat"],
            food["fiber"],
            source_text
        ))
    conn.commit()
    return "saved"

def save_feeling(user_id:str, feeling_type:str, intensity:int, notes:str):
    """
    Save user feelings/symptoms to the database.
    """
    conn.execute("""
        INSERT INTO feelings_log VALUES(?,?,datetime('now'),?,?,?)
    """, (
        str(uuid4()),
        user_id,
        feeling_type,
        intensity,
        notes
    ))
    conn.commit()
    return "feeling_saved"

def save_glucose(user_id:str, glucose_level:float, context:str, notes:str):
    """
    Save glucose level to the database.
    """
    conn.execute("""
        INSERT INTO glucose_log VALUES(?,?,datetime('now'),?,?,?)
    """, (
        str(uuid4()),
        user_id,
        glucose_level,
        context,
        notes
    ))
    conn.commit()
    return "glucose_saved"

def save_sleep(user_id:str, duration_hours:float, quality_score:int, notes:str):
    """
    Save sleep record to the database.
    """
    conn.execute("""
        INSERT INTO sleep_log_entries VALUES(?,?,datetime('now'),?,?,?)
    """, (
        str(uuid4()),
        user_id,
        duration_hours,
        quality_score,
        notes
    ))
    conn.commit()
    return "sleep_saved"

def save_exercise(user_id:str, exercise_type:str, duration_minutes:float, intensity:str, notes:str):
    """
    Save exercise record to the database.
    """
    conn.execute("""
        INSERT INTO exercise_log_entries VALUES(?,?,?,?,?,?)
    """, (
        str(uuid4()),
        user_id,
        exercise_type,
        duration_minutes,
        intensity,
        notes
    ))
    conn.commit()
    return "exercise_saved"



