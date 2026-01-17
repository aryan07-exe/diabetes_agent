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

def save_feeling(user_id:str, feeling_type:str, intensity:int, notes:str, date:str):
    """
    Save user feelings/symptoms to the database.
    """
    conn.execute("""
        INSERT INTO feelings_log VALUES(?,?,?,?,?,?)
    """, (
        str(uuid4()),
        user_id,
        date,
        feeling_type,
        intensity,
        notes
    ))
    conn.commit()
    return "feeling_saved"

def save_glucose(user_id:str, glucose_level:float, context:str, notes:str, date:str):
    """
    Save glucose level to the database.
    """
    conn.execute("""
        INSERT INTO glucose_log VALUES(?,?,?,?,?,?)
    """, (
        str(uuid4()),
        user_id,
        date,
        glucose_level,
        context,
        notes
    ))
    conn.commit()
    return "glucose_saved"

def save_sleep(user_id:str, duration_hours:float, quality_score:int, notes:str, date:str):
    """
    Save sleep record to the database.
    """
    conn.execute("""
        INSERT INTO sleep_log_entries VALUES(?,?,?,?,?,?)
    """, (
        str(uuid4()),
        user_id,
        date,
        duration_hours,
        quality_score,
        notes
    ))

    conn.commit()
    return "sleep_saved"

def save_exercise(user_id:str, exercise_type:str, duration_minutes:float, intensity:str, calories:float, notes:str, date:str):
    """
    Save exercise record to the database.
    """
    conn.execute("""
        INSERT INTO exercise_log_entries VALUES(?,?,?,?,?,?,?,?)
    """, (
        str(uuid4()),
        user_id,
        exercise_type,
        duration_minutes,
        intensity,
        calories,
        date,
        notes
    ))
    conn.commit()
    return "exercise_saved"


def save_water(user_id:str, quantity:str, date:str, notes:str, action:str="add"):
    """
    Save or update water record to the database.
    action: 'add' (append) or 'update' (overwrite)
    """
    # Check if a record already exists for this user and date
    cursor = conn.execute("SELECT log_id, quantity, notes FROM water_log WHERE user_id = ? AND date = ?", (user_id, date))
    existing = cursor.fetchone()

    if existing:
        log_id, old_quantity, old_notes = existing
        
        if action == "update":
            new_quantity = quantity
            new_notes = notes
        else:
            # Append new quantity to the old one.
            new_quantity = f"{old_quantity} + {quantity}"
            new_notes = f"{old_notes}. {notes}" if old_notes else notes
        
        conn.execute("""
            UPDATE water_log 
            SET quantity = ?, notes = ? 
            WHERE log_id = ?
        """, (new_quantity, new_notes, log_id))
        conn.commit()
        return "water_updated"

    conn.execute("""
        INSERT INTO water_log VALUES(?,?,?,?,?)
    """, (
        str(uuid4()),
        user_id,
        quantity,
        date,
        notes
    ))
    conn.commit()
    return "water_saved"


