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

