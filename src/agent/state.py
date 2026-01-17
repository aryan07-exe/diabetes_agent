from typing import TypedDict,List,Optional

class DietState(TypedDict):
    user_id: str
    raw_input: str
    intent: str| None
    extracted_foods: Optional[List[dict]]
    result: str| None
    meal_feedback: str| None
    current_feeling: Optional[dict]
    sleep_duration: Optional[dict]
    glucose_level: Optional[dict]
    exercise: Optional[dict]
    water_quantity: Optional[dict]
    messages: list 