from typing import TypedDict,List,Optional

class DietState(TypedDict):
    user_id: str
    raw_input: str
    intent: str| None
    extracted_foods: Optional[List[dict]]
    result: str| None
    meal_feedback: str| None
    current_feeling: Optional[dict]
    messages: list 