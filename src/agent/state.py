from typing import TypedDict,List,Optional, Annotated
import operator


def reduce_logs(existing: list, new: list) -> list:
    if new == ["CLEAR"]:
        return []
    return existing + new

class DietState(TypedDict):
    user_id: str
    raw_input: str
    intent: List[str]
    extracted_foods: Optional[List[dict]]
    result: str| None
    meal_feedback: str| None
    current_feeling: Optional[dict]
    sleep_duration: Optional[dict]
    glucose_level: Optional[dict]
    exercise: Optional[dict]
    water_quantity: Optional[dict]
    messages: list
    logs: Annotated[list, reduce_logs]