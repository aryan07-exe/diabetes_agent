from pydantic import BaseModel, Field
from typing import Optional, List, Literal

class FoodItem(BaseModel):
    meal_time: Optional[str] = Field(description="Time of the meal. Use ISO 8601 format (YYYY-MM-DD HH:MM:SS) if specific time is known, otherwise today's date with estimated time.")
    food_name: str = Field(description="Name of the food item")
    quantity: str = Field(description="Quantity description (e.g. '1 bowl')")
    grams: Optional[float] = Field(description="Weight in grams (number only)")
    calories: Optional[float] = Field(description="Calories (number only)")
    protein: Optional[float] = Field(description="Protein in grams (number only)")
    carbs: Optional[float] = Field(description="Carbs in grams (number only)")
    fat: Optional[float] = Field(description="Fat in grams (number only)")
    fiber: Optional[float] = Field(description="Fiber in grams (number only)")

class FoodExtraction(BaseModel):
    foods: List[FoodItem]

class RouteDecision(BaseModel):
    intents: List[Literal['log_food', 'ask_history', 'log_feeling', 'log_glucose',
     'log_sleep', 'log_exercise', 'general_health', 
     'log_water']] = Field(description="List of all relevant categories for the user's input")
