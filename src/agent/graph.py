import json
import re
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from agent.state import DietState
from agent.tools import save_food, save_feeling, save_glucose, save_sleep, save_exercise
from agent.db import conn
from agent.models import FoodExtraction
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
    # other params...
)


# llm = ChatOllama(
#     model="phi3:mini",
#     temperature=0
# )
# from langchain_google_genai import ChatGoogleGenerativeAI
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def extract_foods(state: DietState):
    prompt = f"""
You are a nutrition assistant.
Extract food items from the text.
Estimate macros (calories, protein, carbs, fat, fiber) if not explicitly provided.
Convert relative times (like "breakfast", "last night") to strict ISO 8601 timestamps (YYYY-MM-DD HH:MM:SS).
If no time is specified, use the current time.

Text:
{state["raw_input"]}
"""
    structured_llm = llm.with_structured_output(FoodExtraction)
    res = structured_llm.invoke(prompt)
    
    # Convert Pydantic models to dicts for state
    foods = [f.model_dump() for f in res.foods]
    
    return {"extracted_foods": foods}

def save_node(state: DietState):
    save_food(
        user_id=state["user_id"],
        foods=state["extracted_foods"],
        source_text=state["raw_input"],
    )
    return {}

def route_node(state: DietState):
    prompt = f"""
Classify this message into ONE category:
- log_food
- ask_history
- log_feeling
- log_glucose
- log_sleep
- log_exercise
- general_health

Message: {state["raw_input"]}

Return only the category name.
"""
    res = llm.invoke(prompt)
    intent = res.content.strip().lower()

    return {
        "intent": intent,
        "user_id": state["user_id"],
        "raw_input": state["raw_input"],
        "messages": state.get("messages", [])
    }


def route_decision(state: DietState):
    if state["intent"] == "log_food":
        return "extract"
    if state["intent"] == "ask_history":
        return "query"
    if state["intent"] == "log_feeling":
        return "feeling"
    if state["intent"] == "log_glucose":
        return "glucose"
    if state["intent"] == "log_sleep":
        return "sleep"
    if state["intent"] == "log_exercise":
        return "exercise"
    return "advice"

def advice_node(state: DietState):
    prompt =f"""
You are an expert Health & Diabetes Reversal Coach.
Your goal is to help the user reverse diabetes through lifestyle changes, nutrition, sleep, and stress management.

User Input: {state["raw_input"]}

Guidelines:
1. Focus mainly on blood sugar stability, insulin sensitivity, and metabolic health.
2. Provide actionable, practical advice (e.g., walking after meals, prioritizing protein/fiber).
3. Be encouraging but firm about health goals.
4. If the question is about general health, relate it back to metabolic health and diabetes prevention/reversal.
5. Keep answers concise but informative.
"""
    res = llm.invoke(prompt)
    return {"result": res.content}

import re

def query_node(state: DietState):
    # 1) Ask LLM to write SQL only
    sql_prompt = f"""
You are a SQL generator.

Table:
food_log(
    user_id,
    meal_time,
    food_name,
    quantity,
    grams,
    calories,
    protein,
    carbs,
    fat,
    fiber,
    source_text
)

glucose_log(
    user_id,
    timestamp,
    glucose_level,
    context,
    notes
)

User question:
"{state["raw_input"]}"

Rules:
- Output ONE valid SQLite SELECT statement
- Always include: WHERE user_id = ?
- Do NOT include explanations, markdown, or text
- Output SQL only

Examples:
How much protein today?
SELECT SUM(protein) FROM food_log
WHERE user_id = ?
AND date(meal_time) = date('now');

What did I eat yesterday?
SELECT food_name, quantity FROM food_log
WHERE user_id = ?
AND date(meal_time) = date('now','-1 day');
"""

    sql_raw = llm.invoke(sql_prompt).content.strip()

    # Extract SQL if model adds extra text
    match = re.search(r"SELECT .*", sql_raw, re.IGNORECASE | re.DOTALL)
    if not match:
        return {"result": f"Could not generate SQL: {sql_raw}"}

    sql = match.group().strip()

    # 2) Run the SQL
    try:
        rows = conn.execute(sql, (state["user_id"],)).fetchall()
    except Exception as e:
        return {"result": f"Query error: {str(e)}"}

    if not rows or rows[0][0] is None:
        return {"result": "I couldn’t find any food data for that question."}

    # 3) Ask LLM to give advice using the data
    analysis_prompt = f"""
You are a Diabetes Reversal Nutritionist & Health Assistant.

User Question: "{state["raw_input"]}"
Data Results: {rows}

Task:
1. Answer the user's question clearly using the data provided.
2. Analyze the data specifically for diabetes management (e.g., Blood Sugar impact, Carb load, Protein adequacy).
3. If the values are high in carbs/sugar, gently warn and suggest alternatives.
4. If the values are good (high fiber/protein), validate and encourage the behavior.
5. Keep the response supportive, concise, and actionable. Do not mention SQL.
"""

    answer = llm.invoke(analysis_prompt).content.strip()

    return {"result": answer}


def evaluate_meal_node(state: DietState):
    foods = state["extracted_foods"]

    prompt = f"""
You are a nutrition coach helping someone reverse type 2 diabetes.

These foods were just eaten:
{foods}

Use these principles:
- Favor: protein, fiber, vegetables, healthy fats
- Avoid spikes: sugar, refined carbs, white bread, juice
- Stable blood sugar is the goal

Give short, supportive feedback:
1) Was this meal good or risky for blood sugar?
2) What was good?
3) What could be improved next time?

Do not mention databases or macros explicitly.Untill asked specifically

"""

    res = llm.invoke(prompt)
    return {"meal_feedback": res.content}

def feeling_node(state: DietState):
    # 1. Extract feeling details
    prompt = f"""
    You are a medical scribe.
    Extract the following from the user's text:
    1. Main symptom/feeling (e.g., "dizzy", "anxious", "pain").
    2. Intensity (1-10 scale), estimate if not stated (mild=3, severe=8).

    User Text: "{state["raw_input"]}"
    
    Return JSON only: {{ "feeling_type": "...", "intensity": 5 }}
    """
    structured_llm = llm.with_structured_output(dict) # Simplification: asking for dict directly or via JSON 
    extraction = llm.invoke(prompt).content
    
    # Simple cleanup to parse JSON if needed, or use robust extraction
    # For now, we assume LLM behaves or we use a regex (safer to use a Pydantic model usually, but keeping it simple as per style)
    import json
    try:
        data = json.loads(re.search(r"\{.*\}", extraction, re.DOTALL).group())
    except:
        data = {"feeling_type": "general", "intensity": 5}

    # 2. Save
    save_feeling(
        user_id=state["user_id"],
        feeling_type=data.get("feeling_type", "unknown"),
        intensity=data.get("intensity", 5),
        notes=state["raw_input"]
    )

    # 3. Give immediate context-aware feedback
    analysis_prompt = f"""
    You are a Diabetes Coach. The user just reported: {data.get('feeling_type')} (Intensity: {data.get('intensity')}).
    1. Briefly explain if this could be related to blood sugar (high or low).
    2. Suggest a safe immediate action (e.g., "Check glucose", "Drink water").
    3. Be calm and supportive.
    """
    advice = llm.invoke(analysis_prompt).content
    advice = llm.invoke(analysis_prompt).content
    return {"result": advice}

def glucose_node(state: DietState):
    # 1. Extract details from text
    prompt = f"""
    You are a medical scribe.
    Extract the following from the user's text:
    1. Glucose Level (float).
    2. Context (e.g., "fasting", "post-meal", "bedtime").
    
    User Text: "{state["raw_input"]}"
    
    Return JSON only: {{ "glucose_level": 120, "context": "fasting" }}
    """
    res = llm.invoke(prompt).content
    
    try:
        data = json.loads(re.search(r"\{.*\}", res, re.DOTALL).group())
    except:
        data = {"glucose_level": 0.0, "context": "unknown"}

    # 2. Save to DB
    save_glucose(
        user_id=state["user_id"],
        glucose_level=data.get("glucose_level", 0.0),
        context=data.get("context", "unknown"),
        notes=state["raw_input"]
    )

    # 3. Provide Advice
    analysis_prompt = f"""
    You are a Diabetes Coach. The user just reported glucose: {data.get("glucose_level")} ({data.get("context")}).
    1. Briefly analyze this level.
    2. Suggest a safe immediate action or positive reinforcement.
    """
    advice = llm.invoke(analysis_prompt).content
    return {"result": advice}

def sleep_node(state: DietState):
    # 1. Extract sleep details
    prompt = f"""
    You are a medical scribe.
    Extract the following from the user's text:
    1. Sleep Duration in hours (float).
    2. Quality Score (1-10 scale). Estimate if not stated (poor=3, normal=5, good=8).
    3. Notes (context like "interrupted", "napped").

    User Text: "{state["raw_input"]}"

    Return JSON only: {{ "duration_hours": 7.5, "quality_score": 7, "notes": "..." }}
    """
    
    res = llm.invoke(prompt).content

    try:
        data = json.loads(re.search(r"\{.*\}", res, re.DOTALL).group())
    except:
        data = {"duration_hours": 0.0, "quality_score": 5, "notes": "unknown"}

    # 2. Save to DB
    save_sleep(
        user_id=state["user_id"],
        duration_hours=data.get("duration_hours", 0.0),
        quality_score=data.get("quality_score", 5),
        notes=data.get("notes", "unknown")
    )

    # 3. Provide Advice
    analysis_prompt = f"""
    You are a Diabetes Coach. The user reported sleep:
    Duration: {data.get("duration_hours")} hrs
    Quality: {data.get("quality_score")}/10
    Notes: {data.get("notes")}
    
    1. Briefly analyze this sleep in context of diabetes/metabolic health.
    2. Suggest a quick meaningful tip.
    """
    advice = llm.invoke(analysis_prompt).content
    return {"result": advice}

def exercise_node(state: DietState):
    # 1. Extract exercise details
    prompt = f"""
    You are a fitness log assistant. 
    Extract the following details from the user's input:
    
    1. **exercise_type**: e.g. "Walking", "Lifting", "Yoga".
    2. **duration_minutes**: (float). Convert hours to minutes (e.g. "1 hr" -> 60). If not specified, estimate reasonable default (e.g. 30).
    3. **intensity**: "Low", "Medium", "High". Infer from context (e.g. "sprinted" -> High, "stroll" -> Low). Default to "Medium".
    4. **notes**: Any other details.
    
    User Input: "{state["raw_input"]}"
    
    Return strict JSON ONLY. No markdown, no explanations.
    Example: {{ "exercise_type": "Running", "duration_minutes": 45, "intensity": "High", "notes": "felt good" }}
    """
    res = llm.invoke(prompt).content

    try:
        # Robust JSON extraction
        json_str = re.search(r"\{.*\}", res, re.DOTALL).group()
        data = json.loads(json_str)
    except:
        data = {"exercise_type": "General", "duration_minutes": 30.0, "intensity": "Medium", "notes": "unknown"}

    # 2. Save
    save_exercise(
        user_id=state["user_id"],
        exercise_type=data.get("exercise_type", "General"),
        duration_minutes=data.get("duration_minutes", 30.0),
        intensity=data.get("intensity", "Medium"),
        notes=data.get("notes", "unknown")
    )

    # 3. Advice
    analysis_prompt = f"""
    You are a Diabetes Coach. User activity:
    {data.get("exercise_type")} for {data.get("duration_minutes")} mins ({data.get("intensity")} intensity).
    
    1. Acknowledge the activity.
    2. Explain ONE benefit of this specific exercise for blood glucose control.
    3. Keep it short and encouraging.
    """
    advice = llm.invoke(analysis_prompt).content
    return {"result": advice}

############# WorkFlow#####################


builder = StateGraph(DietState)

builder.add_node("extract", extract_foods)
builder.add_node("save", save_node)
builder.add_node("router", route_node)
builder.add_node("query", query_node) 
builder.add_node("advice", advice_node)
builder.add_node("evaluate",evaluate_meal_node)
builder.add_node("feeling", feeling_node)
builder.add_node("glucose", glucose_node)
builder.add_node("sleep", sleep_node)
builder.add_node("exercise", exercise_node)

builder.add_edge(START, "router")
builder.add_conditional_edges("router",route_decision)

builder.add_edge("extract", "evaluate")
builder.add_edge("evaluate", "save")
builder.add_edge("save", END)
builder.add_edge("query", END)
builder.add_edge("advice", END)
builder.add_edge("feeling", END)
builder.add_edge("glucose", END)
builder.add_edge("sleep", END)
builder.add_edge("exercise", END)
graph = builder.compile()
