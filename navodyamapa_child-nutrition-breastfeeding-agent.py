# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import google.generativeai as genai
from kaggle_secrets import UserSecretsClient

api_key = UserSecretsClient().get_secret("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

for m in genai.list_models():
    print(m.name, " | ", m.supported_generation_methods)



#  Gemini setup (Google Generative AI) 
from kaggle_secrets import UserSecretsClient
import google.generativeai as genai

# 1. Load API key from Kaggle Secrets
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("GOOGLE_API_KEY")  

# 2. Configure Gemini
genai.configure(api_key=api_key)

# 3. Create a model object (we'll reuse this everywhere)
gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")


print("Gemini is configured âœ…")



EDU_DISCLAIMER = (
    "âš ï¸� This assistant is for educational purposes only and does NOT replace "
    "a doctor, hospital, or real WHO growth charts."
)

def call_gemini(system_instructions: str, user_message: str) -> str:
    """
    Helper to call Gemini with a small system prompt + user message.
    Returns plain text.
    """
    prompt = f"{system_instructions}\n\nUser question:\n{user_message}\n\n{EDU_DISCLAIMER}"
    response = gemini_model.generate_content(prompt)
    return response.text




session_state = {
    "child_profile": None,  # will store age, weight, height, etc.
    "history": []           # list of past recommendations / messages
}

def set_child_profile(age_months, weight_kg, height_cm,
                      diagnosis=None, feeding_type=None, preferences=None):
    """
    Store the child's basic information in the session state.
    This will be used by all agents (risk, meal plan, breastfeeding).
    """
    session_state["child_profile"] = {
        "age_months": age_months,
        "weight_kg": weight_kg,
        "height_cm": height_cm,
        "diagnosis": diagnosis,
        "feeding_type": feeding_type,
        "preferences": preferences or {}
    }
    return session_state["child_profile"]

print("STEP 1 complete: session state and set_child_profile() are ready.")



# Test Step 1
set_child_profile(
    age_months=18,
    weight_kg=8.0,
    height_cm=76,
    diagnosis="anemia",
    feeding_type="breastfeeding + complementary foods",
    preferences={"avoid_eggs": True}
)

session_state["child_profile"]




def compute_growth_status(age_months, weight_kg, height_cm):
    """
    âš ï¸� Very simplified, NON-MEDICAL demo.
    This is ONLY for educational purposes in this project.
    It does NOT replace real WHO growth charts or a doctor.

    Returns a dict with:
      - category: 'severe_risk' / 'moderate_risk' / 'at_risk' / 'normal' / 'unknown'
      - bmi: approximate BMI
      - message: short explanation
    """
    if age_months is None or weight_kg is None or height_cm is None:
        return {
            "category": "unknown",
            "message": "Not enough data to estimate risk."
        }

    # approximate BMI
    bmi = weight_kg / ((height_cm / 100) ** 2)

    if bmi < 13:
        category = "severe_risk"
        msg = "Measurements suggest a HIGH risk of undernutrition."
    elif bmi < 14:
        category = "moderate_risk"
        msg = "Measurements suggest a MODERATE risk of undernutrition."
    elif bmi < 15:
        category = "at_risk"
        msg = "Child may be slightly at risk. Continue close monitoring."
    else:
        category = "normal"
        msg = "Measurements look within a simple normal range for this demo tool."

    return {
        "category": category,
        "bmi": round(bmi, 2),
        "message": msg + " This is NOT a diagnosis. Please follow your doctor's advice."
    }


# --------- Global-style food database for meal planning ----------

FOODS = [
    # Protein sources
    {"name": "Egg",             "tags": ["protein", "iron"]},
    {"name": "Chicken",         "tags": ["protein"]},
    {"name": "Fish",            "tags": ["protein", "iron"]},
    {"name": "Lentils",         "tags": ["protein", "iron", "cheap"]},
    {"name": "Beans",           "tags": ["protein", "cheap"]},
    {"name": "Chickpeas",       "tags": ["protein", "iron"]},

    # Iron-rich plant foods
    {"name": "Spinach",         "tags": ["iron", "vitamin_c"]},
    {"name": "Moringa leaves",  "tags": ["iron", "vitamin_c"]},
    {"name": "Pumpkin leaves",  "tags": ["iron"]},

    # Staples (common worldwide)
    {"name": "Rice",            "tags": ["carbs"]},
    {"name": "Maize meal",      "tags": ["carbs"]},
    {"name": "Millet porridge", "tags": ["carbs"]},
    {"name": "Sorghum porridge","tags": ["carbs"]},

    # Fruits (vitamin C, useful for iron absorption)
    {"name": "Banana",          "tags": ["carbs", "vitamin_c"]},
    {"name": "Papaya",          "tags": ["vitamin_c"]},
    {"name": "Orange",          "tags": ["vitamin_c"]},

    # Dairy
    {"name": "Yogurt",          "tags": ["protein"]},
]

def get_foods_by_tags(required_tags):
    """
    Return foods that contain at least one of the required tags.
    Example: get_foods_by_tags(['iron', 'protein'])
    """
    required_tags = set(required_tags)
    return [
        f for f in FOODS
        if required_tags.intersection(f["tags"])
    ]

print("STEP 2 complete: tools (growth calculator + global food DB) are ready.")



profile = session_state["child_profile"]  # from Step 1 test, if you set it

print("Growth status:")
print(compute_growth_status(
    profile["age_months"],
    profile["weight_kg"],
    profile["height_cm"]
))

print("\nIron/protein-rich foods:")
get_foods_by_tags(["iron", "protein"])




def save_history(entry):
    session_state["history"].append(entry)

def risk_assessment_agent():
    profile = session_state.get("child_profile")

    if profile is None:
        return "â�— No child profile found. Please enter child details first."

    # 1) Use our simple Python tool
    risk = compute_growth_status(
        profile["age_months"],
        profile["weight_kg"],
        profile["height_cm"],
    )

    # 2) Build a small user message for Gemini
    user_message = (
        f"Child age: {profile['age_months']} months\n"
        f"Weight: {profile['weight_kg']} kg\n"
        f"Height: {profile['height_cm']} cm\n"
        f"Simplified BMI category: {risk['category']}\n"
        f"Tool message: {risk['message']}\n\n"
        "Please explain this in simple language for a parent. "
        "Be very clear that this is NOT medical advice and they must follow their doctor."
    )

    system_instructions = (
        "You are an educational nutrition assistant for parents of young children. "
        "You must:\n"
        "- Stay general and reassuring.\n"
        "- NEVER give diagnoses or treatment plans.\n"
        "- Always recommend seeing a health professional for real decisions.\n"
        "- Keep the answer short, friendly, and easy to read."
    )

    reply = call_gemini(system_instructions, user_message)

    # 3) Save to history
    session_state["history"].append({
        "agent": "risk_assessment",
        "user_message": "check risk",
        "reply": reply,
    })

    return reply



def meal_planner_agent():
    profile = session_state.get("child_profile")

    if profile is None:
        return "â�— No child profile found. Please enter child details first."

    # Use our food tools
    foods_protein = get_foods_by_tags(["protein"])
    foods_iron = get_foods_by_tags(["iron"])
    foods_vitc = get_foods_by_tags(["vitamin_c"])
    staples = get_foods_by_tags(["carbs"])

    base_plan = (
        "**1-Day Meal Plan (Demo)**\n\n"
        "Breakfast: "
        f"{staples[0]['name']} + {foods_protein[0]['name']}, "
        f"fruit: {foods_vitc[0]['name']}\n"
        "Lunch: "
        f"{staples[1]['name']} + {foods_protein[1]['name']}, "
        f"greens: {foods_iron[0]['name']}\n"
        "Snack: "
        f"{foods_vitc[1]['name']} or Yogurt\n"
        "Dinner: "
        f"{staples[2]['name']} + {foods_protein[2]['name']}, "
        f"greens: {foods_iron[1]['name']}\n"
    )

    user_message = (
        f"Child age: {profile['age_months']} months.\n"
        "Here is a rough skeleton of a 1-day meal plan using global foods:\n\n"
        f"{base_plan}\n\n"
        "Please rewrite this in bullet points for a parent. "
        "Keep it simple, emphasize iron, protein and vitamin C, "
        "and add a reminder that this is not individualized medical advice."
    )

    system_instructions = (
        "You are an educational assistant helping parents understand simple child meal plans. "
        "Use friendly tone, short bullets, and keep the foods I provided. "
        "Do not invent medical treatments or strict diets."
    )

    reply = call_gemini(system_instructions, user_message)

    session_state["history"].append({
        "agent": "meal_planner",
        "user_message": "meal plan",
        "reply": reply,
    })

    return reply

def breastfeeding_advisor_agent():
    profile = session_state.get("child_profile")

    if profile is None:
        return "â�— No child profile found. Please enter child details first."

    age = profile["age_months"]

    if age <= 6:
        base = (
            "Exclusive breastfeeding is usually recommended from 0â€“6 months. "
            "Feed on demand, about 8â€“12 times per day, and no water or other foods "
            "unless a health professional advises it."
        )
    elif age <= 24:
        base = (
            "From 6â€“24 months, breastfeeding can continue together with solid foods. "
            "Children often need 2â€“3 meals/day at 6â€“8 months and 3â€“4 meals/day at 9â€“24 months, "
            "including protein, iron-rich foods and fruits."
        )
    else:
        base = (
            "After 24 months, breastfeeding may continue if both mother and child want it, "
            "with a strong focus on balanced solid foods."
        )

    user_message = (
        f"Child age: {age} months.\n"
        f"Base guidance:\n{base}\n\n"
        "Please explain this clearly for a parent. "
        "Be supportive, non-judgmental, and remind them to follow local guidelines "
        "and their doctor or midwife."
    )

    system_instructions = (
        "You are a gentle breastfeeding education assistant. "
        "Explain calmly, avoid guilt, and never give strict medical orders."
    )

    reply = call_gemini(system_instructions, user_message)

    session_state["history"].append({
        "agent": "breastfeeding_advisor",
        "user_message": "breastfeeding help",
        "reply": reply,
    })

    return reply


print("STEP 3 complete: Risk Agent, Meal Plan Agent, and Breastfeeding Agent are ready.")



def orchestrator_agent(user_message: str) -> str:
    """
    Simple intent router.
    Chooses which Gemini-powered agent to call.
    """
    msg = user_message.lower().strip()

    # 1. No child profile yet?
    if session_state.get("child_profile") is None:
        return (
            "I don't have your child's details yet.\n"
            "Please set them using:\n"
            "set_child_profile(age_months, weight_kg, height_cm, diagnosis, feeding_type)\n\n"
            "Then ask things like:\n"
            "â€¢ 'check risk'\n"
            "â€¢ 'meal plan'\n"
            "â€¢ 'breastfeeding help'"
        )

    # 2. Show history
    if "history" in msg:
        if not session_state["history"]:
            return "No conversation history yet."
        lines = ["ğŸ“� Conversation history:"]
        for i, h in enumerate(session_state["history"], start=1):
            lines.append(f"{i}. Agent: {h['agent']}")
        return "\n".join(lines)

    # 3. Route to correct agent
    if any(word in msg for word in ["risk", "growth", "weight", "height", "z-score"]):
        agent_name = "risk_assessment"
        reply = risk_assessment_agent()

    elif any(word in msg for word in ["meal", "food", "diet", "plan", "menu"]):
        agent_name = "meal_planner"
        reply = meal_planner_agent()

    elif any(word in msg for word in ["breast", "breastfeed", "milk", "latch"]):
        agent_name = "breastfeeding_advisor"
        reply = breastfeeding_advisor_agent()

    else:
        # Default fallback message
        agent_name = "help"
        reply = (
            "I can help with:\n"
            "- Checking simple growth status (try: 'check risk')\n"
            "- Creating a simple meal plan (try: 'meal plan for today')\n"
            "- Giving general breastfeeding guidance (try: 'breastfeeding help')\n"
            "What would you like to do?"
        )

    # 4. Save to history
    session_state["history"].append({
        "agent": agent_name,
        "user_message": user_message,
        "reply": reply,
    })

    return reply

print("STEP 4 complete: Orchestrator agent is ready.")


    



# Make sure Step 1 ran:
# set_child_profile(...)

print(">>> User: check risk")
print(orchestrator_agent("Can you check my child's risk?"))
print("\n---------------------------\n")

print(">>> User: meal plan")
print(orchestrator_agent("Please give a meal plan for today"))
print("\n---------------------------\n")

print(">>> User: breastfeeding help")
print(orchestrator_agent("I need breastfeeding help"))




# STEP 5 â€“ DEMO / TEST CASES


# 1) Set a sample child profile (this also tests Step 1)
print("ğŸ§’ Setting child profile...\n")
set_child_profile(
    age_months=18,
    weight_kg=8.0,
    height_cm=76.0,
    diagnosis="anemia (reported by clinic)",
    feeding_type="breastfeeding + complementary foods",
    preferences={"avoid_eggs": False}
)
print("Child profile stored in session_state:\n", session_state["child_profile"])
print("\n" + "-"*70 + "\n")

# 2) Ask orchestrator to check risk
print(">>> User: \"Can you check my child's growth risk?\"\n")
response_risk = orchestrator_agent("Can you check my child's growth risk?")
print(response_risk)
print("\n" + "-"*70 + "\n")

# 3) Ask orchestrator for a meal plan
print(">>> User: \"Please give a meal plan for today.\"\n")
response_meal = orchestrator_agent("Please give a meal plan for today.")
print(response_meal)
print("\n" + "-"*70 + "\n")

# 4) Ask orchestrator for breastfeeding help
print(">>> User: \"I need breastfeeding help.\"\n")
response_bf = orchestrator_agent("I need breastfeeding help.")
print(response_bf)
print("\n" + "-"*70 + "\n")

# 5) Show conversation history recorded by agents
print(">>> User: \"Show history\"\n")
response_hist = orchestrator_agent("show history")
print(response_hist)
print("\nInternal history objects:")
for i, h in enumerate(session_state["history"], start=1):
    agent = h.get("agent", "?")
    text = h.get("reply", "")
    print(f"{i}. agent = {agent}, output length = {len(text)} characters")

print("\nâœ… Demo complete: multi-agent system works end-to-end.")




print(orchestrator_agent("check risk"))




print(orchestrator_agent("meal plan"))



print(orchestrator_agent("breastfeeding help"))




print(orchestrator_agent("show history"))

