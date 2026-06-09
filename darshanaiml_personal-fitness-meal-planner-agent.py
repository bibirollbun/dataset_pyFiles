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

user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("GOOGLE_API_KEY")

genai.configure(api_key=api_key)

models = genai.list_models()
for m in models:
    print(m.name)


model = genai.GenerativeModel("models/gemini-2.0-flash")
print("MODEL LOADED")


response = model.generate_content("Say READY")
print(response.text)


# ---------------------------
# FITNESS & MEAL PLANNER AGENT
# ---------------------------

system_prompt = """
You are a Personal Fitness & Vegetarian Meal Planner AI.

Your tasks:
1. Understand user's fitness goal (fat loss, muscle gain, general fitness).
2. Generate a simple workout plan (beginner level).
3. Generate a vegetarian meal plan (Indian veg allowed).
4. Adjust based on equipment, time, and difficulty.

Output format:
- Goal summary
- Workout plan
- Meal plan
- Tips
"""

def create_agent_response(user_input):
    final_prompt = system_prompt + "\nUser: " + user_input
    response = model.generate_content(final_prompt)
    return response.text

# Test the agent
test_output = create_agent_response(
    "I want to lose fat. I have dumbbells only. I am vegetarian. Give me 1-day workout and meal plan."
)

print(test_output)


import google.generativeai as genai

def make_workout_tool(goal, equipment, time, level):
    return f"""Workout Plan
Goal: {goal}
Equipment: {equipment}
Time: {time}
Level: {level}

Plan:
1. Warm-up (3â€“5 min)
2. Main workout (based on equipment)
3. Cooldown (3 min)
"""

def make_meal_tool(diet, calories):
    return f"""Meal Plan
Diet: {diet}
Calories: {calories}

Breakfast: Oats + Fruits  
Lunch: Rice + Dal + Veggies  
Dinner: Roti + Paneer + Salad  
Snacks: Nuts / Fruits
"""

tools = {
    "workout_tool": make_workout_tool,
    "meal_tool": make_meal_tool,
}
print("TOOLS READY")



# Simple Fitness & Meal Planner Agent (no SDK tools, Python-orchestrated)

system_prompt = """
You are a Personal Fitness & Vegetarian Meal Planner AI.

You:
- Read the user's goal, equipment, and time.
- Use the workout and meal helpers already defined in Python.
- Then write a clear, structured plan.

Always answer in this structure:
1. Goal Summary
2. Workout Plan
3. Meal Plan
4. Tips
"""

def fitness_meal_agent(goal, equipment, time_available, level, diet_type, calories):
    # Use our Python helper "tools"
    workout = make_workout_tool(goal, equipment, time_available, level)
    meals = make_meal_tool(diet_type, calories)

    prompt = f"""{system_prompt}

Here is auto-generated workout data:
{workout}

Here is auto-generated meal data:
{meals}

Now rewrite everything nicely for the user.
"""

    response = model.generate_content(prompt)
    return response.text

# Test the agent once
output = fitness_meal_agent(
    goal="Fat loss",
    equipment="Dumbbells only",
    time_available="45 minutes",
    level="Beginner",
    diet_type="Indian vegetarian",
    calories="1800â€“2000 kcal"
)

print(output)



# Demo 2 â€“ Different goal

output2 = fitness_meal_agent(
    goal="Muscle gain",
    equipment="No equipment (bodyweight only)",
    time_available="30 minutes",
    level="Beginner",
    diet_type="Indian vegetarian, high protein",
    calories="2200â€“2400 kcal"
)

print(output2)


# ----------------------------------------------
# USER TEST CELL â€“ Run your own custom prompts
# ----------------------------------------------

def run_agent(
    goal="Fat loss",
    equipment="Dumbbells only",
    time_available="45 minutes",
    level="Beginner",
    diet_type="Indian vegetarian",
    calories="1800â€“2000 kcal"
):
    prompt = f"""
Goal: {goal}
Equipment: {equipment}
Time: {time_available}
Level: {level}
Diet: {diet_type}
Calories: {calories}

Generate a workout plan + meal plan.
"""

    # Use longer timeout so Kaggle doesn't stop it
    response = model.generate_content(
        prompt,
        request_options={"timeout": 60}
    )

    return response.text


# Example Test Run
output_test = run_agent(
    goal="Fat loss",
    equipment="Dumbbells only",
    time_available="45 minutes",
    level="Beginner",
    diet_type="Indian vegetarian",
    calories="1800â€“2000 kcal"
)

print(output_test)


# ----------------------------------------------
# INTERACTIVE CHAT MODE (disabled for commit)
# ----------------------------------------------

ENABLE_CHAT = False   # <-- change to True only when YOU want to chat manually

if ENABLE_CHAT:
    print("Fitness & Meal Planner Agent is ready! Type 'exit' to stop.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit", "stop"]:
            print("Agent: Goodbye! Stay consistent, stay strong!")
            break

        prompt = f"""
You are a personal fitness and vegetarian meal planning assistant.

User says: {user_input}

Respond like a friendly fitness coach.
Give helpful details.
Ask follow-up questions if needed.
"""

        response = model.generate_content(
            prompt,
            request_options={"timeout": 60}
        )

        print("\nAgent:", response.text, "\n")

