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


import os
import sys
from google.colab import userdata

# --- API Key Configuration (Matches Day 1 Pattern) ---
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = api_key
    print("âœ… API Key found in Kaggle Secrets.")
except Exception:
    # Fallback for manual input if secrets aren't set
    print("âš ï¸� Key not found in secrets.")
    api_key = input("Please enter your Google API Key: ").strip()
    os.environ["GOOGLE_API_KEY"] = api_key

print("âœ… Setup complete.")


from google.adk.agents import Agent, LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, google_search
from google.adk.code_executors import BuiltInCodeExecutor
from google.genai import types

# Configure Retry Options (Standard Course Pattern)
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

print("âœ… ADK Components Imported.")


def calculate_tdee(weight_kg: float, height_cm: float, age: int, gender: str, activity_level: str) -> str:
    """
    Calculates Total Daily Energy Expenditure (TDEE) using the Mifflin-St Jeor equation.
    
    Args:
        weight_kg: Weight in kilograms.
        height_cm: Height in centimeters.
        age: Age in years.
        gender: 'male' or 'female'.
        activity_level: One of 'sedentary', 'light', 'moderate', 'active', 'very_active'.
        
    Returns:
        A descriptive string containing the BMR and TDEE values.
    """
    # 1. Calculate BMR
    if gender.lower() == 'male':
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
    else:
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161
        
    # 2. Apply Activity Multiplier
    multipliers = {
        'sedentary': 1.2,
        'light': 1.375,
        'moderate': 1.55,
        'active': 1.725,
        'very_active': 1.9
    }
    multiplier = multipliers.get(activity_level.lower(), 1.2)
    tdee = int(bmr * multiplier)
    
    # --- FIX: Return a descriptive sentence string ---
    # This prevents the Agent from thinking it's done; it prompts it to relay this info.
    return f"SUCCESS: The calculated BMR is {int(bmr)} calories/day. The TDEE (maintenance calories) is {tdee} calories/day."

print("Custom Tool 'calculate_tdee' defined.")


# --- Sub-Agent 1: The Fitness Coach ---
fitness_agent = LlmAgent(
    name="FitnessCoach",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    You are an expert Fitness Coach.
    Your goal is to create workout plans based on the user's goals and restrictions.
    Use the 'google_search' tool to find specific exercises if the user mentions unique equipment or injuries.
    Provide concise, actionable routines (sets/reps).
    """,
    tools=[google_search] 
)

# --- Sub-Agent 2: The Nutritionist ---
nutrition_agent = LlmAgent(
    name="Nutritionist",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    You are a Clinical Nutritionist.
    Your goal is to calculate calorie needs and suggest meal plans.
    
    CRITICAL INSTRUCTION:
    1. If the user provides body stats, USE the `calculate_tdee` tool.
    2. AFTER using the tool, you MUST generate a text response explaining the results to the user.
    3. DO NOT just stop after the tool call. You must speak the result.
    """,
    tools=[calculate_tdee]
)

print("Sub-Agents (Fitness & Nutrition) created.")


# --- The Manager: Vitality Orchestrator ---
# Uses AgentTool to call the sub-agents defined above.
orchestrator_agent = LlmAgent(
    name="VitalityOrchestrator",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    You are 'Vitality', a holistic Personal Health Orchestrator.
    You manage a team of specialists: a FitnessCoach and a Nutritionist.
    
    Your Workflow:
    1. Analyze the user's request.
    2. Delegate exercise questions to the `FitnessCoach`.
    3. Delegate diet/calorie questions to the `Nutritionist`.
    4. If the user asks for a comprehensive plan, call BOTH agents and synthesize their advice into one summary.
    
    Always be encouraging and professional.
    """,
    # This acts as the routing layer (Multi-Agent System Requirement)
    tools=[
        AgentTool(fitness_agent),
        AgentTool(nutrition_agent)
    ]
)

print("âœ… Root Orchestrator created with Sub-Agent tools.")


# Initialize Runner
runner = InMemoryRunner(agent=orchestrator_agent)

async def run_vitality_demo():
    print("Starting Vitality Health Session...\n")
    
    # --- Query 1 ---
    query1 = "Hi, I'm Alex. I am a 30 year old male, 180cm tall, and weigh 85kg. I work a desk job (sedentary)."
    print(f"USER: {query1}")
    response1 = await runner.run_debug(query1)
    
    # Handle response safely
    if response1 and response1[-1].content and response1[-1].content.parts:
        # Check if the last part is actually text (not a function call)
        last_part = response1[-1].content.parts[0]
        if hasattr(last_part, 'text'):
            print(f"VITALITY: {last_part.text}\n")
    
    # --- Query 2 ---
    query2 = "I want to lose weight. How many calories should I eat?"
    print(f"USER: {query2}")
    response2 = await runner.run_debug(query2)
    
    if response2 and response2[-1].content and response2[-1].content.parts:
        last_part = response2[-1].content.parts[0]
        if hasattr(last_part, 'text'):
            print(f"VITALITY: {last_part.text}\n")
    
    # --- Query 3 ---
    query3 = "I have a pair of dumbbells. Can you give me a simple full body workout?"
    print(f"USER: {query3}")
    response3 = await runner.run_debug(query3)
    
    if response3 and response3[-1].content and response3[-1].content.parts:
        last_part = response3[-1].content.parts[0]
        if hasattr(last_part, 'text'):
            print(f"VITALITY: {last_part.text}\n")

# Run the async function
await run_vitality_demo()




