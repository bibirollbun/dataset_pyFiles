# --- Setup and Imports ---
import os
import json
import uuid
import re
from typing import List, Dict, Any

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService 
from google.adk.tools import FunctionTool, ToolContext 
from google.genai import types


# Configure Model Retry
retry_config = types.HttpRetryOptions(
    attempts=5, exp_base=7, initial_delay=1, http_status_codes=[429, 500, 503, 504]
)


# Global identifiers for session context
USER_ID = "test_user"
APP_NAME = "meal_planner_app"

# Access the API Key
try:
    from kaggle_secrets import UserSecretsClient
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
except Exception:
    # Fallback if running locally and env var is already set
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("FATAL ERROR: GOOGLE_API_KEY not found.")


# --- Tools ---

# Calorie Database
CALORIE_DATABASE = {
    "chicken breast": 165, "rice": 206, "broccoli": 55, "salmon": 208,
    "sweet potato": 103, "mixed greens": 5, "avocado": 160, "egg": 78,
    "oats": 150, "banana": 105, "beef": 250, "pork": 242, "cod": 105,
    "tuna": 132, "shrimp": 99, "pasta": 131, "cheese": 402
}

def calculate_calories(ingredients_list: List[str]) -> Dict[str, Any]:
    """
    Tool: Calculates the total calorie count based on a list of primary ingredients.
    """
    total_calories = 0
    breakdown = {}
    
    for item in ingredients_list:
        item_lower = item.lower().strip()
        found = False
        for key, calories in CALORIE_DATABASE.items():
            if key in item_lower:
                total_calories += calories
                breakdown[item] = calories
                found = True
                break
        if not found:
            breakdown[item] = "Unknown (Estimate: 0)"

    return {
        "status": "success",
        "total_calories_estimate": total_calories,
        "breakdown": breakdown
    }

def save_dietary_focus(tool_context: ToolContext, focus: str) -> Dict[str, Any]:
    """
    Tool: Saves the user's primary dietary focus to the session state.
    """
    tool_context.state["user:diet:focus"] = focus
    return {"status": "success", "message": f"Dietary focus saved: {focus}"}

def get_user_focus(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Tool: Retrieves the user's stored dietary focus.
    """
    focus = tool_context.state.get("user:diet:focus", "No previous diet focus specified.")
    return {"dietary_focus": focus}

# Register the custom tools
CalorieCalculatorTool = FunctionTool(func=calculate_calories)
SaveDietaryFocusTool = FunctionTool(func=save_dietary_focus)
GetUserFocusTool = FunctionTool(func=get_user_focus)


# --- Agents ---

# Meal Planner 
meal_planner_agent = LlmAgent(
    name="MealPlanner",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config, api_key=GOOGLE_API_KEY),
    instruction="""You are a specialized 7-Day Meal Planner.
    
    CRITICAL SEQUENCE: 
    1. **Tool Use for Memory:** You MUST call the 'get_user_focus' tool to retrieve the user's established preferences. 
    2. **Tool Use for Saving:** If the user introduces a *new* diet focus, use the 'save_dietary_focus' tool.
    3. **Core Task (NO TOOL CALL):** **Based on the results from the 'get_user_focus' tool**, generate a 7-day dinner plan. 
    
    If the diet focus is 'low-carb' (from memory), you MUST include chicken and salmon meals in the plan.
    
    Your task is to generate the plan, and **your ONLY final response MUST be a STRICT JSON object containing the plan.**
    
    Output Format:
    {
      "meal_plan": [
        {"day": "Monday", "meal": "Grilled Salmon with Roasted Asparagus and Quinoa"},
        // ... more days ...
      ],
      "all_ingredients": [
        "Salmon", "Asparagus", "Quinoa", 
        // ... ALL unique, raw ingredients needed for the week. List each one once.
      ]
    }
    
    DO NOT include any text, explanations, or conversational filler outside the JSON block.
    """,
    tools=[SaveDietaryFocusTool, GetUserFocusTool], 
    output_key="weekly_meal_plan_data", 
)


# Grocery List Generator 
grocery_generator_agent = LlmAgent(
    name="GroceryListGenerator",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config, api_key=GOOGLE_API_KEY),
    instruction="""Analyze the raw JSON data in the **session state under the key 'weekly_meal_plan_data'**.
    
    1. Extract ONLY the 'all_ingredients' list from that data.
    2. Format this list into a clean, consolidated, unnumbered bullet-point grocery list. Use sensible quantities.
    3. Include a friendly introductory sentence.
    4. Do not include the meal plan details, only the ingredients list.
    """,
    output_key="final_grocery_list", 
)


# Calorie Calculator & Summarizer
calorie_summarizer_agent = LlmAgent(
    name="CalorieSummarizer",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config, api_key=GOOGLE_API_KEY),
    instruction="""You are a Health Summary Agent.
    
    Your task is to:
    1. Retrieve the 'all_ingredients' list from the session state (key: 'weekly_meal_plan_data').
    2. Pass this exact list of raw ingredients to the 'calculate_calories' tool.
    3. Use the tool's output to find the 'total_calories_estimate' for the week.
    4. Generate a polite, final response to the user that clearly summarizes the ENTIRE week's plan:
        - The overall goal/focus of the plan (read the session state key 'user:diet:focus').
        - The total estimated calorie count for the week (use the result from the tool).
        - Confirm that the full grocery list is ready (referencing the previous agent's work).
        
    Ensure the final output is conversational and easy to read.
    """,
    tools=[CalorieCalculatorTool],
    output_key="final_summary_report",
)


# Orchestration ---
root_agent = SequentialAgent(
    name="MealPlannerAssistant",
    sub_agents=[
        meal_planner_agent,      
        grocery_generator_agent, 
        calorie_summarizer_agent,
    ],
)


# Execution Logic ---

session_service = InMemorySessionService() 
runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)

async def run_planner(user_prompt: str, session_id: str):
    print(f"\n{'='*60}")
    print(f"STARTING SESSION: {session_id}")
    print(f"User > {user_prompt}\n")
    
    # Create or get the session
    try:
        await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
    except:
        pass 
    
    # Run the sequential agent
    query_content = types.Content(role="user", parts=[types.Part(text=user_prompt)])
    async for _ in runner.run_async(user_id=USER_ID, session_id=session_id, new_message=query_content):
         pass 

    # Re-fetch the session to get outputs
    session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)

    # Extract raw outputs
    raw_plan_json = session.state.get("weekly_meal_plan_data", "")
    final_summary = session.state.get("final_summary_report", "Error: Final Summary not found.")
    grocery_list = session.state.get("final_grocery_list", "Error: Grocery list not found.")
    
    # PARSE and PRINT the Meal Plan
    print("## Weekly Meal Plan ##")
    if raw_plan_json:
        try:
            # Clean up potential markdown formatting from LLM
            cleaned_json = raw_plan_json.replace("```json", "").replace("```", "").strip()
            plan_data = json.loads(cleaned_json)
            
            # Iterate and print meals
            if "meal_plan" in plan_data:
                for entry in plan_data["meal_plan"]:
                    print(f"* **{entry['day']}:** {entry['meal']}")
            else:
                print("No meal plan found in JSON data.")
        except json.JSONDecodeError:
            print("Error parsing meal plan JSON.")
    else:
        print("No meal plan data generated.")

    # Print remaining reports
    print("\n## Summary Report ##")
    print(final_summary)
    print("\n--- Consolidated Grocery List ---")
    print(grocery_list)
    
    print(f"\n[DEBUG] Memory State: {session.state}")
    print(f"{'='*60}\n")


# Initial Setup
# The agent will generate a plan, calculate calories, and SAVE 'low-carb' to memory.
await run_planner(
    "Generate a healthy 7-day dinner plan for me. I need it to be low-carb and include chicken and salmon.", 
    "user_preferences_session"
)


# Memory Test
# We give a vague prompt. The agent MUST look at memory, see 'low-carb', 
# and generate a compliant plan WITHOUT asking clarifying questions.
await run_planner(
    "Please generate a new 7-day plan, focusing on meals I already prefer.", 
    "user_preferences_session"
)


# --- Export Grocery List to CSV ---
import csv
import re

# 1. Retrieve the session data using global session_service and session ID
target_session_id = "user_preferences_session"

try:
    # Retrieve session asynchronously
    session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=target_session_id)
    
    # 2. Get the raw grocery list text from the session state
    raw_grocery_text = session.state.get("final_grocery_list", "")
    
    # 3. Parse the text to find list items (bullets starting with * or -)
    grocery_items = []
    if raw_grocery_text:
        for line in raw_grocery_text.split('\n'):
            line = line.strip()
            # Regex to match bullets (*, -) or numbers (1.)
            if re.match(r'^[\*\-]\s+', line) or re.match(r'^\d+\.\s+', line):
                # Remove the bullet/number and clean up
                clean_item = re.sub(r'^[\*\-]\s+|^\d+\.\s+', '', line).strip()
                grocery_items.append(clean_item)
    
    if grocery_items:
        print(f"Found {len(grocery_items)} grocery items to export.")
        
        # 4. Write to CSV
        filename = 'grocery_list.csv'
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Grocery Item']) # Header
            for item in grocery_items:
                writer.writerow([item])

        print(f"file generated successfully: {filename}")
        print("Check the 'Output' section of your notebook to download it.")
    else:
        print("No grocery items found in the session state to export.")
        print("Raw text found:", raw_grocery_text)

except Exception as e:
    print(f"❌ Error exporting to CSV: {e}")

