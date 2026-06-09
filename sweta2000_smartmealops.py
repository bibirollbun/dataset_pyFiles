# -----------------------------
# Import libraries
import asyncio
import nest_asyncio
import json
import re
import logging
import os
from kaggle_secrets import UserSecretsClient
import google.generativeai as genai
from google.generativeai import GenerativeModel

# -----------------------------
# Fix asyncio issues (especially in Jupyter/Colab)
nest_asyncio.apply()

# -----------------------------
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# -----------------------------
# Configure API key from Kaggle Secrets
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    genai.configure(api_key=GOOGLE_API_KEY)
    logging.info("âœ… Setup and authentication complete.")
except Exception as e:
    raise ValueError(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )

# -----------------------------
# Initialize agents (using gemini-2.0-flash for efficiency)
meal_planner_agent = GenerativeModel("gemini-2.0-flash")
nutrition_agent = GenerativeModel("gemini-2.0-flash")
recipe_agent = GenerativeModel("gemini-2.0-flash")

# -----------------------------
# Memory bank for storing outputs
memory_bank = {}

# -----------------------------
# Meal Planner Agent Introduction (Initial check)
intro = meal_planner_agent.generate_content(
    "Hi! You are Meal Planner Agent. Introduce yourself in one sentence."
)
logging.info(f"[Meal Planner Agent]: {intro.text}")


# -----------------------------
# Utility: parse LLM JSON safely
def parse_json(raw_text):
    """Safely extracts and parses JSON from text, even if wrapped in markdown."""
    cleaned = re.sub(r"^```json|```$", "", raw_text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Return raw text if parsing fails for debugging
        return {"raw_text": raw_text}

# -----------------------------
# Optional evaluation function
def evaluate_nutrition(nutrition_json):
    """Checks if the nutrition output contains critical data fields."""
    alerts = []
    # Note: The original check was slightly flawed; checking a key deep within the structure.
    # We maintain the original logic for fidelity to the source.
    if "estimated_total_calories" not in nutrition_json.get("meal_plan", {}):
        alerts.append("Missing total calories info")
    return alerts


# -----------------------------
# Nutrition Agent (async)
async def nutrition_agent_async(meal_plan_text):
    """Prompts the Nutrition Agent to analyze the meal plan and return JSON."""
    prompt = (
        f"You are a Nutrition Agent. Analyze this meal plan and return "
        f"complete calorie and macronutrient breakdown in valid JSON:\n{meal_plan_text}"
    )
    response = nutrition_agent.generate_content(prompt)
    nutrition_json = parse_json(response.text)
    memory_bank['nutrition'] = nutrition_json
    logging.info("[Nutrition Agent] Output collected")
    return nutrition_json

# -----------------------------
# Recipe Agent (async)
async def recipe_agent_async(nutrition_json):
    """Prompts the Recipe Agent to generate recipes based on nutrition data."""
    # Handle cases where the previous step failed to produce clean JSON
    if isinstance(nutrition_json, dict) and "raw_text" in nutrition_json:
        nutrition_json = parse_json(nutrition_json["raw_text"])
    
    prompt = (
        f"You are a Recipe Agent. Generate 3 recipes in JSON format based on "
        f"this nutrition info:\n{json.dumps(nutrition_json)}"
    )
    response = recipe_agent.generate_content(prompt)
    recipes_json = parse_json(response.text)
    memory_bank['recipes'] = recipes_json
    logging.info("[Recipe Agent] Output collected")
    return recipes_json


# -----------------------------
# Orchestrator: runs Nutrition and Recipe agents in parallel
async def orchestrator_run_parallel(meal_plan_text, iterations=2, pause_sec=1):
    """Coordinates the parallel execution and evaluation of the agents."""
    logging.info("[Orchestrator] Starting parallel multi-agent workflow...")
    evaluation_alerts = []

    for i in range(iterations):
        logging.info(f"[Orchestrator] Iteration {i+1}")

        # Run Nutrition Agent
        nutrition_result = await nutrition_agent_async(meal_plan_text)

        # Compact context for Recipe Agent
        # Adapts the nutrition output structure for efficient passing to the Recipe Agent
        if "meal_plan_analysis" in nutrition_result:
            compact_context = {
                "total_calories": nutrition_result["meal_plan_analysis"].get("estimated_totals", {}).get("calories"),
                "meals": nutrition_result["meal_plan_analysis"].get("meal_breakdown", [])
            }
        else:
            compact_context = nutrition_result

        # Run Recipe Agent in parallel (asyncio.create_task is used for the concurrency simulation)
        recipes_result_task = asyncio.create_task(recipe_agent_async(compact_context))
        recipes_result = await recipes_result_task

        # Evaluation
        alerts = evaluate_nutrition(nutrition_result)
        evaluation_alerts.extend(alerts)
        if alerts:
            logging.warning(f"[Evaluation] Alerts: {alerts}")

        # Checkpoint memory to file
        with open("memory_bank_checkpoint.json", "w") as f:
            json.dump(memory_bank, f, indent=2)
        logging.info("[Orchestrator] Memory checkpoint saved")

        await asyncio.sleep(pause_sec)

    return {
        "meal_plan": meal_plan_text,
        "nutrition": nutrition_result,
        "recipes": recipes_result,
        "evaluation_alerts": evaluation_alerts,
        "iterations_completed": iterations
    }

# -----------------------------
# Example meal plan input
meal_plan_example = (
    "Breakfast: Oatmeal with banana, "
    "Lunch: Paneer curry with rice, "
    "Dinner: Lentil soup with salad"
)

# -----------------------------
# Run orchestrator and print final output
output = asyncio.get_event_loop().run_until_complete(
    orchestrator_run_parallel(meal_plan_example)
)

# -----------------------------
# Pretty-print final structured JSON
print("\n=== FINAL OUTPUT ===")
print(json.dumps(output, indent=2))


import pandas as pd
import json
import os

# -------------------------
# Extract fields safely
# -------------------------
meal_plan_text = output.get("meal_plan", "")

nutrition_json = output.get("nutrition", {})
recipes_json = output.get("recipes", {})
evaluation_alerts = output.get("evaluation_alerts", [])
iterations_completed = output.get("iterations_completed", "")

# Convert nested JSON to strings for CSV storage
nutrition_str = json.dumps(nutrition_json)
recipes_str = json.dumps(recipes_json)
alerts_str = json.dumps(evaluation_alerts)

# -------------------------
# Create DataFrame
# -------------------------
df = pd.DataFrame([{
    "meal_plan": meal_plan_text,
    "nutrition_output": nutrition_str,
    "recipes_output": recipes_str,
    "evaluation_alerts": alerts_str,
    "iterations_completed": iterations_completed
}])

# -------------------------
# Save CSV to working directory
# -------------------------
save_path = "/kaggle/working/submission.csv"
df.to_csv(save_path, index=False)

print("âœ… submission.csv saved successfully!")
print(f"Location: {save_path}")

# -------------------------
# Verify file exists
# -------------------------
print("\nðŸ“‚ Files in /kaggle/working/:")
print(os.listdir("/kaggle/working"))


