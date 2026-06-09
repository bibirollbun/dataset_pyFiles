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


import pandas as pd
from datetime import datetime

# --- 1. Custom Tool Implementation: Nutrition Lookup ---
# This serves as your mock external API/database for the Analyzer Agent (A2)

MOCK_NUTRITION_DB = {
    "apple": {"calories": 95, "protein": 0.5, "fiber": 4.4},
    "chicken breast": {"calories": 165, "protein": 31.0, "fiber": 0.0},
    "brown rice": {"calories": 215, "protein": 4.5, "fiber": 3.5},
    "egg": {"calories": 70, "protein": 3.0, "fiber": 0.0}, 
    "toast": {"calories": 80, "protein": 1.5, "fiber": 1.0},
    "salmon": {"calories": 208, "protein": 20.0, "fiber": 0.0},
    "beans": {"calories": 113, "protein": 7.0, "fiber": 7.7},
    "default": {"calories": 100, "protein": 5.0, "fiber": 2.0} 
}


def lookup_nutrition_facts(food_item: str) -> dict:
    """
    CUSTOM TOOL: Looks up the nutritional data for a single food item.
    Called by the Analyzer Agent (A2).
    """
    # Normalize input and extract quantity
    item_key = food_item.lower().split(',')[0].strip() 
    data = MOCK_NUTRITION_DB.get(item_key, MOCK_NUTRITION_DB['default'])
    
    quantity = 1
    if 'two' in food_item.lower() or '2' in food_item.lower():
        quantity = 2
    elif 'three' in food_item.lower() or '3' in food_item.lower():
        quantity = 3

    return {
        "item": food_item,
        "calories": data["calories"] * quantity,
        "protein": data["protein"] * quantity,
        "fiber": data["fiber"] * quantity
    }

print("Custom Tool (lookup_nutrition_facts) Defined.")


# --- 2. Long-Term Memory Setup ---
# Long-Term Memory: User Profile and Meal History DataFrame
user_profile = {
    "goal": "Weight Loss",
    "target_calories": 2000,
    "target_protein_g": 100,
    "target_fiber_g": 30
}
meal_history_df = pd.DataFrame(columns=['date', 'meal_description', 'calories', 'protein', 'fiber'])

# Function to simulate saving to memory
def save_log_to_memory(new_log: dict):
    """Saves a new meal log to the global history DataFrame."""
    global meal_history_df
    new_row = pd.DataFrame([new_log])
    meal_history_df = pd.concat([meal_history_df, new_row], ignore_index=True)

# Add some sample history logs for demonstration (Memory Bank data)
sample_logs = [
    {'date': '2025-11-30 08:00', 'meal_description': 'old chicken breast', 'calories': 165, 'protein': 31.0, 'fiber': 0.0},
    {'date': '2025-12-01 13:00', 'meal_description': 'brown rice + salmon', 'calories': 423, 'protein': 24.5, 'fiber': 3.5}
]

for log in sample_logs:
    meal_history_df = pd.concat([meal_history_df, pd.DataFrame([log])], ignore_index=True)





# --- Context Engineering: Context Compaction ---
def get_context_compaction_summary(history_df: pd.DataFrame, profile: dict) -> dict:
    """
    Compacts the raw meal history into a concise summary to save LLM tokens (Context Compaction).
    """
    print("\n[CONTEXT COMPACTION] Compacting raw history data...")
    
    if history_df.shape[0] < 2:
        return {"average_protein": 0, "last_meal": "N/A", "goal_target_cal": profile['target_calories']}
    
    # Calculate overall average protein intake
    avg_protein = history_df['protein'].mean()
    
    # Get the last logged meal for immediate context
    last_meal_desc = history_df.iloc[-1]['meal_description']
    
    # Return a concise summary
    summary = {
        "average_protein": round(avg_protein, 1),
        "last_meal": last_meal_desc,
        "goal_target_cal": profile['target_calories']
    }
    
    print(f"[CONTEXT COMPACTION] Summary: Avg Protein={summary['average_protein']}g")
    return summary

print("Long-term Memory Initialized and Context Compaction Logic Defined.")


# --- 3. Sequential Multi-Agent System (A1, A2, A3, A4) ---

# A1: Logging Agent
def logging_agent_parse(user_input: str) -> list:
    """A1: Parses user text into a structured list of food items (Conceptual LLM parsing)."""
    print(f"\n[A1: LOGGING AGENT] Received input: '{user_input}'")
    
    # Placeholder Logic: LLM should enforce structure.
    parsed_items = []
    
    if 'apple' in user_input.lower(): parsed_items.append("apple")
    if 'egg' in user_input.lower(): parsed_items.append("two eggs")
    if 'chicken' in user_input.lower() or 'sandwich' in user_input.lower(): parsed_items.append("chicken breast")
    if 'toast' in user_input.lower(): parsed_items.append("toast")
    if 'beans' in user_input.lower(): parsed_items.append("beans")
    
    if not parsed_items: return ["default"]
        
    print(f"[A1] Structured items for A2: {parsed_items}")
    return parsed_items

# A2: Analyzer Agent
def analyzer_agent_analyze(parsed_items: list) -> dict:
    """A2: Calls the Custom Tool and calculates totals."""
    print("\n[A2: ANALYZER AGENT] Starting nutritional analysis...")
    
    total_cal, total_protein, total_fiber = 0, 0, 0
    
    for item in parsed_items:
        # *** TOOL CALL DEMONSTRATION ***
        facts = lookup_nutrition_facts(item) 
        total_cal += facts['calories']
        total_protein += facts['protein']
        total_fiber += facts['fiber']
        print(f"  - Analyzed '{facts['item']}': {facts['calories']} Kcal, {facts['protein']}g Protein")
        
    log = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "meal_description": " + ".join(parsed_items),
        "calories": total_cal,
        "protein": total_protein,
        "fiber": total_fiber
    }
    print(f"[A2] Analysis Complete. Total Calories: {total_cal}")
    return log

# A4: Evaluation Agent (for the Loop)
def evaluation_agent_check(suggestion: str) -> tuple:
    """
    A4: Simulates user feedback for the Loop Agent structure.
    Returns (is_accepted, rejection_reason)
    """
    print("\n[A4: EVALUATION AGENT] Checking user feedback on suggestion...")
    
    # Simulate Rejection on the first pass (if salmon is suggested)
    if 'salmon' in suggestion.lower() and 'REVISION' not in suggestion:
        print("[A4] Simulating User Feedback: 'I don't like fish/salmon.' -> REJECTED")
        return False, "User stated they dislike fish/salmon and asked for a non-fish protein source."
    else:
        print("[A4] Simulating User Feedback: 'That sounds great, thanks!' -> ACCEPTED")
        return True, "Accepted"


# A3: Coach Agent
def coach_agent_respond(analyzed_log: dict, context_summary: dict, feedback: str="") -> str:
    """
    A3: Updates Memory, considers feedback, and generates personalized coaching.
    """
    
    print(f"\n[A3: COACH AGENT] Generating response. Feedback: {bool(feedback)}")
    
    # 1. Update Memory (Only save the log if it's the initial run)
    if not feedback: 
        save_log_to_memory(analyzed_log)
        print(f"[A3] New log saved to Long-term Memory: {analyzed_log['calories']} Kcal")
    
    # 2. Check against goal using COMPACTED CONTEXT
    suggestion = ""
    avg_protein = context_summary['average_protein']
    
    if feedback:
        print(f"[A3] Revising suggestion based on feedback: '{feedback}'")
        suggestion += f"[REVISION] Based on your feedback ({feedback}), here is a new suggestion: "
        if 'fish' in feedback:
            suggestion += "Since you dislike fish, try adding **beans** and a side of **chicken breast** to your dinner for a lean, high-fiber, non-fish protein source."
        else:
            suggestion += "Try adding a handful of **walnuts or beans** to your next meal."
            
    else:
        # Suggest salmon/fish based on low historical protein
        suggestion += "Your current meal is insufficient for your high protein goal. "
        suggestion += "To boost your historical average (currently 27.8g), try a portion of **salmon** with a side of brown rice next."
        
    # Context Compaction is used here in the final output
    return f"**Coach's Feedback (Avg Protein: {avg_protein}g):**\n{suggestion}"


## 4. Running the Agent Pipeline (Demonstration)

def run_nutrition_coach(user_input: str):
    """Executes the full Sequential Multi-Agent System with a Loop."""
    
    print("=========================================================")
    print(f"       STARTING COACH RUN for: '{user_input}'")
    print("=========================================================")
    
    # Agent 1: Logging Agent
    parsed_items = logging_agent_parse(user_input)
    
    # Agent 2: Analyzer Agent (A2 executes the Custom Tool)
    analyzed_log = analyzer_agent_analyze(parsed_items)
    
    # Agent 3&4 LOOP START (Demonstrates Loop Agents)
    is_accepted = False
    rejection_reason = ""
    run_count = 0
    
    while not is_accepted and run_count < 3: # limit loop for safety
        run_count += 1
        print(f"\n--- LOOP ITERATION {run_count} ---")
        
        #Context Compaction (Executed before A3)
        context_summary = get_context_compaction_summary(meal_history_df, user_profile)
        
        # Agent 3: Coach Agent (core decision-maker)
        final_response = coach_agent_respond(analyzed_log, context_summary, rejection_reason)
        print("\n" + final_response)
        
        # Agent 4: Evaluation Agent (checks if we need to loop)
        is_accepted, rejection_reason = evaluation_agent_check(final_response)
        
        if is_accepted:
            print("\n[LOOP TERMINATED] Coach suggestion accepted.")
            break
            
        print("\n[LOOP CONTINUING] Re-running Coach Agent with rejection feedback...")
        
    print("=========================================================")
    print("DEMO END")
    print("=========================================================")
    
# --- Test Run ---
# This run will trigger the loop agent because the initial suggestion includes 'salmon' 
# and the A4 Evaluation Agent is programmed to reject 'salmon' once.
run_nutrition_coach("I had a big breakfast: two eggs and two pieces of toast.")

# Show the Long-term Memory update
print("\n--- Current Long-term Meal History (Proof of updated Memory) ---")
print(meal_history_df.tail(4))

