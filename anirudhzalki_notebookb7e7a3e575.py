import os
import json
import pandas as pd
from datetime import datetime, timedelta
import asyncio

# Google & Auth Imports
from google.oauth2 import service_account
from googleapiclient.discovery import build
from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types
#from discord_webhook import DiscordWebhook

# Kaggle Secrets
from kaggle_secrets import UserSecretsClient
import nest_asyncio
nest_asyncio.apply()

# --- 1. Configuration & Secrets ---

print("ğŸ”‘ Loading Secrets...")
try:
    secrets = UserSecretsClient()
    #D-ISCORD_WEBHOOK_URL = secrets.get_secret("DISCORD_WEBHOOK_URL")
    SPREADSHEET_ID = secrets.get_secret("GOOGLE_SHEET_ID")
    MY_API_KEY = secrets.get_secret("GEMINI_API_KEY")
    
    # We get the JSON string from secrets and write it to a real file
    service_account_json_string = secrets.get_secret("GCP_SERVICE_ACCOUNT")
    
    if service_account_json_string is None:
        raise ValueError("GCP_SERVICE_ACCOUNT secret is missing!")
        
    SERVICE_ACCOUNT_FILE = 'service_account.json'
    with open(SERVICE_ACCOUNT_FILE, "w") as f:
        f.write(service_account_json_string)
    print("âœ… Successfully created service_account.json from secrets.")
    
except Exception as e:
    print(f"âš ï¸� Error loading secrets: {e}")
    print("Ensure you have added GEMINI_API_KEY, GOOGLE_SHEET_ID, and GCP_SERVICE_ACCOUNT in Add-ons -> Secrets")
    # Stop execution if secrets are missing
    raise e

# Constants
SHEET_RANGE = "Sheet1!A:K"

# Gemini Config
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# Initialize Model
model = Gemini(
    model="gemini-2.5-flash", 
    api_key=MY_API_KEY, 
    retry_config=retry_config
)


# --- 2. Tool Definitions ---

def fetch_recent_grocery_data():
    """
    Connects to the Google Sheet with SERVER-SIDE filtering.
    Fetches items from the last 4 days for specific categories.
    """
    print(f"ğŸ“Š Connecting to Google Sheet ID: {SPREADSHEET_ID}...")
    
    cutoff_date = datetime.now() - timedelta(days=4)
    
    try:
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, 
            scopes=['https://www.googleapis.com/auth/spreadsheets']
        )
        service = build('sheets', 'v4', credentials=creds)
        
        main_sheet = SHEET_RANGE.split('!')[0] if '!' in SHEET_RANGE else 'Sheet1'
        helper_sheet = 'MealPlannerFilteredData'
        
        # QUERY formula for server-side filtering
        query_formula = f"""=QUERY({main_sheet}!A:K, "SELECT A, B, C, D, E, F, G, H, I, J, K WHERE (K > {cutoff_date.year} OR (K = {cutoff_date.year} AND J > {cutoff_date.month}) OR (K = {cutoff_date.year} AND J = {cutoff_date.month} AND I >= {cutoff_date.day})) AND (D contains 'Vegetable' OR D contains 'Spice' OR D contains 'Condiment' OR D contains 'Poultry')", 1)"""
        
        # Check/Create Helper Sheet
        spreadsheet = service.spreadsheets().get(spreadsheetId=SPREADSHEET_ID).execute()
        sheets = spreadsheet.get('sheets', [])
        helper_exists = any(sheet['properties']['title'] == helper_sheet for sheet in sheets)
        
        if not helper_exists:
            print(f"ğŸ“� Creating helper sheet '{helper_sheet}'...")
            request_body = {
                'requests': [{'addSheet': {'properties': {'title': helper_sheet}}}]
            }
            service.spreadsheets().batchUpdate(spreadsheetId=SPREADSHEET_ID, body=request_body).execute()
        
        # Update Query
        print(f"ğŸ”„ Updating QUERY formula...")
        service.spreadsheets().values().update(
            spreadsheetId=SPREADSHEET_ID,
            range=f"{helper_sheet}!A1",
            valueInputOption='USER_ENTERED',
            body={'values': [[query_formula]]}
        ).execute()
        
        # Wait for calculation
        import time
        time.sleep(2)
        
        # Fetch Result
        print(f"ğŸ“¥ Fetching filtered data from server...")
        result = service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=f"{helper_sheet}!A2:K"
        ).execute()
        
        values = result.get('values', [])
        
        if not values:
            print("âš ï¸� No ingredients found in the last 4 days")
            return "No ingredients found in the last 4 days. Check if 'CATEGORY' column matches target categories."
        
        inventory_list = []
        for row in values:
            if len(row) >= 6: # Basic check to ensure columns exist
                item = row[1]
                qty = row[4]
                unit = row[5]
                category = row[3]
                date_str = row[0]
                inventory_list.append(f"{item} ({qty} {unit}) - {category} - bought on {date_str}")
        
        print(f"âœ… Found {len(inventory_list)} ingredients from last 4 days")
        return "\n".join(inventory_list)
        
    except Exception as e:
        print(f"â�Œ Error with server-side filtering: {str(e)}")
        print("âš ï¸� Falling back to client-side filtering...")
        return fetch_recent_grocery_data_fallback()

def fetch_recent_grocery_data_fallback():
    """Fallback method using client-side filtering."""
    print(f"ğŸ“Š Using client-side filtering fallback...")
    
    try:
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, 
            scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
        )
        service = build('sheets', 'v4', credentials=creds)
        
        sheet = service.spreadsheets()
        result = sheet.values().get(spreadsheetId=SPREADSHEET_ID, range=SHEET_RANGE).execute()
        values = result.get('values', [])
        
        if not values:
            return "No data found in the spreadsheet."
        
        expected_cols = ["DATE", "ITEM", "STORE", "CATEGORY", "QTY", "UNIT", "PRICE", "COMMENT", "DAY", "MONTH", "YEAR"]
        
        # Robust dataframe creation
        if len(values) > 1:
            df = pd.DataFrame(values[1:], columns=expected_cols[:len(values[0])])
        else:
            return "Spreadsheet is empty."
        
        # Ensure DATE column exists and is parsed
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
            four_days_ago = datetime.now() - timedelta(days=4)
            recent_items = df[df['DATE'] >= four_days_ago]
            
            target_categories = ["Vegetable", "Spice", "Condiment", "Poultry"]
            category_mask = recent_items['CATEGORY'].astype(str).str.contains(
                '|'.join(target_categories), case=False, na=False
            )
            filtered_items = recent_items[category_mask]
            
            inventory_list = []
            for _, row in filtered_items.iterrows():
                inventory_list.append(
                    f"{row['ITEM']} ({row['QTY']} {row['UNIT']}) - {row['CATEGORY']}"
                )
            
            if not inventory_list:
                return "No ingredients found in the last 4 days."
            return "\n".join(inventory_list)
        else:
            return "DATE column missing in spreadsheet."
            
    except Exception as e:
        return f"Fallback failed: {str(e)}"

def read_memory_bank():
    try:
        with open("memory_bank.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"dislikes": [], "favorites": ["Daal Chawal", "Chicken Handi White"], "last_7_days_suggestions": []}

def write_memory_bank(data: dict):
    with open("memory_bank.json", "w") as f:
        json.dump(data, f, indent=2)
    return "Memory bank updated successfully."

def update_preferences(favorite: str = None, dislike: str = None):
    memory = read_memory_bank()
    if favorite and favorite not in memory.get("favorites", []):
        memory.setdefault("favorites", []).append(favorite)
        print(f"âœ… Added '{favorite}' to favorites")
    if dislike and dislike not in memory.get("dislikes", []):
        memory.setdefault("dislikes", []).append(dislike)
        print(f"â�Œ Added '{dislike}' to dislikes")
    write_memory_bank(memory)
    return f"Preferences updated."

def save_selected_meal(meal_name: str):
    memory = read_memory_bank()
    suggestion_entry = {"meal": meal_name, "date": datetime.now().strftime("%Y-%m-%d")}
    
    if "last_7_days_suggestions" not in memory:
        memory["last_7_days_suggestions"] = []
        
    memory["last_7_days_suggestions"].append(suggestion_entry)
    
    cutoff_date = datetime.now() - timedelta(days=7)
    memory["last_7_days_suggestions"] = [
        entry for entry in memory["last_7_days_suggestions"]
        if datetime.strptime(entry["date"], "%Y-%m-%d") >= cutoff_date
    ]
    
    write_memory_bank(memory)
    print(f"ğŸ’¾ Saved '{meal_name}' to meal history")
    return f"Saved '{meal_name}' to last 7 days suggestions."

def send_discord_notification(message: str):
    print("ğŸ“¨ Sending notification to Discord...")
    webhook = DiscordWebhook(url=DISCORD_WEBHOOK_URL, content=message)
    response = webhook.execute()
    return f"Notification sent. Status: {response.status_code}"

# --- 3. Agents ---

data_agent_prompt = """
You are the **Inventory & Context Manager**.
1. Call `fetch_recent_grocery_data` to see what was bought recently.
2. Call `read_memory_bank` to see what was suggested in the last 7 days.
3. Report: AVAILABLE INGREDIENTS and FORBIDDEN MEALS.
"""

data_agent = Agent(
    name="KitchenManager",
    model=model,
    instruction=data_agent_prompt,
    tools=[fetch_recent_grocery_data, read_memory_bank],
    output_key="kitchen_state",
)

planner_agent_prompt = """
You are the **Creative Chef**.
Generate 3 distinct Lunch Options based **ONLY** on {kitchen_state}.
Constraints:
1. MUST use at least one 'Available Ingredient'.
2. MUST NOT be a 'Forbidden Meal'.
3. Prioritize 'Favorites'.
Output format:
1. [Meal Name] - [Main Ingredients] - [Reason]
"""

planner_agent = Agent(
    name="CreativeChef",
    model=model,
    instruction=planner_agent_prompt,
    tools=[update_preferences],
    output_key="meal_options"
)

selection_agent_prompt = """
You are the **Final Decision Maker**.
Review: {meal_options}
1. Pick the SINGLE best lunch option.
2. Use `save_selected_meal` to record it.
3. Draft a message and use `send_discord_notification`.
"""

selection_agent = Agent(
    name="DecisionMaker",
    model=model,
    instruction=selection_agent_prompt,
    tools=[save_selected_meal, send_discord_notification],
)

# --- 4. Execution ---

async def run_meal_planner():
    print("ğŸš€ Starting Agentic Meal Planner...")
    
    root_agent = SequentialAgent(
        name="MealPlanner",
        sub_agents=[data_agent, planner_agent, selection_agent]
    )

    runner = InMemoryRunner(agent=root_agent)
    # Using run_async instead of run_debug for better compatibility in some envs, 
    # but run_debug is fine if nest_asyncio is applied.
    response = await runner.run_debug("Check the fridge and plan tomorrow's lunch.")
    return response

if __name__ == "__main__":
    # async.run() is standard python, nest_asyncio makes it work in notebooks
    asyncio.run(run_meal_planner())


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("GCP_SERVICE_ACCOUNT")
secret_value_1 = user_secrets.get_secret("GEMINI_API_KEY")
secret_value_2 = user_secrets.get_secret("GOOGLE_SHEET_ID")
secret_value_3 = user_secrets.get_secret("GOOGLE_SHEET_RANGE")



!pip install google-adk pandas python-dotenv google-auth google-api-python-client nest_asyncio

