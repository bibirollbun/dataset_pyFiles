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


# --- CELL 1: SETUP & AUTH ---
import google.generativeai as genai
import requests
import json
import datetime
from kaggle_secrets import UserSecretsClient

print("--- โ๏ธ INITIALIZING SETUP ---")

try:
    # 1. Connect to Kaggle Secrets
    user_secrets = UserSecretsClient()
    
    # 2. Authenticate Gemini (The Brain)
    api_key = user_secrets.get_secret("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    print("โ Google Gemini API: Connected")
    
    # 3. Authenticate SerpApi (The Travel Agent)
    try:
        serp_api_key = user_secrets.get_secret("SERPAPI_KEY")
        print("โ SerpApi (Flights): Connected")
    except:
        serp_api_key = None
        print("โ�๏ธ Warning: 'SERPAPI_KEY' missing. Flights will use mock data.")

    # 4. Authenticate Spoonacular (The Chef) <--- NEW SECTION
    try:
        spoon_api_key = user_secrets.get_secret("SPOONACULAR_KEY")
        print("โ Spoonacular (Recipes): Connected")
    except:
        spoon_api_key = None
        print("โ�๏ธ Warning: 'SPOONACULAR_KEY' missing. Recipes will use mock data.")
        
except Exception as e:
    print(f"โ Critical Setup Error: {e}")


# --- CELL 2: MEMORY SYSTEM ---
# The agent uses this dictionary to "remember" facts.
user_profile = {
    "name": "Guest",
    "diet": "None",
    "home_airport": "Unknown"
}

# We add 'name' as an optional argument here
def update_profile(name: str = None, diet: str = None, home_airport: str = None):
    """Updates the user's profile memory."""
    if name: user_profile["name"] = name  # <--- Logic to save the name
    if diet: user_profile["diet"] = diet
    if home_airport: user_profile["home_airport"] = home_airport
    return f"โ Memory Updated: {user_profile}"

def get_profile():
    """Reads the current user profile."""
    return user_profile

print("โ Memory System Initialized")


# --- CELL 3: CHEF AGENT TOOL (REAL API) ---
def search_recipes(query: str):
    """
    Finds REAL recipes using Spoonacular API, filtering by the user's diet.
    """
    # 1. Get User Context
    diet = user_profile["diet"]
    print(f"\n[TOOL] ๐ณ Spoonacular Search: '{query}' | Filter: {diet}...")

    # 2. Check if Key Exists
    if not spoon_api_key:
        return ["Error: SPOONACULAR_KEY missing. Cannot fetch real recipes."]

    # 3. Prepare the API Request
    # We use 'complexSearch' which allows strict dietary filtering
    url = "https://api.spoonacular.com/recipes/complexSearch"
    params = {
        "query": query,
        "diet": diet if diet != "None" else "", # Only send diet if it exists
        "number": 3, # Get top 3 results
        "addRecipeInformation": "true", # Get details like cooking time
        "apiKey": spoon_api_key
    }

    # 4. Call the API
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        results = []
        if "results" in data and len(data["results"]) > 0:
            for item in data["results"]:
                title = item['title']
                time = item.get('readyInMinutes', '?')
                # We format it nicely so the LLM understands
                results.append(f"Recipe: {title} (Takes {time} mins)")
            return results
        else:
            return [f"No {diet} recipes found for '{query}'."]
            
    except Exception as e:
        return [f"API Error: {str(e)}"]

print("โ Real-Time Chef Tool (Spoonacular) Ready")


# --- CELL 4: SHOPPER AGENT TOOL ---
def make_shopping_list(recipe_name: str):
    """Generates a strict JSON shopping list."""
    print(f"\n[TOOL] ๐ Generating JSON list for {recipe_name}...")
    
    # Returns raw JSON string for downstream apps
    return json.dumps([
        {"item": "Main Ingredient", "qty": "2 lbs", "aisle": "Produce"},
        {"item": "Spices", "qty": "1 jar", "aisle": "Baking"},
        {"item": "Olive Oil", "qty": "1 bottle", "aisle": "Condiments"}
    ])

print("โ Shopper Tool Ready")


# --- CELL 5: TRAVEL AGENT TOOL (FIXED FOR ONE-WAY) ---
def search_flights(destination: str, date: str):
    """Finds REAL-TIME flight prices using Google Flights via SerpApi."""
    
    # 1. Check prerequisites
    origin = user_profile["home_airport"]
    if origin == "Unknown":
        return "Error: I don't know your home airport yet. Please tell me where you live first."
    
    if not serp_api_key:
        return "Error: SERPAPI_KEY is missing. Cannot search flights."

    print(f"\n[TOOL] โ๏ธ Checking SerpApi: {origin} -> {destination} on {date}...")
    
    # 2. Construct Request
    params = {
        "engine": "google_flights",
        "departure_id": origin,
        "arrival_id": destination,
        "outbound_date": date,
        "type": "2", # <--- CRITICAL FIX: 2 = One-Way Flight (Default is 1=Round Trip)
        "currency": "USD",
        "hl": "en",
        "api_key": serp_api_key
    }

    # 3. Hit the API
    try:
        response = requests.get("https://serpapi.com/search", params=params)
        data = response.json()
        
        # Debug: If something fails, show the error
        if "error" in data:
            print(f"[API ERROR] {data['error']}")
            return f"Google Flights Error: {data['error']}"

        # 4. Parse Results (Robustly)
        flight_list = data.get("best_flights", [])
        if not flight_list:
            flight_list = data.get("other_flights", [])
            
        if flight_list:
            best = flight_list[0]
            flight_info = best["flights"][0]
            price = best.get("price", "Check Link")
            return {
                "airline": flight_info.get("airline", "Unknown"),
                "price": f"${price}" if isinstance(price, int) else price,
                "duration": f"{best.get('total_duration', 0)} min",
                "link": data.get("search_metadata", {}).get("google_flights_url", "")
            }
        else:
            return "No flights found. Hint: Are you using 3-letter Airport Codes (e.g. 'JFK', 'LHR')?"
            
    except Exception as e:
        return f"API Connection Error: {str(e)}"

print("โ Travel Tool Updated (One-Way Support Enabled)")


# --- CELL 6: AGENT INIT (WITH DATE AWARENESS) ---
import datetime

# 1. Combine Tools
tools_list = [update_profile, get_profile, search_recipes, make_shopping_list, search_flights]

# 2. Initialize Model
model = genai.GenerativeModel(
    model_name='models/gemini-2.0-flash',
    tools=tools_list
)

chat = model.start_chat(enable_automatic_function_calling=True)

# 3. Get Real-Time Date
# We calculate today's date so the Agent knows "when" it is.
today_date = datetime.date.today().strftime("%Y-%m-%d")

# 4. System Prompt with Date Injection
system_prompt = f"""
You are a High-End Concierge Agent.
CURRENT CONTEXT: Today's date is {today_date}.

CRITICAL RULES:
1. **Memory:** Always check `get_profile` first.
2. **Dates:** When the user says "tomorrow" or "next week", YOU must calculate the exact date (YYYY-MM-DD) based on Today's Date ({today_date}).
3. **Flight Tool:** The `search_flights` tool ONLY accepts dates in 'YYYY-MM-DD' format. Never send words like 'tomorrow'.
4. **Locations:** Convert city names (Seattle) to IATA codes (SEA) for the `home_airport`.
"""

chat.send_message(system_prompt)

print(f"โ Concierge Online (Date Synced: {today_date})")


# --- CELL 7: INTERACTIVE CHAT LOOP (IMPROVED) ---
import time

# 1. RESET MEMORY (Fixes the "Leftover Info" bug)
# We force the profile back to blank every time you start this cell.
user_profile = {
    "name": "Guest",
    "diet": "None",
    "home_airport": "Unknown"
}
print("๐งน Memory Wiped Clean. Starting fresh session.")

print("\n--- ๐ฌ INTERACTIVE CONCIERGE MODE ONLINE ---")
print("๐ก Commands to try:")
print("   1. 'My name is Nithya and I live in Seattle' (Updates Name & Location)")
print("   2. 'I am Vegan' (Updates Diet)")
print("   3. 'Check flights to NYC for 2025-12-12' (Uses your stored location)")

while True:
    try:
        # Show the user what the Agent knows (The "Debug" View)
        print(f"\n[๐ง� MEMORY STATE]: {user_profile}")
        
        user_input = input("User: ")
        
        if user_input.lower() in ['exit', 'quit', 'stop']:
            print("\n๐ Concierge signing off.")
            break
        
        response = chat.send_message(user_input)
        print(f"Concierge: {response.text}\n")
        
    except Exception as e:
        print(f"โ Error: {e}")

