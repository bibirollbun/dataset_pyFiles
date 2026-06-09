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


# 1. Install necessary libraries
!pip cache purge
!pip install -U -q "google-genai"

# 2. Imports
import google.genai as genai
import google.genai.types as types
from google.api_core import retry
import json
import logging
import time
import re
from datetime import datetime, timedelta
from IPython.display import display, HTML, Markdown
from kaggle_secrets import UserSecretsClient

# --- CAPSTONE REQ: OBSERVABILITY & LOGGING ---
# We set up a custom logger to trace the execution flow of our agents.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TravelConciergeSystem")

class AgentTracer:
    """
    A Context Manager to trace agent performance (Latency & Status).
    This helps in debugging and demonstrating agent orchestration.
    """
    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"ğŸŸ¢ [START] {self.agent_name} activated.")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        duration = time.time() - self.start_time
        if exc_type:
            logger.error(f"ğŸ”´ [ERROR] {self.agent_name} failed after {duration:.2f}s: {exc_value}")
        else:
            logger.info(f"ğŸ�� [COMPLETE] {self.agent_name} finished in {duration:.2f}s.")

# 3. API Configuration
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    client = genai.Client(api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"âš ï¸� Error accessing secrets: {e}. Please ensure 'GOOGLE_API_KEY' is set in Kaggle Secrets.")

# Retry policy for robustness
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})
genai.models.Models.generate_content = retry.Retry(predicate=is_retriable)(genai.models.Models.generate_content)


# --- CAPSTONE REQ: SCHEMA DEFINITION ---
# This schema enforces the structure needed for the Validator to check the budget.
itinerary_schema = {
    "type": "object",
    "properties": {
        "trip_title": {"type": "string"},
        "destination": {"type": "string"},
        # Crucial for the Validator Agent:
        "total_estimated_trip_cost": {"type": "number"}, 
        "transport_options": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "plan_title": {"type": "string"},
                    "total_transport_cost": {"type": "number"},
                    "justification": {"type": "string"},
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "mode": {"type": "string"},
                                "from": {"type": "string"},
                                "to": {"type": "string"},
                                "cost": {"type": "number"}
                            }
                        }
                    }
                }
            }
        },
        "daily_plan": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "day": {"type": "number"},
                    "theme": {"type": "string"},
                    "activities": {"type": "string"}
                }
            }
        },
        "overall_budget_note": {"type": "string"}
    },
    "required": ["trip_title", "total_estimated_trip_cost", "transport_options", "daily_plan"]
}


# --- AGENT 1: RESEARCHER (Uses Google Search Tool) ---
def research_agent(destination: str, start_date: str) -> dict:
    """
    Role: Fetches real-time weather and safety data.
    Tool: Google Search.
    """
    with AgentTracer("Research_Agent"):
        month_year = datetime.strptime(start_date, "%Y-%m-%d").strftime("%B %Y")
        query = f"Weather forecast and travel safety advisories for {destination} in {month_year}"
        print(f"ğŸ•µï¸� Research Agent is searching: '{query}'...")
        
        try:
            # CAPSTONE REQ: TOOLS (Google Search)
            search_config = types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
            # We use gemini-1.5-flash for speed and tool capability
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=query,
                config=search_config
            )
            
            # Extract the grounded text or summary
            research_summary = response.text if response.text else "No specific data found, proceed with general knowledge."
            return {"status": "success", "context_data": research_summary[:1500]}
            
        except Exception as e:
            logger.error(f"Research failed: {e}")
            return {"status": "error", "context_data": "Real-time data unavailable."}

# --- AGENT 2: PLANNER (The Brain) ---
def planner_agent(user_request: dict, research_context: str, feedback: str = None) -> dict:
    """
    Role: Generates the itinerary. 
    Capability: Self-Correction based on 'feedback'.
    """
    with AgentTracer("Planner_Agent"):
        # Handle the Feedback Loop
        if feedback:
            print(f"ğŸ”„ Planner is refining the plan based on feedback: '{feedback}'...")
            instruction_prefix = f"âš ï¸� CRITICAL: Your previous plan was rejected. REASON: {feedback}. You MUST fix the plan to meet the budget constraints."
        else:
            print("ğŸ§  Planner Agent is building the initial itinerary...")
            instruction_prefix = "Create a detailed travel plan."

        prompt = f"""
        {instruction_prefix}
        
        You are an expert Travel Logistician. Create a travel plan based on this data:
        
        USER REQUEST:
        Origin: {user_request['origin']}
        Destination: {user_request['destination']}
        Max Budget: â‚¹{user_request.get('max_budget')}
        Duration: {user_request['duration']} days
        Interests: {user_request['interests']}
        
        REAL-TIME CONTEXT:
        {research_context}
        
        INSTRUCTIONS:
        1. Calculate a 'total_estimated_trip_cost'. This MUST include Transport + Accommodation/Food (Approx â‚¹3000/day).
        2. Provide 2 distinct transport options.
        3. Be realistic with costs.
        4. Output JSON only.
        """
        
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=itinerary_schema
            )
        )
        return json.loads(response.text)

# --- AGENT 3: VALIDATOR (The Gatekeeper) ---
def validator_agent(itinerary_json: dict, max_budget: int) -> tuple[bool, str]:
    """
    Role: Checks if the plan adheres to the user's constraints.
    Returns: (is_valid, feedback_message)
    """
    with AgentTracer("Validator_Agent"):
        estimated_cost = itinerary_json.get("total_estimated_trip_cost", 0)
        
        print(f"âš–ï¸� Validator checking: Plan Cost â‚¹{estimated_cost:,} vs Budget â‚¹{max_budget:,}")
        
        if estimated_cost <= max_budget:
            return True, "Plan looks great and is within budget."
        else:
            diff = estimated_cost - max_budget
            feedback = (f"The plan is too expensive. Total cost is â‚¹{estimated_cost}, "
                        f"which exceeds the budget of â‚¹{max_budget} by â‚¹{diff}. "
                        "Reduce transport costs (switch flight to train) or reduce daily expense estimates.")
            print(f"â�Œ Validation Failed: {feedback}")
            return False, feedback

# --- AGENT 4: PUBLISHER (The Formatter) ---
def publisher_agent(itinerary_json: dict) -> str:
    """
    Role: Converts the JSON into a readable HTML format.
    """
    with AgentTracer("Publisher_Agent"):
        print(f"ğŸ�¨ Publisher Agent is formatting the final report...")
        
        # Simple HTML Template
        html = f"""
        <div style="font-family: sans-serif; max-width: 800px; margin: auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px; background-color: #f9f9f9;">
            <h1 style="color: #2c3e50; text-align: center;">{itinerary_json.get('trip_title', 'Trip Plan')}</h1>
            <div style="background: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 5px solid #17a2b8;">
                <h3>ğŸ’° Budget Overview</h3>
                <p><b>Total Estimated Cost:</b> â‚¹{itinerary_json.get('total_estimated_trip_cost', 0):,}</p>
                <p><i>{itinerary_json.get('overall_budget_note', '')}</i></p>
            </div>
            
            <h3>âœˆï¸� Transport Options</h3>
            {''.join([f"<div style='background:white; padding:10px; border:1px solid #eee; margin-bottom:10px;'><b>{opt['plan_title']}</b> (â‚¹{opt.get('total_transport_cost'):,}): {opt.get('justification')}</div>" for opt in itinerary_json.get('transport_options', [])])}
            
            <h3>ğŸ“… Daily Itinerary</h3>
            {''.join([f"<div style='margin-bottom:10px; padding-left:10px; border-left: 3px solid #28a745;'><b>Day {day['day']} - {day['theme']}:</b> {day['activities']}</div>" for day in itinerary_json.get('daily_plan', [])])}
        </div>
        """
        return html


# --- CAPSTONE REQ: LOOP AGENTS (Self-Correction) ---

def run_smart_travel_system(origin, destination, date, duration, interests, max_budget):
    print(f"ğŸš€ Initializing Smart Travel System for: {destination}")
    print(f"ğŸ’° Constraint: Max Budget â‚¹{max_budget:,}")
    print("="*60)
    
    # 1. RESEARCH
    research_result = research_agent(destination, date)
    context = research_result['context_data']
    
    # 2. PLANNING LOOP
    user_req = {
        "origin": origin,
        "destination": destination,
        "interests": interests,
        "duration": duration,
        "max_budget": max_budget
    }
    
    itinerary_json = None
    feedback = None
    max_retries = 3
    is_valid = False
    
    for attempt in range(max_retries):
        print(f"\n--- ğŸ”„ Planning Attempt {attempt + 1}/{max_retries} ---")
        
        # A. Call Planner (passes feedback if previous attempt failed)
        itinerary_json = planner_agent(user_req, context, feedback)
        
        # B. Call Validator
        is_valid, feedback_msg = validator_agent(itinerary_json, max_budget)
        
        if is_valid:
            print("âœ… Plan Validated! Proceeding to publish.")
            break
        else:
            feedback = feedback_msg # Store feedback for the next loop
            
    if not is_valid:
        print("âš ï¸� Warning: Could not fully meet budget constraints after max retries. Showing best effort.")

    # 3. PUBLISH
    final_html = publisher_agent(itinerary_json)
    
    print("="*60)
    print("ğŸ�‰ Workflow Complete.")
    display(HTML(final_html))
    return itinerary_json

# === TEST CASE: TIGHT BUDGET ===
# We set a budget of 20,000 for a trip to Ladakh. 
# This usually costs >30k with flights, so it should force the agent 
# to "loop" and switch to train/bus options to meet the budget.
final_plan = run_smart_travel_system(
    origin="Bengaluru, Karnataka",
    destination="Leh, Ladakh",
    date="2025-09-17",
    duration=5,
    interests="Monasteries, Pangong Lake",
    max_budget=20000 
)

