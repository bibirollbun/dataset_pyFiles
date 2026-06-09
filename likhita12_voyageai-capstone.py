# --- 1. SETUP & INSTALLATION ---
# Install the Google Agent Development Kit
!pip install -q google-adk

import os
import logging
from kaggle_secrets import UserSecretsClient

# Setup Authentication
try:
    user_secrets = UserSecretsClient()
    # Ensure you have set 'GOOGLE_API_KEY' in the Add-ons -> Secrets menu
    api_key = user_secrets.get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = api_key
    print("âœ… Google API Key configured.")
except Exception as e:
    print(f"âš ï¸� Error: {e}. Please ensure 'GOOGLE_API_KEY' is added to Kaggle Secrets.")

# Setup Logging (Optional: helpful for seeing agent thought process)
logging.basicConfig(level=logging.ERROR)


# --- 2. DEFINE AGENTS & TOOLS ---
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools.agent_tool import AgentTool
from google.genai import types

# Configuration for reliability (retry logic from your course notebook)
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# Common Model Config
model_config = Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config)

# --- 2.1 Researcher Agent ---
# (Responsible for finding facts. In a real app, give this agent a Google Search tool!)
researcher_agent = LlmAgent(
    name="researcher_agent",
    model=model_config,
    instruction="""
    You are an expert Travel Researcher.
    Given a location and dates, provide:
    1. Typical weather.
    2. Top 3 cultural attractions.
    3. Local customs to be aware of.
    Be concise and factual.
    """
)

# --- 2.2 Planner Agent ---
# (Responsible for logistics and schedule)
planner_agent = LlmAgent(
    name="planner_agent",
    model=model_config,
    instruction="""
    You are a Senior Travel Planner.
    Input: Research data about a destination.
    Task: Create a day-by-day itinerary (Morning, Afternoon, Evening).
    Ensure the flow is logical (group nearby activities).
    """
)

# --- 2.3 Budget Agent ---
# (Responsible for financial estimation)
budget_agent = LlmAgent(
    name="budget_agent",
    model=model_config,
    instruction="""
    You are a Travel Budget Analyst.
    Input: A proposed itinerary.
    Task:
    1. Estimate total cost (Accommodation, Food, Transport, Entry fees).
    2. Compare against the user's budget limit.
    3. Suggest one money-saving tip.
    """
)

print("âœ… Sub-agents defined successfully.")


# --- 3. ORCHESTRATION (THE ROOT AGENT) ---

# We wrap our sub-agents as Tools so the Root Agent can call them.
# This follows the pattern from the observability notebook where agents call other agents.

tools_list = [
    AgentTool(agent=researcher_agent),
    AgentTool(agent=planner_agent),
    AgentTool(agent=budget_agent)
]

# The Concierge Agent (Root)
# We give it strict instructions to run the pipeline sequentially.
concierge_agent = LlmAgent(
    name="voyage_concierge_agent",
    model=model_config,
    tools=tools_list,
    instruction="""
    You are VoyageAI, an intelligent travel concierge.
    You must orchestrate a travel plan by invoking your tools in this STRICT sequence:

    1. Call 'researcher_agent' to get information about the destination.
    2. Pass that research to 'planner_agent' to generate an itinerary.
    3. Pass that itinerary to 'budget_agent' to evaluate costs.

    Final Output: Present the final Itinerary and Budget Analysis to the user.
    """
)

print("âœ… Root Concierge Agent ready.")


# --- 4. EXECUTION ---
from google.adk.runners import InMemoryRunner
import asyncio

# Create the Runner
runner = InMemoryRunner(agent=concierge_agent)

# Define the User's Request
user_request = "Plan a 3-day trip to Paris for a couple in May. Budget: $2000."

print(f"ğŸš€ Starting VoyageAI for request: '{user_request}'\n")

# FIX: Use run_debug() to pass the user request. 
# This matches the pattern used in the 'day-4a-agent-observability' notebook.
response = await runner.run_debug(user_request)

print("\n--------------------------------------------------")
print("ğŸ�‰ FINAL VOYAGE PLAN")
print("--------------------------------------------------")

# Try to print .text if available, otherwise print the full response object
try:
    print(response.text)
except AttributeError:
    print(response)

