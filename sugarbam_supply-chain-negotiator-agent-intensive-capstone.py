


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


# Cell 1 â€” Install & import dependencies
!pip install --upgrade google-adk

from pathlib import Path
import json
from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent, BaseAgent
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools import FunctionTool
from google.adk.runners import InMemoryRunner

print("ADK version:", LlmAgent)  # just to check import works


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print("ðŸ”‘ Authentication Error: Please add 'GOOGLE_API_KEY' to your Kaggle secrets.")
    raise e


from typing import Any, Dict, List
import json

from google.genai import types

from google.adk.agents import Agent, LlmAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import AgentTool
from google.adk.tools.tool_context import ToolContext

print("âœ… ADK components imported successfully.")

# Retry configuration
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# Session service (conceptual)
session_service = InMemorySessionService()
print("âœ… Session service (conceptual) created.")


def pretty_print_json(data: Any):
    print(json.dumps(data, indent=2, ensure_ascii=False))

print("âœ… Helper pretty printer ready.")


# --- Configuration ---
# Use a consistent model for the whole pipeline
GEMINI_MODEL = "gemini-2.5-flash" # High capability, low latency model
APP_NAME = "SupplyChainNegotiatorApp"
USER_ID = "simulated_buyer"
SESSION_ID = "order_001_session"

print("Environment setup complete.")


import os
import asyncio
from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent, BaseAgent
from google.adk.tools.base_toolset import BaseToolset
from google.adk.runners import InMemoryRunner
from google.genai import types

# Global storage for simulation results
SCOUT_MEMORY = []
FINAL_MEMORY = None


toolset = NegotiationToolset()
print("Toolset and Memory defined.")

# Global storage for simulation results
SCOUT_MEMORY = []
FINAL_MEMORY = None

class NegotiationToolset(BaseToolset):
    # Change the method signature to take only 'self'
    def get_tools(self): 
        """Returns a list of tools available in this toolset."""
        return [self.save_final_choice]

    def save_final_choice(self, vendor_json: str) -> str:
        # ... (rest of the method remains the same) ...
        global FINAL_MEMORY
        FINAL_MEMORY = vendor_json
        return f"Successfully saved final choice: {vendor_json}"

toolset = NegotiationToolset()
print("Toolset and Memory defined.")





# --- 1. Vendor Scout Agents (Run in Parallel) ---
# We simulate 3 distinct vendors with slightly different personas/data
def create_vendor_scout(vendor_id, lead_time_info, price_info):
    return LlmAgent(
        name=f"VendorCheck_{vendor_id}",
        model=GEMINI_MODEL,
        instruction=f"""
        You are a supply chain assistant for Vendor {vendor_id}. 
        You check the user's requirements against your data: {lead_time_info}, {price_info}.
        If you meet the requirements, output a JSON object with keys: 'vendor_id', 'eligible' (bool), 'lead_time_days' (int), 'price_unit' (int).
        If you don't meet requirements, output a JSON object with 'eligible': false and an 'explanation'.
        Always save your output to a shared list called SCOUT_MEMORY.
        """,
        # This saves the agent's full response JSON directly into the global SCOUT_MEMORY list
        output_key="SCOUT_MEMORY" 
    )

agents = [
    create_vendor_scout("V001", "Lead time is 5 days", "Price is $110"),
    create_vendor_scout("V002", "Lead time is 8 days", "Price is $100"), # Fails lead time
    create_vendor_scout("V003", "Lead time is 6 days", "Price is $125"),
]

# --- 2. Aggregator Agent (Sequential Step 2) ---
aggregator = LlmAgent(
    name="VendorAggregator",
    model=GEMINI_MODEL,
    instruction="""
    You have vendor evaluations in SCOUT_MEMORY.
    Your job is to read all entries from SCOUT_MEMORY.
    1. Identify eligible vendors ('eligible': true).
    2. Sort them by 'lead_time_days' (ASC) then 'price_unit' (ASC).
    3. Produce a summary JSON list of ONLY the eligible vendors under the key "eligible_vendors".
    4. Do NOT save anything back to memory yet.
    """,
    output_key="eligible_vendors" # This saves the output to the session state for the next agent
)

# --- 3. Final Decision Agent (Sequential Step 3) ---
final_decision = LlmAgent(
    name="VendorDecisionAgent",
    model=GEMINI_MODEL,
    instruction="""
    You take the list of eligible_vendors from the previous step.
    Pick the single best option based on the sorting criteria (fastest, then cheapest).
    Then, you MUST call the `save_final_choice` tool with the chosen vendor's JSON object.
    Finally, output a clear Markdown summary explaining the final choice.
    """,
    output_key="final_decision_md",
    tools=[toolset] # Provide the toolset defined in Cell 2
)

print("Agents defined.")



# 1. Parallel Scout Phase
parallel_scout = ParallelAgent(
    name="ParallelVendorScout",
    sub_agents=agents # Runs V001, V002, V003 concurrently
)

# 2. Root Sequential Pipeline
root_agent = SequentialAgent(
    name="VendorPipeline",
    sub_agents=[parallel_scout, aggregator, final_decision]
)

print("Pipeline orchestrated.")



# Initialize the runner with the root agent
runner = InMemoryRunner(
    agent=root_agent, 
    app_name=APP_NAME
)

# Input Prompt from the user/system
user_description = """
We need a vendor for shipping electronics.
Max acceptable lead time: 7 days
Max acceptable price: 130 dollars per unit
"""

prompt = f"""
SHIPMENT_ID: order_001

REQUIREMENTS:
{user_description}
"""

print("Memory cleared and execution started in async function.")

# In Cell 5, inside the async function:
async def run_pipeline_simulation():
    try:
        # Use run_debug with only the prompt string as a positional argument (FIXED PREVIOUS ERROR)
        events = await runner.run_debug(prompt)

        # Process the events generator to ensure it runs to completion and print final output
        final_response_text = ""
        async for event in events:
            if event.is_final_response():
                final_response_text = event.content.parts.text
                print("\n--- FINAL AGENT RESPONSE (Markdown Summary) ---")
                print(final_response_text)
            # The run_debug command prints internal traces automatically

        print("\n--- SIMULATION COMPLETE ---")
        print(f"Final saved choice (via Tool): {FINAL_MEMORY}")
        return final_response_text

    except ExceptionGroup as eg:
        print("\n--- CAUGHT EXCEPTION GROUP (Details Below) ---")
        # Print each underlying exception to see what actually failed
        for i, exc in enumerate(eg.exceptions):
            print(f"ERROR {i+1}: {type(exc).__name__} - {exc}")
        print("--------------------------------------------------\n")
        raise eg # Re-raise if you want the notebook cell to fail



# Initialize the runner with the root agent
runner = InMemoryRunner(
    agent=root_agent, 
    app_name=APP_NAME
)

# Input Prompt from the user/system
user_description = """
We need a vendor for shipping electronics.
Max acceptable lead time: 7 days
Max acceptable price: 130 dollars per unit
"""

prompt = f"""
SHIPMENT_ID: order_001

REQUIREMENTS:
{user_description}
"""

print("Memory cleared and execution started in async function.")

# In Cell 5, inside the async function:
async def run_pipeline_simulation():
    try:
        # Use run_debug with only the prompt string as a positional argument (FIXED PREVIOUS ERROR)
        events = await runner.run_debug(prompt)

        # Process the events generator to ensure it runs to completion and print final output
        final_response_text = ""
        async for event in events:
            if event.is_final_response():
                final_response_text = event.content.parts.text
                print("\n--- FINAL AGENT RESPONSE (Markdown Summary) ---")
                print(final_response_text)
            # The run_debug command prints internal traces automatically

        print("\n--- SIMULATION COMPLETE ---")
        print(f"Final saved choice (via Tool): {FINAL_MEMORY}")
        return final_response_text

    except ExceptionGroup as eg:
        print("\n--- CAUGHT EXCEPTION GROUP (Details Below) ---")
        # Print each underlying exception to see what actually failed
        for i, exc in enumerate(eg.exceptions):
            print(f"ERROR {i+1}: {type(exc).__name__} - {exc}")
        print("--------------------------------------------------\n")
        raise eg # Re-raise if you want the notebook cell to fail

# *** THIS LINE IS REQUIRED TO START THE SIMULATION IN JUPYTER ***
results = await run_pipeline_simulation()


