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
# You can also write temporary files to /kaggle/temp/, but they won't be saved outsid



import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Auth Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")



# Install the library first (Run this line in a separate cell if needed, or keep here)
# !pip install google-adk

from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.runners import InMemoryRunner,Runner
#from gloogle.adk.session import InMemorySessionService
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types

print("âœ… ADK components imported successfully.")



import os
import time
from google.adk.agents import Agent, SequentialAgent, LoopAgent
from google.adk.tools import ToolContext,FunctionTool,AgentTool
from google.adk.models.google_llm import Gemini
from google.genai import types
from google.adk.tools.tool_context import ToolContext
from google.adk.runners import InMemoryRunner
from google.adk.runners import Runner


print("âœ… ADK components imported successfully.")



# [A] THE ANALYST (Sequential Step 1)
    # Extracts structured intent from the raw prompt
analyst_agent = Agent(
     name="Analyst",
     model="gemini-2.5-flash-lite",
     instruction=""" Extract Event Type.Budget, Date and Guests from user query and pass it to Strategist  """,
     output_key="analysis_result"
    )
print("âœ…Analyst Agent created")


def fetch_budget_guidelines(event_category:str)->dict:
    
    """
    Retrieves industry-standard budget split ratios (venue/food/decor) 
    for a given event category (e.g., 'corporate', 'wedding', 'party').
    """
    time.sleep(5)
    strategies = {
        "corporate": {"venue": 0.60, "catering": 0.30, "decor": 0.10},
        "party":     {"venue": 0.20, "catering": 0.60, "decor": 0.20},
        "general":   {"venue": 0.40, "catering": 0.40, "decor": 0.20}
    }
    return strategies.get(event_category.lower(), strategies["general"])
    
print("âœ…fetch_budget_guidelines created")


# [B] THE REFINEMENT LOOP (Sequential Step 2)
    
    # 1. Strategist: 
strategist_agent = Agent(
        name="Strategist",
        model="gemini-2.5-flash-lite",
        instruction="""
        You are the Budget Strategist.
        GOAL: Output integer budgets for Venue, Catering, and Decor.
        
        LOGIC:
        1. Read 'analysis_result'.
        2. Check for 'reviewer_feedback' (from previous failed loops).
           - IF FEEDBACK EXISTS: You MUST ignore standard ratios and solve the specific complaint (e.g., cut food budget to pay for venue).
           - IF NO FEEDBACK: Use 'fetch_budget_guidelines'.
        """,        
        #time.sleep(5),
        tools=[fetch_budget_guidelines],
        output_key="budget_plan"
)
    
print("âœ… Strategist Agent created")


def check_venue_availability(budget_limit:int)->str:
    """
    Checks venue inventory. Returns FAILURE if the budget_limit is below 5000.
    """
    time.sleep(5) 
    if budget_limit < 5000:
        
        return "FAILURE: Budget too low (Minimum required: 5000)."
        
        return f"SUCCESS: Grand Hall Reserved (Cost: {budget_limit})"
    
print("âœ…check_venue_availability created")   


# 2 Execution Team member 1
venue_agent = Agent(
    name="Venue",
    model="gemini-2.5-flash-lite",
    instruction="Use 'check_venue_availability' based on 'budget_plan'.",
    tools=[check_venue_availability], 
    output_key="venue_status"
)
print("âœ…Venue Agent created")


def check_catering_options(budget_limit:int)->str:
    """
    Checks catering menus. Returns FAILURE if the budget_limit is below 2000.
    """
    time.sleep(5) 
    if budget_limit < 2000:
        return "FAILURE: Below Minimum Order Value (Minimum required: 2000)."
    return f"SUCCESS: Premium Buffet Confirmed (Cost: {budget_limit})"

print("âœ…check_catering_options created")


#Execution Team member 2
catering_agent = Agent(
    name="Catering",
    model="gemini-2.5-flash-lite",
    instruction="Use 'check_catering_options' based on 'budget_plan'.",
    tools=[check_catering_options],
    output_key="catering_status"
)
print("âœ…Catering Agent Created")


def check_decor_packages(budget_limit:int)->str:
    """
    Checks decor options. Skips ordering if the budget_limit is 0 or less.
    """
    time.sleep(5) 
    if budget_limit <= 0:
        return "SKIPPED: No Decor Ordered"
    return f"SUCCESS: Standard Decor Package (Cost: {budget_limit})"

print("âœ…check_decor_packages created")


#Execution Team member 3
decor_agent = Agent(
    name="Decor",
    model="gemini-2.5-flash-lite",
    instruction="Use 'check_decor_packages' based on 'budget_plan'.", 
    tools=[check_decor_packages], output_key="decor_status"
)
print("âœ…Decor Agent created")


execution_agent = SequentialAgent(
    name="ExecutionTeam",
    sub_agents=[venue_agent,catering_agent,decor_agent],
)
print("âœ…Sequential agent / ExecutionTeam created")
    


def approve_plan():
    """ 
    Finalizes the workflow. Call this ONLY when the Reviewer confirms 
    that all venue_agent, catering_agent, and decor_agent tasks are 'SUCCESS' or 'SKIPPED'.
    """  
    return {"status": "Success","message":"Budget for venue,catering and decoration successfully planned for your event"}

print("âœ…approve_plan created")   


# 3. Reviewer: The Critic
    # Has the power to call 'approve_plan' to break the loop

reviewer_agent = Agent(
    name="Reviewer",
    model="gemini-2.5-flash-lite",
    instruction="""
      Audit the status of Venue, Catering, and Decor.
      - If ANY failure: Output 'REJECTED: <Reason>' to 'reviewer_feedback'.
      - If ALL success: Call 'approve_plan' to finalize.
     """,
    tools=[approve_plan],
    output_key="reviewer_feedback"
)

print("âœ…Reviewer_agent created")



#LOOP AGENT
# Wrap 1, 2, 3 into a LoopAgent (Self-Correction Mechanism)
refinement_loop = LoopAgent(
    name="SelfCorrectionLoop",
    sub_agents=[strategist_agent, execution_agent, reviewer_agent],
    max_iterations=4 
)

print("âœ… LoopAgent / SelfCorrectionLoop created")



# [C] ROOT AGENT
    # Connects Analyst -> Loop
root_agent = SequentialAgent(
        name="EventPlannerSystem",
        sub_agents=[analyst_agent, refinement_loop]
    )
print("âœ… EventPlannerSystem /SequentialAgent created")


runner = InMemoryRunner(agent=root_agent)
response = await runner.run_debug( "Plan a birth day Party for 50 people on 6 July 2026 . Total Budget: Rs 10,000")


