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


pip install google-adk


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    
except Exception as e:
    print(f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")


import os
import pandas as pd
import numpy as np
import math
import datetime
from datetime import timedelta, date
from google.adk.models.google_llm import Gemini
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.adk.tools import google_search
from google.genai import types
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.tools.tool_context import ToolContext
from google.adk.code_executors import BuiltInCodeExecutor
import json
import sqlite3

from IPython.core.display import display, HTML
from jupyter_server.serverapp import list_running_servers

import warnings
import logging
logger = logging.getLogger("google_genai.types")

# Suppress WARNING-level messages for this logger
logger.setLevel(logging.ERROR)
logger.propagate = False

print("ADK components imported successfully.")


db_url = "sqlite:///my_sop_gelato_agent.db"  # Local SQLite file
session_service = DatabaseSessionService(db_url=db_url)


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


SKU_LIST = ['PROD-A', 'PROD-B', 'PROD-C']

def generate_simulated_data():
    """Generates synthetic data (constraints and costs) for the S&OP process."""
    
    
    SIMULATED_DATA = {}
    
    # Constraints Data
    SIMULATED_DATA['constraints'] = pd.DataFrame({
        'SKU': SKU_LIST,
        'Unconstrained_Demand': [1150, 165, 350],  # Input from Sales 
        'Max_Production_Standard': [950, 200, 400], 
        'Max_Production_Overtime': [1200, 220, 500], # Higher cap using overtime
        'Max_Inventory_Value': [200000, 35000, 15000] # Financial constraint
    })

    # Cost Data
    SIMULATED_DATA['cost_data'] = pd.DataFrame({
        'SKU': SKU_LIST,
        'Unit_Cost': [200.00, 45.00, 18.00],
    })
    
    return SIMULATED_DATA

GLOBAL_SIMULATED_DATA = generate_simulated_data()


def get_planning_constraints(sku: str) -> dict:
    """Retrieves all planning constraints for a specific SKU."""
    row = GLOBAL_SIMULATED_DATA['constraints'][GLOBAL_SIMULATED_DATA['constraints']['SKU'] == sku].iloc[0]
    return row.to_dict()

def get_unit_cost(sku: str) -> float:
    """Retrieves unit cost data."""
    return GLOBAL_SIMULATED_DATA['cost_data'][GLOBAL_SIMULATED_DATA['cost_data']['SKU'] == sku]['Unit_Cost'].iloc[0]

def get_demand(sku: str) -> float:
    """Retrieves unconstrained demand (number) on the base of the SKU provided by the user."""
    
    demand_value = GLOBAL_SIMULATED_DATA['constraints'][GLOBAL_SIMULATED_DATA['constraints']['SKU']== sku]['Unconstrained_Demand'].iloc[0]
    return float(demand_value)


demand_retreiver_agent = Agent(
    name="DemandRetreiverAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config, temperature=0.5),
    instruction=f"""You are a specialized Demand assistant. Your only job is to use the
    function tool to retrieve the unconstrained demand for the SKU provided by the user.
    **CRITICAL**: You must retreive the product name and only use the function. product name can be one of the following {SKU_LIST}. When providing the final answer (the tool's result), you **must output only the number**
    representing the unconstrained demand for the SKU provided by the user, with no leading text, trailing text, or explanation.""",
    tools=[FunctionTool(get_demand)],
    output_key="demand_findings", # The result of this agent will be stored in the session state with this key.
)

print("retreiver_agent created.")


meteo_agent = Agent(
    name="MeteoAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config, temperature=0.5),
    instruction="""You are a specialized research agent. 
    **CRITICAL**: FIRST You must retreive only the day from the user input and then execute your job.
    Your only job is to retreive the day from the user prompt and then use the google_search tool to check if for the date specified by the user in Milan (Italy) will rain or it will be a sunny day or cloudy. you can return only one of the these three statement and a singol word "RAIN" or "SUN" or "CLOUDS"
    .""",
    tools=[google_search],
    output_key="meteo_findings", # The result of this agent will be stored in the session state with this key.
)

print("Meteo_agent created.")


temperature_agent = Agent(
    name="TemperatureAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config, temperature=0.5),
    instruction="""You are a specialized research agent. 
    **CRITICAL**: First You must retreive only the day from the user prompt and then execute your job.
    Your only job is to retreive the day from the user prompt and then use the
    google_search tool to check if for the date specified by the user the average temperature in Milan (Italy). you can return only one of the these three statement and a singol word on the base of the temperature range: 
    #1 return "HOT" if the temperature is greather than 22Â°C 
    #2 return "COLD" if the temperature is lower than 15Â°C
    #3 retrurn "NORMAL" in all the other cases.""",
    tools=[google_search],
    output_key="temperature_findings", # The result of this agent will be stored in the session state with this key.
)

print("Temperature_agent created.")


Demand_agent = Agent(
    name="DemandCoordinator",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config, temperature=0.5),
    # This instruction tells the agent HOW to use its tools and generate a result.
    instruction="""You are the expert Demand coordinator. Your goal is to answer the user's query by orchestrating a workflow.
1. First, you MUST call the `get_demand` tool to find the unconstrained demand on the SKU provided by the user.
2. Next, you MUST call the `meteo_agent` agent tool  to get the {meteo_findings}
3. Next, you MUST call the `temperature_agent` agent tool  to get the {temperature_findings}
4. apply STRICTLY ONE OF THE RULES: 

**RULES:**
    - If the Weather is 'SUN' AND the Temperature is 'HOT', increase the unconstrained Demand by 15%.
    - If the Weather is 'RAIN' AND the Temperature is 'COLD', decrease the unconstrained Demand by 20%.
    - If the Weather is 'CLOUDS' AND the Temperature is 'COLD', decrease the unconstrained Demand by 10%.
    - If the Weather is 'RAIN' AND the Temperature is 'NORMAL', decrease the unconstrained Demand by 5%.
    - In all other cases, the Base Demand remains unchanged (0% adjustment).


5. Return the Output

**Output:** State the calculation performed (e.g., 1500 * 1.10) and provide ONLY the final calculated demand figure (a number) in the format 'FINAL_DEMAND: [NUMBER]'.
    """,
     
    output_key="final_adjusted_demand",  # This stores the result in memory/session state
)
print("Demand_agent created.")


# The ParallelAgent runs all its sub-agents simultaneously.
parallel_demand_team = ParallelAgent(
    name="ParallelDemandTeam",
    sub_agents=[demand_retreiver_agent, meteo_agent, temperature_agent]
)

# The Sequential demand_manager_agent completes the execution.
demand_manager_agent = SequentialAgent(
    name="DemandManagerPipeline",
    sub_agents=[parallel_demand_team, Demand_agent],
)


APP_NAME = "DEMAND_PLANNER"
USER_ID = "Senior_Planner"
current_time = datetime.datetime.now()
SESSION_ID = f"{USER_ID}_{current_time.strftime('%Y%m%d_%H%M%S')}"

runner = Runner(agent=demand_manager_agent, app_name=APP_NAME, session_service=session_service)
response = await runner.run_debug("need the demand for PROD-A, given the meteo and temperature the day is: 19 November 2025", session_id=SESSION_ID)


print("Upgraded to persistent sessions!")
print(f"   - Application: {APP_NAME}")
print(f"   - User: {USER_ID}")
print(f"    - **Generated Session ID**: {SESSION_ID}")


def get_last_final_demand():
    """
    Connects to the database, retrieves the content of the most recent 
    'DemandCoordinator' event, and extracts the FINAL_DEMAND value.
    """
    with sqlite3.connect("my_sop_gelato_agent.db") as connection:
        cursor = connection.cursor()
        
        cursor.execute(
            f"""
            SELECT content 
            FROM events 
            WHERE author = 'DemandCoordinator'
            ORDER BY ROWID DESC
            LIMIT 1
            """
        )
        
        # Fetch the single result (content string)
        result = cursor.fetchone()

    if not result:
        return None, "No 'DemandCoordinator' entries found."

    # The result is a tuple, so we extract the string
    content_json_str = result[0]
    
    # 2. Parse the JSON and extract the text
    try:
        content_data = json.loads(content_json_str)
        text_content = content_data.get('parts', [{}])[0].get('text', '')
        
        # 3. Extract the final demand number from the text
        if 'FINAL_DEMAND:' in text_content:
            # Splits the string by 'FINAL_DEMAND:' and takes the second part, then strips whitespace
            final_demand_str = text_content.split('FINAL_DEMAND:')[1].strip()
            
            # Converts the string value to a float
            final_demand_value = float(final_demand_str)
            return final_demand_value, "Success"
        else:
            return None, "FINAL_DEMAND key not found in the content text."
            
    except (json.JSONDecodeError, IndexError, ValueError) as e:
        return None, f"Error parsing content: {e}"

# Run the function
last_demand, status = get_last_final_demand()

# Print the result
if last_demand is not None:
    print(f"The last FINAL_DEMAND extracted is: {last_demand}")
else:
    print(f"Could not retrieve final demand. Status: {status}")


SKU_TO_UPDATE = 'PROD-A'

row_index = GLOBAL_SIMULATED_DATA['constraints'][GLOBAL_SIMULATED_DATA['constraints']['SKU'] == SKU_TO_UPDATE].index[0]

GLOBAL_SIMULATED_DATA['constraints'].loc[row_index, 'Unconstrained_Demand'] = last_demand

print(f"Updated Demand for {SKU_TO_UPDATE}:")
GLOBAL_SIMULATED_DATA['constraints']


data_ingestion_agent = Agent(
    name="DataIngestionAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config, temperature=0.5),
    instruction=f"""Your job is to gather all four required data points for the S&OP consensus process:
    **CRITICAL** You must retreive the product name and only use the functions. product name can be one of the following {SKU_LIST}.

    ** RULES **
    
    First use the tool get_planning_constraints to get the constraints data:
    1. Unconstrained Demand (Input)
    2. Max Standard Production Capacity (Custom Data Access)
    3. Max Overtime Production Capacity (Custom Data Access)
    4. Max Inventory Value (Custom Data Access)

    Second use the tool get_unit_cost to get the cost data:
    5. Unit Cost (Custom Data Access)

    
    Output all five values clearly for the Consensus LLM Agent.""",
    
    tools=[FunctionTool(get_planning_constraints), FunctionTool(get_unit_cost)],
    output_key="conflicting_data",
)
print("DataIngestionAgent created.")


calculation_agent = Agent(
      name="CalculationAgent",
      model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config, temperature=0.5),
      instruction="""You are a specialized calculator that ONLY responds with Python code. You are forbidden from providing any text, explanations, or conversational responses.
 
     Your task is to take a request for a calculation and translate it into a single block of Python code that calculates the answer.
     
     **RULES:**
    1.  Your output MUST be ONLY a Python code block.
    2.  Do NOT write any text before or after the code block.
    3.  The Python code MUST calculate the result.
    4.  The Python code MUST print the final result to stdout.
    5.  You are PROHIBITED from performing the calculation yourself. Your only job is to generate the code that will perform the calculation.
   
    Failure to follow these rules will result in an error.
       """,
        code_executor=BuiltInCodeExecutor(),
    )
print("CalculationAgent created.")


consensus_manager_agent = Agent(
    name="ConsensusLLMAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config, temperature=0.5),

    
    instruction="""You are a smart S&OP Manager. You must strictly follow these steps and use the available tools.

    **CRITICAL** You must retreive the product name from the user prompt.
    **CRITICAL** You are strictly prohibited from performing any arithmetic calculations yourself. You must use the calculation_agent to generate Python code that calculates all the following steps.
    
    For any product provided you must:

    1. Use the DataIngestionAgent to get all the data for the SKU provided by the user
    #IMPORTANT# State all the data retrieved

     2. Use the calculation_agent to Calculate the Financial Limit:
    Determine the Financial Limit (units) by calculating: Maximum allowable inventory value / Unit cost of the product
    #IMPORTANT# at the end of this step you MUST State only the Financial Limit, do not show the code of the calculation_agent.
    
    3.  Use the calculation_agent to Calculate the Feasible Production Capacity as follow:
    IF 'Unconstrained Demand' is less than or equal to `Max Standard Production Capacity`:
        Set the `Feasible Production Capacity` euqal to `Max Standard Production Capacit`.
        
    ELSE IF `Unconstrained Demand` is greater than `Max Standard Production Capacity` AND less than or equal to `Max Overtime Production Capacity`:
        Set the `Feasible Production Capacity` to `Max Overtime Production Capacity`.

    ELSE IF `Unconstrained Demand` is greater than `Max Overtime Production Capacity`:
        Set the `Feasible Production Capacity` to `Max Overtime Production Capacity`.

    #IMPORTANT# MUST State only the Feasible Production Capacity at the end of this step and the reason, do not show the code of the calculation_agent.

    4.  Use the calculation_agent to Calculate the Final Constrained Plan:
    Final Constrained Plan is the lowest numerical value among the following three numbers:
        * The Unconstrained Demand
        * Feasible Production Capacity (from Step 3)
        * The Financial Limit (from Step 2)

    #IMPORTANT# MUST State only the Final Constrained Plan and explain the reason, do not show the code of the calculation_agent.
    **Output:** finally provide ONLY the final calculated demand figure (a number) in the format 'FINAL_CONSTRAINED_PLAN: [NUMBER]'.
   
    """,
      tools=[AgentTool(agent=data_ingestion_agent), AgentTool(agent=calculation_agent)]
)

print("SOP_manager_agent.")


APP_NAME = "consensus_Manager"
USER_ID = "Senior_S&OP"
current_time = datetime.datetime.now()
SESSION_ID = f"{USER_ID}_{current_time.strftime('%Y%m%d_%H%M%S')}"

session_service = DatabaseSessionService(db_url=db_url)

runner = Runner(agent=consensus_manager_agent, app_name=APP_NAME, session_service=session_service)
response = await runner.run_debug("I need all the data for PROD-A", session_id=SESSION_ID)


print("Upgraded to persistent sessions!")
print(f"   - Application: {APP_NAME}")
print(f"   - User: {USER_ID}")
print(f"    - **Generated Session ID**: {SESSION_ID}")


def get_last_final_demand():
    """
    Connects to the database, retrieves the content of the most recent 
    'DemandCoordinator' event, and extracts the FINAL_DEMAND value.
    """
    with sqlite3.connect("my_sop_gelato_agent.db") as connection:
        cursor = connection.cursor()

        cursor.execute(
            f"""
            SELECT content 
            FROM events 
            WHERE author = 'ConsensusLLMAgent'
            ORDER BY ROWID DESC
            LIMIT 1
            """
        )
        
        # Fetch the single result (content string)
        result = cursor.fetchone()

    if not result:
        return None, "No 'ConsensusLLMAgent' entries found."

    # The result is a tuple, so we extract the string
    content_json_str = result[0]
    
    # 2. Parse the JSON and extract the text
    try:
        content_data = json.loads(content_json_str)
        text_content = content_data.get('parts', [{}])[0].get('text', '')
        
        # 3. Extract the final demand number from the text
        if 'FINAL_CONSTRAINED_PLAN:' in text_content:
  
            final_demand_str = text_content.split('FINAL_CONSTRAINED_PLAN:')[1].strip()
            

            final_demand_value = float(final_demand_str)
            return final_demand_value, "Success"
        else:
            return None, "FINAL_CONSTRAINED_PLAN key not found in the content text."
            
    except (json.JSONDecodeError, IndexError, ValueError) as e:
        return None, f"Error parsing content: {e}"

last_plan, status = get_last_final_demand()

if last_plan is not None:
    print(f"FINAL_CONSTRAINED_PLAN extracted is: {last_plan}")
else:
    print(f"Could not retrieve the FINAL_CONSTRAINED_PLAN. Status: {status}")


# STEP 1: DEFINE THE INPUTS
data = get_planning_constraints('PROD-A')
cost_data = get_unit_cost('PROD-A')

UNCONSTRAINED_DEMAND = data['Unconstrained_Demand']
MAX_STANDARD_PRODUCTION_CAPACITY = data['Max_Production_Standard']
MAX_OVERTIME_PRODUCTION_CAPACITY = data['Max_Production_Overtime']
FINANCIAL_LIMIT = data['Max_Inventory_Value'] / cost_data

# STEP 2: DEFINE THE AGENT'S ACTUAL OUTPUT
agent_fcp_output = last_plan 


# STEP 3: CALCULATE THE MATHEMATICALLY CORRECT FCP (Expected Answer)
FPC = MAX_STANDARD_PRODUCTION_CAPACITY
if UNCONSTRAINED_DEMAND > MAX_STANDARD_PRODUCTION_CAPACITY:
    FPC = MAX_OVERTIME_PRODUCTION_CAPACITY

# Calculate the Final Constrained Plan (FCP) - the minimum of all three limitations.
expected_fcp = min(UNCONSTRAINED_DEMAND, FPC, FINANCIAL_LIMIT)

print(f"[System Check] Calculated Feasible Production Capacity (FPC): {FPC}")
print(f"[System Check] Calculated Expected Final Constrained Plan (FCP): {expected_fcp}")

# STEP 4: PERFORM THE EVALUATION AND REPORT RESULT

if expected_fcp == agent_fcp_output:
    print(f"PASS: The Agent's FCP ({agent_fcp_output}) is mathematically correct.")
else:
    print(f"FAIL: The Agent's FCP ({agent_fcp_output}) is incorrect.")
    print(f"   - Expected Value: {expected_fcp}")
    print(f"   - Check the minimum constraint: Demand ({UNCONSTRAINED_DEMAND}), Production ({FPC}), or Financial Limit ({FINANCIAL_LIMIT})")

