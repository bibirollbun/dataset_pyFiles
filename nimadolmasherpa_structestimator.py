import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    BRIGHTDATA_API_TOKEN = UserSecretsClient().get_secret("BRIGHTDATA_API_TOKEN")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["BRIGHTDATA_API_TOKEN"] = BRIGHTDATA_API_TOKEN
    print("✅ Gemini API key and BRIGHTDATA_API_TOKEN setup complete.")
except Exception as e:
    print(
        f"🔑 Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' and 'BRIGHTDATA_API_TOKEN' to your Kaggle secrets. Details: {e}"
    )


# ---- ADK Core Models & Agents ----
from google.adk.models.google_llm import Gemini
from google.adk.agents import SequentialAgent, LlmAgent

# ---- Tools ----
from google.adk.tools import AgentTool, FunctionTool, google_search, ToolContext
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams

# ---- Sessions, Memory, Execution ----
from google.adk.sessions import DatabaseSessionService, InMemorySessionService
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.tools import load_memory

# ---- MCP ----
from mcp import StdioServerParameters

# ---- App & Resumability ----
from google.adk.apps.app import App, ResumabilityConfig, EventsCompactionConfig

# ---- Execution Runner ----
from google.adk.runners import Runner

# ---- Standard Library ----
import uuid
from typing import Any, Dict
from google.genai import types

print("✅ ADK components imported successfully.")



async def run_session(    # async → this function uses async/await, meaning it runs asynchronously.
    runner_instance: Runner,
    user_queries: list[str] | str = None,
    session_name: str = "default",
):
    print(f"\n ### Session: {session_name}")

    # Get app name from the Runner
    app_name = runner_instance.app_name

    # Attempt to create a new session or retrieve an existing one
    try:
        session = await session_service.create_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
    except:
        session = await session_service.get_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )

    # Process queries if provided
    if user_queries:
        # Convert single query to list for uniform processing
        if type(user_queries) == str:
            user_queries = [user_queries]

        # Process each query in the list sequentially
        for query in user_queries:
            print(f"\nUser > {query}")

            # Convert the query string to the ADK Content format
            query = types.Content(role="user", parts=[types.Part(text=query)])

            # Stream the agent's response asynchronously
            async for event in runner_instance.run_async(
                user_id=USER_ID, session_id=session.id, new_message=query
            ):
                # Check if the event contains valid content
                if event.content and event.content.parts:
                    # Filter out empty or "None" responses before printing
                    if (
                        event.content.parts[0].text != "None"
                        and event.content.parts[0].text
                    ):
                        print(f"{MODEL_NAME} > ", event.content.parts[0].text)
    else:
        print("No queries!")


print("✅ Helper functions defined.")


def show_python_code_and_result(response):
    for i in range(len(response)):
        # Check if the response contains a valid function call result from the code executor
        if (
            (response[i].content.parts)
            and (response[i].content.parts[0])
            and (response[i].content.parts[0].function_response)
            and (response[i].content.parts[0].function_response.response)
        ):
            response_code = response[i].content.parts[0].function_response.response
            if "result" in response_code and response_code["result"] != "```":
                if "tool_code" in response_code["result"]:
                    print(
                        "Generated Python Code >> ",
                        response_code["result"].replace("tool_code", ""),
                    )
                else:
                    print("Generated Python Response >> ", response_code["result"])


print("✅ Helper functions defined.") #This function scans the agent's response and prints any Python code or Python execution results returned by ADK tools.


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
) # If a request fails because of a temporary issue, automatically try again after waiting for a bit.


APP_NAME = "default"  # Application
USER_ID = "default"  # User
SESSION = "default"  # Session

MODEL_NAME = "gemini-2.5-flash-lite"

# Calculation Agent: It takes in a Python code and uses the `BuiltInCodeExecutor` to run it.
calculation_agent = LlmAgent(
    name="CalculationAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
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


# This function ensures that all user-provided building parameters (like plinth area or rates) 
# are numeric and valid before proceeding with calculations.
def validate_numeric_input(value: str) -> dict:
    """Validates that the user input is a valid numeric value.

    Args:
        value: The input string provided by the user.

    Returns:
        Dictionary:
            Success example:
            {"status": "success", "value": 1500.0}

            Error example:
            {"status": "error", "error_message": "Input must be a number"}
    """
    try:
        cleaned = value.replace("sq.m", "").replace("m2", "").strip()
        numeric_value = float(cleaned)
        return {"status": "success", "value": numeric_value}
    except Exception:
        return {
            "status": "error",
            "error_message": f"Invalid numeric value: {value}"
        }

# This agent collects building details from the user (plinth area, plinth rate) 
# and validates them using tools like calculator_agent and validate_numeric_input.
input_agent = LlmAgent(
    name="UserInputAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="A text chatbot that collects data for rough building estimates ",
    instruction="""
You are a user input collection agent for a Building Cost Estimator.

Your Job:
1. Ask the user for essential inputs:
    - Plinth area (in sq.m), Plinth area rate (cost per sq.m)
2. For each value:
    - Ask → Receive → Validate using `validate_numeric_input`
3. End by saying: "All inputs collected successfully."
4. You will receive 2 values for Plinth area and Plinth area rate

""",
    tools=[validate_numeric_input, AgentTool(agent=calculation_agent)],
    output_key="Plinth_estimates"
)

print("✅ User Input Agent created")
print("🔧 Tools loaded:")
print("  • validate_numeric_input")


# Base Cost Agent: Calculates the core building cost using the formula:
# base_cost = plinth_area × plinth_rate and leverages the calculator_agent for precise computation.
base_cost_agent = LlmAgent(
    name="BaseCostCalculatorAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),

    instruction="""
Your job is to compute the base construction cost.

You will receive from session state {Plinth estimates}:
- plinth_area      (example: 1500)
- plinth_rate      (example: 9500)

Your responsibilities:

1. Validate that both values exist.
2. Reads plinth_area and plinth_rate from the provided variables.
3. Computes: base_cost = plinth_area * plinth_rate
4. Prints ONLY the computed numeric result.

Failure to follow these rules results in an error.
    """,
    tools=[AgentTool(agent=calculation_agent)],
    output_key="base_cost"
)

print("✅ base_cost_agent created")



# Climate Agent: Uses the MCP-powered Bright Data server to fetch real-time weather data (temperature, humidity, rainfall) 
# and adjusts add-on building costs accordingly.

# MCP integration with Bright Data Server (Weather Specific)
weather_mcp_server = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=[
                "@brightdata/mcp",
            ],
            env={
                "API_TOKEN": BRIGHTDATA_API_TOKEN,
                "PRO_MODE": "false",      # Optional: enable all tools if needed
            },
            # Only expose weather-relevant tools
            tool_filter=[
                "web_scraper",            # For scraping weather pages
                "web_unblocker",          # For sites that block requests
                "request",                # Simple HTTP weather API fetch
                "browser",                # Render JS-heavy weather sites
            ],
        ),
        timeout=300,
    ),
)

print("✅ Weather MCP Tool created")

climate_agent = LlmAgent(
    name="ClimateAdjustmentAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
You adjust building cost multipliers based on real-time weather data.

Steps:
1. Use the Bright Data MCP Weather Server to fetch annual average weather condition for the building location.
2. Extract rainfall, humidity, and temperature from the MCP response.
3. Adjust multipliers for add-on costs:
   - If rainfall > 20mm → water_sanitary_cost *= 1.10
   - If humidity > 70% → electrical_cost *= 1.05
   - If temperature > 35°C → services_cost *= 1.05
4. Output only a JSON object with the updated multipliers, e.g.:

{
  "architectural": 1.0,
  "water_sanitary": 1.10,
  "electrical": 1.05,
  "services": 1.05,
  "contingencies": 1.0,
  "supervision": 1.0
}

""",
    tools=[weather_mcp_server],  # Use the MCP Bright Data server here
    output_key="adjusted_multipliers"
)

print("✅ Climate Adjustment Agent created with MCP server")



# Add-On Cost Agent: Computes additional building costs (architectural, electrical, water & sanitary, services, contingencies, supervision) 
# and integrates climate-based adjustments from the Climate Agent.
addon_cost_agent = LlmAgent(
    name="AddonCostAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),

    instruction="""
Your job:
1. You will receive base_cost (state key: {base_cost})
2. Calculate:
        architectural = 0.015 * base_cost
        water_sanitary = 0.05 * base_cost
        electrical = 0.14 * base_cost
        services = 0.06 * base_cost
        contingencies = 0.03 * base_cost
        supervision = 0.08 * base_cost
3. You adjust building cost multipliers using climate_agent tool.
   - stores all values in a dictionary named `addon_costs`
   - prints the dictionary
""",
    tools=[
        AgentTool(agent=calculation_agent), 
        AgentTool(agent=climate_agent)
          ],
    output_key="addon_costs"
)

print("✅ Add-On Cost Agent created!")



# Final Cost Calculator Agent: Computes the final building cost by summing base and add-on costs
final_cost_agent = LlmAgent(
    name="FinalCostAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
You will receive:
- base_cost (numeric)
- addon_costs (dictionary with keys: architectural, water_sanitary, electrical, services, contingencies, supervision)

Task:
1. Compute final_total_cost = base_cost + sum(all add-on costs)
2. Return a concise, machine-readable dictionary like:
   {
       "final_total_cost": ...,
       "base_cost": ...,
       "addon_costs": {...}
   }
""",
    code_executor=BuiltInCodeExecutor(),
    output_key="final_building_cost"
)

print("✅ Final Cost Calculator Agent created!")



# Result Collector Agent: Aggregates outputs from the User Input, Base Cost, and Add-On Cost agents, 
# computes the final_total_cost, and prepares structured results for further approval.
result_collector_agent = SequentialAgent(
    name="ResultCollectorAgent",
    sub_agents=[input_agent, base_cost_agent, addon_cost_agent, final_cost_agent],
)

print("✅ Sequential Agent created.")


# Google search agent: for real time information
google_search_agent = LlmAgent(
    name="google_search_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="Searches for information using Google search",
    instruction="Use the google_search tool to find information on the given topic. Return the raw search results.",
    tools=[google_search],
)


# Approval Agent: Reviews the final building estimate and determines if human approval is required, 
# supporting a pausable workflow for large or sensitive projects.

# Threshold for requiring human approval
LARGE_COST_THRESHOLD = 20000  # Example: ₹20,000

def approve_building_estimate(
    final_total_cost: float, tool_context: "ToolContext"
) -> dict:
    """
    Handles approval logic for building cost estimation.

    Requires human approval if the final_total_cost exceeds LARGE_COST_THRESHOLD.

    Args:
        final_total_cost: The total estimated cost of the building
        tool_context: Context object for handling pausable tool logic

    Returns:
        Dictionary with approval status:
        - "approved" if auto-approved or human approved
        - "pending" if waiting for human approval
        - "rejected" if human rejected
    """

    # -----------------------------------------------------------------------------------------------
    # SCENARIO 1: Cost ≤ threshold → auto-approve
    if final_total_cost <= LARGE_COST_THRESHOLD:
        return {
            "status": "approved",
            "final_total_cost": final_total_cost,
            "message": f"Estimate auto-approved: ₹{final_total_cost:,.2f}",
        }

    # -----------------------------------------------------------------------------------------------
    # SCENARIO 2: First time tool is called → require human approval
    if not tool_context.tool_confirmation:
        tool_context.request_confirmation(
            hint=f"⚠️ Large estimate: ₹{final_total_cost:,.2f}. Do you approve?",
            payload={"final_total_cost": final_total_cost},
        )
        return {
            "status": "pending",
            "message": f"Building estimate requires human approval: ₹{final_total_cost:,.2f}",
        }

    # -----------------------------------------------------------------------------------------------
    # SCENARIO 3: Tool is called again after human interaction → handle approval response
    if tool_context.tool_confirmation.confirmed:
        return {
            "status": "approved",
            "final_total_cost": final_total_cost,
            "message": f"Estimate approved by human: ₹{final_total_cost:,.2f}",
        }
    else:
        return {
            "status": "rejected",
            "final_total_cost": final_total_cost,
            "message": f"Estimate rejected by human: ₹{final_total_cost:,.2f}",
        }

print("✅ Long-running building estimate approval tool created!")



# Wrap approval logic in a pausable LlmAgent using the approve_building_estimate tool
approval_agent = LlmAgent(
    name="ApprovalAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
You review the final building estimate and decide if human approval is needed.

Steps:
1. Receive {final_building_cost} from the final_cost_agent.
2. Call the `approve_building_estimate` tool.
3. If status is 'pending', inform the user that human approval is required.
4. If approved, confirm the project can proceed.
5. If rejected, inform the user and halt further steps.
6. Keep responses concise and clear.
""",
    tools=[FunctionTool(func=approve_building_estimate)],
    output_key="approval_status"
)


# Final Aggregator with Approval: Sequentially combines outputs from the Result Collector Agent and 
# the Approval Agent to produce a finalized, structured, and approved building estimate.
final_aggregator_with_approval = SequentialAgent(
    name="FinalAggregatorWithApproval",
    sub_agents=[result_collector_agent, approval_agent],
)

print("✅ Final Aggregator Agent with Pausable Approval created!")


# Root Agent: Oversees the entire workflow, delegating tasks to either the Final Aggregator with 
# Approval or the Google Search Agent based on the user’s query, ensuring efficient and context-aware execution.
root_agent = LlmAgent(
    name="FinalAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),

    instruction="""
You are the Root Orchestrator Agent. Your job is to understand the user's request and route it to the correct tool or sub-agent.

### Available Tools:
1. **FinalAggregatorWithApproval**
   - Full building estimation pipeline (base cost → add-ons → climate → final cost -> result → approval).

2. **GoogleSearchAgent**
   - Use for real-time, factual, up-to-date information.

3. **LoadMemory**
   - Retrieve past session memory when the user refers to previous interactions.

### How to Decide:
- If the query requires *real-time facts* → use GoogleSearchAgent.
- If the task needs *multi-step agent pipeline, merging, climate logic, or approval* → use FinalAggregatorWithApproval.
- If the user asks about something from the *past session* → use LoadMemory.
- If none of the tools apply → answer directly.

Be concise, delegate tasks properly, and never perform a tool's job yourself.
""",

    tools=[
        AgentTool(agent=final_aggregator_with_approval),
        AgentTool(agent=google_search_agent),
        load_memory
    ],
)

print("✅ Root Agent created")



# Wrap the root_agent in a resumable app and Events Compaction enabled
StructEstimator = App(
    name="StructEstimator",
    root_agent=root_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=3,  # Trigger compaction every 3 invocations
        overlap_size=1,  # Keep 1 previous turn for context
    ),
)

print("✅ Resumable Building Estimation App created!")



# Implementing Persistent Sessions to DatabaseSessionService` using SQLite
db_url = "sqlite:///my_agent_data.db"  # Local SQLite file
session_service = DatabaseSessionService(db_url=db_url)

# Step 3: Create a runner with persistent storage
runner = Runner(agent=root_agent, app_name="StructEstimator", session_service=session_service)

print("✅ Upgraded to persistent sessions!")
print(f"   - Database: my_agent_data.db")
print(f"   - Sessions will survive restarts!")


# Test the 
response=await run_session(
    runner,
    ["Hi, I am Nima! Can you provide me a rough estimate of building with plinth area 80 m2 and plinth area rate 9 per m2?"],
    "compaction_demo"
)


response=await run_session(
    runner,
    ["The location is in Bangaluru, India"],
    "compaction_demo"
)


response=await run_session(
    runner,
    ["Can you provide me a rough estimate of building with plinth area 800000 m2 and plinth area rate 900 per m2 and Delhi, India? Show me the detail calculation please."],
    "compaction_demo"
)


response=await run_session(
    runner,
    ["Can you tell me what is the annual average weather condition in Delhi, India?"],
    "compaction_demo"
)


response=await run_session(
    runner,
    ["Can you tell me the current time in India?"],
    "compaction_demo"
)


# Check the event storage in the Database
import sqlite3

def check_data_in_db():
    with sqlite3.connect("my_agent_data.db") as connection:
        cursor = connection.cursor()
        result = cursor.execute(
            "select app_name, session_id, author, content from events"
        )
        print([_[0] for _ in result.description])
        for each in result.fetchall():
            print(each)


check_data_in_db()


# Verifying Compaction in the Session History
# Get the final session state
final_session = await session_service.get_session(
    app_name=runner.app_name,
    user_id=USER_ID,
    session_id="compaction_demo",
)

print("--- Searching for Compaction Summary Event ---")
found_summary = False
for event in final_session.events:
    # Compaction events have a 'compaction' attribute
    if event.actions and event.actions.compaction:
        print("\n✅ SUCCESS! Found the Compaction Event:")
        print(f"  Author: {event.author}")
        print(f"\n Compacted information: {event}")
        found_summary = True
        break

if not found_summary:
    print(
        "\n❌ No compaction event found. Try increasing the number of turns in the demo."
    )


# Clean up any my_agent_data database to start fresh (if Notebook is restarted)
import os

if os.path.exists("my_agent_data.db"):
    os.remove("my_agent_data.db")
print("✅ Cleaned up old database files")

