import os
import uuid
import json
import asyncio
from kaggle_secrets import UserSecretsClient
# CRITICAL FIX: Import Optional for LRO parameters
from typing import List, Dict, Optional 

# --- Authentication & Setup ---
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("Gemini API key setup complete.")
except Exception as e:
    print(
        f"Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )

# --- ADK Imports for LRO, App, and Sessions ---
from google.adk.agents import Agent, SequentialAgent, LlmAgent 
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService 
from google.adk.apps.app import App, ResumabilityConfig 
from google.adk.tools import google_search, FunctionTool, AgentTool 
from google.adk.code_executors import BuiltInCodeExecutor 
from google.adk.tools.tool_context import ToolContext 
from google.genai import types

# Retry Configuration
retry_config=types.HttpRetryOptions(
    attempts=5, exp_base=7, initial_delay=1, http_status_codes=[429, 500, 503, 504] 
)

# --- 3. Custom LRO Tool with Pause/Resume Logic ---
def get_farmer_data(
    location: Optional[str] = None, 
    crops: Optional[List[str]] = None, 
    tool_context: ToolContext = None 
) -> bool: 
    """
    Checks for location/crops. If missing, PAUSES and asks the user for input (LRO).
    Returns: Boolean (watered status) if data is complete.
    """
    watering_history = {
        "california, usa": False,
        "chennai, india": True, 
        "paris, france": False,
    }
    
    # --- SCENARIO 1: MISSING MANDATORY INPUTS (Trigger LRO Pause) ---
    if (not location or not crops) and (not tool_context or not tool_context.tool_confirmation):
        if tool_context:
            tool_context.request_confirmation(
                hint=f"I am missing the location and/or crop list. Please provide the missing information.",
                payload={"missing_data": True} 
            )
            # [cite_start]Return 'pending' status to the Agent [cite: 826-829]
            return {
                "status": "pending",
                "message": "Missing mandatory data. Requires user input to continue."
            }
        
        raise ValueError("Location and Crops are mandatory inputs.")

    # --- SCENARIO 2: ALL INPUTS ARE PRESENT (Final execution) ---
    loc_key = location.lower().strip()
    return watering_history.get(loc_key, False) 

FarmerDataTool = FunctionTool(get_farmer_data) 
print("\nFarmerDataTool (Custom LRO Tool) created.")


# --- 4. Specialized Agents (Optimized Prompts and Robust Flow) ---

# Calculation Agent remains the same
calculator_agent = LlmAgent(
    name="CalculationAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""PERSONA: You are a specialized calculator that ONLY responds with Python code. TASK: Translate the arithmetic request into a single Python code block that prints the final result to stdout. RULES: 1. Output MUST be ONLY a Python code block. 2. You are PROHIBITED from performing the calculation yourself.""", 
    code_executor=BuiltInCodeExecutor() 
)
print("CalculatorAgent created.")

# A. InputProcessorAgent (Step 1: Executes LRO Tool for Status)
input_processor_agent = Agent(
    name="InputProcessorAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    PERSONA: You are a dedicated Data Ingestion Agent. Your task is to extract the 'location' and 'crops' from the user's query and call the get_farmer_data tool. If the tool returns 'pending', tell the user you are waiting for data.
    """,
    tools=[FarmerDataTool],
    output_key="watered_status" # Saves the True/False result
)
print("InputProcessorAgent created.")

# B. ResearchAgent (Step 2: Reads location from history and executes enhanced searches)
research_agent = Agent(
    name="ResearchAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="A specialist in finding current, external information like weather forecasts and seasonality.",
    instruction="""
    PERSONA: You are a Geospatial Research Engine. **You MUST identify the location and crops from the chat history.**
    
    TASK: Use google_search to find and consolidate the results for the following specific data points related to the identified location and crops:
    1. Today's exact date and day.
    2. The 3-day rain chance (as percentages).
    3. The dominant agricultural season.
    4. Current **Soil Temperature** and **Evapotranspiration (ET)** rate.
    5. **Typical agricultural soil pH range**.
    6. **Price per kg of Urea and DAP** in the local market.
    
    OUTPUT RULE: Format all findings into a strict, dense 'Research Report'. Do not offer opinions or advice.
    """,
    tools=[google_search],
    output_key="research_report"
)
print("ResearchAgent created.")

# C. AdvisorAgent (Step 3: Synthesis with Final Risk-Based Logic)
advisor_agent = Agent(
    name="AdvisorAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="A specialist in synthesizing data to provide the final irrigation and fertilizer advice.",
    instruction="""
    PERSONA: You are the Chief Agronomy Advisor.
    INPUTS: Watering Status ({watered_status}) and Research Report ({research_report}).
    TASK: Generate a final farmer briefing. You MUST use the 'CalculationAgent' tool for any arithmetic.
    
    LOGIC (Risk-Based Weighing):
    1. **Primary Risk Check (Overwatering):** If the 'Watering Status' is **True** (recently watered), the decision is **NO**. This is the highest priority rule to prevent root rot.
    2. **Secondary Need Check:** If the 'Watering Status' is **False**, proceed to recommend watering unless the '3-day rain chance' in the Research Report is greater than 50%.
    
    FINAL OUTPUT FORMAT (Strict):
    - Start with the **CURRENT DATE AND DAY** extracted from the Research Report.
    - Follow with a clear, bold **WATERING DECISION: YES/NO**.
    - Immediately follow the decision with a section titled **WATERING ANALYSIS** (Brief reason: Explaining how you weighed the risk of overwatering against the current ET rate and rain chance).
    - Conclude with a detailed **Fertilizer Recommendation** (2-3 key points based on crops/season, pH, and cost-effectiveness).
    """,
    output_key="final_advice",
    tools=[AgentTool(agent=calculator_agent)]
)
print("AdvisorAgent created.")


# [cite_start]--- 5. LRO App & Runner Setup [cite: 890-894, 900-903] ---
root_agent = SequentialAgent(
    name="IrrigationAdvisorPipeline",
    description="Orchestrates data collection, external research, and final irrigation advice.",
    sub_agents=[
        input_processor_agent, 
        research_agent,        
        advisor_agent          
    ]
)

# 1. Wrap the root agent in a resumable App 
shipping_app = App(
    name="irrigation_coordinator",
    root_agent=root_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)
session_service = InMemorySessionService() 

# 2. Use the base Runner with the App and Session Service 
lro_runner = Runner(
    app=shipping_app, 
    session_service=session_service,
)
print("\nLRO Runner created, ready for resumable execution.")

# Define the helper functions for LRO execution
def check_for_approval(events):
    """Check if events contain an approval/confirmation request (PAUSE signal)."""
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if (
                    hasattr(part, "function_call")
                    and part.function_call
                    and part.function_call.name == "adk_request_confirmation"
                ):
                    return {
                        "approval_id": part.function_call.id,
                        "invocation_id": event.invocation_id,
                        "hint": part.function_call.args.get("hint"),
                    }
    return None

def create_approval_response(approval_info, response_text):
    """Creates a FunctionResponse that ADK understands for resumption."""
    confirmation_response = types.FunctionResponse(
        id=approval_info["approval_id"],
        name="adk_request_confirmation",
        response={"confirmed": True, "message": "Input received."} 
    )
    # The new user text is the key to resuming the agent's logic thread
    return types.Content(
        role="user", 
        parts=[
            types.Part(function_response=confirmation_response),
            types.Part(text=response_text)
        ]
    )



# [cite_start]Custom LRO Workflow Function [cite: 970-989]
async def run_lro_workflow(query: str):
    user_id = "farmer_user"
    session_id = f"session_{uuid.uuid4().hex[:8]}" 

    await session_service.create_session(
        app_name="irrigation_coordinator", user_id=user_id, session_id=session_id
    )
    query_content = types.Content(role="user", parts=[types.Part(text=query)])
    events = []
    final_response_text = ""
    
    print(f"\n{'='*60}\nUser > {query}\n{'='*60}")
    
    # --- STEP 1: Send initial request (PAUSE point) ---
    async for event in lro_runner.run_async(
        user_id=user_id, session_id=session_id, new_message=query_content
    ):
        events.append(event)
        if event.content and hasattr(event.content.parts[0], 'text'):
            print(f"Agent > {event.content.parts[0].text}")
    
    # --- STEP 2: Detect Pause Event ---
    approval_info = check_for_approval(events)

    if approval_info:
        print("\n*** AGENT PAUSED: Missing Mandatory Data (LRO Detected) ***")
        
        # User Interaction: This is where the workflow pauses for interactive input
        missing_input = input(f"Agent prompt: {approval_info['hint']}\nYour Answer (e.g., 'corn and tomatoes in Paris'): ")
        
        # [cite_start]--- STEP 3: Call Agent AGAIN to RESUME [cite: 1021-1030] ---
        resume_content = create_approval_response(approval_info, missing_input)
        
        print(f"\n*** RESUMING AGENT with new input: '{missing_input}' ***")
        
        async for event in lro_runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=resume_content, # Send the combined approval/text payload
            invocation_id=approval_info["invocation_id"], # CRITICAL: Resumes thread
        ):
            if event.content and event.content.parts and hasattr(event.content.parts[0], 'text'):
                final_response_text = event.content.parts[0].text
                print(f"Agent > {final_response_text}")
    else:
        # If no pause, the final answer should be in the last event's text content
        if events and events[-1].content and hasattr(events[-1].content.parts[0], 'text'):
            final_response_text = events[-1].content.parts[0].text
            print(f"Agent > {final_response_text}")
        
    print(f"\n{'='*60}")
    print("LRO Workflow complete.")
    
    return final_response_text



# --- Execution Test (Will trigger the LRO pause) ---
# This query is intentionally missing the location and crops, forcing the LRO PAUSE.
await run_lro_workflow("What should I do about watering?")



!adk create advisor-agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY
print("Advisor agent created successfully")



from IPython.core.display import display, HTML 
from jupyter_server.serverapp import list_running_servers 

# Helper function to get proxied URL for ADK Web UI: Day 4
def get_adk_proxy_url():
    PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
    ADK_PORT = "8000"
    servers = list(list_running_servers())
    if not servers:
        raise Exception("No running Jupyter servers found.")
    baseURL = servers[0]["base_url"]
    try:
        path_parts = baseURL.split("/")
        kernel = path_parts[2]
        token = path_parts[3]
    except IndexError:
        raise Exception(f"Could not parse kernel/token from base URL: {baseURL}")
    url_prefix = f"/k/{kernel}/{token}/proxy/proxy/{ADK_PORT}"
    url = f"{PROXY_HOST}{url_prefix}"
    
    # [cite_start]Styled HTML for the button: Day 4 [cite: 1327-1347]
    styled_html = f"""
    <div style="padding: 15px; border: 2px solid #f0ad4e; border-radius: 8px; background-color: #fef9f0; margin: 20px 0;">
    <div style="font-family: sans-serif; margin-bottom: 12px; color: #333; font-size: 1.1em;">
    <strong> IMPORTANT: Action Required</strong>
    </div>
    <div style="font-family: sans-serif; margin-bottom: 15px; color: #333; line-height: 1.5;">
    The ADK web Ul is <strong>not running yet</strong>. You must start it in the next cell.
    <ol style="margin-top: 10px; padding-left: 20px;">
    <li style="margin-bottom: 5px;"><strong>Run the next cell</strong> (the one with <code>!adk web ...</code>) to start the ADK web UI.</li>
    <li style="margin-bottom: 5px;">Wait for that cell to show it is "Running" (it will not "complete").</li>
    <li>Once it's running, <strong>return to this button</strong> and click it to open the UI.</li>
    </ol>
    <em style="font-size: 0.9em; color: #555;">(If you click the button before running the next cell, you will get a 500 error.)</em>
    </div>
    <a href='{url}' target='_blank' style="
    display: inline-block; background-color: #1a73e8; color: white; padding: 10px 20px;
    text-decoration: none; border-radius: 25px; font-family: sans-serif; font-weight: 500;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.2s ease;">
    Open ADK Web UI (after running cell below)
    </a>
    </div>
    """
    display(HTML(styled_html))
    return url_prefix

# Get the URL prefix and display the instruction button
url_prefix = get_adk_proxy_url()

print("\n--- Running ADK Web UI ---")
!adk web --url_prefix {url_prefix} --host 0.0.0.0 --port 8000 --log_level DEBUG


