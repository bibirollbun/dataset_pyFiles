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


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


import datetime
import logging
import uuid
import textwrap
import functools
from typing import Dict, Any, Optional
import asyncio

# ==============================================================================
# PROJECT: GlucoGuard AI
# ARCHITECTURE: Google Cloud ADK (Native Patterns)
# CONCEPTS APPLIED:
#   1. Multi-Agent System (LlmAgent + AgentTool for delegation)
#   2. Tools (Custom Python functions with Context injection)
#   3. Sessions & Memory (InMemory Services)
#   4. Observability (Custom Decorator Tracing)
#   5. Deployment (AdkApp Runner)
# ==============================================================================
from google.adk.agents import Agent, LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.google_search_tool import google_search
from google.adk.tools.tool_context import ToolContext
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.runners import InMemoryRunner, Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import load_memory, preload_memory
from google.adk.memory import InMemoryMemoryService
from google.genai import types

USER_ID="user-01"
MODEL_NAME="gemini-2.5-flash-lite"
APP_NAME="glucoguard-app"

# ------------------------------------------------------------------------------
# 1. Observability Layer (Logging & Tracing)
# ------------------------------------------------------------------------------

# Configure structured logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s [%(name)s] %(message)s')
logger = logging.getLogger("GlucoGuard")

def trace_execution(func):
    """Decorator to log agent/tool execution flow."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        caller_id = getattr(args[0], 'name', 'System') if args else 'System'
        logger.info(f"EXEC_START: {func.__name__} | Caller: {caller_id}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"EXEC_END: {func.__name__} | Status: Success")
            return result
        except Exception as e:
            logger.error(f"EXEC_FAIL: {func.__name__} | Error: {str(e)}")
            raise e
    return wrapper

# ------------------------------------------------------------------------------
# 2. Custom Tools (The "Hands" of the Agent)
# ------------------------------------------------------------------------------

@trace_execution
def log_biometric(tool_context: ToolContext, metric_type: str, value: float, unit: str) -> str:
    """
    Records biometric data (e.g., blood glucose, weight) to the session state.
    
    Args:
        metric_type: The type of measurement (e.g., 'glucose', 'insulin').
        value: The numerical value.
        unit: The unit of measurement (e.g., 'mg/dL', 'units').
    """
    # Utilization of ADK Session Service for state management
    session = tool_context.session
    if "biometrics" not in tool_context.state:
        tool_context.state["biometrics"] = []
    
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "type": metric_type,
        "value": value,
        "unit": unit
    }
    tool_context.state["biometrics"].append(entry)
    # print(tool_context.state)
    
    # Also persist to Long Term Memory if critical
    if metric_type == "glucose" and (value < 70 or value > 250):
        tool_context.memory.add(f"Critical Event: {metric_type} reading of {value} {unit}")

    return f"Recorded {metric_type}: {value} {unit}"

@trace_execution
def fetch_last_readings(tool_context: ToolContext) -> str:
    """Retrieves the last 3 biometric readings from the current session."""
    session = tool_context.session
    readings = tool_context.state.get("biometrics", [])
    if not readings:
        return "No readings recorded in this session."
    
    recent = readings[-3:]
    return f"Recent history: {recent}"


async def run_session(
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


print("âœ… Helper functions defined.")

# ------------------------------------------------------------------------------
# 3. Multi-Agent Setup (The "Brains")
# ------------------------------------------------------------------------------

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

# --- SUB-AGENT: Clinical Risk Evaluator ---
# Specialized agent focused purely on medical triage logic.
risk_evaluator = LlmAgent(
    name="risk_specialist",
    model= Gemini(model="gemini-2.5-flash-lite",retry_options=retry_config),  # Faster, lower latency model for triage
    instruction=textwrap.dedent("""
        You are the Clinical Risk Evaluator.
        INPUT: Patient symptoms and/or glucose levels.
        OUTPUT: 
        - Risk Level: (LOW, MODERATE, HIGH, CRITICAL)
        - Immediate Action: (e.g., "Consume 15g fast-acting carbs", "Call 911")
        
        RULES:
        - Glucose < 70 mg/dL is HYPOGLYCEMIA (High/Critical).
        - Glucose > 250 mg/dL is HYPERGLYCEMIA (Moderate/High).
        - Be concise. No conversational filler.
    """)
)


# --- ROOT AGENT: Care Orchestrator ---
# The main interface that manages the conversation and delegates tasks.
orchestrator = LlmAgent(
    name="care_orchestrator",
    model=Gemini(model="gemini-2.5-flash",retry_options=retry_config),  # Higher reasoning capability
    instruction=textwrap.dedent("""
        You are 'GlucoGuard', a compassionate diabetes assistant.
        
        WORKFLOW:
        1. If user provides health data (glucose, weight), use 'log_biometric'.
        2. If user describes feeling unwell or provides dangerous numbers, 
           you MUST consult the 'risk_specialist' agent immediately.
        3. If user asks about history, use 'fetch_last_readings'.
        
        Always retain a supportive tone, but defer medical advice to the specialist.
    """),
    tools=[
        # CONCEPT: Agent-as-a-Tool (Native ADK delegation)
        # This exposes the risk_evaluator agent as a callable tool to the Orchestrator
        AgentTool(agent=risk_evaluator),
        preload_memory,
        # Custom Tools
        log_biometric,
        fetch_last_readings
    ]
)

# ------------------------------------------------------------------------------
# 4. Deployment & Execution (The Runner)
# ------------------------------------------------------------------------------

# The AdkApp wraps the agent for deployment (FastAPI compatible)
# This enables the A2A (Agent-to-Agent) protocol endpoints.
app = App(
    name="glucoguard-app",
    root_agent=orchestrator,
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=3,  # Trigger compaction every 3 invocations
        overlap_size=1,  # Keep 1 previous turn for context
    ),
)

session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()
runner = Runner(agent=orchestrator, app_name="glucoguard-app", session_service=session_service,memory_service=memory_service,)

async def run_simulation():
    print("\n>>> GlucoGuard AI System Startup (Google ADK Mode)...")
    
    session_id = str(uuid.uuid4())
    print(f">>> Session ID: {session_id}\n")

    scenarios = [
        # Scenario 1: Routine Logging (Uses Custom Tool)
        "I just checked my sugar, it's 110 mg/dL.",
        
        # Scenario 2: Context Retrieval (Uses Session State)
        "What was that last reading I gave you?",
        
        # Scenario 3: Critical Event (Triggers AgentTool Delegation -> Risk Specialist)
        "I'm feeling really shaky and sweating. My sugar dropped to 55."
    ]

    # for user_input in scenarios:
    #     print(f"ğŸ‘¤ Patient: {user_input}")
        
        # Invoke the ADK Runner
        # In a real deployment, this would be an HTTP POST to the Agent endpoint
        # response = app.agent.invoke(
        #     input=user_input,
        #     session_id=session_id,
        #     # Explicitly passing services to ensure tool context works
        #     session_service=app.session_service,
        #     memory_service=app.memory_service
        # )
        

    try:  # run_debug() requires ADK Python 1.18 or higher:
        response = await run_session(runner,scenarios,session_id,)
        

    except Exception as e:
        print(f"An error occurred during agent execution: {e}")
    
        print(f"ğŸ¤– GlucoGuard: {response.text}\n")
        print("-" * 60)

    # Observability: Dump Session State at end
    final_session = await session_service.get_session(
    app_name=APP_NAME, user_id=USER_ID, session_id=session_id
)
    print(">>> [DEBUG] Final Session State Dump:")
    print("ğŸ“� Session contains:")
    # print(final_session)
    # print(final_session.state)
    for event in final_session.events:
        if event.content and event.content.parts:
            if (
                        event.content.parts[0].text != "None"
                        and event.content.parts[0].text
                    ):
                text = (
                    event.content.parts[0].text[:60]
                    if event.content and event.content.parts
                    else "(empty)"
                )
                print(f"  {event.content.role}: {text}...")
    await memory_service.add_session_to_memory(final_session)
    print("âœ… Session added to memory!")

if __name__ == "__main__":
    await run_simulation()


# Confirming data is stored in memory.
response = await run_session(runner,["What did i tell you before?"],"session-02",)




