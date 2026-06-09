!pip install google-adk google-genai uvicorn python-dotenv fastapi nest-asyncio


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… GOOGLE_API_KEY has been configured.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


import asyncio
import uuid
import uvicorn
import nest_asyncio
from multiprocessing import Process

from dotenv import load_dotenv

# Google GenAI and ADK Imports
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.tools import ToolContext, AgentTool, google_search, FunctionTool
from google.adk.agents import LlmAgent, ParallelAgent, LoopAgent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini
from google.genai import types

# Apply a patch to allow asyncio to run within the notebook's existing event loop
nest_asyncio.apply()


# Custom Tools, Long-Running Operations

def calculate_cycle_phase(days_since_period: int) -> dict:
    """
    Calculates the menstrual cycle phase based on days since last period.
    """
    if days_since_period <= 5:
        phase = "Menstrual"
    elif days_since_period <= 13:
        phase = "Follicular"
    elif days_since_period == 14:
        phase = "Ovulation"
    else:
        phase = "Luteal"
    return {"phase": phase, "status": "success"}

def book_appointment_tool(doctor_name: str, time: str, tool_context: ToolContext) -> dict:
    """
    Books an appointment. 
    CRITICAL: This tool requires HUMAN APPROVAL for a deposit.
    """
    if not tool_context.tool_confirmation:
        tool_context.request_confirmation(
            hint=f"Booking with {doctor_name} at {time} requires a $50 deposit. Do you authorize this charge?",
            payload={"doctor": doctor_name, "time": time}
        )
        return {"status": "pending", "message": "Waiting for user approval for deposit."}

    if tool_context.tool_confirmation.confirmed:
        return {"status": "confirmed", "message": f"SUCCESS: Appointment confirmed with {doctor_name} at {time}. Deposit charged."}
    else:
        return {"status": "rejected", "message": "Booking cancelled by user."}

def exit_loop_tool() -> dict:
    """
    Call this tool ONLY when the text has been successfully simplified and approved.
    """
    return {"status": "loop_exited", "message": "Text simplified successfully."}

print("âœ… Custom tools defined.")


# Agent2Agent

# Define the Knowledge Base Agent
retry_config = types.HttpRetryOptions(attempts=3, initial_delay=1)
library_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="medical_library",
    description="An external medical encyclopedia service. Use this to define complex terms.",
    instruction="""
    You are a Medical Library. Your goal is to provide precise, dictionary-style definitions 
    for women's health terms (e.g., 'Hirsutism', 'Dysmenorrhea').
    Keep definitions concise (under 50 words) and strictly factual.
    """
)

# Expose the agent via the A2A Protocol
remote_app = to_a2a(library_agent, port=8001)

def run_server():
    """Function to run the Uvicorn server."""
    print("ğŸ�¥ Starting Medical Library Server on port 8001...")
    uvicorn.run(remote_app, host="0.0.0.0", port=8001)

print("âœ… Remote agent server defined.")


# Multi-Agent Systems

# --- 1. The Remote Specialist (A2A) ---
remote_library_agent = RemoteA2aAgent(
    name="medical_library_proxy",
    description="Use this to get precise definitions of medical terms.",
    agent_card="http://localhost:8001/.well-known/agent-card.json"
)

# --- 2. The Holistic Care Team (Parallel) ---
diet_agent = LlmAgent(name="Dietitian", model="gemini-2.5-flash-lite", instruction="Provide 3 nutrition tips for the user's condition.", tools=[google_search])
fitness_agent = LlmAgent(name="FitnessCoach", model="gemini-2.5-flash-lite", instruction="Provide 3 safe exercises for the user's condition.", tools=[google_search])
mental_agent = LlmAgent(name="Therapist", model="gemini-2.5-flash-lite", instruction="Provide 1 mindfulness technique.", tools=[google_search])
holistic_team = ParallelAgent(name="HolisticCareTeam", sub_agents=[diet_agent, fitness_agent, mental_agent])

# --- 3. The Jargon Simplifier (Loop) ---
simplifier = LlmAgent(name="Simplifier", model="gemini-2.5-flash-lite", instruction="Rewrite the provided medical text for a 12-year-old.", output_key="simplified_text")
critic = LlmAgent(name="Critic", model="gemini-2.5-flash-lite", instruction="Check {simplified_text}. If simple, call 'exit_loop_tool'. If not, explain why.", tools=[FunctionTool(exit_loop_tool)])
jargon_loop = LoopAgent(name="JargonLoop", sub_agents=[simplifier, critic], max_iterations=3)

# --- 4. The Orchestrator (Root) ---
root_agent = LlmAgent(
    name="FemHealth_Orchestrator",
    model="gemini-2.5-flash-lite",
    description="Main coordinator for FemHealth 360.",
    instruction="""
    You are the FemHealth 360 Concierge.
    1. For a condition (e.g., PCOS), use 'HolisticCareTeam'.
    2. For cycle dates, use 'calculate_cycle_phase'.
    3. For definitions, use 'medical_library_proxy'.
    4. To book a doctor, use 'book_appointment_tool'.
    5. Always simplify medical advice with 'JargonLoop'.
    """,
    tools=[
        AgentTool(holistic_team),
        AgentTool(jargon_loop),
        AgentTool(remote_library_agent),
        FunctionTool(calculate_cycle_phase),
        FunctionTool(book_appointment_tool)
    ]
)

print("âœ… Multi-agent system defined.")


# Sessions, Memory, Observability, App/Workflow

APP_NAME_DISPLAY = "FemHealth360"
APP_NAME_DB_KEY = "agents"

# --- Configuration ---
session_service = DatabaseSessionService(db_url="sqlite:///femhealth_sessions.db")
memory_service = InMemoryMemoryService()

app = App(
    name=APP_NAME_DB_KEY,
    root_agent=root_agent,
    plugins=[LoggingPlugin()],
    resumability_config=ResumabilityConfig(is_resumable=True)
)

runner = Runner(
    app=app,
    session_service=session_service,
    memory_service=memory_service,
)

# --- Workflow Logic ---
def check_for_approval(events):
    """Scans events for the human-in-the-loop trigger."""
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.function_call and part.function_call.name == "adk_request_confirmation":
                    return {"approval_id": part.function_call.id, "invocation_id": event.invocation_id, "hint": "Booking requires approval."}
    return None

def create_approval_response(approval_info, approved: bool):
    """Creates the payload to resume the agent."""
    response_payload = types.FunctionResponse(id=approval_info["approval_id"], name="adk_request_confirmation", response={"confirmed": approved})
    return types.Content(role="user", parts=[types.Part(function_response=response_payload)])

print("âœ… Application runner configured.")


async def run_chat_loop():
    user_id = "user_123"
    session_id = f"session_{uuid.uuid4().hex[:6]}"
    
    try:
        await session_service.create_session(app_name=APP_NAME_DB_KEY, user_id=user_id, session_id=session_id)
    except Exception as e:
        print(f"[System Warning: Could not create session (likely exists): {e}]")

    print(f"ğŸ�¥ {APP_NAME_DISPLAY} initialized (Session: {session_id})")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("You > ")
        if user_input.lower() in ["quit", "exit"]:
            break

        print("\nğŸ¤– Processing...")
        
        # 1. Send Message
        events = []
        user_msg = types.Content(role="user", parts=[types.Part(text=user_input)])
        
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=user_msg):
            events.append(event)
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(f"Agent > {part.text}")

        # 2. Check for Pauses (Human-in-the-Loop)
        approval_req = check_for_approval(events)
        
        if approval_req:
            print(f"\nâš ï¸� SYSTEM: {approval_req['hint']}")
            decision = input("Do you approve? (yes/no) > ").strip().lower()
            is_approved = (decision == "yes")
            
            print("ğŸ¤– Resuming workflow...")
            
            # 3. Resume Execution
            resume_msg = create_approval_response(approval_req, is_approved)
            
            async for event in runner.run_async(
                user_id=user_id, 
                session_id=session_id, 
                new_message=resume_msg,
                invocation_id=approval_req["invocation_id"]
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            print(f"Agent > {part.text}")

        # 4. Post-Turn Memory Ingestion
        current_session = await session_service.get_session(app_name=APP_NAME_DB_KEY, session_id=session_id, user_id=user_id)
        await memory_service.add_session_to_memory(current_session)
        print("\n[System: Interaction saved to Memory Bank]\n")

print("âœ… Main chat loop defined.")


# --- Start the Remote Agent Server in the Background ---
# I use multiprocessing to run the server without blocking the notebook.
server_process = Process(target=run_server)
server_process.start()

# Give the server a moment to initialize
import time
time.sleep(5) 

# --- Start the Chat Application ---
try:
    # The nest_asyncio patch allows to run this directly in the notebook
    asyncio.run(run_chat_loop())
except KeyboardInterrupt:
    print("\nExiting chat loop.")
finally:
    # --- Clean up the background server process ---
    print("Shutting down remote server...")
    server_process.terminate()
    server_process.join()
    print("Cleanup complete.")

