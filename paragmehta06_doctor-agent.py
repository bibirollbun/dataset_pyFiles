# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import os
from kaggle_secrets import UserSecretsClient
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google.adk.agents import LlmAgent
from google.adk.agents.remote_a2a_agent import (
    RemoteA2aAgent,
    AGENT_CARD_WELL_KNOWN_PATH,
)

from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

print("âœ… ADK components imported successfully.")


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


print("âœ… Helper functions defined.")


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


import os
from google import genai
from google.genai.errors import APIError
from typing import Callable
# Assuming you have the helper and retry functions defined as per your description
# If you didn't define the helper/retry, you might need to use genai.Client() directly
# and handle errors as shown in the retry_call function below.

# --- Configuration ---
# The client should pick up the API key from the environment variable 
# 'GEMINI_API_KEY' which you've mentioned you set up.
try:
    client = genai.Client()
except Exception as e:
    print(f"Error initializing Gemini client. Make sure GEMINI_API_KEY is set: {e}")
    # Fallback to a client without environment variable check (if your setup allows)
    # client = genai.Client(api_key="YOUR_API_KEY_HERE") 

# Define the model to use
MODEL_NAME = 'gemini-2.5-flash' 
# You may want to use a model like gemini-2.5-pro for more complex medical reasoning.

# --- Helper Function to Call Gemini API ---

# The retry function you mentioned would wrap the generate_content call.
# A simplified, non-decorator version is used here for demonstration.
def safe_generate_content(
    prompt: str, 
    max_retries: int = 3, 
    model_name: str = MODEL_NAME
) -> str:
    """A helper function to call Gemini API with a system instruction and retry logic."""
    for attempt in range(max_retries):
        try:
            # Set a clear system instruction for a medical context
            system_instruction = (
                "You are an AI medical assistant. All your responses MUST begin with "
                "a **DISCLAIMER: I am an AI and cannot provide professional medical advice. "
                "Consult a qualified healthcare professional for any health concerns.** "
                "Be concise, helpful, and focus only on providing the requested information."
            )
            
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt],
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.4 # Lower temperature for factual and consistent advice
                )
            )
            
            return response.text
        
        except APIError as e:
            print(f"Attempt {attempt + 1} failed with APIError: {e}")
            if attempt < max_retries - 1 and (e.code == 429 or e.code >= 500):
                # Retry on Rate Limit (429) or Server Error (5xx)
                import time
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            return f"An error occurred while fetching information: {e}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"
            
    return "Failed to get a response after multiple retries."


# --- Custom Tools for Agent 1 (Suggestions) and Agent 2 (Causes) ---

def suggest_tool(symptoms: str) -> str:
    """
    Suggests preliminary home care or medical next steps for the given symptoms.
    This tool is for the 'Suggestion Agent'.
    
    Args:
        symptoms: A string describing the user's medical symptoms.
        
    Returns:
        A string containing suggestions or next steps from the Gemini model.
    """
    prompt = (
        f"Based on the following symptoms: '{symptoms}', provide concise and general "
        "preliminary home care suggestions and a clear next step (e.g., when to see a doctor/urgent care). "
        "Format the response clearly with bullet points."
    )
    return safe_generate_content(prompt)

# Register the function name for clarity
suggest_tool.__name__ = "suggest_tool"
suggest_tool.__doc__ = suggest_tool.__doc__

def causes_tool(symptoms: str) -> str:
    """
    Lists the most common and relevant potential causes for the given symptoms.
    This tool is for the 'Causes Agent'.
    
    Args:
        symptoms: A string describing the user's medical symptoms.
        
    Returns:
        A string containing a list of potential causes from the Gemini model.
    """
    prompt = (
        f"List the 3 to 5 most common and relevant potential causes for the following symptoms: '{symptoms}'. "
        "Provide a brief one-sentence description for each cause. "
        "Format the response as a numbered list."
    )
    return safe_generate_content(prompt)

# Register the function name for clarity
causes_tool.__name__ = "causes_tool"
causes_tool.__doc__ = causes_tool.__doc__

# --- Example Usage (Optional) ---
# NOTE: You will use these functions when defining your agents, 
# but here is how they work in isolation:
# if __name__ == '__main__':
#     test_symptoms = "severe headache, fever, and stiff neck"
#     print("--- Suggestions Tool Output ---")
#     suggestions = suggest_tool(test_symptoms)
#     print(suggestions)
#     print("\n--- Causes Tool Output ---")
#     causes = causes_tool(test_symptoms)
#     print(causes)


# --- IMPORTANT: Ensure these objects are available in your current scope ---
# You must have defined:
# 1. The custom tool function: causes_tool (from the previous step)
# 2. The Gemini model configuration: Gemini (from your framework)
# 3. The retry configuration: retry_config (from your framework)
# 4. The Agent class: LlmAgent (from your framework)
# ---

# Define the Causes Agent
causes_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    name="CausesAgent",
    description="An AI medical agent specializing in finding the common causes of a medical symptom.",
    instruction="""
    You are a medical causes specialist. 
    Your sole task is to determine the most common and relevant potential causes 
    for the symptoms provided by the user (or another agent).
    You MUST use the 'causes_tool' to fetch this information.
    Provide the list of causes clearly and concisely.
    Always start your response with a medical disclaimer.
    """,
    tools=[causes_tool],  # Register the custom causes tool
)

print("âœ… Causes Agent created successfully!")
print("    Name: CausesAgent")
print("    Model: gemini-2.5-flash")
print("    Tool: causes_tool()")
print("    Ready for A2A communication...")


# Convert the product catalog agent to an A2A-compatible application
# This creates a FastAPI/Starlette app that:
#   1. Serves the agent at the A2A protocol endpoints
#   2. Provides an auto-generated agent card
#   3. Handles A2A communication protocol
from google.adk.a2a.utils.agent_to_a2a import to_a2a
causes_a2a_app = to_a2a(
    causes_agent, port=8001  # Port where this agent will be served
)

print("âœ… Cases Agent is now A2A-compatible!")
print("   Agent will be served at: http://localhost:8001")
print("   Agent card will be at: http://localhost:8001/.well-known/agent-card.json")
print("   Ready to start the server...")


import os
import subprocess
import requests
import time
from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini
from google.genai import types

# --- 1. Define the custom tool (assuming it's defined and imported/accessible) ---
# NOTE: This definition is included here for the self-contained server script string.
# In your full notebook, ensure the actual function is defined outside the string
# or is imported if you are saving this in a separate file.
def causes_tool(symptoms: str) -> str:
    """
    Lists the most common and relevant potential causes for the given symptoms.
    
    Args:
        symptoms: A string describing the user's medical symptoms.
        
    Returns:
        A string containing a list of potential causes from the Gemini model.
    """
    # In a real scenario, this would call the Gemini API client.
    # For this example, we'll return a placeholder to ensure the agent runs.
    # Replace this with your actual safe_generate_content(prompt) call.
    return f"**DISCLAIMER: I am an AI and cannot provide professional medical advice.** Potential causes for '{symptoms}' are: 1. Common Cold, 2. Flu, 3. Allergies."


# --- 2. Define the Agent and Server Script String ---
# Using the same retry_config and LlmAgent structure as your example
causes_agent_port = 8002 # Use a different port than the example (8001)

causes_agent_code = f'''
import os
import time
from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini
from google.genai import types

# Re-define retry_config (must be defined inside the server module)
retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

# Re-define causes_tool (must be defined inside the server module)
def causes_tool(symptoms: str) -> str:
    """
    Calls the Gemini API to get the most common causes for the symptoms.
    (Placeholder implementation for server module)
    """
    # IMPORTANT: In a real environment, you MUST set up the 'client' here
    # and call the Gemini API as defined in the previous step.
    # E.g., return safe_generate_content(prompt)
    return f"**DISCLAIMER:** Causes for '{{symptoms}}' are: 1. Viral Infection, 2. Bacterial Infection, 3. Stress."


# Define the Causes Agent
causes_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    name="CausesAgent",
    description="An AI medical agent specializing in finding the common causes of a medical symptom.",
    instruction="""
    You are a medical causes specialist. 
    Your sole task is to determine the most common and relevant potential causes 
    for the symptoms provided by the user (or another agent).
    You MUST use the 'causes_tool' to fetch this information.
    Provide the list of causes clearly and concisely.
    Always start your response with a medical disclaimer.
    """,
    tools=[causes_tool],  # Register the custom causes tool
)

# Create the A2A app
app = to_a2a(causes_agent, port={causes_agent_port})
'''

# --- 3. Write the agent to a temporary file ---
temp_file_name = "/tmp/causes_agent_server.py"
with open(temp_file_name, "w") as f:
    f.write(causes_agent_code)

print(f"ğŸ“� Causes Agent code saved to {temp_file_name}")

# --- 4. Start uvicorn server in background ---
server_process = subprocess.Popen(
    [
        "uvicorn",
        "causes_agent_server:app",  # Module:app format
        "--host",
        "localhost",
        "--port",
        str(causes_agent_port),
    ],
    cwd="/tmp",  # Run from /tmp where the file is
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env={**os.environ},  # Pass environment variables (including GOOGLE_API_KEY)
)

print(f"ğŸš€ Starting Causes Agent server on port {causes_agent_port}...")
print("    Waiting for server to be ready...", end="", flush=True)

# --- 5. Wait for server to start ---
max_attempts = 30
for attempt in range(max_attempts):
    try:
        response = requests.get(
            f"http://localhost:{causes_agent_port}/.well-known/agent-card.json", timeout=1
        )
        if response.status_code == 200:
            print(f"\nâœ… Causes Agent server is running!")
            print(f"    Server URL: http://localhost:{causes_agent_port}")
            break
    except requests.exceptions.RequestException:
        time.sleep(5)
        print(".", end="", flush=True)
else:
    print("\nâš ï¸�  Server may not be ready yet. Check manually if needed.")

# Save process reference to stop it later
globals()["causes_agent_server_process"] = server_process


import requests
import json
import time

# --- Configuration ---
CAUSES_AGENT_PORT = 8002 
AGENT_CARD_URL = f"http://localhost:{CAUSES_AGENT_PORT}/.well-known/agent-card.json"

print(f"Attempting to fetch Causes Agent Card from: {AGENT_CARD_URL}")
print("-" * 50)

# --- Fetch Agent Card ---
try:
    # Set a timeout in case the server is not running
    response = requests.get(AGENT_CARD_URL, timeout=5) 

    if response.status_code == 200:
        agent_card = response.json()
        
        print("âœ… Causes Agent Card Fetched Successfully!")
        print("-" * 50)
        print(json.dumps(agent_card, indent=2))
        
        # Display Key Information
        print("\nâœ¨ Key Agent Information:")
        print(f"    Name: **{agent_card.get('name')}**")
        print(f"    Description: {agent_card.get('description')}")
        print(f"    URL: {agent_card.get('url')}")
        print(f"    Skills: **{len(agent_card.get('skills', []))}** capability exposed")
        
        # Verify Tool
        skill_names = [s.get('name') for s in agent_card.get('skills', [])]
        print(f"    Exposed Tool: {skill_names[0] if skill_names else 'None'}")
        
    else:
        print(f"â�Œ Failed to fetch agent card. HTTP Status Code: **{response.status_code}**")

except requests.exceptions.RequestException as e:
    print(f"â�Œ Error fetching Causes Agent Card: **{e}**")
    print("    **Action required:** Make sure the Causes Agent server is running on port 8002.")


remote_causes_agent = RemoteA2aAgent(
    name="causes_agent",
    description="provides Causes of the medical symptoms",
    # Point to the agent card URL - this is where the A2A protocol metadata lives
    agent_card=f"http://localhost:8002{AGENT_CARD_WELL_KNOWN_PATH}",
)

print("âœ… Remote Product Catalog Agent proxy created!")
print(f"   Connected to: http://localhost:8001")
print(f"   Agent card: http://localhost:8001{AGENT_CARD_WELL_KNOWN_PATH}")
print("   The Customer Support Agent can now use this like a local sub-agent!")



# --- IMPORTANT: Ensure these objects are available in your current scope ---
# You must have defined:
# 1. The custom tool function: suggest_tool (from previous steps)
# 2. The remote agent proxy: remote_causes_agent (from the previous step, after fixing the ModuleNotFoundError)
# 3. The framework components: LlmAgent, Gemini, retry_config
# ---

# 1. Define the Suggestion Agent (Orchestrator)
suggestion_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    name="DoctorAgent",  # Renamed to DoctorAgent for a clearer role as the main orchestrator
    description="A primary AI medical assistant that provides both common causes and home care suggestions for medical symptoms.",
    instruction="""
    You are a professional AI Doctor Assistant. 
    Your primary goal is to provide comprehensive and balanced advice.
    
    When a user provides symptoms, your procedure is:
    1. **First, use the 'remote_causes_agent' sub-agent** to quickly fetch the most common potential causes.
    2. **Then, use the 'suggest_tool'** (your local tool) to provide preliminary home care suggestions and clear medical next steps.
    3. **Combine the information** from both tools/agents into a single, cohesive, and easy-to-read response.
    4. **Always include a strong medical disclaimer** at the beginning of your response.
    """,
    tools=[suggest_tool],                 # Local tool for suggestions
    sub_agents=[remote_causes_agent],     # Remote agent for causes (via A2A proxy)
)

print("âœ… Doctor/Suggestion Agent (Orchestrator) created!")
print("    Model: gemini-2.5-flash")
print(f"    Local Tool: suggest_tool()")
print(f"    Sub-agents: 1 (remote Causes Agent via A2A on port 8002)")
print("    Ready to coordinate and provide comprehensive medical advice!")


import uuid
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import asyncio

# --- IMPORTANT: Ensure these objects are available in your current scope ---
# 1. suggestion_agent (your LlmAgent defined as the orchestrator)
# 2. remote_causes_agent (your RemoteA2aAgent proxy)
# 3. suggest_tool (your local custom tool)
# ---

# Use the orchestrator agent defined in the previous step
doctor_agent = suggestion_agent 

async def test_a2a_communication(user_query: str):
    """
    Test the A2A communication between the Doctor Agent (Orchestrator) 
    and the Causes Agent (Remote Sub-Agent).

    This function:
    1. Creates a new session for this conversation.
    2. Sends the query to the Doctor Agent.
    3. Doctor Agent communicates with Causes Agent via A2A proxy 
       and uses its local suggest_tool.
    4. Displays the combined response.

    Args:
        user_query: The symptoms question to ask the Doctor Agent.
    """
    # Setup session management (required by ADK)
    session_service = InMemorySessionService()

    # Session identifiers
    app_name = "doctor_app"
    user_id = "patient_user"
    # Use unique session ID for each test to avoid conflicts
    session_id = f"demo_session_{uuid.uuid4().hex[:8]}"

    # CRITICAL: Create session BEFORE running agent (synchronous, not async!)
    session = await session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )

    # Create runner for the Doctor Agent
    # The runner manages the agent execution and session state
    runner = Runner(
        agent=doctor_agent, app_name=app_name, session_service=session_service
    )

    # Create the user message
    test_content = types.Content(parts=[types.Part(text=user_query)])

    # Display query
    print(f"\nğŸ‘¤ Patient: {user_query}")
    print(f"\nğŸ©º Doctor Agent response:")
    print("-" * 70)

    # Run the agent asynchronously (handles streaming responses and A2A communication)
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=test_content
    ):
        # Print final response only (skip intermediate events like tool/sub-agent calls)
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                if hasattr(part, "text"):
                    print(part.text)

    print("-" * 70)

# --- Run the Test ---
print("ğŸ§ª Testing Agent-to-Agent (A2A) Communication...\n")

# Example query that requires both the remote CausesAgent and the local SuggestTool
await test_a2a_communication("I have a persistent  dry cough  What are the common causes and what should I do?")

# Example query focusing more on suggestions


