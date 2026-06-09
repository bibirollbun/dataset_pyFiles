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


import json
import requests
import subprocess
import time
import uuid

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

# Hide additional warnings in the notebook
import warnings

warnings.filterwarnings("ignore")

print("âœ… ADK components imported successfully.")


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


# Create the Advice Agent
# This agent specializes in providing advice related to activities or excercise required to manage blood sugar
advice_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="advice_agent",
    description="Advice agent that provides information about activities or excercise required for end user.",
    instruction="""
    You are a diabetes exercise specialist. Provide personalized activity recommendations and safety tips. Respond in below format:
      "recommendations": ["tip1", "tip2", "tip3"],
      "glucoseImpact": "expected impact description",
      "safetyTips": ["safety1", "safety2"]
    """,
    tools=[],  # Register tool
)

print("âœ… Agent created successfully!")
print("   Model: gemini-2.5-flash-lite")
print("   Ready to be exposed via A2A...")


# Create the Nutrition Agent
# This agent specializes in recommending nutritional meals to manage blood sugar levels
nutrition_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="nutrition_agent",
    description="Nutrition agent that recommends nutritional and meals information to end user.",
    instruction="""
    You are a diabetes nutrition expert. Analyze meals and provide detailed nutritional information and diabetes-specific advice. Always respond in valid JSON format with the following structure:
      
        "name": "meal name",
        "calories": number,
        "carbs": number (grams),
        "protein": number (grams), 
        "fat": number (grams),
        "fiber": number (grams),
        "glycemicImpact": "Low|Medium|High",
        "glucoseImpact": number (estimated mg/dL increase),
        "healthScore": number (0-100),
        "recommendations": ["recommendation1", "recommendation2", "recommendation3"]
      
    """,
    tools=[],  # Register tool
)

print("âœ… Agent created successfully!")
print("   Model: gemini-2.5-flash-lite")
print("   Ready to be exposed via A2A...")


advice_agent_a2a_app = to_a2a(
    advice_agent, port=8001  # Port where this agent will be served
)

nutrition_agent_a2a_app = to_a2a(
    nutrition_agent, port=8002  # Port where this agent will be served
)
print("âœ… Agent is now A2A-compatible!")
print("   Ready to start the server...")


advice_agent_code = '''
import os
from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini
from google.genai import types

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

advice_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="advice_agent",
    description="Advice agent that provides information about activities or excercise required for end user.",
    instruction="""
    You are a diabetes exercise specialist. Provide personalized activity recommendations and safety tips. Respond in below format:
      "recommendations": ["tip1", "tip2", "tip3"],
      "glucoseImpact": "expected impact description",
      "safetyTips": ["safety1", "safety2"]
    """,
    tools=[],  # Register tool
)

# Create the A2A app
app = to_a2a(
    advice_agent, port=8001  # Port where this agent will be served
)
'''

# Write the product catalog agent to a temporary file
with open("/tmp/advice_agent_server.py", "w") as f:
    f.write(advice_agent_code)

print("ğŸ“� Advice agent code saved to /tmp/advice_agent_server.py")

# Start uvicorn server in background
# Note: We redirect output to avoid cluttering the notebook
server_process = subprocess.Popen(
    [
        "uvicorn",
        "advice_agent_server:app",  # Module:app format
        "--host",
        "localhost",
        "--port",
        "8001",
    ],
    cwd="/tmp",  # Run from /tmp where the file is
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env={**os.environ},  # Pass environment variables (including GOOGLE_API_KEY)
)

print("ğŸš€ Starting Advice Agent server...")
print("   Waiting for server to be ready...")

# Wait for server to start (poll until it responds)
max_attempts = 30
for attempt in range(max_attempts):
    try:
        response = requests.get(
            "http://localhost:8001/.well-known/agent-card.json", timeout=1
        )
        if response.status_code == 200:
            print(f"\nâœ… Advice Agent server is running!")
            print(f"   Server URL: http://localhost:8001")
            print(f"   Agent card: http://localhost:8001/.well-known/agent-card.json")
            break
    except requests.exceptions.RequestException:
        time.sleep(5)
        print(".", end="", flush=True)
else:
    print("\nâš ï¸�  Server may not be ready yet. Check manually if needed.")

# Store the process so we can stop it later
globals()["advice_agent_server_process"] = server_process


nutrition_agent_code = '''
import os
from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini
from google.genai import types

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

nutrition_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="nutrition_agent",
    description="Nutrition agent that recommends nutritional and meals information to end user.",
    instruction="""
    You are a diabetes nutrition expert. Analyze meals and provide detailed nutritional information and diabetes-specific advice. Always respond in valid JSON format with the following structure:
      
        "name": "meal name",
        "calories": number,
        "carbs": number (grams),
        "protein": number (grams), 
        "fat": number (grams),
        "fiber": number (grams),
        "glycemicImpact": "Low|Medium|High",
        "glucoseImpact": number (estimated mg/dL increase),
        "healthScore": number (0-100),
        "recommendations": ["recommendation1", "recommendation2", "recommendation3"]
      
    """,
    tools=[],  # Register tool
)

# Create the A2A app
app = to_a2a(
    nutrition_agent, port=8002  # Port where this agent will be served
)
'''

# Write the product catalog agent to a temporary file
with open("/tmp/nutrition_agent_server.py", "w") as f:
    f.write(nutrition_agent_code)

print("ğŸ“� Nutrition agent code saved to /tmp/nutrition_agent_server.py")

# Start uvicorn server in background
# Note: We redirect output to avoid cluttering the notebook
server_process = subprocess.Popen(
    [
        "uvicorn",
        "nutrition_agent_server:app",  # Module:app format
        "--host",
        "localhost",
        "--port",
        "8002",
    ],
    cwd="/tmp",  # Run from /tmp where the file is
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env={**os.environ},  # Pass environment variables (including GOOGLE_API_KEY)
)

print("ğŸš€ Starting Nutri Agent server...")
print("   Waiting for server to be ready...")

# Wait for server to start (poll until it responds)
max_attempts = 30
for attempt in range(max_attempts):
    try:
        response = requests.get(
            "http://localhost:8002/.well-known/agent-card.json", timeout=1
        )
        if response.status_code == 200:
            print(f"\nâœ… Nutri Agent server is running!")
            print(f"   Server URL: http://localhost:8002")
            print(f"   Agent card: http://localhost:8002/.well-known/agent-card.json")
            break
    except requests.exceptions.RequestException:
        time.sleep(5)
        print(".", end="", flush=True)
else:
    print("\nâš ï¸�  Server may not be ready yet. Check manually if needed.")

# Store the process so we can stop it later
globals()["nutrition_agent_server_process"] = server_process


# Fetch the agent card from the running server
try:
    response = requests.get(
        "http://localhost:8001/.well-known/agent-card.json", timeout=5
    )

    if response.status_code == 200:
        agent_card = response.json()
        print("ğŸ“‹ Advice Agent Card:")
        print(json.dumps(agent_card, indent=2))

        print("\nâœ¨ Key Information:")
        print(f"   Name: {agent_card.get('name')}")
        print(f"   Description: {agent_card.get('description')}")
        print(f"   URL: {agent_card.get('url')}")
        print(f"   Skills: {len(agent_card.get('skills', []))} capabilities exposed")
    else:
        print(f"â�Œ Failed to fetch agent card: {response.status_code}")

except requests.exceptions.RequestException as e:
    print(f"â�Œ Error fetching agent card: {e}")
    print("   Make sure the Advice Agent server is running (previous cell)")


# Fetch the agent card from the running server
try:
    response = requests.get(
        "http://localhost:8002/.well-known/agent-card.json", timeout=5
    )

    if response.status_code == 200:
        agent_card = response.json()
        print("ğŸ“‹ Nutri Agent Card:")
        print(json.dumps(agent_card, indent=2))

        print("\nâœ¨ Key Information:")
        print(f"   Name: {agent_card.get('name')}")
        print(f"   Description: {agent_card.get('description')}")
        print(f"   URL: {agent_card.get('url')}")
        print(f"   Skills: {len(agent_card.get('skills', []))} capabilities exposed")
    else:
        print(f"â�Œ Failed to fetch agent card: {response.status_code}")

except requests.exceptions.RequestException as e:
    print(f"â�Œ Error fetching agent card: {e}")
    print("   Make sure the Advice Agent server is running (previous cell)")


remote_advice_agent = RemoteA2aAgent(
    name="advice_agent",
    description="Remote advice agent that provides activity information.",
    # Point to the agent card URL - this is where the A2A protocol metadata lives
    agent_card=f"http://localhost:8001{AGENT_CARD_WELL_KNOWN_PATH}",
)

print("âœ… Remote Advice Agent proxy created!")
print(f"   Connected to: http://localhost:8001")
print(f"   Agent card: http://localhost:8001{AGENT_CARD_WELL_KNOWN_PATH}")
print("   The Advice Agent can now use this like a local sub-agent!")


remote_nutrition_agent = RemoteA2aAgent(
    name="nutrition_agent",
    description="Remote Nutrition agent that provides activity information.",
    # Point to the agent card URL - this is where the A2A protocol metadata lives
    agent_card=f"http://localhost:8002{AGENT_CARD_WELL_KNOWN_PATH}",
)

print("âœ… Remote Nutrition Agent proxy created!")
print(f"   Connected to: http://localhost:8002")
print(f"   Agent card: http://localhost:8002{AGENT_CARD_WELL_KNOWN_PATH}")
print("   The Nutrition Agent can now use this like a local sub-agent!")


root_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="main_agent",
    description="A assistant that helps customers with inquiries and information.",
    instruction="""
    You are a comprehensive diabetes management AI assistant named Type1D. You ONLY help users with diabetes-related topics and care.

      âš ï¸� SCOPE RESTRICTIONS:
      You MUST ONLY answer questions related to:
      - Diabetes management, treatment, and care
      - Blood glucose monitoring and insulin management
      - Diabetic-friendly foods, meals, and nutrition
      - Exercise and physical activity for diabetics
      - Medication and treatment adherence
      - Diabetes complications and prevention
      - Living with diabetes (daily management, lifestyle)
      - Diabetes education and health information
      - Healthcare provider consultations for diabetes
      
      YOUR CAPABILITIES:

      1. MEDICATION MANAGEMENT: Remind about medications and provide timing advice
      2. ACTIVITY & EXERCISE: Use the advice_agent sub-agent to suggest workouts, track activities, and advise on glucose management during exercise.
      3. MEAL ANALYSIS: Use the nutrition_agent to analyze foods, provide carb counts, suggest portion sizes, and give nutritional advice

      CONVERSATION STYLE:
      - Be warm, supportive, and encouraging
      - Ask follow-up questions to gather necessary details
      - Provide specific, actionable advice
      - Use emojis appropriately to make conversations friendly
      - Remember context from the conversation history

      SAFETY REMINDERS:
      - Always remind users to consult healthcare providers for medical decisions
      - You provide educational information, not medical diagnosis or treatment
      - In emergencies, direct users to seek immediate medical attention

      RESPONSE FORMAT:
      - Give clear, structured responses
      - Use bullet points for lists
      - Offer next steps or follow-up questions`
    """,
    sub_agents=[remote_advice_agent,remote_nutrition_agent],  # Add the remote agent as a sub-agent!
)

print("âœ… Main Agent created!")
print("   Model: gemini-2.5-flash-lite")
print("   Sub-agents: 1 (remote Advice Agent & remote Nutrition Agent via A2A)")
print("   Ready to help patients!")


async def test_a2a_communication(user_query: str):
    """
    Test the A2A communication between Advice Agent and Main Agent.

    This function:
    1. Creates a new session for this conversation
    2. Sends the query to the Main Agent
    3. Support Agent communicates with Advice Agent via A2A
    4. Displays the response

    Args:
        user_query: The question to ask the Main Agent
    """
    # Setup session management (required by ADK)
    session_service = InMemorySessionService()

    # Session identifiers
    app_name = "support_app"
    user_id = "demo_user"
    # Use unique session ID for each test to avoid conflicts
    session_id = f"demo_session_{uuid.uuid4().hex[:8]}"

    # CRITICAL: Create session BEFORE running agent (synchronous, not async!)
    # This pattern matches the deployment notebook exactly
    session = await session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )

    # Create runner for the Customer Support Agent
    # The runner manages the agent execution and session state
    runner = Runner(
        agent=main_agent, app_name=app_name, session_service=session_service
    )

    # Create the user message
    # This follows the same pattern as the deployment notebook
    test_content = types.Content(parts=[types.Part(text=user_query)])

    # Display query
    print(f"\nğŸ‘¤ Customer: {user_query}")
    print(f"\nğŸ�§ Main Agent response:")
    print("-" * 60)

    # Run the agent asynchronously (handles streaming responses and A2A communication)
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=test_content
    ):
        # Print final response only (skip intermediate events)
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                if hasattr(part, "text"):
                    print(part.text)

    print("-" * 60)


# Run the test
print("ğŸ§ª Testing A2A Communication...\n")
await test_a2a_communication("I am having a tinking at my foot, Can you help me or suggest some excersize?")


await test_a2a_communication("Yes, please transfer me to advice agent")


from IPython.core.display import display, HTML
from jupyter_server.serverapp import list_running_servers


# Gets the proxied URL in the Kaggle Notebooks environment
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

    styled_html = f"""
    <div style="padding: 15px; border: 2px solid #f0ad4e; border-radius: 8px; background-color: #fef9f0; margin: 20px 0;">
        <div style="font-family: sans-serif; margin-bottom: 12px; color: #333; font-size: 1.1em;">
            <strong>âš ï¸� IMPORTANT: Action Required</strong>
        </div>
        <div style="font-family: sans-serif; margin-bottom: 15px; color: #333; line-height: 1.5;">
            The ADK web UI is <strong>not running yet</strong>. You must start it in the next cell.
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
            Open ADK Web UI (after running cell below) â†—
        </a>
    </div>
    """

    display(HTML(styled_html))

    return url_prefix


print("âœ… Helper functions defined.")


!adk create diabetes_management_agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY


%%writefile diabetes_management_agent/agent.py

import os
import json
import requests
import subprocess
import time
import uuid

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

# Hide additional warnings in the notebook
import warnings

warnings.filterwarnings("ignore")

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

# Create the Advice Agent
# This agent specializes in providing advice related to activities or excercise required to manage blood sugar
advice_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="advice_agent",
    description="Advice agent that provides information about activities or excercise required for end user.",
    instruction="""
    You are a diabetes exercise specialist. Provide personalized activity recommendations and safety tips. Respond in JSON format:
      {
        "recommendations": ["tip1", "tip2", "tip3"],
        "glucoseImpact": "expected impact description",
        "safetyTips": ["safety1", "safety2"]
      }
    """,
    tools=[],  # Register tool
)

# Create the Nutrition Agent
# This agent specializes in recommending nutritional meals to manage blood sugar levels
nutrition_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="nutrition_agent",
    description="Nutrition agent that recommends nutritional and meals information to end user.",
    instruction="""
    You are a diabetes nutrition expert. Analyze meals and provide detailed nutritional information and diabetes-specific advice. Always respond in valid JSON format with the following structure:
      
        "name": "meal name",
        "calories": number,
        "carbs": number (grams),
        "protein": number (grams), 
        "fat": number (grams),
        "fiber": number (grams),
        "glycemicImpact": "Low|Medium|High",
        "glucoseImpact": number (estimated mg/dL increase),
        "healthScore": number (0-100),
        "recommendations": ["recommendation1", "recommendation2", "recommendation3"]
      
    """,
    tools=[],  # Register tool
)

advice_agent_a2a_app = to_a2a(
    advice_agent, port=8001  # Port where this agent will be served
)

nutrition_agent_a2a_app = to_a2a(
    nutrition_agent, port=8002  # Port where this agent will be served
)

advice_agent_code = '''
import os
from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini
from google.genai import types

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

advice_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="advice_agent",
    description="Advice agent that provides information about activities or excercise required for end user.",
    instruction="""
    You are a diabetes exercise specialist. Provide personalized activity recommendations and safety tips. Respond in JSON format:
      {
        "recommendations": ["tip1", "tip2", "tip3"],
        "glucoseImpact": "expected impact description",
        "safetyTips": ["safety1", "safety2"]
      }
    """,
    tools=[],  # Register tool
)

# Create the A2A app
app = to_a2a(
    advice_agent, port=8001  # Port where this agent will be served
)
'''

# Write the product catalog agent to a temporary file
with open("/tmp/advice_agent_server.py", "w") as f:
    f.write(advice_agent_code)

print("ğŸ“� Advice agent code saved to /tmp/advice_agent_server.py")

# Start uvicorn server in background
# Note: We redirect output to avoid cluttering the notebook
server_process = subprocess.Popen(
    [
        "uvicorn",
        "advice_agent_server:app",  # Module:app format
        "--host",
        "localhost",
        "--port",
        "8001",
    ],
    cwd="/tmp",  # Run from /tmp where the file is
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env={**os.environ},  # Pass environment variables (including GOOGLE_API_KEY)
)

print("ğŸš€ Starting Advice Agent server...")
print("   Waiting for server to be ready...")

# Wait for server to start (poll until it responds)
max_attempts = 30
for attempt in range(max_attempts):
    try:
        response = requests.get(
            "http://localhost:8001/.well-known/agent-card.json", timeout=1
        )
        if response.status_code == 200:
            print(f"\nâœ… Advice Agent server is running!")
            print(f"   Server URL: http://localhost:8001")
            print(f"   Agent card: http://localhost:8001/.well-known/agent-card.json")
            break
    except requests.exceptions.RequestException:
        time.sleep(5)
        print(".", end="", flush=True)
else:
    print("\nâš ï¸�  Server may not be ready yet. Check manually if needed.")

# Store the process so we can stop it later
globals()["advice_agent_server_process"] = server_process


nutrition_agent_code = '''
import os
from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini
from google.genai import types

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

nutrition_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="nutrition_agent",
    description="Nutrition agent that recommends nutritional and meals information to end user.",
    instruction="""
    You are a diabetes nutrition expert. Analyze meals and provide detailed nutritional information and diabetes-specific advice. Always respond in valid JSON format with the following structure:
      
        "name": "meal name",
        "calories": number,
        "carbs": number (grams),
        "protein": number (grams), 
        "fat": number (grams),
        "fiber": number (grams),
        "glycemicImpact": "Low|Medium|High",
        "glucoseImpact": number (estimated mg/dL increase),
        "healthScore": number (0-100),
        "recommendations": ["recommendation1", "recommendation2", "recommendation3"]
      
    """,
    tools=[],  # Register tool
)

# Create the A2A app
app = to_a2a(
    nutrition_agent, port=8002  # Port where this agent will be served
)
'''

# Write the product catalog agent to a temporary file
with open("/tmp/nutrition_agent_server.py", "w") as f:
    f.write(nutrition_agent_code)

print("ğŸ“� Nutrition agent code saved to /tmp/nutrition_agent_server.py")

# Start uvicorn server in background
# Note: We redirect output to avoid cluttering the notebook
server_process = subprocess.Popen(
    [
        "uvicorn",
        "nutrition_agent_server:app",  # Module:app format
        "--host",
        "localhost",
        "--port",
        "8002",
    ],
    cwd="/tmp",  # Run from /tmp where the file is
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env={**os.environ},  # Pass environment variables (including GOOGLE_API_KEY)
)

print("ğŸš€ Starting Nutri Agent server...")
print("   Waiting for server to be ready...")

# Wait for server to start (poll until it responds)
max_attempts = 30
for attempt in range(max_attempts):
    try:
        response = requests.get(
            "http://localhost:8002/.well-known/agent-card.json", timeout=1
        )
        if response.status_code == 200:
            print(f"\nâœ… Nutri Agent server is running!")
            print(f"   Server URL: http://localhost:8002")
            print(f"   Agent card: http://localhost:8002/.well-known/agent-card.json")
            break
    except requests.exceptions.RequestException:
        time.sleep(5)
        print(".", end="", flush=True)
else:
    print("\nâš ï¸�  Server may not be ready yet. Check manually if needed.")

# Store the process so we can stop it later
globals()["nutrition_agent_server_process"] = server_process

remote_advice_agent = RemoteA2aAgent(
    name="advice_agent",
    description="Remote advice agent that provides activity information.",
    # Point to the agent card URL - this is where the A2A protocol metadata lives
    agent_card=f"http://localhost:8001{AGENT_CARD_WELL_KNOWN_PATH}",
)


remote_nutrition_agent = RemoteA2aAgent(
    name="nutrition_agent",
    description="Remote Nutrition agent that provides activity information.",
    # Point to the agent card URL - this is where the A2A protocol metadata lives
    agent_card=f"http://localhost:8002{AGENT_CARD_WELL_KNOWN_PATH}",
)

root_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="main_agent",
    description="A assistant that helps customers with inquiries and information.",
    instruction="""
    You are a comprehensive diabetes management AI assistant named Type1D. You ONLY help users with diabetes-related topics and care.

      âš ï¸� SCOPE RESTRICTIONS:
      You MUST ONLY answer questions related to:
      - Diabetes management, treatment, and care
      - Blood glucose monitoring and insulin management
      - Medication and treatment adherence
      - Diabetes complications and prevention
      - Living with diabetes (daily management, lifestyle)
      - Diabetes education and health information
      - Healthcare provider consultations for diabetes
      
      YOUR CAPABILITIES:

      1. MEDICATION MANAGEMENT: Remind about medications and provide timing advice
      2. ACTIVITY & EXERCISE: Use the advice_agent sub-agent.
      3. MEAL ANALYSIS: Use the nutrition_agent
      
      Always get activity and excercise related information from the advice_agent and
      meal analysis diet related information from nutrition_agent before answering question.
      And whenever using sub agents inform end user about it.
      
      CONVERSATION STYLE:
      - Be warm, supportive, and encouraging
      - Ask follow-up questions to gather necessary details
      - Provide specific, actionable advice
      - Use emojis appropriately to make conversations friendly
      - Remember context from the conversation history

      SAFETY REMINDERS:
      - Always remind users to consult healthcare providers for medical decisions
      - You provide educational information, not medical diagnosis or treatment
      - In emergencies, direct users to seek immediate medical attention

      RESPONSE FORMAT:
      - Give clear, structured responses
      - Use bullet points for lists
      - Offer next steps or follow-up questions`
    """,
    sub_agents=[remote_advice_agent,remote_nutrition_agent],  # Add the remote agent as a sub-agent!
)



url_prefix = get_adk_proxy_url()


!adk web --url_prefix {url_prefix}

