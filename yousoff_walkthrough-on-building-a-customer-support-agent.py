import os
import logging
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")

# Clean up any previous logs
for log_file in ["logger.log", "web.log", "tunnel.log"]:
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"ğŸ§¹ Cleaned up {log_file}")

# Configure logging with INFO log level.
logging.basicConfig(
    filename="logger.log",
    level=logging.INFO,
    format="%(filename)s:%(lineno)s %(levelname)s:%(message)s",
)

print("âœ… Logging configured")


from google.genai import types
from typing import Any, Dict
import json
import requests
import subprocess
import time
import uuid

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner, InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.tools.tool_context import ToolContext
from google.adk.plugins.logging_plugin import (
    LoggingPlugin,
)
from google.adk.agents.remote_a2a_agent import (
    RemoteA2aAgent,
    AGENT_CARD_WELL_KNOWN_PATH,
)
from google.adk.a2a.utils.agent_to_a2a import to_a2a

# Hide additional warnings in the notebook
import warnings

warnings.filterwarnings("ignore")

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

print("âœ… ADK components imported successfully.")


import json

with open('/kaggle/input/product/capstone_product.json', 'r') as f:
    product_dataset = json.load(f)

print("âœ… Dataset loaded.")


def get_available_products() -> dict:
    """Get list of available products

    Returns:
        List of available products.
        
        Success: {"status": "success", "data": ['productA', 'productB']}
        Error: {"status": "error", "error_message": "Not able to fetch products"}
    """
    return {"status": "success", "data": product_dataset}


print("âœ… get_available_products created")
print(f"ğŸ’± Test: {get_available_products()}")


basic_product_support_agent = LlmAgent(
    name="basic_product_support_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    
    You are a product support assistance to handle any customer enquiry from the question.

    For any product enquiry :
    1. Use `get_available_products()` to get list of available products. Then reply to the user straight away.

    If any tool returns status "error", explain the issue to the user clearly.
    """,
    tools=[get_available_products]
)

print("âœ… Basic Product Support Agent defined.")


runner = InMemoryRunner(agent=basic_product_support_agent)
print("âœ… Runner created.")


response = await runner.run_debug("Is Product E available ? ")


def save_context_selected_product(
    tool_context: ToolContext, user_name: str, product: str
) -> Dict[str, Any]:
    """
    Tool to record chosen product
    Args:
        user_name: The username to store in session state
        product: The name of the product
    """
    # Write to session state using the 'context:' prefix for user data
    tool_context.state["context:user"] = user_name
    tool_context.state["context:product"] = product

    return {"status": "success"}

# This demonstrates how tools can read from session state.
def retrieve_context_selected_product(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Tool to retrieve user selected product
    """
    # Read from session state
    user_name = tool_context.state.get("context:user", "Username not found")
    product = tool_context.state.get("context:product", "Product not found")

    return {"status": "success", "user_name": user_name, "product": product}


print("âœ… Tools created.")


product_support_agent = LlmAgent(
    name="product_support_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    
    You are a product support assistance to handle any customer enquiry from the question below
    {user_question}

    For any product enquiry :
    1. Use `get_available_products()` to get list of available products. Then reply to the user.
    2. Use `save_context_selected_product()` once user chosen a product
    3. Use `retrieve_context_selected_product()` to show the current chosen product

    If any tool returns status "error", explain the issue to the user clearly.
    """,
    tools=[get_available_products, save_context_selected_product, retrieve_context_selected_product]
)

print("âœ… Product Support Agent defined.")


customer_support_agent_input = LlmAgent(
    name="customer_support_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    
    You are a customer support assistance to handle any customer enquiry
    Summarize the user question to be handled by product support agent. Format this way. 

    Question :
    <Summary of the user question>
    
     

    If any tool returns status "error", explain the issue to the user clearly.
    """,
    output_key = 'user_question'
)

print("âœ… Customer Support Agent Input defined.")


customer_support_agent_output = LlmAgent(
    name="customer_support_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    
    You are a customer support assistance to handle any customer enquiry
    Summarize the user question to be handled by product support agent.
    Then based on the response from product support agent, summarize it to customer.
     

    If any tool returns status "error", explain the issue to the user clearly.
    """,
    output_key = 'answer'
)

print("âœ… Customer Support Agent Output defined.")


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


APP_NAME = "Customer Support Chatbot"
USER_ID = "default"
SESSION = "default"
MODEL_NAME = "gemini-2.5-flash-lite"

root_agent = SequentialAgent(
    name="SupportPipeline",
    sub_agents=[customer_support_agent_input, product_support_agent, customer_support_agent_output],
)

print("âœ… Sequential Agent created.")

session_service = InMemorySessionService()
runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service, plugins=[
        LoggingPlugin()
    ])

print("âœ… Runner with session is created.")

# To prevent from memory keeping a lot of conversation, we will use compacting
customer_support_compacting = App(
    name="customer_support_compacting",
    root_agent=root_agent,
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=3,
        overlap_size=1,
    ),
)




# We will use session-1 to simulate questions from Andy
# The output will be long so that you can see the flow between agents and tools.
# Once run, you can right-click to clear the cell output
await run_session(
    runner,
    [
        "Which product is available ?"
    ],
    "session-1",
)


# We will use session-1 to simulate questions from Andy
# The output will be long so that you can see the flow between agents and tools.
# Once run, you can right-click to clear the cell output
await run_session(
    runner,
    [
        "I will choose Product B. My name is Andy by the way."
    ],
    "session-1",
)


# We will use session-2 to simulate questions from Bob
# The output will be long so that you can see the flow between agents and tools.
# Once run, you can right-click to clear the cell output
await run_session(
    runner,
    [
        "My name is Bob. I would like to buy Product A"
    ],
    "session-2",
)


# Now let's check each session to see if the agent able to decide what is the current context
await run_session(
    runner,
    [
        "What is my name ? And which product did i choose"
    ],
    "session-1",
)


# First, let's save the product_development_support_code to a file that uvicorn can import
product_development_support_code = '''
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

def get_product_info(product_name: str) -> str:
    """Get product information for a given product.

    Args:
        product_name: Name of the product (e.g., "Product A", "Product B")

    Returns:
        Product information as a string
    """
    product_catalog = {
        "product a": "Current version is 1.0.0-alpha. Next version 1.0.1 will be available to all customers on 1st Jan 2026",
        "product b": "Current version is 1.0.0-alpha. Next version 1.0.1 will be available to all customers on 1st Feb 2026",
        "product c": "Current version is 1.0.0-alpha. Next version 1.0.1 will be available to all customers on 1st March 2026",
        "product d": "Not available. This product only available to first 20 early access customers.",
        "product e": "Discontinued due to outdated tech stack",
    }

    print(product_name.lower().strip())
    product_lower = product_name.lower().strip()

    if product_lower in product_catalog:
        return f"Product: {product_catalog[product_lower]}"
    else:
        available = ", ".join([p.title() for p in product_catalog.keys()])
        return f"Sorry, I don't have information for {product_name}. Available products: {available}"

    product_lower = product_name.lower().strip()

    if product_lower in product_catalog:
        return f"Product: {product_catalog[product_lower]}"
    else:
        available = ", ".join([p.title() for p in product_catalog.keys()])
        return f"Sorry, I don't have information for {product_name}. Available products: {available}"

product_development_support_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="product_development_support_agent",
    description="External vendor's product catalog agent that provides product information and availability.",
    instruction="""
    You are a product development support agent from Company Techies.

    For product enquiry, use the get_product_info tool to fetch data from the catalog.
    And then answer accordingly based on the catalog description for availability 
    """,
    tools=[get_product_info],  # Register the product lookup tool
)

# Create the A2A app
app = to_a2a(product_development_support_agent, port=8001)
'''

# Write the product catalog agent to a temporary file
with open("/tmp/product_development_support_agent_server.py", "w") as f:
    f.write(product_development_support_code)

print("ğŸ“� product_development_support_code saved to /tmp/product_development_support_agent_server.py")

# Start uvicorn server in background
# Note: We redirect output to avoid cluttering the notebook
server_process = subprocess.Popen(
    [
        "uvicorn",
        "product_development_support_agent_server:app",  # Module:app format
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

print("ğŸš€ Starting product_development_support_code server...")
print("   Waiting for server to be ready...")

# Wait for server to start (poll until it responds)
max_attempts = 30
for attempt in range(max_attempts):
    try:
        response = requests.get(
            "http://localhost:8001/.well-known/agent-card.json", timeout=1
        )
        if response.status_code == 200:
            print(f"\nâœ… product_development_support_code is running!")
            print(f"   Server URL: http://localhost:8001")
            print(f"   Agent card: http://localhost:8001/.well-known/agent-card.json")
            break
    except requests.exceptions.RequestException:
        time.sleep(5)
        print(".", end="", flush=True)
else:
    print("\nâš ï¸�  Server may not be ready yet. Check manually if needed.")

# Store the process so we can stop it later
globals()["product_development_support_agent_server_process"] = server_process


# See whether we can fetch the agent card from the running server
try:
    response = requests.get(
        "http://localhost:8001/.well-known/agent-card.json", timeout=5
    )

    if response.status_code == 200:
        agent_card = response.json()
        print("ğŸ“‹ product_development_support_code Card:")
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
    print("   Make sure the server is running (previous cell)")


# Create a RemoteA2aAgent that connects to our Product Catalog Agent
# This acts as a client-side proxy - the Customer Support Agent can use it like a local agent
remote_product_development_support_agent = RemoteA2aAgent(
    name="remote_product_development_support_agent",
    description="Remote product development support agent from external vendor that provides product information.",
    # Point to the agent card URL - this is where the A2A protocol metadata lives
    agent_card=f"http://localhost:8001{AGENT_CARD_WELL_KNOWN_PATH}",
)

print("âœ… Remote remote_product_development_support_agent proxy created!")
print(f"   Connected to: http://localhost:8001")
print(f"   Agent card: http://localhost:8001{AGENT_CARD_WELL_KNOWN_PATH}")
print("   The Customer Support Agent can now use this like a local sub-agent!")


customer_support_agent_v2 = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="customer_support_agent_v2",
    description="A customer support assistant that helps customers with product inquiries and information.",
    instruction="""
    You are a friendly and professional customer support agent.
    
    When customers ask about products:
    1. Use the remote_product_development_support_agent sub-agent to look up product information
    2. Provide clear answers about pricing, availability, and specifications
    3. If a product is out of stock, mention the expected availability
    4. Be helpful and professional!
    
    Always get product information from the remote_product_development_support_agent before answering customer questions.
    """,
    sub_agents=[remote_product_development_support_agent],  # Add the remote agent as a sub-agent!
)

print("âœ… Customer Support Agent created!")


async def test_a2a_communication(user_query: str):
    """
    Test the A2A communication between Customer Support Agent and Product Development Support Agent.

    This function:
    1. Creates a new session for this conversation
    2. Sends the query to the Customer Support Agent
    3. Support Agent communicates with Product Development Support Agent via A2A
    4. Displays the response

    Args:
        user_query: The question to ask the Product Development Support Agent
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
        agent=customer_support_agent_v2, app_name=app_name, session_service=session_service,
        plugins=[
        LoggingPlugin()
        ]
    )

    # Create the user message
    # This follows the same pattern as the deployment notebook
    test_content = types.Content(parts=[types.Part(text=user_query)])

    # Display query
    print(f"\nğŸ‘¤ Customer: {user_query}")
    print(f"\nğŸ�§ Support Agent response:")
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


await test_a2a_communication("What happen to Product D ?")

