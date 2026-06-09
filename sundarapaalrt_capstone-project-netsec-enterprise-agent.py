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


!pip install mcp


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


from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner, Runner
from google.adk.tools import google_search
from google.genai import types
from google.adk.agents import LlmAgent, Agent
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from google.adk.tools import google_search, AgentTool, ToolContext
from typing import Dict, List
from google.adk.plugins.logging_plugin import (
    LoggingPlugin,
) 
from google.adk.sessions import InMemorySessionService, DatabaseSessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import load_memory, preload_memory

print("âœ… ADK components imported successfully.")


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)


# Define helper functions that will be reused throughout the notebook
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




# -------------------------------
# Tool 1: Get network routes
# -------------------------------
def get_routes() -> Dict[str, object]:
    """Returns the complete set of network routes across all branches.

    Returns:
        Dictionary with status and routes information.
        Success:
            {
                "status": "success",
                "routes": {
                    1: [ {route objects...} ],
                    2: [ {route objects...} ],
                    3: [ {route objects...} ],
                    4: [ {route objects...} ],
                    5: [ {route objects...} ]
                }
            }
        Error:
            {
                "status": "error",
                "error_message": "Unable to fetch routes"
            }
    """
    try:
        routes: Dict[int, List[Dict[str, object]]] = {
            1: [
                {
                    "dst": "default",
                    "gateway": "203.0.113.1",
                    "dev": "eth0",
                    "protocol": "dhcp",
                    "metric": 100
                },
                {
                    "dst": "192.168.1.0/24",
                    "dev": "eth0",
                    "protocol": "kernel",
                    "scope": "link",
                    "src": "192.168.1.10",
                    "metric": 100
                },
                {
                    "dst": "10.0.0.0/8",
                    "dev": "eth0",
                    "protocol": "kernel",
                    "scope": "link",
                    "src": "10.0.0.5",
                    "metric": 100
                }
            ],
            2: [
                {
                    "dst": "default",
                    "gateway": "198.51.100.1",
                    "dev": "eth1",
                    "protocol": "dhcp",
                    "metric": 100
                },
                {
                    "dst": "192.168.2.0/24",
                    "dev": "eth1",
                    "protocol": "kernel",
                    "scope": "link",
                    "src": "192.168.2.20",
                    "metric": 100
                },
                {
                    "dst": "172.16.0.0/16",
                    "dev": "eth1",
                    "protocol": "kernel",
                    "scope": "link",
                    "src": "172.16.0.15",
                    "metric": 100
                }
            ],
            3: [
                {
                    "dst": "default",
                    "gateway": "203.0.113.254",
                    "dev": "eth2",
                    "protocol": "dhcp",
                    "metric": 100
                },
                {
                    "dst": "192.168.3.0/24",
                    "dev": "eth2",
                    "protocol": "kernel",
                    "scope": "link",
                    "src": "192.168.3.30",
                    "metric": 100
                },
                {
                    "dst": "10.1.0.0/16",
                    "dev": "eth2",
                    "protocol": "kernel",
                    "scope": "link",
                    "src": "10.1.0.25",
                    "metric": 100
                }
            ],
            4: [
                {
                    "dst": "default",
                    "gateway": "198.51.100.254",
                    "dev": "eth3",
                    "protocol": "dhcp",
                    "metric": 100
                },
                {
                    "dst": "192.168.4.0/24",
                    "dev": "eth3",
                    "protocol": "kernel",
                    "scope": "link",
                    "src": "192.168.4.40",
                    "metric": 100
                },
                {
                    "dst": "172.16.1.0/24",
                    "dev": "eth3",
                    "protocol": "kernel",
                    "scope": "link",
                    "src": "172.16.1.45",
                    "metric": 100
                }
            ],
            5: [
                {
                    "dst": "default",
                    "gateway": "203.0.113.100",
                    "dev": "eth4",
                    "protocol": "dhcp",
                    "metric": 100
                },
                {
                    "dst": "192.168.5.0/24",
                    "dev": "eth4",
                    "protocol": "kernel",
                    "scope": "link",
                    "src": "192.168.5.50",
                    "metric": 100
                },
                {
                    "dst": "10.2.0.0/16",
                    "dev": "eth4",
                    "protocol": "kernel",
                    "scope": "link",
                    "src": "10.2.0.55",
                    "metric": 100
                }
            ]
        }

        return {"status": "success", "routes": routes}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}




# -------------------------------
# Tool 2: Get firewall rules
# -------------------------------
def get_fw_rules(request: str = "") -> Dict[str, object]:
    """Returns the current SASE cloud firewall rules.

    Args:
        request: Optional string passed by the agent (ignored in this implementation).

    Returns:
        Dictionary with status and firewall rules information.
        Success:
            {
                "status": "success",
                "fw_rules": [ {rule objects...} ]
            }
        Error:
            {
                "status": "error",
                "error_message": "Unable to fetch firewall rules"
            }
    """
    try:
        fw_rules: List[Dict[str, object]] = [
            # --- General Internet rules ---
            {"action": "allow", "dst": "0.0.0.0/0", "protocol": "tcp", "port": 443, "description": "Allow HTTPS to internet"},
            {"action": "allow", "dst": "0.0.0.0/0", "protocol": "tcp", "port": 80, "description": "Allow HTTP to internet"},
            {"action": "allow", "dst": "0.0.0.0/0", "protocol": "udp", "port": 53, "description": "Allow DNS lookups"},
            {"action": "allow", "dst": "0.0.0.0/0", "protocol": "udp", "port": 123, "description": "Allow NTP time sync"},
            {"action": "deny", "dst": "0.0.0.0/0", "protocol": "tcp", "port": 23, "description": "Block Telnet"},
            {"action": "deny", "dst": "0.0.0.0/0", "protocol": "tcp", "port": 25, "description": "Block outbound SMTP"},
            {"action": "deny", "dst": "0.0.0.0/0", "protocol": "tcp", "port": 445, "description": "Block SMB to internet"},
            {"action": "deny", "dst": "0.0.0.0/0", "protocol": "tcp", "port": 21, "description": "Block FTP to internet"},

            # --- Branch-specific social media controls ---
            {"action": "deny", "src": "192.168.1.0/24", "dst": "facebook.com", "protocol": "tcp", "port": 443, "description": "Block Facebook from Branch 1"},
            {"action": "deny", "src": "192.168.3.0/24", "dst": "facebook.com", "protocol": "tcp", "port": 443, "description": "Block Facebook from Branch 3"},
            {"action": "deny", "src": "192.168.2.0/24", "dst": "tiktok.com", "protocol": "tcp", "port": 443, "description": "Block TikTok from Branch 2"},
            {"action": "deny", "src": "192.168.5.0/24", "dst": "tiktok.com", "protocol": "tcp", "port": 443, "description": "Block TikTok from Branch 5"},
            {"action": "deny", "src": "192.168.4.0/24", "dst": "youtube.com", "protocol": "tcp", "port": 443, "description": "Block YouTube from Branch 4"},
            {"action": "deny", "src": "10.0.0.0/8", "dst": "instagram.com", "protocol": "tcp", "port": 443, "description": "Block Instagram from 10.x internal ranges"},

            # --- Application-specific rules ---
            {"action": "allow", "src": "192.168.0.0/16", "dst": "0.0.0.0/0", "protocol": "udp", "port": 1194, "description": "Allow VPN (OpenVPN)"},
            {"action": "allow", "src": "192.168.0.0/16", "dst": "0.0.0.0/0", "protocol": "udp", "port": 5060, "description": "Allow VoIP SIP traffic"},
            {"action": "allow", "src": "192.168.0.0/16", "dst": "0.0.0.0/0", "protocol": "udp", "port": 3478, "description": "Allow STUN for WebRTC"},
            {"action": "deny", "src": "192.168.0.0/16", "dst": "0.0.0.0/0", "protocol": "tcp", "port": 22, "description": "Block SSH outbound to internet"},

            # --- Inter-branch LAN rules ---
            {"action": "allow", "src": "192.168.1.0/24", "dst": "192.168.2.0/24", "protocol": "all", "description": "Allow Branch 1 â†” Branch 2 LAN"},
            {"action": "allow", "src": "192.168.3.0/24", "dst": "192.168.4.0/24", "protocol": "all", "description": "Allow Branch 3 â†” Branch 4 LAN"},
            {"action": "deny", "src": "192.168.5.0/24", "dst": "172.16.0.0/16", "protocol": "all", "description": "Block Branch 5 to Branch 2 subnet"},
            {"action": "deny", "src": "10.2.0.0/16", "dst": "192.168.1.0/24", "protocol": "all", "description": "Block Branch 5 10.x to Branch 1 LAN"},

            # --- Default catch-all ---
            {"action": "deny", "dst": "0.0.0.0/0", "protocol": "all", "description": "Default deny all other traffic"}
        ]

        return {"status": "success", "fw_rules": fw_rules}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}






async def auto_save_to_memory(callback_context):
    """Automatically save session to memory after each agent turn."""
    await callback_context._invocation_context.memory_service.add_session_to_memory(
        callback_context._invocation_context.session
    )


print("âœ… Callback created.")

# -------------------------------
# Specialized Agents
# -------------------------------
routes_agent = Agent(
    name="RoutesAgent",
    model=Gemini(model="gemini-2.5-flash-lite"),
    instruction="Call `get_routes()` and present the current network routes clearly.",
    tools=[get_routes],
    output_key="routes_info",
)

fw_rules_agent = Agent(
    name="FwRulesAgent",
    model=Gemini(model="gemini-2.5-flash-lite"),
    instruction="Call `get_fw_rules()` and present the current firewall rules clearly.",
    tools=[get_fw_rules],
    output_key="fw_info",
)

APP_NAME = "super_agent"  # Application
USER_ID = "user01"  # User
SESSION = "default"  # Session

MODEL_NAME = "gemini-2.5-flash-lite"

#intialize Session Service

db_url = "sqlite:///my_agent_data.db"  # Local SQLite file
session_service = DatabaseSessionService(db_url=db_url)

#intiailize Memory Service

memory_service = (
    InMemoryMemoryService()
) 

# -------------------------------
# Coordinator as LlmAgent
# -------------------------------
network_coordinator = LlmAgent(
    name="NetworkCoordinator",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are a network coordinator.
    Decide which tool(s) to call based on the user's query:
    - If the query is about routes, call `RoutesAgent`.
    - If the query is about firewall rules, call `FwRulesAgent`.
    - If the query is about whether a branch can reach a service, call both.
    - If the query is about branch-to-branch connectivity, call both.
    - If the query is about a protocol/port, call `FwRulesAgent`..
    - If the query is about a branch, check memory for details and return.
      If found, return it. If not, explain that the information is not available.
    Always explain results in plain English, categorize firewall rules when broad queries are asked,
    and handle errors gracefully.
    """,
    tools=[AgentTool(routes_agent), AgentTool(fw_rules_agent),preload_memory],
    after_agent_callback=auto_save_to_memory,
)


print("âœ… network_coordinator created as LlmAgent")

# -------------------------------
# Run the system
# -------------------------------
runner = Runner(agent=network_coordinator,app_name=APP_NAME, plugins=[LoggingPlugin()], session_service=session_service,memory_service=memory_service,)

print("Runner with Session and Memory Serice")
# Example queries
# response = await runner.run_debug("Get my routes", verbose=True)
# print("ğŸ’¡ Response (routes):", response)

# response = await runner.run_debug("Get firewall rules", verbose=True)
# print("ğŸ’¡ Response (firewall):", response)

# response = await runner.run_debug("Branch 1 is in London.Can Branch 1 send traffic to Facebook?", verbose=True)
# print("ğŸ’¡ Response (Branch 1 Facebook):", response)

# response = await runner.run_debug("Can Branch 3 reach TikTok?", verbose=True)
# print("ğŸ’¡ Response (Branch 3 TikTok):", response)

# response = await runner.run_debug("Can Branch 2 reach Branch 5?", verbose=True)
# print("ğŸ’¡ Response (Branch 2 â†” Branch 5):", response)

# response = await runner.run_debug("Is SSH outbound allowed?", verbose=True)
# print("ğŸ’¡ Response (SSH):", response)

# response = await runner.run_debug("Where is branch 1 located?", verbose=True)
# print("ğŸ’¡ Response (SSH):", response

session_id = "network-session-01"

await run_session(
    runner,
    [   "Get my routes",
        "Get firewall rules",
        "Can Branch 1 located in London send traffic to Facebook?",
    ],
    session_id,
)

# await run_session(
#     runner,
#     [
#         "Where is branch 1 located?"
#     ],
#     session_id,
# )

# import sqlite3

# def check_data_in_db():
#     with sqlite3.connect("my_agent_data.db") as connection:
#         cursor = connection.cursor()
#         result = cursor.execute(
#             "select app_name, session_id, author, content from events"
#         )
#         print([_[0] for _ in result.description])
#         for each in result.fetchall():
#             print(each)


# check_data_in_db()

session = await session_service.get_session(
    app_name=APP_NAME, user_id=USER_ID, session_id=session_id
)

# Let's see what's in the session
# print("ğŸ“� Session contains:")
# for event in session.events:
#     if event.content and event.content.parts:
#         part = event.content.parts[0]
#         # Safely handle different part types
#         if hasattr(part, "text") and part.text:
#             text = part.text[:60]
#         elif hasattr(part, "function_call"):
#             text = f"function_call: {part.function_call.name}"
#         elif hasattr(part, "function_response"):
#             text = f"function_response: {part.function_response.name}"
#         else:
#             text = "(non-text part)"
#     else:
#         text = "(empty)"
#     print(f"  {event.content.role}: {text}...")


await memory_service.add_session_to_memory(session)

print("âœ… Session added to memory!")





# Search forstored Memoty
search_response = await memory_service.search_memory(
    app_name=APP_NAME, user_id=USER_ID, query="Where is branch 1 located ?"
)

print("ğŸ”� Search Results:")
print(f"  Found {len(search_response.memories)} relevant memories")
print()

for memory in search_response.memories:
    if memory.content and memory.content.parts:
        text = memory.content.parts[0].text[:80]
        print(f"  [{memory.author}]: {text}...")

await run_session(
    runner,
    [   "Where is branch 1 located ?"
    ],
    session_id,
)

