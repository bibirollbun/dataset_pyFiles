import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    # Use os.getenv instead if you're running the notebook locally outside of kaggle
    # GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


import logging
import os

# Clean up any previous logs
for log_file in ["logger.log", "web.log", "tunnel.log"]:
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"ğŸ§¹ Cleaned up {log_file}")

# Configure logging with DEBUG log level.
logging.basicConfig(
    filename="logger.log",
    level=logging.DEBUG,
    format="%(filename)s:%(lineno)s %(levelname)s:%(message)s",
)

print("âœ… Logging configured")


# # Run this cell to debug using the ADK web UI

# !mkdir research-agent

# !rm -rf /kaggle/working/research-agent

# !adk create research-agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY


# run_session() helper fx

from typing import Any, Dict

from google.adk.agents import Agent, LlmAgent, SequentialAgent
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.models.google_llm import Gemini
from google.adk.sessions import DatabaseSessionService

from google.adk.runners import Runner
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from google.genai import types


APP_NAME = "default"  # Application
USER_ID = "default"  # User
SESSION = "default"  # Session
MODEL_NAME = "gemini-2.5-flash-lite"

async def run_session(
    runner_instance: Runner, user_queries: list[str] | str, session_id: str = "default"
):
    """Helper function to run queries in a session and display responses."""
    print(f"\n### Session: {session_id}")

    # Create or retrieve session
    try:
        session = await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )
    except:
        session = await session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )

    # Convert single query to list
    if isinstance(user_queries, str):
        user_queries = [user_queries]

    # Process each query
    for query in user_queries:
        print(f"\nUser > {query}")
        query_content = types.Content(role="user", parts=[types.Part(text=query)])

        # Stream agent response
        async for event in runner_instance.run_async(
            user_id=USER_ID, session_id=session.id, new_message=query_content
        ):
            if event.is_final_response() and event.content and event.content.parts:
                text = event.content.parts[0].text
                if text and text != "None":
                    print(f"Model: > {text}")



# %%writefile researchagent/agent.py

from typing import Any, Dict

from google.adk.agents import Agent, LlmAgent, SequentialAgent
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.models.google_llm import Gemini
from google.adk.sessions import DatabaseSessionService

from google.adk.runners import Runner
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from google.genai import types


APP_NAME = "default"  # Application
USER_ID = "default"  # User
SESSION = "default"  # Session
MODEL_NAME = "gemini-2.5-flash-lite"

retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)

# MCP integration
mcp_pubmed_server = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",  # Run MCP server via npx
            args=[
                "-y",  # Argument for npx to auto-confirm install
                "@cyanheads/pubmed-mcp-server", # https://github.com/cyanheads/pubmed-mcp-server
            ],
            env={
                "MCP_LOG_LEVEL": "debug",
            },
        ),
        timeout=90,
    )
)

# Root agent
root_agent = LlmAgent(
    name="research_paper_finder_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are a medical expert explaining health topics to a regular person.

    Use the 'pubmed_search_articles' tool to find PMID's for research papers on the given topic using the following filters:
        "maxResults": 5,
        "sortBy": "pub_date",
        "filterByPublicationTypes": ["Review", "Journal Article"],

    Then, use the 'mcp_pubmed_server.pubmedFetchContents' tool to get the full text of the papers based on the PMIDs.

    Return the summarize the search results and include citations and a link to each paper used based on the PMIDs.
    """,
    tools=[mcp_pubmed_server],
)



# Uncomment this cell if you want to run the agent directly from the notebook instead of running the ADK web UI
# Also, comment out `%%writefile researchagent/agent.py` from the previous cell so that it runs the agent code instead of saving it to a file

from google.adk.plugins.logging_plugin import (
    LoggingPlugin,
)
from google.adk.plugins.reflect_retry_tool_plugin import (
    ReflectAndRetryToolPlugin,
)

# SQLite database will be created automatically
# Use db_url = "sqlite+aiosqlite:///my_agent_data.db" when running locally
db_url = "sqlite:///my_agent_data.db"
session_service = DatabaseSessionService(db_url=db_url)

# Create a new runner with persistent storage
runner = Runner(
    agent=root_agent,
    app_name=APP_NAME, 
    session_service=session_service,
    plugins=[
        LoggingPlugin(), # Add the plugin. Handles standard Observability logging across ALL agents
        ReflectAndRetryToolPlugin(), # This plugin fixes errors sometimes by retrying the tool call automatically
    ],
)

# There are 2 different ways to run the agent 
# Option 1: run agent in the notebook here

QUERY = "I want to learn about stage 4 prostate cancer. What can cause it?"
# QUERY = "Summarize the most relevant 2 research papers on stage 4 prostate cancer."
# QUERY = "Summarize some relevant research papers about the best treatments for stage 4 prostate cancer?"
# QUERY = "Can you tell me more about genomically targeted therapies"
# QUERY = "What are some common fungal diseases for broccoli plants that I should know about?"

await run_session(runner, QUERY, "conversation-09")



# View session events

# session = await session_service.get_session(
#     app_name=APP_NAME, user_id=USER_ID, session_id="conversation-01"
# )

# # Let's see what's in the session
# print("ğŸ“� Session contains:")
# for event in session.events:
#     # print(event) # Uncomment this line to view the full event
#     text = (
#         event.content.parts[0].text[:6000]
#         if event.content and event.content.parts and event.content.parts[0].text
#         else "(empty)"
#     )
#     print(f"  {event.content.role}: {text}")


# Option 2: 
# runs the ADK web UI

# from IPython.core.display import display, HTML
# from jupyter_server.serverapp import list_running_servers


# # Gets the proxied URL in the Kaggle Notebooks environment
# def get_adk_proxy_url():
#     PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
#     ADK_PORT = "8000"

#     servers = list(list_running_servers())
#     if not servers:
#         raise Exception("No running Jupyter servers found.")

#     baseURL = servers[0]["base_url"]

#     try:
#         path_parts = baseURL.split("/")
#         kernel = path_parts[2]
#         token = path_parts[3]
#     except IndexError:
#         raise Exception(f"Could not parse kernel/token from base URL: {baseURL}")

#     url_prefix = f"/k/{kernel}/{token}/proxy/proxy/{ADK_PORT}"
#     url = f"{PROXY_HOST}{url_prefix}"

#     styled_html = f"""
#     <div style="padding: 15px; border: 2px solid #f0ad4e; border-radius: 8px; background-color: #fef9f0; margin: 20px 0;">
#         <div style="font-family: sans-serif; margin-bottom: 12px; color: #333; font-size: 1.1em;">
#             <strong>âš ï¸� IMPORTANT: Action Required</strong>
#         </div>
#         <div style="font-family: sans-serif; margin-bottom: 15px; color: #333; line-height: 1.5;">
#             The ADK web UI is <strong>not running yet</strong>. You must start it in the next cell.
#             <ol style="margin-top: 10px; padding-left: 20px;">
#                 <li style="margin-bottom: 5px;"><strong>Run the next cell</strong> (the one with <code>!adk web ...</code>) to start the ADK web UI.</li>
#                 <li style="margin-bottom: 5px;">Wait for that cell to show it is "Running" (it will not "complete").</li>
#                 <li>Once it's running, <strong>return to this button</strong> and click it to open the UI.</li>
#             </ol>
#             <em style="font-size: 0.9em; color: #555;">(If you click the button before running the next cell, you will get a 500 error.)</em>
#         </div>
#         <a href='{url}' target='_blank' style="
#             display: inline-block; background-color: #1a73e8; color: white; padding: 10px 20px;
#             text-decoration: none; border-radius: 25px; font-family: sans-serif; font-weight: 500;
#             box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.2s ease;">
#             Open ADK Web UI (after running cell below) â†—
#         </a>
#     </div>
#     """

#     display(HTML(styled_html))
#     return url_prefix


# url_prefix = get_adk_proxy_url()


# !adk web --log_level DEBUG --reload_agents

