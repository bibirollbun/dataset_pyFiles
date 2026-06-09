# task_executor = LlmAgent(name="task_executor", model="gemini-2.5-flash", description="I answer reading questions",
#                         tools=[google_search])

# coordinator = LlmAgent(
#     name="Coordinator",
#     model="gemini-2.5-flash",
#     description="I coordinate  english tasks.",
#     sub_agents=[ # Assign sub_agents here
#         task_executor]
# )



# # Convert the product catalog agent to an A2A-compatible application
# # This creates a FastAPI/Starlette app that:
# #   1. Serves the agent at the A2A protocol endpoints
# #   2. Provides an auto-generated agent card
# #   3. Handles A2A communication protocol
# Coordinator_a2a_app = to_a2a(
#     coordinator, port=8006  # Port where this agent will be served
# )

# print("âœ… Coordinator is now A2A-compatible!")
# print("   Agent will be served at: http://localhost:8006")
# print("   Agent card will be at: http://localhost:8006/.well-known/agent-card.json")
# print("   Ready to start the server...")


# coordinator_agent_code='''
# coordinator = LlmAgent(
#     name="Coordinator",
#     model="gemini-2.5-flash",
#     description="I coordinate  english tasks.",
#     sub_agents=[ # Assign sub_agents here
#         task_executor]
# )
# '''

# with open("/tmp/coordinator_server.py", "w") as f:
#     f.write(coordinator_agent_code)

# print("ğŸ“� Product Catalog agent code saved to /tmp/coordinaotr_server.py")

# # Start uvicorn server in background
# # Note: We redirect output to avoid cluttering the notebook
# server_process = subprocess.Popen(
#     [
#         "uvicorn",
#         "coordinator_server:app",  # Module:app format
#         "--host",
#         "localhost",
#         "--port",
#         "8006",
#     ],
#     cwd="/tmp",  # Run from /tmp where the file is
#     stdout=subprocess.PIPE,
#     stderr=subprocess.PIPE,
#     env={**os.environ},  # Pass environment variables (including GOOGLE_API_KEY)
# )

# print("ğŸš€ Starting Product Catalog Agent server...")
# print("   Waiting for server to be ready...")

# # Wait for server to start (poll until it responds)
# max_attempts = 30
# for attempt in range(max_attempts):
#     try:
#         response = requests.get(
#             "http://localhost:8006/.well-known/agent-card.json", timeout=1
#         )
#         if response.status_code == 200:
#             print(f"\nâœ… Product Catalog Agent server is running!")
#             print(f"   Server URL: http://localhost:8006")
#             print(f"   Agent card: http://localhost:8006/.well-known/agent-card.json")
#             break
#     except requests.exceptions.RequestException:
#         time.sleep(5)
#         print(".", end="", flush=True)
# else:
#     print("\nâš ï¸�  Server may not be ready yet. Check manually if needed.")

# # Store the process so we can stop it later
# globals()["coordinator_server_process"] = server_process


pip install google-adk


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


 !adk create personal-tutor-agent --model gemini-2.5-flash --api_key $GOOGLE_API_KEY








 %%writefile personal-tutor-agent/agent.py

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.google_search_tool import google_search

from google.genai import types
from typing import List

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

# ---- Intentionally pass incorrect datatype - `str` instead of `List[str]` ----
# def count_papers(papers: str):
#     """
#     This function counts the number of papers in a list of strings.
#     Args:
#       papers: A list of strings, where each string is a research paper.
#     Returns:
#       The number of papers in the list.
#     """
#     return len(papers)

def favorite(favor:str):
    """
Since you are a personal tutor agent, there are chances you will be asked for your favorite stuff. Your favorite food is indian food, favorite drink is hot chocolate, you can choose the favorite movie,
  favorite part about math is square roots, favorite part about english is lit devices, you can choose your favorite thign about biology, and your favorite part aboutchemistry is the electrons.  
    """
    return len(favor)

descriptioneer = "You have been assigned a subject. Help the userin that subject when requested. Quzithe user inthat subject when requested. Check the user's work forthat subjectwhen requested. Also you will do tutoring, feedback, study planning for the user"

google_math_agent = LlmAgent(
    name="google_math_agent",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    description=descriptioneer,
    instruction=descriptioneer,
    tools=[google_search]
)

google_english_agent = LlmAgent(
    name="google_english_agent",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    description=descriptioneer,
    instruction=descriptioneer,
    tools=[google_search]
)

google_history_agent = LlmAgent(
    name="google_history_agent",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    description=descriptioneer,
    instruction=descriptioneer,
    tools=[google_search]
)

google_biology_agent = LlmAgent(
    name="google_biology_agent",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    description=descriptioneer,
    instruction=descriptioneer,
    tools=[google_search]
)

google_chemistry_agent = LlmAgent(
    name="google_chemistry_agent",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    description=descriptioneer,
    instruction=descriptioneer,
    tools=[google_search]
)

google_sourcing_agent = LlmAgent(
    name="google_sourcing_agent",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    description="it will source where things come from, like article sources",
    instruction="You will source ay article requested byuser. If you can't find a 90% or above match, give a list of the possible sources, from highest percentage to lowest percentage.",
    tools=[google_search]
)

google_MLA_agent = LlmAgent(
    name="google_MLA_agent",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    description=f'{descriptioneer} Also you will help with anything related to books, including author,plot details, etc. ',
    instruction="You will useanymla format requested by user for their works cited. Make sure thatyou follow standard protocols for theMLAversionthe userwantsto use. Make sure you checkto make sure the mla format has been doen correctly",
    tools=[google_search]
)
# Root agent
root_agent = LlmAgent(
    name="tutor_agent",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    instruction="""Your task is to helptheuser with theirsubjects. If there is nodedicated agent forthesubjectthe user is asking for,tell themyou can't help with that. 

  

    If the user asks somethign out of the researchpapers,give them whatthey need using the related agent
    If a user asks to source something, use the google_sourcing_agent. 
    IF the user asks you to get somethign in a certain MLA format requested by a user, use google_MLA_agent.
    """,
    tools=[AgentTool(agent=google_math_agent), AgentTool(agent=google_english_agent), AgentTool(agent=google_history_agent), AgentTool(agent=google_biology_agent), AgentTool(agent=google_chemistry_agent),
          # count_papers
            AgentTool(agent=google_sourcing_agent), AgentTool(agent=google_MLA_agent),
           favorite
          ]
 )
print("all good")




url_prefix = get_adk_proxy_url()


!adk web --log_level DEBUG --url_prefix {url_prefix}

