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
import sys
import time
import subprocess
import requests
import logging

from google.genai import types
from google.adk.agents import Agent, LlmAgent, SequentialAgent, ParallelAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent, AGENT_CARD_WELL_KNOWN_PATH
from kaggle_secrets import UserSecretsClient

# 1. Configure API Key
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… API Key Configured")
except Exception as e:
    print(f"âš ï¸� Authentication Error: {e}")

# 2. Configure Retry Logic
# Handles temporary API glitches automatically
retry_config = types.HttpRetryOptions(
    attempts=3, exp_base=2, initial_delay=1, http_status_codes=[429, 500, 503]
)

# 3. Configure Observability (Logging)
# This plugin records agent thoughts and actions
logging.basicConfig(level=logging.INFO)
logging_plugin = LoggingPlugin()

print("âœ… Setup & Observability Complete")


# Define the code for the separate microservice
trend_analyzer_code = """
import os
from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini
from google.genai import types

retry_config = types.HttpRetryOptions(attempts=3, exp_base=2, initial_delay=1, http_status_codes=[429, 500, 503])

# The Specialist Agent
trend_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="trend_analyzer",
    description="Analyzes a list of skills to identify market trends.",
    instruction=\"\"\"
    You are a Job Market Trend Analyst.
    Input: A list of technical skills extracted from job descriptions.
    Task: Identify 3 'Rising Stars' (skills gaining popularity) and 3 'Foundational Staples'.
    Output: A concise analysis report.
    \"\"\"
)

# Expose via A2A on port 8001
app = to_a2a(trend_agent, port=8001)
"""

# Write the server code to a file
with open("trend_server.py", "w") as f:
    f.write(trend_analyzer_code)

# Start the server in the background
print("ğŸš€ Starting Remote Trend Analyzer Service...")
server_process = subprocess.Popen(
    ["uvicorn", "trend_server:app", "--host", "localhost", "--port", "8001"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env={**os.environ}
)

# Health check: Wait for the server to come online
for _ in range(15):
    try:
        if requests.get("http://localhost:8001/.well-known/agent-card.json").status_code == 200:
            print("âœ… Remote Trend Analyzer is Online (A2A)")
            break
    except:
        time.sleep(1)

# Connect to the remote agent using the Client Proxy
remote_trend_analyzer = RemoteA2aAgent(
    name="RemoteTrendAnalyzer",
    agent_card=f"http://localhost:8001{AGENT_CARD_WELL_KNOWN_PATH}"
)


# Agent 1: Search for AI Engineering jobs
ai_job_searcher = Agent(
    name="AI_Job_Searcher",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    tools=[google_search],
    instruction="Search for 'most in-demand AI engineering skills 2025'. List the top 5 technical skills found.",
    output_key="ai_skills"  # Saves result to session state
)

# Agent 2: Search for Data Science jobs
data_job_searcher = Agent(
    name="Data_Job_Searcher",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    tools=[google_search],
    instruction="Search for 'top Data Science skills 2025'. List the top 5 technical skills found.",
    output_key="data_skills" # Saves result to session state
)

# The Parallel Team Manager
parallel_search_team = ParallelAgent(
    name="Market_Intelligence_Team",
    sub_agents=[ai_job_searcher, data_job_searcher]
)

print("âœ… Parallel Search Team Created")


# Agent 3: Skill Extractor (Cleans the data)
skill_extractor = Agent(
    name="Skill_Extractor",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    Review the findings from the search team:
    AI Skills: {ai_skills}
    Data Skills: {data_skills}
    
    Create a single, de-duplicated comma-separated list of all extracted technical skills.
    """,
    output_key="consolidated_skills"
)

# Agent 4: The Bridge to our Remote Service
# Uses the remote A2A agent as a tool
trend_adapter = Agent(
    name="Trend_Analysis_Bridge",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    tools=[AgentTool(remote_trend_analyzer)], 
    instruction="""
    Take this list of skills: {consolidated_skills}
    
    Call the 'trend_analyzer' tool to analyze these skills. 
    Return the analysis report exactly as provided by the tool.
    """,
    output_key="trend_report"
)

# Agent 5: The Career Advisor
advisor_agent = Agent(
    name="Career_Advisor",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    Based on this Trend Analysis: 
    {trend_report}
    
    Recommend 3 specific projects a candidate should build to demonstrate these skills.
    Be brief and high-impact.
    """,
    output_key="final_recommendation"
)

# The Master Pipeline
job_market_pipeline = SequentialAgent(
    name="AI_Job_Market_Insider_Pipeline",
    sub_agents=[
        parallel_search_team, # Step 1: Search (Parallel)
        skill_extractor,      # Step 2: Extract
        trend_adapter,        # Step 3: Analyze (Remote A2A)
        advisor_agent         # Step 4: Recommend
    ]
)

print("âœ… Pipeline Agents Created")


print("\nğŸ”¥ Initializing AI Job Market Insider Pipeline...")
runner = InMemoryRunner(
    agent=job_market_pipeline, 
    plugins=[logging_plugin] # Attach Observability
)

# Run the pipeline
# We don't need a complex prompt here because the agents have specific instructions
response = await runner.run_debug("Generate a job market report.")

# Cleanup: Always stop your background processes!
try:
    server_process.terminate()
    print("\nğŸ›‘ Remote Server Stopped.")
except:
    pass


from IPython.core.display import display, HTML
from jupyter_server.serverapp import list_running_servers

# Helper to get the proxy URL for the UI
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


# Create the folder structure for the UI agent
!adk create job_market_agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY


%%writefile job_market_agent/agent.py
import os
from google.genai import types
from google.adk.agents import Agent, LlmAgent, SequentialAgent, ParallelAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent, AGENT_CARD_WELL_KNOWN_PATH

# 1. Configure Retry Logic
retry_config = types.HttpRetryOptions(
    attempts=3, exp_base=2, initial_delay=1, http_status_codes=[429, 500, 503]
)

# 2. Define the Search Team (Parallel)
ai_job_searcher = Agent(
    name="AI_Job_Searcher",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    tools=[google_search],
    instruction="Search for 'most in-demand AI engineering skills 2025'. List the top 5 technical skills found.",
    output_key="ai_skills"
)

data_job_searcher = Agent(
    name="Data_Job_Searcher",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    tools=[google_search],
    instruction="Search for 'top Data Science skills 2025'. List the top 5 technical skills found.",
    output_key="data_skills"
)

parallel_search_team = ParallelAgent(
    name="Market_Intelligence_Team",
    sub_agents=[ai_job_searcher, data_job_searcher]
)

# 3. Define the Processor (Sequential)
skill_extractor = Agent(
    name="Skill_Extractor",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    Review the findings from the search team:
    AI Skills: {ai_skills}
    Data Skills: {data_skills}
    
    Create a single, de-duplicated comma-separated list of all extracted technical skills.
    """,
    output_key="consolidated_skills"
)

# 4. Define the Remote Adapter
# Connects to your locally running Trend Analyzer service on port 8001 (Must be running!)
remote_trend_analyzer = RemoteA2aAgent(
    name="RemoteTrendAnalyzer",
    agent_card=f"http://localhost:8001{AGENT_CARD_WELL_KNOWN_PATH}"
)

trend_adapter = Agent(
    name="Trend_Analysis_Bridge",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    tools=[AgentTool(remote_trend_analyzer)], 
    instruction="""
    Take this list of skills: {consolidated_skills}
    
    Call the 'trend_analyzer' tool to analyze these skills. 
    Return the analysis report exactly as provided by the tool.
    """,
    output_key="trend_report"
)

# 5. Define the Advisor
advisor_agent = Agent(
    name="Career_Advisor",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction=""
    Based on this Trend Analysis: 
    {trend_report}
    
    Recommend 3 specific projects a candidate should build to demonstrate these skills.
    Be brief and high-impact.
    """,
    output_key="final_recommendation"
)

# 6. Create the Pipeline (The Root Agent)
# This variable 'root_agent' will be automatically picked up by the UI
root_agent = SequentialAgent(
    name="AI_Job_Market_Insider_Pipeline",
    sub_agents=[
        parallel_search_team, # Step 1: Search 
        skill_extractor,      # Step 2: Extract
        trend_adapter,        # Step 3: Analyse (Remote A2A)
        advisor_agent         # Step 4: Recommend
    ]
)


# Get the URL prefix
url_prefix = get_adk_proxy_url()

# Start the ADK Web UI
print(f"ğŸš€ Starting UI at proxy: {url_prefix}")
!adk web --log_level DEBUG --url_prefix {url_prefix}




