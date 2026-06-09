import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")


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


!adk create eco_rescue_coordinator_agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY


%%writefile eco_rescue_coordinator_agent/agent.py

from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools import AgentTool, google_search
from google.genai import types

retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)

# Encounter Safety Agent : It's job is to guide users safely during unexpected wildlife sightings or confrontations
encounter_safety_agent = Agent(
    name="EncounterSafetyAgent",
    model=Gemini(
        model='gemini-2.5-flash-lite',
        retry_options=retry_config
    ),
    instruction="""Your job is to guide users safely during wildlife encounters.
    Nvere suggest approaching, feeding or touching animals.""",
    tools=[google_search],
    # output_key="agent_output", #The result of this agent will be stored in the session state with this key.
)

# Injury Triage Agent : It's job is to help users access injured/sick wildlife safely and responsibly.
injury_triage_agent = Agent(
    name="InjuryTriageAgent",
    model=Gemini(
        model='gemini-2.5-flash-lite',
        retry_options=retry_config
    ),
    instruction="""Ask upto 3 diagnostic questions, determine severity, and give hands-off safety instructions only.""",
    tools=[google_search],
    # output_key="agent_output", #The result of this agent will be stored in the session state with this key.
)

# EcoVisitor Guide Agent : It's job is to improve wildlife understanding and safety for park visitors, tourists, and fieldworks.
ecovisitor_guide_agent = Agent(
    name="EcoVisitorGuideAgent",
    model=Gemini(
        model='gemini-2.5-flash-lite',
        retry_options=retry_config
    ),
    instruction="""Explain the species, give 2-3 safety tips and add one fun fact.
    Tone should be friendly and educational.""",
    tools=[google_search],
    # output_key="agent_output", #The result of this agent will be stored in the session state with this key.
)

# Study Buddy Agent : It's job is to help students understand zoology concepts easily.
study_buddy_agent = Agent(
    name="StudyBuddyAgent",
    model=Gemini(
        model='gemini-2.5-flash-lite',
        retry_options=retry_config
    ),
    instruction="""Explain zoology concepts simply with examples.
    Use bullets and avoid jargon.""",
    tools=[google_search],
    # output_key="agent_output", #The result of this agent will be stored in the session state with this key.
)

# Awareness Creator Agent : It's job is to create stories, blogs, and awareness posts to create empathy for wildlife.
awareness_creator_agent = Agent(
    name="AwarenessCreatorAgent",
    model=Gemini(
        model='gemini-2.5-flash-lite',
        retry_options=retry_config
    ),
    instruction="""Create short, factual,positive wildlife content.
    Use simple language and avoid exaggeration.""",
    tools=[google_search],
    # output_key="agent_output", #The result of this agent will be stored in the session state with this key.
)

#Root coordinator: Orchaestrates the workflow by calling sub-agents as tools.
root_agent = Agent(
    name="EcoRescueCoordinator",
    model=Gemini(
        model="gemini-2.5-flash-lite", 
        retry_options=retry_config
    ),
    description="Analyzes user query and routes it to appropriate sub-agent.",
    #This instruction tells the root agent HOW to use it's tools (which are the other agents).
    instruction="""You are a ecorescue coordinator. You must route the query to the correct sub-agent and collect responses:

    INPUT: User's queries

    TASK: ROUTE TO APPROPRIATE AGENT
        1. Wildlife encounter - EncounterSafetyAgent
        2. Injured animal - InjuryTriageAgent
        3. Visitor/park/species info - EcoVisitorGuideAgent
        4. Study help - StudyBuddyAgent
        5. Story, blog or awareness - AwarenessCreatorAgent

    OUTPUT:
        Collect the response from appropriate agent and present to user.
        YOU MUST OUTPUT THE SUBAGENT RESPONSES TO THE USER
    """,
       #We wrap the sub-agents in 'AgentTool' to make them callable tools for the root agent.
    tools=[
        AgentTool(agent=encounter_safety_agent), 
        AgentTool(agent=injury_triage_agent), 
        AgentTool(agent=ecovisitor_guide_agent), 
        AgentTool(agent=study_buddy_agent), 
        AgentTool(agent=awareness_creator_agent)
    ],    
)

print("root_agent created.")


url_prefix = get_adk_proxy_url()


!adk web --log_level DEBUG --url_prefix {url_prefix}




