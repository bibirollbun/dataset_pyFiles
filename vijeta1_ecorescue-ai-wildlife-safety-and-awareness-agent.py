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
    os.environ["GOOGLE_API_KEY"] = 'GOOGLE_API_KEY'
    print( "Gemini API key setup complete.")
except Exception as e:
    print (
    f" Authentification Error: Please make sure you have added 'GOOGLE_API_KEY' to your kaggle secrets. Details: {e}"
    )


from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner, Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import AgentTool, google_search
from google.genai import types
from google.ai.generativelanguage_v1beta import DiscussServiceClient
print ("ADK components imported successfully.")


client = DiscussServiceClient(client_options={"api_key":os.environ["GOOGLE_API_KEY"]})
print ("Client ready.")

session_id = "369"
print("session_service.")


retry_config=types.HttpRetryOptions(
    attempts=5,     # Maximum retry attempts
    exp_base=7,     # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], #Retry on these HTTP errors
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
    output_key="output", #The result of this agent will be stored in the session state with this key.
)

print ("encounter_safety_agent created.")


# Injury Triage Agent : It's job is to help users access injured/sick wildlife safely and responsibly.
injury_triage_agent = Agent(
    name="InjuryTriageAgent",
    model=Gemini(
        model='gemini-2.5-flash-lite',
        retry_options=retry_config
    ),
    instruction="""Ask upto 3 diagnostic questions, determine severity, and give hands-off safety instructions only.""",
    tools=[google_search],
    output_key="output", #The result of this agent will be stored in the session state with this key.
)

print ("injury_triage_agent created.")


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
    output_key="output", #The result of this agent will be stored in the session state with this key.
)

print ("ecovisitor_guide_agent created.")


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
    output_key="output", #The result of this agent will be stored in the session state with this key.
)

print ("study_buddy_agent created.")


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
    output_key="output", #The result of this agent will be stored in the session state with this key.
)

print ("awareness_creator_agent created.")


#Root coordinator: Orchaestrates the workflow by calling sub-agents as tools.
root_agent = Agent(
    name="EcoRescueCoordinator",
    model=Gemini(
        model="Gemini-2.5-flash-lite",
        client=client,
        retry_options=retry_config
    ),
    #This instruction tells the root agent HOW to use it's tools (which are the other agents).
    instruction="""You are a ecorescue coordinator. You must route the query to the correct sub-agent:
    1. Wildlife encounter - EncounterSafetyAgent
    2. Injured animal - InjuryTriageAgent
    3. Visitor/park/species info - EcoVisitorGuideAgent
    4. Study help - StudyBuddyAgent
    5. Story, blog or awareness - AwarenessCreatorAgent
    """,
       #We wrap the sub-agents in 'AgentTool' to make them callable tools for the root agent.
    tools=[
        AgentTool(encounter_safety_agent), 
        AgentTool(injury_triage_agent), 
        AgentTool(ecovisitor_guide_agent), 
        AgentTool(study_buddy_agent), 
        AgentTool(awareness_creator_agent)],    
)

print("root_agent created.")


runner = InMemoryRunner(agent=root_agent)

print("Runner created.")


response = await runner.run_debug(
    "I saw a tiger on trail. What should I do?"
)


#Helper functions being defined.
from IPython.core.display import display, HTML
from jupyter_server.serverapp import list_running_servers

#Get the proxied url for the kaggle notebooks environment.
def get_adk_proxy_url():
    PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
    ADK_PORT = "8000"

    servers = list(list_running_servers())
    if not servers:
        raise Exception("No running jupyter servers found.")

baseURL = 'servers'[0],["base_url"]

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
     <div style="font-family: sans-serif; margin-bottom: 12px; colour: #333; font-size: 1.1em;">
         <strong>  IMPORTANT: Action Required</strong>
     </div>
     <div style="font-family: sans-serif; margin-bottom: 15px; colour: #333; line-height: 1.5;">
          The ADK web UI is <strong>not running yet</strong>. You must start it in the next cell.
          <ol style="margin-top: 10px; padding-left: 20px;">
              <li style="margin-bottom: 5px;"><strong>Run the next cell</strong> (the one with the <code>!adk web ...</code>) to start the ADK web UI.</li>
              <li style="margin-bottom: 5px;">Wait for that cell to show it is "Running" (it will not "complete").</li>
              <li style="Once it's running, <strong>return to this button</strong> and click it to open the UI.</li>
          </ol>
          <em style="font-size: 0.9em; color: #555;">(If you click the button before next cell, you will get a 500 error.)</em>
     </div>
     <a href='{url}' target='blank' style='
         display: inline-block; background-color: #1a73e8; color: white; padding: 10px 20px;
         text-decortion: none; border-radius: 25px; font-family: sans-serif; font-weight: 500;
         box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.2s ease;">
         Open ADK web UI (after running cell below)
     </a>
 </div>
 """

display(HTML(styled_html))

return url_prefix


print("Helper functions defined.")


!adk create sample-agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY


url_prefix = get_adk_proxy_url()


!adk web --url_prefix {url_prefix}

