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
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
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


!adk create ER-agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY


%%writefile ER-agent/agent.py
# Import Necessary Components
from google.adk.agents import Agent, SequentialAgent, ParallelAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search, AgentTool, BaseTool
from google.genai import types

# print("âœ… ADK components imported successfully.")

#Configure Retry Options
retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)
# print("âœ… Retry options configured successfully.")

# Create Caching Tool
class CachedAgentTool(BaseTool):
    def __init__(self, agent):
        super().__init__(name=f"cached_{agent.name}", description=f"Cached output of {agent.name}")
        self.agent = agent
        self._cache = None

    @classmethod
    def input_schema(cls):
        # Pass a user prompt (optional)
        return {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "User prompt for the agent"}
            },
            "required": ["prompt"]
        }

    def __call__(self, prompt: str):
        if self._cache is None:
            self._cache = runner.run(self.agent, prompt)
        return self._cache

#Create OpenAPI Elden Ring Tool
class EldenRingAPITool(BaseTool):
    def __init__(self):
        super().__init__(
            name="elden_ring_api",
            description="Fetches data from the Elden Ring API"
        )

    @classmethod
    def input_schema(cls):
        return {
            "type": "object",
            "properties": {
                "endpoint": {
                    "type": "string",
                    "description": "API endpoint like 'weapons', 'bosses', 'items'"
                },
                "query": {
                    "type": "string",
                    "description": "Optional query string like '?name=Moonveil'",
                    "nullable": True
                }
            },
            "required": ["endpoint"]
        }

    def __call__(self, endpoint: str, query: str = None):
        import requests
        base = "https://eldenring.fanapis.com/api/"
        url = base + endpoint + (query or "")
        response = requests.get(url)
        return response.json()

elden_tool = EldenRingAPITool()

#Create Fantasy Theme Assistant: Its job is to translate the user's request
fantasy_theme_assistant = Agent(
    name="fantasythemeassistant",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="A simple agent that uses Google Search to turn a user prompt into specific fantasy themes/ideas",
    instruction="""You are a fantasy theme generator. Use Google Search to turn the user prompt 
    into specific fantasy themes, ideas, gear, and stats remembering that your output will be 
    used to build a character in Elden Ring, e.g. A user says they want to be a ronin and you 
    use Google Search to find what weapons/fighting styles ronins typically have."""
,
    tools=[google_search],
    output_key="character_info",
)
# print("âœ… Fantasy Theme Assistant defined.")

# Convert Agent to Tool
cached_fantasy_theme_assistant_tool = CachedAgentTool(fantasy_theme_assistant)


#Create Class Agent: Its job is to decide what class to give the user
class_agent = Agent(
    name="ClassAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""Read the provided character information: {character_info}. 
    Choose the best fitting 2 classes using the Elden Ring tool.""",
        tools=[elden_tool],
        output_key="final_classes",
)
# print("âœ… class_agent created.")

# Convert Agent to Tool
cached_class_agent_tool = CachedAgentTool(class_agent)


#Create Armor Agent: Its job is to decide what armor to give the user
armor_agent = Agent(
    name="ArmorAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""Read the provided character information: {character_info}.
    Choose the best fitting armor using the Elden Ring tool.""",
        tools=[elden_tool],
        output_key="armor",    
)
# print("âœ… armor_agent created.")

# Convert Agent to Tool
cached_armor_agent_tool = CachedAgentTool(armor_agent)


#Create Item/Spirit Agent: Its job is to decide what Items/Spirits to give the user
item_spirit_agent = Agent(
    name="ItemSpiritAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""Read the provided character information: {character_info}.
    Choose the best fitting items/spirits using the Elden Ring tool. Remember the term items in elden ring 
    is used to describe consumables. Exclude all weapons, armors, talismans, spells. return both the 
    items(consumables) and spirits(ashes)""",
        tools=[elden_tool],
        output_key="item_spirit",
)
# print("âœ… item_spirit_agent created.")

# Convert Agent to Tool
cached_item_spirit_agent_tool = CachedAgentTool(item_spirit_agent)


#Create Incantations/Sorceries Agent: Its job is to decide what magic to give the user
magic_agent = Agent(
    name="MagicAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""Read the provided character information: {character_info}.
    Choose the best-fitting sorceries/incantations using the Elden Ring tool. 
    Make sure to mention if it's an incantation or a sorcery. 
    You can decide to not pick any sorceries or incantations if it doesn't fit the character_info.
    The combined total number of sorceries and incantations must not exceed 12.""",
        tools=[elden_tool],
        output_key="magic",
)
# print("âœ… magic_agent created.")

# Convert Agent to Tool
cached_magic_agent_tool = CachedAgentTool(magic_agent)


#Create Weapons/Shield Agent: Its job is to decide what weapon to give the user
weapon_agent = Agent(
    name="WeaponAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""Read the provided magic information: {magic}.
    Also read the provided character information using 
    the cached_fantasy_theme_assistant_tool: {character_info}. 
    Choose the best-fitting weapons/shields using the Elden Ring tool. 
    Each hand can have at max 3 objects (meaning weapons and shields).
    If there is sorcery, a staff is needed.
    If there is incantation, a seal is needed.
    Try to select the best staff/seal for the schools of spells.""",
        tools=[cached_fantasy_theme_assistant_tool, elden_tool],
        output_key="weapons",
)
# print("âœ… weapon_agent created.")

# Convert Agent to Tool
cached_weapon_agent_tool = CachedAgentTool(weapon_agent)


#Create Elden Wiki Agent: Its job is to use Google Search Tool to only access elden ring wiki to figure out what weapons/ideas correspond to specific Ashes of Wars, and Ammos
elden_wiki_agent = Agent(
    name="EldenWikiAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are ensuring weapon compatibility.
    Read the provided weapons information: {weapons}.
Use the google_search tool but restrict it to â€œsite:eldenring.wiki.fextralife.comâ€�.
Based on the weapons information, determine what Ammos are definitely needed depending on the type of weapon.
Based on the weapons information, also give a list of Ashes of Wars that could be applied depending on the type of weapon.
    """,
        tools=[google_search],
        output_key="eldenwiki",
)
# print("âœ… elden_wiki_agent created.")

# Convert Agent to Tool
cached_elden_wiki_agent_tool = CachedAgentTool(elden_wiki_agent)


#Create Ashes of War Agent: Its job is to decide what Ashes of War to give the user
ashes_of_war_agent = Agent(
    name="AshesofWarAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""Read the provided Elden Ring Wiki information: {eldenwiki}.
    Also read the provided character information using 
    the cached_fantasy_theme_assistant_tool: {character_info}. 
    Choose the best-fitting Ashes of Wars using the Elden Ring tool. 
    You can only have 1 Ashes of War per weapon.""",
        tools=[cached_fantasy_theme_assistant_tool, elden_tool],
        output_key="ashes_of_war",
)
# print("âœ… ashes_of_war_agent created.")

# Convert Agent to Tool
cached_ashes_of_war_agent_tool = CachedAgentTool(ashes_of_war_agent)


#Create Ammo Agent: Its job is to decide what Ammo to give the user
ammo_agent = Agent(
    name="AmmoAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""Read the provided Elden Ring Wiki information: {eldenwiki}.
    Also read the provided character information using 
    the cached_fantasy_theme_assistant_tool: {character_info}. 
    Choose the best-fitting Ammos using the Elden Ring tool. 
    You can only have 2 types of arrows,including 2 of the same type or 1 each of 2 different types.
    You can also only have 2 types of bows, including 2 of the same type or 1 each of 2 different types.""",
        tools=[cached_fantasy_theme_assistant_tool, elden_tool],
        output_key="ammo",
)
# print("âœ… ammo_agent created.")

# Convert Agent to Tool
cached_ammo_agent_tool = CachedAgentTool(ammo_agent)


# Create ParallelAgent that runs the ammos and ashes of war agents simultaneously.
parallel_ammo_ashesofwar = ParallelAgent(
    name="ParallelAmmoAshesofWar",
    sub_agents=[ashes_of_war_agent, ammo_agent],
)

# Create SequentialAgent that ensures weapon compatibility with the different Elden Ring characteristics.
sequential_weapon_compatibility = SequentialAgent(
    name="WeaponCompatibility",
    sub_agents=[magic_agent, weapon_agent, elden_wiki_agent, parallel_ammo_ashesofwar],
)

# Create ParallelAgent that compiles inputs for the talisman agent
parallel_classes_except_talisman = ParallelAgent(
    name="ParallelClassesExceptTalisman",
    sub_agents=[sequential_weapon_compatibility, class_agent, armor_agent, item_spirit_agent], 
)

# print("âœ… Parallel and Sequential Weapons Agents created.")



# Create Talisman Agent: Its job is to decide what Talismans to give the user
talisman_agent = Agent(
    name="TalismanAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    Read the provided  information: {ashes_of_war}, {ammo}, {item_spirit}.
    Also call the existing information using the 
    cached_fantasy_theme_assistant_tool, cached_magic_agent_tool, 
    cached_weapon_agent_tool: {character_info}, {weapons}, {magic}.
    Choose the best-fitting Talismans using the Elden Ring tool. 
    You can only have 4 total talismans.""",
        tools=[cached_fantasy_theme_assistant_tool, cached_magic_agent_tool, 
               cached_weapon_agent_tool, elden_tool],
        output_key="talisman",
)

# print("âœ… talisman_agent created.")


# Create Output Agent: Its job is to give an output
output_agent = Agent(
    name="OutputAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are an agent that takes the outputs from the previous agents and 
    compiles it together in a presentable manner.
    Read the provided  information: {talisman}.
    Call the cached_fantasy_theme_assistant_tool, cached_weapon_agent_tool,
    cached_magic_agent_tool, cached_item_spirit_agent_tool,
    cached_asges_of_war_agent_tool, cached_ammo_agent_tool, 
    cached_class_agent_tool, cached_armor_agent_tool to find the 
    existing information: {character_info}, {weapons}, {magic}, {item_spirit}, 
    {ashes_of_war}, {ammo}, {final_classes}, {armor}
    """,
        tools=[cached_fantasy_theme_assistant_tool, cached_weapon_agent_tool,
    cached_magic_agent_tool, cached_item_spirit_agent_tool,
    cached_ashes_of_war_agent_tool, cached_ammo_agent_tool, 
    cached_class_agent_tool, cached_armor_agent_tool],
        output_key="final_output",
)

# print("âœ… output_agent created.")


# Create the total sequence
root_agent = SequentialAgent(
    name="ERCharacterCreation",
    sub_agents=[fantasy_theme_assistant, parallel_classes_except_talisman, talisman_agent, output_agent],
)
# print lines are a relic of the agents being separated into individual code blocks


url_prefix = get_adk_proxy_url()


# This cell will not "complete", but will remain
# running and serving the ADK web UI until you manually stop the cell.
!adk web --log_level DEBUG --url_prefix {url_prefix}


# this is the test runner before using the adk web ui
# runner = InMemoryRunner(agent=root_agent)
# response = await runner.run_debug(
#    "I want to be a necromancer"
# )

