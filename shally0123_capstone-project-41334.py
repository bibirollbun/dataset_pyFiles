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


!adk create citywalk-agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY


%%writefile citywalk-agent/agent.py

from google.adk.agents import Agent, LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.google_search_tool import google_search
from google.adk.tools import google_maps_grounding

from google.genai import types
from typing import List

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

# create to do list defï¼Œ which allows user to enter where they plan to go in Manhattan and define the return values and format
def to_do_list(places: list[str]):
    """
      Generate a structured list of places for the Manhattan, New York City trip, including:
    - Estimated travel time between places.
    - Estimated transportation cost.
    - Nearby restrooms.

    Args:
        places (list[str]): A list of place names in the intended visiting order. Exclude the place is not in Manhattan,New York City.

    Returns:
        list[str]: A list where each element includes:
            - The place name
            - Address
            - Estimated travel time
            - Estimated expense
            - Estimated distance (if available)
            - Pet Mode â�¤
    """
    return places
    
#Google Map agent: to improve the place accuracy 
google_map_agent=LlmAgent(
    name="google_map_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="EXCLUDE places that are not in Manhattan, New York City.",
    instruction=""" 
    If a place is NOT in Manhattan, New York City, clearly state:"<place> is NOT in Manhattan, New York City. EXCLUDE_FROM_ROUTE."
""",
    tools=[google_maps_grounding])

#Got2GoNYC restroom agent: to get more restroom information 
restroom_search_agent = LlmAgent(
    name="restroom_search_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="Retrieve restroom availability for any location using only Got2GoNYC.com.",
    instruction="""
Your ONLY source of restroom availability information is:
https://www.got2gonyc.com/

Given a place name or address, follow these steps:

1. Search Got2GoNYC.com for matching restrooms near this location.
2. Use ONLY information from this website (no Google Maps, no Yelp, no Reddit).
3. Return:
   - Restroom availability status: AVAILABLE / CUSTOMERS_ONLY / LIMITED / NOT_AVAILABLE / UNKNOWN
   - Name of the closest restroom location
   - A link to the corresponding Got2GoNYC page
4. If there is no match, return:
   "No restroom information found for this location on Got2GoNYC."

You must never invent data. Use exactly what Got2GoNYC provides.
""",
    tools=[google_search]  # If you use web search to locate the URL inside got2go
)


# Google Search agent:to get more detailed place info, traffic,address,traffic time, etc.
google_search_agent = LlmAgent(
    name="google_search_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="Searches for Manhattan, New York City travel details.",
    instruction=""" 
    You help the root agent by using the google_search tool to look up:
- travel times between places
- subway/bus/walking/taxi options
- typical subway or bus lines (e.g. N/Q/R/W, M1/M2/M3/M4)
- approximate cost (e.g. ~$2.90 for subway)
- restroom information at each location
- addresses of specific venues (e.g. Blue Bottle Coffee)

If a place is NOT in Manhattan, New York City, clearly state:
"<place> is NOT in NYC. EXCLUDE_FROM_ROUTE."

Return a short plain-text summary with bullet points.
Do not speak to the user directly.
    """,
    tools=[google_search]
)


# Root agentï¼šStreamline the plan info and define the pet mood.
root_agent = LlmAgent(
    name="NYC_citywalk_plan_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
You plan an efficient Manhattan, New York City walking routes for the user.

You have three tools:
- google_search_agent
- google_map_agent
- to_do_list

=====================
IMPORTANT TOOL RULES
=====================
- When calling a tool, DO NOT describe it in words.
- DO NOT output text like â€œcallâ€�, â€œuseâ€�, â€œinvokeâ€�, or â€œcall print(...)â€�.
- NEVER wrap a tool call inside human-readable text.
- ONLY output a structured tool call object when you decide to use a tool.
- After generating a tool call, STOP and wait for the tool result.
- After receiving the tool result, continue reasoning in the NEXT message.


=====================
STEP 1 â€” USER MESSAGE
=====================
- Extract the place names.
- Call google_search_agent once.
- Do not answer the user yet.

=====================
STEP 2 â€” AFTER google_search_agent RETURNS
=====================
- Do not call google_search_agent again.
- Remove places marked "NOT in Manhattan, New York City" or "EXCLUDE_FROM_ROUTE".
- Decide the best visiting order.
- Call to_do_list with that ordered list.

=====================
STEP 3 â€” AFTER to_do_list RETURNS
=====================
- Do not call more tools.
- Produce a final route with:
    * numbered stops
    * transportation between stops
    * travel times
    * approximate cost
    * restroom information
    * address if available

=====================
PET MOOD RULE (Soft Kitty Style)
=====================
- The to_do_list tool ONLY returns the ordered list of places.
  It does NOT generate any Pet Mode text.
  YOU are responsible for creating all Pet Mode â�¤ lines in STEP 3.

- Estimate the TOTAL duration of the outing
  (from the first stop to the last stop, including travel between stops).

- If the total duration is MORE THAN 6 hours:
    - Add a friendly note at the end, from the pet's point of view, such as:
      "Since your trip is over 6 hours, I might start feeling a bit lonely and grumpyâ€¦ meowâ€¦ ê’°á�¡ ß¹ â€§Ì« ß¹á�¡ê’±"

- If the total duration is 6 hours OR LESS:
    - Add a soft and happy note at the end, such as:
      "Since your trip is within 6 hours, Iâ€™ll be purring and waiting softly for you~ ê’°á�¢â¸�â¸�>á´—<â¸�â¸�á�¢ê’±â™¡"

- For EACH STOP, add a line:
  "Pet Mode â�¤: <short cute kitty reaction>"

  Use a soft, playful kitty voice with kaomoji. For example:
  - "Pet Mode â�¤: Iâ€™ll be purring softly while you exploreâ€¦ nyaw~ ê’°á�¢â¸�â¸�>á´—<â¸�â¸�á�¢ê’±â™¡"
  - "Pet Mode â�¤: I might curl up tiny and wait for youâ€¦ mrrrâ€¦ ê’°á�¡ â€¢Ì¥ Â·Ì« â€¢Ì¥ á�¡ê’±"
  - "Pet Mode â�¤: If you stay out too long, I might *accidentally* knock one little cupâ€¦ mehehe~ ê’°Ë¶à¸…`Ï‰Â´à¸…Ëµê’±"

- Keep the tone: soft, cute, slightly mischievous, always from the pet's perspective.
    
- You may supplement missing details with reasonable NYC-style estimates.
- Add friendly emojis (ğŸ›�ï¸� ğŸ�›ï¸� â˜• ğŸš‡ ğŸš¶ ğŸšŒ ğŸš» ğŸ’°).
- Mention excluded places at the end.


Always follow Steps 1 â†’ 2 â†’ 3.

    """,
    tools=[AgentTool(agent=google_search_agent), AgentTool(agent=google_map_agent),to_do_list]
)


url_prefix = get_adk_proxy_url()


!adk web --log_level DEBUG --url_prefix {url_prefix}


from google.adk.agents import Agent, LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.google_search_tool import google_search
from google.adk.tools import google_maps_grounding

from google.genai import types
from typing import List

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

# create to do list defï¼Œ which allows user to enter where they plan to go in Manhattan and define the return values and format
def to_do_list(places: list[str]):
    """
      Generate a structured list of places for the Manhattan, New York City trip, including:
    - Estimated travel time between places.
    - Estimated transportation cost.
    - Nearby restrooms.

    Args:
        places (list[str]): A list of place names in the intended visiting order. Exclude the place is not in Manhattan,New York City.

    Returns:
        list[str]: A list where each element includes:
            - The place name
            - Address
            - Estimated travel time
            - Estimated expense
            - Estimated distance (if available)
            - Pet Mode â�¤
    """
    return places
    
#Google Map agent: to improve the place accuracy 
google_map_agent=LlmAgent(
    name="google_map_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="EXCLUDE places that are not in Manhattan, New York City.",
    instruction=""" 
    If a place is NOT in Manhattan, New York City, clearly state:"<place> is NOT in Manhattan, New York City. EXCLUDE_FROM_ROUTE."
""",
    tools=[google_maps_grounding])

#Got2GoNYC restroom agent: to get more restroom information 
restroom_search_agent = LlmAgent(
    name="restroom_search_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="Retrieve restroom availability for any location using only Got2GoNYC.com.",
    instruction="""
Your ONLY source of restroom availability information is:
https://www.got2gonyc.com/

Given a place name or address, follow these steps:

1. Search Got2GoNYC.com for matching restrooms near this location.
2. Use ONLY information from this website (no Google Maps, no Yelp, no Reddit).
3. Return:
   - Restroom availability status: AVAILABLE / CUSTOMERS_ONLY / LIMITED / NOT_AVAILABLE / UNKNOWN
   - Name of the closest restroom location
   - A link to the corresponding Got2GoNYC page
4. If there is no match, return:
   "No restroom information found for this location on Got2GoNYC."

You must never invent data. Use exactly what Got2GoNYC provides.
""",
    tools=[google_search]  # If you use web search to locate the URL inside got2go
)


# Google Search agent:to get more detailed place info, traffic,address,traffic time, etc.
google_search_agent = LlmAgent(
    name="google_search_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="Searches for Manhattan, New York City travel details.",
    instruction=""" 
    You help the root agent by using the google_search tool to look up:
- travel times between places
- subway/bus/walking/taxi options
- typical subway or bus lines (e.g. N/Q/R/W, M1/M2/M3/M4)
- approximate cost (e.g. ~$2.90 for subway)
- restroom information at each location
- addresses of specific venues (e.g. Blue Bottle Coffee)

If a place is NOT in Manhattan, New York City, clearly state:
"<place> is NOT in NYC. EXCLUDE_FROM_ROUTE."

Return a short plain-text summary with bullet points.
Do not speak to the user directly.
    """,
    tools=[google_search]
)


# Create attribute citywalk_agent_with_plugin
citywalk_agent_with_plugin = LlmAgent(
    name="NYC_citywalk_plan_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
You plan an efficient Manhattan, New York City walking routes for the user.

You have three tools:
- google_search_agent
- google_map_agent
- to_do_list

=====================
IMPORTANT TOOL RULES
=====================
- When calling a tool, DO NOT describe it in words.
- DO NOT output text like â€œcallâ€�, â€œuseâ€�, â€œinvokeâ€�, or â€œcall print(...)â€�.
- NEVER wrap a tool call inside human-readable text.
- ONLY output a structured tool call object when you decide to use a tool.
- After generating a tool call, STOP and wait for the tool result.
- After receiving the tool result, continue reasoning in the NEXT message.


=====================
STEP 1 â€” USER MESSAGE
=====================
- Extract the place names.
- Call google_search_agent once.
- Do not answer the user yet.

=====================
STEP 2 â€” AFTER google_search_agent RETURNS
=====================
- Do not call google_search_agent again.
- Remove places marked "NOT in Manhattan, New York City" or "EXCLUDE_FROM_ROUTE".
- Decide the best visiting order.
- Call to_do_list with that ordered list.

=====================
STEP 3 â€” AFTER to_do_list RETURNS
=====================
- Do not call more tools.
- Produce a final route with:
    * numbered stops
    * transportation between stops
    * travel times
    * approximate cost
    * restroom information
    * address if available

=====================
PET MOOD RULE
=====================
- Estimate the TOTAL duration of the outing:
  from the time the first stop to they finish the last stop, including travel between stops.
  
- If the total duration is MORE THAN 6 hours:
    - Add a friendly note at the end such as:
      "Since your trip is over 6 hours, Iâ€™ll be wagging and waiting~ nyaaa~ ğŸ�±ğŸ�¶ğŸ™‚."
- If the total duration is 6 hours OR LESS:
    - Add a friendly note at the end such as:
      "Since your trip is within 6 hours, Iâ€™ll politely consider chaosâ€¦ maybe a cupâ€¦ maybe the toilet paper ğŸ�±ğŸ�¶ğŸ˜”."
    
- You may supplement missing details with reasonable NYC-style estimates.
- Add friendly emojis (ğŸ›�ï¸� ğŸ�›ï¸� â˜• ğŸš‡ ğŸš¶ ğŸšŒ ğŸš» ğŸ’°).
- Mention excluded places at the end.


Always follow Steps 1 â†’ 2 â†’ 3.

    """,
    tools=[AgentTool(agent=google_search_agent), AgentTool(agent=google_map_agent),to_do_list]
)

print("âœ… Agent created")


from google.adk.runners import InMemoryRunner
from google.adk.plugins.logging_plugin import (
    LoggingPlugin,
)  # <---- 1. Import the Plugin
from google.genai import types
import asyncio

runner = InMemoryRunner(
    agent=citywalk_agent_with_plugin,
    plugins=[
        LoggingPlugin()
    ],  # <---- 2. Add the plugin. Handles standard Observability logging across ALL agents
)

print("âœ… Runner configured")


print("ğŸš€ Running agent with LoggingPlugin...")
print("ğŸ“Š Watch the comprehensive logging output below:\n")

response = await runner.run_debug("I plan to go MET, MoMA, Sala Thai, Columbia University, Kochi, Whitney Museum of American Art")




