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
from google import genai
from google.genai import types

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    client = genai.Client(api_key=GOOGLE_API_KEY)

    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )



from google.genai import types

from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.tools import AgentTool, FunctionTool, google_search

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search, AgentTool, ToolContext
from google.adk.code_executors import BuiltInCodeExecutor

from google.adk.runners import Runner

from google.adk.memory import InMemoryMemoryService
from google.adk.tools import load_memory, preload_memory

print("âœ… ADK components imported successfully.")


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)
print("âœ… Retry options configured successfully.")


import requests

# Fetch weather data
WEATHER_API_KEY = UserSecretsClient().get_secret("WEATHER_API_KEY")
city = "Bengaluru"
url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q={city}&aqi=yes"

response = requests.get(url)
weather_data = response.json()
print("âœ… weather data extracted.")


# Fetch required weather data
condition = weather_data['current']['condition']['text']
temperature = weather_data['current']['temp_c']
rain = weather_data['current']['precip_mm']
aqi = weather_data['current']['air_quality']['pm2_5']

current_weather = (
    f"Current weather in {city}:\n"
    f"- Condition: {condition}\n"
    f"- Temperature: {temperature}Â°C\n"
    f"- Precipitation: {rain} mm\n"
    f"- Air Quality: {aqi} PM2.5"
)
output_key="current_weather"
print(current_weather)


# Define constants used throughout the notebook
APP_NAME = "RecreationApp"
USER_ID = "demo_user"

# Recreation Agent: Its job is to generate plan based on the instrcution it receives.
recreation_agent = Agent(
    name="RecreationAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    # The instruction is modified to request a bulleted list for a clear output format.


# Suggest plans for today based on the current weather
instruction = f"""
You are a personal assistant to the executive. Based on the current weather suggest when the executive can go out for recreational activities and list top 3 activities executive can do.
Here is the current weather {current_weather}
Suggest what precautions executive needs to take based on air quality and precipitation. Executive prioritise comfort over cost and is allergic to pollution.
Given these facts, suggest outdoor or indoor activities accordingly.""",
    output_key="recreation_plan",





)


from IPython.display import Markdown
Markdown(response.text)


print("âœ… Recreation Agent created.")
print(output_key)



# This agent summarizes the plan.
summarizer_agent = Agent(
    name="SummarizerAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a plan summarizer, You have a recreation plan.
    
    Plan: {recreation_plan}
        
    Your task is to analyze the plan and generate summary. List top 3 activities""",
    output_key="plan_summary",  # It overwrites the story with the new, refined version.
)  

print("âœ… Summarizer Agent created.")
print(output_key)




# Define constants used throughout the notebook
APP_NAME = "RecreationPlanApp"
USER_ID = "demo_user"

# The root agent is a Agent that defines the overall workflow: Initial plan -> summary.
root_agent = SequentialAgent(
    name="RecreationPlanFinal",
    sub_agents=[summarizer_agent],
    
)

print("âœ… Sequential Agent created.")

print("âœ… Plan and Summarizer Agents created.")


memory_service = (
    InMemoryMemoryService()
)  # ADK's built-in Memory Service for development and testing
print("âœ… Memory service defined.")


from google.adk.plugins.logging_plugin import (
    LoggingPlugin,
)  # <---- 1. Import the Plugin
from google.genai import types
import asyncio

# Create Session Service
session_service = InMemorySessionService()  # Handles conversations

# Create runner with BOTH services
runner = Runner(
    agent=root_agent,
    app_name="RecreationPlanApp",
    session_service=session_service,
    memory_service=memory_service,  # Memory service is now available!
    plugins=[
        LoggingPlugin()
    ],  # <---- 2. Add the plugin. Handles standard Observability logging across ALL agents
)

print("âœ… Agent and Runner created with memory support!")
from IPython.display import Markdown
Markdown(response.text)


