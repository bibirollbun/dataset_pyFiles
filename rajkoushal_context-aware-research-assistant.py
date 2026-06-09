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


# Install the Google Agent Development Kit
!pip install -q google-adk

import os
import sys
from kaggle_secrets import UserSecretsClient

# Setup the API Key from Kaggle Secrets
user_secrets = UserSecretsClient()
try:
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… API Key successfully loaded.")
except Exception as e:
    print("â�Œ Error: Could not find 'GOOGLE_API_KEY' in Kaggle Secrets.")


import asyncio
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools import google_search
from google.adk.runners import InMemoryRunner
from google.genai import types

# Define the model we want to use
MODEL_NAME = "gemini-2.0-flash-001"
print("âœ… Imports complete.")


# Run this to list valid model names for your key
from google.genai import Client
import os

client = Client(api_key=os.environ["GOOGLE_API_KEY"])
print("--- Available Models ---")
for model in client.models.list():
    if "flash" in model.name:
        print(f"Name: {model.name}")


def create_agent():
    # 1. Define the Agent
    # We allow it to use the 'google_search' tool to get real data.
    research_agent = Agent(
        name="research_assistant",
        model=Gemini(model=MODEL_NAME), # Requirement 1: Agent powered by LLM
        description="A smart assistant that searches the web and remembers context.",
        instruction=(
            "You are a helpful Research Assistant. "
            "1. When asked a question, use the 'google_search' tool to find the latest info. "
            "2. Summarize the results clearly. "
            "3. If the user asks a follow-up question (like 'tell me more about it'), "
            "use your conversation history to understand what 'it' refers to."
        ),
        tools=[google_search] # Requirement 2: Tools
    )
    return research_agent

print("âœ… Agent function defined.")


async def run_demo():
    agent = create_agent()
    
    # Requirement 3: Sessions & Memory
    runner = InMemoryRunner(agent=agent)
    
    print("--- ğŸ¤– DEMO STARTED ---")
    
    # QUESTION 1
    q1 = "What are the key features of Python 3.13?"
    print(f"\nUser: {q1}")
    
    # FIX: Use run_debug() instead of run()
    response1 = await runner.run_debug(q1)
    
    # Note: run_debug prints the output automatically in most versions, 
    # but returns the final state/response object.
    
    # QUESTION 2 (Memory Test)
    q2 = "When was it released?"
    print(f"\nUser: {q2}")
    
    # FIX: Use run_debug() again
    response2 = await runner.run_debug(q2)
    
    print("\n--- âœ… DEMO FINISHED ---")

# Run the async function
await run_demo()

