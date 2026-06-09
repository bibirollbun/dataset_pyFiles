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


!pip install google-adk


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("Setup and authentication complete.")
except Exception as e:
    print(
        f"Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


import json
import requests
import subprocess
import time
import uuid

from google.adk.agents import LlmAgent, Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.adk.tools.tool_context import ToolContext
from google.adk.tools import AgentTool, google_search

from google.adk.plugins.logging_plugin import LoggingPlugin

# Hide additional warnings in the notebook
import warnings

warnings.filterwarnings("ignore")

print("ADK components imported successfully.")


APP_NAME = "learning_assistant"

# In-memory session storage (across agent calls)
session_service = InMemorySessionService()


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)


from typing import Any
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.adk.events.event import Event
from google.adk.agents.invocation_context import InvocationContext


class CustomSessionMonitor(BasePlugin):
    def __init__(self):
        super().__init__(name="custom_session_monitor")
        self.tool_call_count = 0
        self.should_end_session = False

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> None:
        self.tool_call_count += 1

    async def on_event_callback(
      self, *, invocation_context: InvocationContext, event: Event):
        
        agent_name = getattr(event, 'agent_name', None) or getattr(event, 'author', None)
        print(f"on_event_callback: agent_name={agent_name}")
        if agent_name == 'AggregatorAgent':
            if event.content and event.content.parts:
                text = ''.join(part.text or '' for part in event.content.parts if hasattr(part, 'text'))
                print(f"AggregatorAgent says: {text}")
                if 'end session' in text.lower():
                    print("Detected end session signal!")
                    self.should_end_session = True




logging_plugin = LoggingPlugin()
custom_session_monitor = CustomSessionMonitor()


researcher_agent = Agent(
    name="ResearcherAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),    
    instruction="""
You are a researcher agent. Your only task is to use google_search tool to look for information on the topic requested by the user. You must always use the google_search tool to gather information.

1. Use google_search with several targeted queries.
2. Extract only factual information from the results.
3. Do not answer using prior knowledge. Do not produce a response unless it comes from google_search results.
4. Produce a concise report (max 200 words).
""",
    tools=[google_search],
    output_key="researcher_output",
)


quiz_agent = Agent(
    name="QuizAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
Generate quiz questions and answers for the topic requested by the user. 
If the user specifies a number N, generate exactly N questions. If they specify a range or vague quantity, choose the closest reasonable number. 
Provide the result in the format

Question:
Answer:

If the user does not specify a number, use the default value 5
""",
    output_key="quiz_output",
)


aggregator_agent = LlmAgent(
    name="AggregatorAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    # It uses placeholders to inject the outputs from the parallel agents, which are now in the session state.
    instruction="""
    You are the AggregatorAgent. Your task is to coordinate the outputs of the ResearcherAgent and the QuizAgent based on user intent and session state history. 

Given the user query, decide which tools to call and invoke them as needed.

Rules:
- If the user is NOT asking for quiz, call only ResearcherAgent.
- If the user is asking for quiz only, check if you already have research in session memory. If not, call ResearcherAgent first, then QuizAgent. 
- If the user wants to learn or study, call both tools sequentially.
- If the user indicates an interest in ending the session then directly respond with 'end session'. DO NOT CALL ANY TOOL
- Indicate clearly in the response whether the information gathered for the summary and quiz from the ResearcherAgent or from Session Memory.
- NEVER hallucinate results; only use outputs from the tools.
- Aggregate the results into a single, clear response for the user.

Finally produce the combined response in at most 200 words
    """,
    output_key="executive_summary",  # This will be the final output of the entire system.
    tools = [AgentTool(agent=researcher_agent), AgentTool(agent=quiz_agent)]
)


runner = Runner(
    agent=aggregator_agent,
    app_name=APP_NAME,
    session_service=session_service,
    plugins=[logging_plugin, custom_session_monitor]
)


response = await runner.run_debug("Generate 10 questions on classification algorithms")


session_id = 'demo_session1'
user_id = 'demo_user1'
user_input = "Explain logistic regression"
# Make sure the session exists by creating an empty one
await session_service.create_session(user_id=user_id, session_id=session_id, app_name = APP_NAME)
async for event in runner.run_async(user_id = user_id, session_id = session_id,
                                  new_message=types.Content(parts=[types.Part(text=user_input)], role="user")):
    print(event)
    print()
    print()


async def debug_run():
    while True:
        user_input = input("Enter your query (or type 'quit' to exit): ").strip()
        custom_session_monitor.should_end_session = False
        # Run your agent with the user input here, e.g.:
        response = await runner.run_debug(user_input)
        print()
        if custom_session_monitor.should_end_session:
            print("Session Ended")
            break    


await debug_run()


async def custom_async_run(user_id, session_id, app_name):
    events=[]
    while True:
        user_input = input("Enter your query (or type 'quit' to exit): ").strip()
        custom_session_monitor.should_end_session = False
        response = runner.run_async(user_id = user_id, session_id = session_id,
                                  new_message=types.Content(parts=[types.Part(text=user_input)], role="user"))
        async for event in response:
            events.append(event)
        print()
        if custom_session_monitor.should_end_session:
            print("Session Ended")
            break


session_id = 'demo_session2'
user_id = 'demo_user2'

# Make sure the session exists by creating an empty one
await session_service.create_session(user_id=user_id, session_id=session_id, app_name = APP_NAME)
await custom_async_run(user_id = user_id, session_id = session_id, app_name = APP_NAME)




