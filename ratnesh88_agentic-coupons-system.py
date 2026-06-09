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


!cat /kaggle/input/agents-intensive-capstone-project/Hackathon dataset.txt


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from typing import Any, Dict
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search, AgentTool, ToolContext
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.agents import Agent, LlmAgent
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.models.google_llm import Gemini
from google.adk.sessions import DatabaseSessionService
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools.tool_context import ToolContext
from google.genai import types

print("âœ… ADK components imported successfully.")


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)


users = [
    { "user_id": "USR001", "name": "Amit Sharma", "age": 29, "location": "Delhi" },
    { "user_id": "USR002", "name": "Neha Verma", "age": 34, "location": "Mumbai" },
    { "user_id": "USR003", "name": "Rohit Kumar", "age": 22, "location": "Bangalore" },
    { "user_id": "USR004", "name": "Priya Singh", "age": 27, "location": "Hyderabad" },
    { "user_id": "USR005", "name": "Sandeep Das", "age": 41, "location": "Kolkata" }
  ]
criteria_definitions =[
    { "criteria_id": "CRT001", "description": "User age should be greater than 25" },
    { "criteria_id": "CRT002", "description": "User location must be metro city" },
    { "criteria_id": "CRT003", "description": "User must have completed KYC" },
    { "criteria_id": "CRT004", "description": "User must have minimum 5 purchases" },
    { "criteria_id": "CRT005", "description": "User last login should be within 7 days" }
  ]

coupons = [
    { "coupon_id": "CPN10", "discount": "10%", "title": "Flat 10% Off" },
    { "coupon_id": "CPN20", "discount": "20%", "title": "Mega 20% Deal" },
    { "coupon_id": "CPN50", "discount": "50%", "title": "Half Price Sale" },
    { "coupon_id": "CPN100", "discount": "â‚¹100", "title": "â‚¹100 Cashback" }
]





def get_users() -> dict:
    """
    Returns a list of user records.

    This function retrieves a predefined list of user dictionaries,
    where each dictionary contains basic information such as user ID,
    name, age, and location.

    Returns:
        list[dict]: A list of user information objects. Each dictionary contains:
            - user_id (str): Unique identifier for the user.
            - name (str): Full name of the user.
            - age (int): Age of the user.
            - location (str): City where the user resides.

    Example:
        [
            {"user_id": "USR001", "name": "Amit Sharma", "age": 29, "location": "Delhi"},
            {"user_id": "USR002", "name": "Neha Verma", "age": 34, "location": "Mumbai"},
            ...
        ]
    """
    users = [
        {"user_id": "USR001", "name": "Amit Sharma", "age": 29, "location": "Delhi"},
        {"user_id": "USR002", "name": "Neha Verma", "age": 34, "location": "Mumbai"},
        {"user_id": "USR003", "name": "Rohit Kumar", "age": 22, "location": "Bangalore"},
        {"user_id": "USR004", "name": "Priya Singh", "age": 27, "location": "Hyderabad"},
        {"user_id": "USR005", "name": "Sandeep Das", "age": 41, "location": "Kolkata"}
    ]
    return users



# user_indentify_agents

# criteria_mactching_agent

# validity_check_agent

# coupon_information_agent


# Research Agent: Its job is to use the google_search tool and present findings.
user_indentify_agent = Agent(
    name="UserIdentifyAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a specialized user identifier agent. Your only job is to use the users data and with help of keys and value understand and describe user information""",
    tools=[get_users],
    output_key="user_findings",  # The result of this agent will be stored in the session state with this key.
)

print("âœ… user_indentify_agent created.")


# Summarizer Agent: Its job is to summarize the text it receives.
criteria_mactching_agent = Agent(
    name="CriteriaMatchAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    # The instruction is modified to request a bulleted list for a clear output format.
    instruction="""Understand information the provided in user findings: {user_findings} 
    and check criteria in [
    { "criteria_id": "CRT001", "description": "User age should be greater than 25" },
    { "criteria_id": "CRT002", "description": "User location must be metro city" },
    { "criteria_id": "CRT003", "description": "User must have completed KYC" },
    { "criteria_id": "CRT004", "description": "User must have minimum 5 purchases" },
    { "criteria_id": "CRT005", "description": "User last login should be within 7 days" }
  ] to provide which user is eligible for coupons based which criteria description""",
    output_key="criteria_match",
)

print("âœ… criteria_mactching_agent created.")


# Root Coordinator: Orchestrates the workflow by calling the sub-agents as tools.
coupon_agent = Agent(
    name="CouponCoordinator",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    # This instruction tells the root agent HOW to use its tools (which are the other agents).
    instruction="""You are a coupon coordinator. Your goal is to provide the coupon to eligible user by orchestrating a workflow.
1. First, you MUST call the `UserIdentifyAgent` tool to find relevant information of the by the user.
2. Next, after receiving the user information findings, you MUST call the `CriteriaMatchAgent` tool to find which is eligible to coupons based on which criteria.
3. Finally, present the final information clearly to the user as your response.""",
    # We wrap the sub-agents in `AgentTool` to make them callable tools for the root agent.
    tools=[AgentTool(user_indentify_agent), AgentTool(criteria_mactching_agent)],
)

print("âœ… coupon_agent created.")


runner = InMemoryRunner(agent=coupon_agent)
response = await runner.run_debug(
    "Is Priya Singh is eligible for any coupon please information?"
)


# Define helper functions that will be reused throughout the notebook
async def run_session(
    runner_instance: Runner,
    user_queries: list[str] | str = None,
    session_name: str = "default",
):
    print(f"\n ### Session: {session_name}")

    # Get app name from the Runner
    app_name = runner_instance.app_name

    # Attempt to create a new session or retrieve an existing one
    try:
        session = await session_service.create_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
    except:
        session = await session_service.get_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )

    # Process queries if provided
    if user_queries:
        # Convert single query to list for uniform processing
        if type(user_queries) == str:
            user_queries = [user_queries]

        # Process each query in the list sequentially
        for query in user_queries:
            print(f"\nUser > {query}")

            # Convert the query string to the ADK Content format
            query = types.Content(role="user", parts=[types.Part(text=query)])

            # Stream the agent's response asynchronously
            async for event in runner_instance.run_async(
                user_id=USER_ID, session_id=session.id, new_message=query
            ):
                # Check if the event contains valid content
                if event.content and event.content.parts:
                    # Filter out empty or "None" responses before printing
                    if (
                        event.content.parts[0].text != "None"
                        and event.content.parts[0].text
                    ):
                        print(f"{MODEL_NAME} > ", event.content.parts[0].text)
    else:
        print("No queries!")


print("âœ… Helper functions defined.")


APP_NAME = "default"  # Application
USER_ID = "default"  # User
SESSION = "default"  # Session

MODEL_NAME = "gemini-2.5-flash-lite"


# Step 2: Set up Session Management
# InMemorySessionService stores conversations in RAM (temporary)
session_service = InMemorySessionService()

# Step 3: Create the Runner
runner = Runner(agent=coupon_agent, app_name=APP_NAME, session_service=session_service)

print("âœ… Stateful agent initialized!")
print(f"   - Application: {APP_NAME}")
print(f"   - User: {USER_ID}")
print(f"   - Using: {session_service.__class__.__name__}")



# Run a conversation with two queries in the same session
# Notice: Both queries are part of the SAME session, so context is maintained
await run_session(
    runner,
    [
        "Hi, I am Neha Verma! Please share for how many coupons I am eligible?",
        "Hello! What is my name?",  # This time, the agent should remember!
    ],
    "stateful-agentic-session",
)


# Configuration
APP_NAME = "default"
USER_ID = "default"
MODEL_NAME = "gemini-2.5-flash-lite"

# Create an agent with session state tools
root_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="text_chat_bot",
    description="""A text chatbot.
    Tools for managing user context:
    * To record username and country when provided use `save_userinfo` tool. 
    * To fetch username and country when required use `retrieve_userinfo` tool.
    """,
    tools=[save_userinfo, retrieve_userinfo],  # Provide the tools to the agent
)

# Set up session service and runner
session_service = InMemorySessionService()
runner = Runner(agent=root_agent, session_service=session_service, app_name="default")

print("âœ… Agent with session state tools initialized!")










