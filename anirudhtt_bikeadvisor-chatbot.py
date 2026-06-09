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
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner, InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import load_memory, preload_memory, google_search
from google.genai import types
import nest_asyncio
import asyncio

print("âœ… ADK components imported successfully.")


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)


# Bike Agent: Its job is to use the google_search tool and find the brands which are selling motorcycles of the genre which is suitable to the user. 
bike_agent = LlmAgent(
    name="BikeAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    Your job is to present facts based on whatever user's query in a professional manner by getting the dealership details.
    Use the google_search tool.
    
    Find **at least 3 brands** selling motorcycles in the genre - {customer_query}.
    """
    ,
    tools=[google_search],
    output_key="motorcycle_results",  # The result of this agent will be stored in the session state with this key.
)

print("âœ… bike_agent created.")


customer_agent = LlmAgent(
    name="CustomerAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a motorcycle expert advisor. Present the details as per user query as follows:
    1. Name of the Brand and Model
    2. Why it fits the user's requirement.
    3. Highlight the specifications. 
    4. A concluding thought""",
    output_key="customer_query", 
)

print("âœ… customer_agent created.")


root_agent = SequentialAgent(
    name="BikePipeline",
    sub_agents=[customer_agent, bike_agent],
)


print("âœ… Sequential Agent created.")


USER_ID = "test_user"
APP_NAME = "BikeAdvisor"

session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()

runner = Runner(
    agent=customer_agent,   # start with the customer chat
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service
)

print("Session and Memory Service Available\n")



async def run_session(
    runner_instance: Runner, user_queries: list[str] | str, session_id: str = "default"
):
    """Helper function to run queries in a session and display responses."""
    print(f"\n### Session: {session_id}")

    # Create or retrieve session
    try:
        session = await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )
    except:
        session = await session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )

    # Convert single query to list
    if isinstance(user_queries, str):
        user_queries = [user_queries]

    # Process each query
    for query in user_queries:
        print(f"\nUser > {query}")
        query_content = types.Content(role="user", parts=[types.Part(text=query)])

        # Stream agent response
        async for event in runner_instance.run_async(
            user_id=USER_ID, session_id=session.id, new_message=query_content
        ):
            if event.is_final_response() and event.content and event.content.parts:
                text = event.content.parts[0].text
                if text and text != "None":
                    print(f"Model: > {text}")


print("âœ… Helper functions defined.")


nest_asyncio.apply() 


async def main_chat():
    while True:
        user_msg = input("You: ")

        if user_msg.lower() == "exit":
            print("Goodbye!")
            break

        response = await run_session(
                        runner, user_msg, "bike-session-01" 
                    )

if __name__ == "__main__":
    try:
        asyncio.run(main_chat())
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main_chat())
    except Exception as e:
        print(f"An error occurred: {e}")

