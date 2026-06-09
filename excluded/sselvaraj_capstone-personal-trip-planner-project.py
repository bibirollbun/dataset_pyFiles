from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.sessions import DatabaseSessionService
from google.adk.sessions import InMemorySessionService
from google.adk.runners import InMemoryRunner
from google.adk.runners import Runner
from google.adk.tools import google_search
from google.genai import types

print("âœ… ADK components imported successfully.")


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


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
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



flight_agent = Agent(
    name="flight_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="A simple flight agent that searches for flights between cities.",
    instruction=""" Use Google Search for finding flights between two cities. You are forbidden from providing any 
    explanations, or conversational responses. 

    Your task is to find flights and list the options with Airline, Price, Duration.

     **RULES:**
    1.  Your output MUST be ONLY a List of flight options.
    2.  Do NOT write any text before or after the flight options.
   
    Failure to follow these rules will result in an error.
    """,
    tools=[google_search],
    output_key="flight_details"
)

print("âœ… flight Agent defined.")


hotel_agent = Agent(
    name="hotel_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="A simple hotel agent that searches for hotels for given city/cities.",
    instruction="""Use Google Search for finding hotel stays for city/cities. You are forbidden from providing any 
    explanations, or conversational responses.

    Your task is to find hotels and list the options with Name and Price.

     **RULES:**
    1.  Your output MUST be ONLY a List of hotel options.
    2.  Do NOT write any text before or after the hotel options.
   
    Failure to follow these rules will result in an error.
    
    """,
    tools=[google_search],
    output_key="hotel_details"
)

print("âœ… hotel Agent defined.")


trip_detail_agent = Agent(
    name="TripDetailAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    # It uses placeholders to inject the outputs from the parallel agents, which are now in the session state.
    instruction="""You are a specialized agent and you are forbidden from providing any explanations, or conversational responses. 
    Use the below flight details and hotel details.

    **Flight Details:**
    {flight_details}
    
    **Hotel Options:**
    {hotel_details}

    **RULES:**
    1.  Your output MUST ONLY display the Trip plan, flight details and hotel details.
    2.  Do NOT write any text before or after.
   
    Failure to follow these rules will result in an error.
    """,
    output_key="trip_detail_summary",  # This will be the final output of the entire system.
)

print("âœ… trip_detail_agent created.")



APP_NAME = "default"  # Application
USER_ID = "default"  # User
SESSION = "default"  # Session

MODEL_NAME = "gemini-2.5-flash-lite"

# The ParallelAgent runs all its sub-agents simultaneously.
parallel_trip_planner_team = ParallelAgent(
    name="TripPlannerTeam",
    sub_agents=[flight_agent, hotel_agent],
)

# This SequentialAgent defines the high-level workflow: run the parallel team first, then run the aggregator.
root_agent = SequentialAgent(
    name="PersonalTripPlanner",
    sub_agents=[parallel_trip_planner_team, trip_detail_agent],
)

# SQLite database will be created automatically
db_url = "sqlite:///tripplan.db"  # Local SQLite file
session_service = DatabaseSessionService(db_url=db_url)

print("âœ… Parallel and Sequential Agents created.")


runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)

print("âœ… Runner created.")


await run_session(
    runner,
    ["Hi, I am Stark! can you please outline a detailed trip plan to Rome,Italy with a start date 14th Jan 2026 and return date 19th Jan 2026 from jfk, newyork? I also want to see Venice, Italy."],
    "italy-02",
)



await run_session(
    runner,
    ["Hello! what is the trip plan for Stark?"],
    "italy-02",
)



import sqlite3

def check_data_in_db():
    with sqlite3.connect("tripplan.db") as connection:
        cursor = connection.cursor()
        result = cursor.execute(
            "select app_name, session_id, author, content from events"
        )
        print([_[0] for _ in result.description])
        for each in result.fetchall():
            print(each)


check_data_in_db()


# Clean up any existing database to start fresh (if Notebook is restarted)
import os

if os.path.exists("tripplan.db"):
    os.remove("tripplan.db")
print("âœ… Cleaned up old database files")

