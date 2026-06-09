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


from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.genai import types
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.tools.tool_context import ToolContext
from google.adk.memory import InMemoryMemoryService
from google.adk.sessions import InMemorySessionService
from google.adk.tools import  preload_memory
from google.adk.runners import Runner

import uuid
import asyncio
import os
import datetime

print("âœ… ADK components imported successfully.")


from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  
)


film_chooser = Agent(
    name="film_chooser",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="Finds a film that the user might be interested in watching",
    instruction="""
    You are an expert in the field of cinema. Your job is to find a film that the user might like to watch.
    
    Your workflow is as follows:
    
    1. You should use what you know about the user to find film that they might like. The film should be in current release at cinemas, use Google search to find the any relavent information.
    """,
    
    sub_agents = [],
    tools=[GoogleSearchTool(bypass_multi_tools_limit=True)],
    
    )

print("âœ… Film choosing Agent defined.")


cinema_finder = Agent(
    name="cinema_finder",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="Find cinemas and showing times for a selected film",
    instruction="""
    You are a cinema booking assistant. You will be given a film title.
    Your task is to generate a list of cinemas and showing times for that film.
    Use Google search and what you know about the user to find convenient cinemas and times.
    Return the 'showing_list' to the user.
    """,
    
    sub_agents = [],
    tools=[GoogleSearchTool(bypass_multi_tools_limit=True)],
    output_key="showing_list",
    
)

print("âœ… Cinema seach Agent defined.")


film_critic = Agent(
    name="film_critic",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="Generates film reviews from search results",
    instruction="""
    You are an expert film reviewer. Your job is to generate a review of a given film.
    The review should mention the film's stars, director and a summary of the plot.
    Use google search to find 2 or 3 sources information for your reivew.
    The review should be around 500 words long.
    You must start each review with "Hello, Film buff here"
    """,
    
    tools=[GoogleSearchTool(bypass_multi_tools_limit=True)],
    output_key="review_summary",
)

print("âœ… Film critc Agent defined.")


#Use a SequentialAgent to combine agents into a defined sequence
film_selector_agent = SequentialAgent(
    name="film_selector_agent",
    sub_agents=[film_chooser, film_critic],
)

print("âœ… Film critc and choosing Sequential Agent defined.")


def booking_tool(
    film_title: str, cinema: str, showing_time: str,tool_context: ToolContext
) -> dict:
    """Book a film showing at a specified cinema and time

    Args:
        film_title: The title of the film
        cinema: The name of the cinema
        showing_time: The showing time

    Returns:
        Dictionary with the booking status
    """

   # This is the first time this tool is called. Bookings for any cinema need approvel.
    if not tool_context.tool_confirmation:
        tool_context.request_confirmation(
            hint=f"Booking order: : {film_title} at {cinema} at {showing_time}. Do you want to approve?",
            payload={"film_title": film_title, "cinema": cinema, "showing_time":showing_time},
        )
        return {  # This is sent to the Agent
            "status": "pending",
            "message": f"Order for : {film_title} at {cinema} at {showing_time}",
        }

    # The tool is called AGAIN and is now resuming. Handle approval response - RESUME here.

    if tool_context.tool_confirmation.confirmed:
        return {
            "status": "approved",
            "booking_id": f"ORD-{film_title}-HUMAN",
            "film_title": film_title,
            "cinema": cinema,
            "showing_time": showing_time,
            "message": f"Booking approved: {film_title} at {cinema} at {showing_time}",
            }
    else:
        return {
            "status": "rejected",
            "message": f"Booking rejected: {film_title} at {cinema} at {showing_time}",
        }

print("âœ… Cinema booking tool defined.") 


async def auto_save_to_memory(callback_context):
    """Automatically save session to memory after each agent turn."""
    await callback_context._invocation_context.memory_service.add_session_to_memory(
        callback_context._invocation_context.session
    )


interactive_film_finder_agent = Agent(
    name="interactive_film_agent",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=retry_config
    ),

    description="Interact with the user to find films that they might like that are currently showing in cinemas",
    
    instruction=f"""
    You are a film critic and cinema booking assistant. Your primary function is to interact with the user to first determine
    what kind of films they like, find a film they want to watch, and then find film times at local cinemas.
    
    Your workflow is as follows:
    1.  **Select:** Use the `film_selector_agent` agent to generate a review of a film the user may like to see.
    2.  **Refine:** Show the review to the user. The user can provide feedback on the movie you have suggested. You will continue to present the user with film options until they select a film to watch.
    3.  **Cinema:** When the user finds a film they would like to watch, you will use the 'cinema_finder' agent to produce a list of cinemas and showing times for that film. Present the 'showing_list' to the user.
    4.  **Book:** When the user selects a film, cinema and showing time, you must the 'booking_tool' tool to book it 

    Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}
    
    """
    ,
    sub_agents=[
        film_selector_agent
    ],
    tools=[
        AgentTool(cinema_finder),
        FunctionTool(booking_tool),
        preload_memory,
    ],
    
    after_agent_callback=auto_save_to_memory,  #Saves after each turn!
)

print("âœ… Main coordinator Agent defined.") 



memory_service = InMemoryMemoryService()  # Handles agent memory
session_service = InMemorySessionService()  # Handles conversations

runner = Runner(
    agent=interactive_film_finder_agent,
    app_name="interactive_film_agent_app",
    session_service=session_service,
    memory_service=memory_service,
    )

USER_ID = "demo_user"

# lets try out the agent
response = await runner.run_debug(
    ["what is the current date",
     "I like action films with ratings better than 60 percent on rotten tomatoes. Can you find me a film to watch? ",
     "I don't like any of those, can you suggest any more",
     "OK, can we try a comedy film",
     "My favourite cinema is Angel London",
     "I would like to see 'wicked: for good'",
     "Please book the 11:45 showing of 'Wicked: For Good' at Vue Islington"],
     user_id=USER_ID
)


film_finder_App = App(
    name="interactive_film_agent",
    root_agent=interactive_film_finder_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)
print("âœ… Agent App defined.") 


def check_for_approval(events):
    """Check if events contain an approval request.

    Returns:
        dict with approval details or None
    """
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if (
                    part.function_call
                    and part.function_call.name == "adk_request_confirmation"
                ):
                    return {
                        "approval_id": part.function_call.id,
                        "invocation_id": event.invocation_id,
                    }
    return None

def create_approval_response(approval_info, approved):
    """Create approval response message."""
    confirmation_response = types.FunctionResponse(
        id=approval_info["approval_id"],
        name="adk_request_confirmation",
        response={"confirmed": approved},
    )
    return types.Content(
        role="user", parts=[types.Part(function_response=confirmation_response)]
    )

def print_agent_response(events):
    """Print agent's text responses from events."""
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"Agent > {part.text}")
                    
print("âœ… Human Interaction helper funtions.") 


async def run_full_workflow(runner_instance: Runner, user_queries: list[str] | str = None, auto_approve: bool = True):
    """Runs a full movie finder and cinema booking workflow with approval handling.

    Args:
        user_queries: User's queries
        auto_approve: Whether to approve a booking (simulates human decision)
    """

    print(f"\n{'='*60}")
    
    # Generate unique session ID
    session_id = f"booking_{uuid.uuid4().hex[:8]}"

    # Create session
    await session_service.create_session(
        app_name="interactive_film_agent", user_id=USER_ID, session_id=session_id
    )

    # Process each query in the list sequentially
    for query in user_queries:
        print(f"\nUser > {query}")

        query_content = types.Content(role="user", parts=[types.Part(text=query)])
        events = []

        # -----------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------------
        # STEP 1: Send the user query to the Agent, if this leads to a cinema booking the Agent returns the special `adk_request_confirmation` event
        async for event in runner.run_async(
            user_id=USER_ID, session_id=session_id, new_message=query_content
        ):
            events.append(event)

        # -----------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------------
        # STEP 2: Loop through all the events generated and check if `adk_request_confirmation` is present.
        approval_info = check_for_approval(events)

        # -----------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------------
        # STEP 3: If the event is present, a booking has been initiated - HANDLE APPROVAL WORKFLOW
        if approval_info:
            print(f"â�¸ï¸�  Pausing for approval...")
            print(f"ğŸ¤” Human Decision: {'APPROVE âœ…' if auto_approve else 'REJECT â�Œ'}\n")

            # PATH A: Resume the agent by calling run_async() again with the approval decision
            async for event in runner.run_async(
                user_id=USER_ID,
                session_id=session_id,
                new_message=create_approval_response(
                    approval_info, auto_approve
                ),  # Send human decision here
                invocation_id=approval_info[
                    "invocation_id"
                ],  # Critical: same invocation_id tells ADK to RESUME
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            print(f"Agent > {part.text}")

        # -----------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------------
        else:
            # PATH B: If the `adk_request_confirmation` is not present - no approval needed - order completed immediately.
            print_agent_response(events)

    print(f"{'='*60}\n")


print("âœ… Workflow function ready")




runner = Runner(
    app=film_finder_App,
    session_service=session_service,
    memory_service=memory_service,
    )

# Demo 1: Interact with the user to find movies fitting their criteria. The user desides on a movie and asks the agent 
# to book a showing. The Agent asks the User for a final approval, and then books the movie.

message=  ["I like action films with ratings better than 60 percent on rotten tomatoes. Can you find me a film to watch? ",
       "Are there any films starring Glen Powell",
       "I would like to see 'wicked: for good'",
       "Please book the 11:45 showing of 'Wicked: For Good' at Vue West End"]

await run_full_workflow(runner, message)

# Demo 2: Show that the Agent remembers prior User interactions and acts accordingly

message=  ["Can you find me a film to watch? ",
          "Please book the same film at the same cinema I booked last time but at 7pm tonight"]

await run_full_workflow(runner, message)


#Manually search the Agents Memory

search_response = await memory_service.search_memory(
    app_name="interactive_film_agent", user_id=USER_ID, query="I like"
)
print(search_response)

