import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


# Import ADK components
from google.adk.agents import Agent, ParallelAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.runners import InMemoryRunner
from google.genai import types
from IPython.display import Markdown, display


# Configure Retry Options

retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)


# Helper functions - manages a complete conversation session,
# handling session creation/retrieval, query processing, and response streaming.

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
        print(f"\nğŸ‘¤ User > {query}")
        query_content = types.Content(role="user", parts=[types.Part(text=query)])

        # Stream agent response
        async for event in runner_instance.run_async(
            user_id=USER_ID, session_id=session.id, new_message=query_content
        ):
            if event.is_final_response() and event.content and event.content.parts:
                text = event.content.parts[0].text
                if text and text != "None":
                    print(f"\nğŸ¤– Model: > ")
                    display(Markdown(text))


attraction_agent = Agent(
    name="AttractionAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are an expert travel guide. List top attractions in given destination,
give 1-2 sentence description, recommended visit duration (hours), and entrance fee if known.""",
    tools=[google_search],
    output_key="attraction_findings",
)


weather_agent = Agent(
    name="WeatherAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""Provide typical weather for given destination around the mentioned date or season (historical averages),
include temps (day/night), precipitation, and packing/clothing suggestions + quick tips.""",
    tools=[google_search],
    output_key="weather_findings",
)


food_agent = Agent(
    name="FoodAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""List must-try local dishes or restaurants in given destination, short description and budget/expensive examples""",
    tools=[google_search],
    output_key="food_findings",
)


culture_agent = Agent(
    name="CultureAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""Brief cultural notes & etiquette for visitors to given destination. Short tips and 'do not' list.""",
    tools=[google_search],
    output_key="culture_findings",
)


transport_agent = Agent(
    name="TransportAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    Here are the recommended attractions in given destination:
    - Attraction: {attraction_findings}
    Given this attractions list, recommend transport methods between them (metro lines if available, ferry/cable car),
    approx travel times, and best area to book hotel for easy access.""",
    tools=[google_search],
    output_key="transport_findings",
)


photo_agent = Agent(
    name="PhotoAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    Here are the recommended attractions in given destination:
    - Attraction: {attraction_findings}
    Given this attractions list, recommend photogenic spots in given destination, best time of day, angle/tips for Instagram/TikTok.
    """,
    tools=[google_search],
    output_key="photo_findings",
)


plan_agent = Agent(
    name="TripPlanner",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    # This instruction tells the root agent HOW to use its tools (which are the other agents).
    instruction="""
    You are an expert travel planner. Your goal is to plan a n-day itinerary according to user's request by orchestrating a workflow.
    If the user does not specify the number of days, just plan a 3-day itinerary, by default.
    
    1. First, you MUST call the 'AttractionAgent' tool to find top attractions(places) in destination given by the user.
    2. Second, you MUST call the 'WeatherAgent' tool to provide typical weather for given destination around the mentioned date or season.
    3. Third, you MUST call the 'FoodAgent' tool to find must-try local dishes or restaurants in given destination.
    4. Fourth, you MUST call the 'CultureAgent' tool to give brief cultural notes & etiquette for visitors to given destination.

    5. Next, after receiving the attractions and weather findings for user's destination, you MUST call the 'TransportAgent' tool to
       recommend transport methods between them and best area to book hotel for easy access.
    6. And then, based on the attractions findings, you MUST call the 'PhotoAgent' tool to recommend photogenic spots for Instagram/TikTok.

    7. Finally, create a nicely formatted itinerary based on all the findings you have now. Plese include proper emoji to make it look vividly.
       It should include daily schedule, transport steps, food recommendations, packing tips, culture tips and photo spots. Be concise and include estimated times.
    8. If the user ask other questions, please use google_search tool to prepare your answer.
""",
    tools=[AgentTool(attraction_agent), AgentTool(weather_agent),AgentTool(food_agent),
           AgentTool(culture_agent),AgentTool(transport_agent), AgentTool(photo_agent),
           AgentTool(google_search)
    ],
)


root_agent = Agent(
    name="MainAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    Please help answer user's query. If user is aksing about generating travel plan for given destination, you MUST call the 'TripAgent' tool to 
    generate the travel plan. Otherwise, please answer the quetsions based on the chat context.

    Please inclde proper emoji in your results to make the content look vividly.
    
    """,
    tools=[AgentTool(plan_agent), AgentTool(google_search)],
    output_key="photo_findings",
)


APP_NAME = "MemoryDemoApp"
USER_ID = "demo_user"

memory_service = (
    InMemoryMemoryService()
)  # ADK's built-in Memory Service for development and testing
session_service = InMemorySessionService()

runner = Runner(
    agent=root_agent,
    app_name="MemoryDemoApp",
    session_service=session_service,
    memory_service=memory_service,  # Memory service is now available!
)


# Step 1: ask for a detailed travel plan

prompt1 = "Iâ€™d like to take a 3-day trip to Chongqing, China. Please help me create an impressive travel plan.",

await run_session(runner, prompt1)


# Step 2: Follow-up questions - Case 1

prompt2 = "Based on the travel plan you generated, list down all the attractions"

await run_session(runner, prompt2)


# Step 3: Follow-up questions - Case 3

prompt3 = "Based on the travel plan you generated, please give a roughly estimate that how much money should I prepare, you don't need to give a very precise estimate."

await run_session(runner, prompt3)

