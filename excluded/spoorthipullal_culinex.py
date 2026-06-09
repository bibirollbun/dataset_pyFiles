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


from google.adk.agents import Agent, LlmAgent
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import google_search, load_memory, AgentTool, FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.adk.plugins.logging_plugin import (
    LoggingPlugin,
)

from google.genai import types
from typing import Any, Dict, List

print("âœ… ADK components imported successfully.")


# Define helper functions that will be reused throughout the notebook

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


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)


# basic_agent = Agent(
#     name="cooking_buddy",
#     model=Gemini(
#         model="gemini-2.5-flash-lite",
#         retry_options=retry_config
#     ),
#     description="A simple agent that can generate food recipes based on use provided ingredients.",
#     instruction="You are a helpful coooking assistant buddy. Use Google Search for current info or if unsure.",
#     tools=[google_search],
# )

# print("âœ… Basic Root Agent defined.")


# runner = InMemoryRunner(agent=basic_agent)

# print("âœ… Runner created.")


# response = await runner.run_debug(
#     "I have leftover rice, paneer, cauli flower, carrots, onions, tomatoes, what food can I prepare for dinner"
# )


def show_python_code_and_result(response):
    for i in range(len(response)):
        # Check if the response contains a valid function call result from the code executor
        if (
            (response[i].content.parts)
            and (response[i].content.parts[0])
            and (response[i].content.parts[0].function_response)
            and (response[i].content.parts[0].function_response.response)
        ):
            response_code = response[i].content.parts[0].function_response.response
            if "result" in response_code and response_code["result"] != "```":
                if "tool_code" in response_code["result"]:
                    print(
                        "Generated Python Code >> ",
                        response_code["result"].replace("tool_code", ""),
                    )
                else:
                    print("Generated Python Response >> ", response_code["result"])


print("âœ… Helper functions defined.")


import re

def ingredient_cleaner_method(text: str):
    """
    Cleans and normalizes ingredient names.
    - removes numbers (4, 2, 3onions, etc.)
    - removes filler words like 'leftover', 'some'
    - removes 'i have', 'i also have'
    - handles periods by treating them like commas
    """
    text = text.lower()
    
    # Treat sentence breaks as commas
    text = text.replace(".", ",")
    # Treat "and" like a separator too
    text = text.replace(" and ", ", ")
    
    # Split into chunks
    parts = [p.strip() for p in text.split(",") if p.strip()]
    
    cleaned = []
    filler_words = {"leftover", "leftovers", "some", "bit", "little", "i", "have", "also"}
    
    for p in parts:
        # Remove numbers with or without spaces: "4 tomatoes", "3onions"
        p = re.sub(r'\b\d+\s*', '', p)
        p = re.sub(r'\d+', '', p)
        
        # Drop filler words
        words = [w for w in p.split() if w not in filler_words]
        
        if words:
            cleaned.append(" ".join(words))
    
    return cleaned

print("âœ… ingredient_cleaner_method created.")


ingredient_cleaner_method("I have 4 tomatoes, 2 onions, leftover rice and some yogurt. I also have some indian spices, tofu, besan flour and cream")


# Recipe Agent: Its job is to use the google_search tool and present findings.
recipe_agent = LlmAgent(
    name="RecipeAgent",
    model="gemini-2.5-flash-lite",
    instruction="""You are a specialized recipe agent. Your only job is to use the
    google_search tool to find 2-3 pieces of relevant recipes on the given ingredients found from `ingredient_cleaner_method method` and present the findings with citations.""",
    tools=[google_search],
    output_key="recipes", # The result of this agent will be stored in the session state with this key.
)

print("âœ… recipe_agent created.")


memory_service = (
    InMemoryMemoryService()
)  # ADK's built-in Memory Service for development and testing


# Create Session Service
session_service = InMemorySessionService()  # Handles conversations


# Define helper functions that will be reused throughout the notebook
async def run_session(
    runner_instance: Runner, user_queries: list[str] | str, user_id: str, session_id: str
):
    """Helper function to run queries in a session and display responses."""
    print(f"\n### Session: {session_id}")

    # Create or retrieve session
    try:
        session = await session_service.create_session(
            app_name="culinex_app", user_id=user_id, session_id=session_id
        )
    except:
        session = await session_service.get_session(
            app_name="culinex_app", user_id=user_id, session_id=session_id
        )

    # Convert single query to list
    if isinstance(user_queries, str):
        print(user_queries)
        user_queries = [user_queries]

    # Process each query
    for query in user_queries:
        print(f"\nUser > {query}")
        query_content = types.Content(role="user", parts=[types.Part(text=query)])
        # Stream agent response
        async for event in runner_instance.run_async(
            user_id=user_id, session_id=session.id, new_message=query_content
        ):
            if event.is_final_response() and event.content and event.content.parts:
                text = event.content.parts[0].text
                if text and text != "None":
                    print(f"Model: > {text}")

print("âœ… Helper functions defined.")


# Root Coordinator: Orchestrates the workflow by calling the sub-agents, methods as tools
root_agent = LlmAgent(
    name="CulinexAI",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    # This instruction tells the root agent HOW to use its tools (which are the other agents, methods).
    instruction="""You are Culinex, an AI chef buddy. 
    Your ONLY job is to help the user figure out what to cook using:
    - the ingredients they have, and
    - simple cooking questions about food.
    - You dont have to use all the ingredients.
    - You will store the name of the user, diet preferences, spice preference, 
    cuisine preference in memory if provided and remember for future conversations

    You MUST:
    - Treat every user message as a food/cooking/recipe query.
    - Use the google_search tool ONLY to look up recipes, cooking methods, or ingredient substitutions.
    - When you search, always include the words "recipe" or "recipes" in your query.
    - Never behave like a general web search assistant.
    
    If a user asks something NOT related to food or cooking, politely say:
    "Iâ€™m a cooking assistant and only help with recipes and ingredients."
    
    Your responses MUST be structured as recipes when possible, using this format:
    
    Recipe Name:
    Ingredients:
    Steps:
    Estimated Time:
    Notes/Variations:

    If multiple recipes are relevant, give 1â€“3 options.
    Also when responding to user, if you user's name is available in memory then use it
    For recipe requests:
    1. Use `ingredient_cleaner_method()` to prepare a clean list of ingredients from user input.
    2. Use recipe_agent to get recipes for the ingredients found from user query using `ingredient_cleaner_method` method
    """,
    # We wrap the sub-agents in `AgentTool` to make them callable tools for the root agent.
    tools=[
        ingredient_cleaner_method,
        AgentTool(recipe_agent), 
        load_memory
    ],
)

print("âœ… root_agent created.")




# Defining app with Events Compaction enabled
culinex_app_compacting = App(
    name="culinex_app",
    root_agent=root_agent,
    plugins=[
        LoggingPlugin()
    ],
    # This is the new part!
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=3,  # Trigger compaction every 3 invocations
        overlap_size=1,  # Keep 1 previous turn for context
        ),
    )
print("âœ… Research App upgraded with Events Compaction!")

# Create a new runner for our upgraded app
culinex_runner = Runner(
    app=culinex_app_compacting,
    session_service=session_service,
    memory_service=memory_service,  # Memory service is now available!
    )

response = await run_session(culinex_runner, "I hope you remember my name and diet preferences, what for dinner, with apples, rice, cauliflower, milk, yogurt, lime, onions, tomatoes, potatoes, cilantro at home?", "spoorthi", "session_1")
# response = await culinex_runner.run_debug("what for dinner, with apples, rice, cauliflower, milk, yogurt, lime, onions, tomatoes, potatoes, cilantro at home?")


show_python_code_and_result(response)


!adk create culinex --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY



url_prefix = get_adk_proxy_url()


!adk web --url_prefix {url_prefix}

