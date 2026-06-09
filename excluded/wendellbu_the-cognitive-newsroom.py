# Install Google ADK and search dependencies
!pip install -q google-adk
!pip install -q langchain-community duckduckgo-search
!pip install -Uq ddgs

import os
import asyncio
from typing import Dict, Any, List

# ADK Core Imports
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from google.adk.apps.app import App, EventsCompactionConfig

# Search Tool
from langchain_community.tools import DuckDuckGoSearchRun

print("âœ… ADK Environment & Dependencies Installed.")


# 1. API Key Setup
# In Kaggle, use: from kaggle_secrets import UserSecretsClient
# secrets = UserSecretsClient()
# os.environ = secrets.get_secret("GEMINI_API_KEY")

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


# 2. Define Custom Tool Function
def exit_loop(tool_context: ToolContext):
  """Call this function ONLY when the critique indicates no further changes are needed, signaling the iterative process should end."""
  print(f"  [Tool Call] exit_loop triggered by {tool_context.agent_name}")
  tool_context.actions.escalate = True
  # Return empty dict as tools should typically return JSON-serializable output
  return {}

search_tool = DuckDuckGoSearchRun()
def web_search(query: str) -> str:
    """
    Performs a web search to find recent news, facts, and data.
    Useful for gathering information on current events or technical topics.

    Args:
        query (str): The search query string.

    Returns:
        str: A summary of search results.
    """
    try:
        # Limit content to avoid token overflow in smaller models
        # You might need to adjust the length or use a more sophisticated summarizer
        return search_tool.run(query)[:2000]
    except Exception as e:
        return f"Search failed: {e}"

# Example of how to use the tool (optional, for testing)
print(web_search("What is the capital of France?"))


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


MODEL_NAME = "gemini-flash-latest"

gemini_flash = Gemini(
    model=MODEL_NAME,
    retry_options=retry_config
)

# --- Group 1: The Researchers (Parallel) ---
# Agent A: Focuses on financial/market aspects
market_researcher = Agent(
    name="Market_Researcher",
    model=gemini_flash,
    tools=[web_search],
    output_key="market_news",
    instruction="""
    You are a Financial Researcher. Search for market trends, stock performance, 
    and economic impact related to the user's topic.
    Output a concise summary of FINANCIAL facts only.
    """
)

# Agent B: Focuses on technical/product aspects
tech_researcher = Agent(
    name="Tech_Researcher",
    model=gemini_flash,
    tools=[web_search],
    output_key="tech_news",
    instruction="""
    You are a Tech Researcher. Search for technical specifications, version updates, 
    and engineering breakthroughs related to the user's topic.
    Output a concise summary of TECHNICAL facts only.
    """
)

# --- Group 2: The Analyst (Sequential) ---
# Synthesizes the parallel research
analyst_agent = Agent(
    name="Chief_Analyst",
    model=gemini_flash,
    # No tools needed, it reads from state/context
    output_key="analysis",
    instruction=f"""
    You are the Chief Analyst. You will receive financial research from: {{market_news}} and technical research from: {{tech_news}}.
    Combine these into a strategic SWOT analysis (Strengths, Weaknesses, Opportunities, Threats).
    Do not add new external info, just synthesize the provided research.
    """
)

# --- Group 3: The Editorial Team (Loop) ---
# Agent D: The Writer
writer_agent = Agent(
    name="Writer",
    model=gemini_flash,
    output_key="current_article",
    instruction=f"""
    You are a Senior Journalist. Write a compelling news article based on the Analyst's SWOT report: {{analysis}}.
    The article must have a headline, a lead paragraph, and bullet points.
    IMPORTANT: You must incorporate feedback from the Critic if provided.
    """
)

# Agent E: The Critic (Quality Control)
# This agent decides if the loop continues or ends
critic_in_loop = Agent(
    name="Critic",
    model=gemini_flash,
    output_key="feedback",
    instruction=f"""
    You are the Editor-in-Chief. Review the Writer's article: 
    ```
    {{current_article}}
    ```
    
    CRITERIA:
    1. Does it have a catchy Headline?
    2. Is the SWOT analysis clear?
    3. Is the tone professional?

    IF you identify 1-2 *clear and actionable* ways the article could be improved to better capture the topic or enhance reader engagement (e.g., "Needs a stronger opening sentence", "Clarify the character's goal"):
    Provide these specific suggestions concisely. Output *only* the critique text.

    ELSE IF the article is coherent, addresses the topic adequately for its length, and has no glaring errors or obvious omissions:
    Respond *exactly* with the phrase "APPROVE" and nothing else. It doesn't need to be perfect, just functionally complete for this stage. Avoid suggesting purely subjective stylistic preferences if the core is sound.
    """
)

# STEP 2b: Refiner/Exiter Agent (Inside the Refinement Loop)
refiner_in_loop = Agent(
    name="Refiner",
    model=gemini_flash,
    instruction=f"""You are a Creative Writing Assistant refining an article based on feedback OR exiting the process.
    **Current Article:**
    ```
    {{current_article}}
    ```
    **Critique/Suggestions:**
    {{feedback}}

    **Task:**
    Analyze the 'Critique/Suggestions'.
    IF the critique is *exactly* "APPROVE":
    You MUST call the 'exit_loop' function. Do not output any text.
    ELSE (the critique contains actionable feedback):
    Carefully apply the suggestions to improve the 'Current Article'. Output *only* the refined article text.

    Do not add explanations. Either output the refined article OR call the exit_loop function.
""",
    description="Refines the document based on critique, or calls exit_loop if critique indicates completion.",
    tools=[exit_loop],
    output_key="current_article"
)

print("âœ… All 6 Agents Configured.")


# 1. Parallel Research Block
# These two run at the same time, saving time.
research_team = ParallelAgent(
    name="Research_Desk",
    sub_agents=[market_researcher, tech_researcher],
    description="Fetches financial and technical data simultaneously."
)

# 2. Editorial Loop Block
# The Writer and Critic loop until the Critic approves or we hit max_iterations.
editorial_loop = LoopAgent(
    name="Editorial_Review_Loop",
    sub_agents=[critic_in_loop, refiner_in_loop],
    max_iterations=3, # Safety valve to prevent infinite loops
    description="Iterative writing and critique cycle."
)

# 3. Master Pipeline
# Research -> Analysis -> Editorial Loop
newsroom_pipeline = SequentialAgent(
    name="NewsRoom_Pipeline",
    sub_agents=[research_team, analyst_agent, writer_agent, editorial_loop],
    description="End-to-end news generation pipeline."
)

print("âœ… Pipeline Assembled: Research (Parallel) -> Analyst -> Editorial (Loop)")


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


# Define our application
research_app = App(
    name="NewsRoom",
    root_agent=newsroom_pipeline,
)

APP_NAME = "NewsRoom"  # Application

# Set up Session Management
# InMemorySessionService stores conversations in RAM (temporary)
session_service = InMemorySessionService()

# Step 3: Create the Runner
research_runner = Runner(
    app=research_app, session_service=session_service
)

print("âœ… Stateful NewsRoom Agent initialized!")
print(f"   - Application: {APP_NAME}")
print(f"   - Using: {session_service.__class__.__name__}")


topic = "Cloudera's Competitive Landscape: Market Dynamics, Key Rivals, and Strategic Imperatives for Growth (2025-2030)"
USER_ID = "board_analyst"
session_id = "cloudera-newsroom-session-1"

await run_session(
    research_runner,
    [f"Create a comprehensive market report on: {topic}"],
    session_id,
)

