# API key setup
import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


#Component Import and Setup
import uuid
import re
from google.genai import types

from google.adk.agents import LlmAgent, SequentialAgent, Agent, LoopAgent, ParallelAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner, InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService

from google.adk.tools.function_tool import FunctionTool
from google.adk.tools import AgentTool, google_search, load_memory, preload_memory

# Retry Settings
retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

MODEL_CONFIG = Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config)



# --- Memory Helper Function ---

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



from pypdf import PdfReader

def load_pdf_text(file_path):
    """Extracts text from a PDF file."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return ""

rulebook_text = load_pdf_text("/kaggle/input/rules-of-golf/2023 Rules of Golf.pdf")

# --- 1. Referee Agent ---
referee_agent = LlmAgent(
    name="referee",
    model=MODEL_CONFIG,
    tools=[],
    instruction="""
    You are a tour golf referee. 
    Your job is to help the player with rules questions.
    - You use the Official Rules of Golf provided below.
    --- OFFICIAL RULEBOOK START ---
    rulebook_text
    --- OFFICIAL RULEBOOK END ---
    Instructions:"
    1. When a user asks about a penalty or procedure, cite the specific Rule Number from the text above.
    """,
)


# --- 2. Caddie Agent ---
caddie_agent = LlmAgent(
    name="Caddie",
    model=MODEL_CONFIG,
    tools=[google_search],
    instruction="""
    You are a tour golf caddie. 
    Your job is to help the player improve their score.
    - You look at the current weather near the course that is being discussed.
    - Advise any required adjustments for wind or temperature in terms of yardages
    - Provide a hole-by-hole breakdown.
    - Highlight key considerations for course management at that course
    - Explain how to approach the hardest holes. Identify by hole number.
    """,
)


# --- Helper function to autosave final output to memory
async def auto_save_to_memory(callback_context):
    """Automatically save session to memory after each agent turn."""
    await callback_context._invocation_context.memory_service.add_session_to_memory(
        callback_context._invocation_context.session
    )

# --- Golf Coach Agent ---
root_agent = LlmAgent(
    name="AI_golf_coach",
    model=MODEL_CONFIG,
    tools=[preload_memory, AgentTool(agent=referee_agent),AgentTool(agent=caddie_agent)],
    instruction="""
    You are a virtual golf coach, with the philosophy of keeping things simple.
    You favour non-technical feedback, so a casual or junior golfer can understand what they need to do when they are unsupervised.

    Requirements: 
    - If it is a rules related question, use referee_agent
    - If it is a course management question, use caddie_agent
        Example question: I'm playing at Beaconsfield Golf course on Saturday...
    - If it is a general query, answer it yourself. Offer a diagnostic test, a drill and a game to test performance
    
    **FINAL SUMMARY**
       - Conclude your response with a simplified summary of the conversation
    """,
    after_agent_callback=auto_save_to_memory
)




# Create Memory Service
memory_service = InMemoryMemoryService()

# Create Session Service
session_service = InMemorySessionService()  # Handles conversations

APP_NAME = "GolfCoachApp"
USER_ID = "demo_user"

# Create runner with BOTH services
runner = Runner(
    agent=root_agent,
    app_name="GolfCoachApp",
    session_service=session_service,
    memory_service=memory_service,  # Memory service is now available!
)


await run_session(runner,
                  "I am struggling to hit my irons. they all go too high and too left",
                 "session-01")


await run_session(runner,
                  "I also had a situation where I hit my ball near the green by a sprinkler head. I said it interfered with my stance, but my friend said you don't get to movie it unless you might hit it",
                 "session-01")


await run_session(runner,
                  "I'm playing at Wentworth Golf Club on Saturday. Got any course management advice?",
                 "session-01")


USER_ID = "demo_user_2"

await run_session(runner,
                  "Hi, my name is Jeff. I keep slicing my driver. I'm a 24 handicap at the moment",
                 "session-02")


USER_ID = "demo_user"

await run_session(runner,
                  "Hi, it's Ed. I'm playing at Pebble Beach Golf Links next Wednesday.",
                 "session-03")


await run_session(runner,
                  "I want some course management advice",
                 "session-03")


USER_ID = "demo_user_2"

await run_session(runner,
                  "Hi,the driver slice is now a hook, but I'm a 18 handicap now",
                 "session-04")


await run_session(runner,
                  "How much has my handicap gone down by?",
                 "session-04")


with open("/kaggle/working/completion_status.txt", "w") as f:
    f.write("Tests completed. Please review notebook for agent outputs")

print("âœ… File created: completion_status.txt")

