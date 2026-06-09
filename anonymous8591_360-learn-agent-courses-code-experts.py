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
import google.generativeai as genai
from kaggle_secrets import UserSecretsClient

# Get the key from Kaggle Secrets
try:
    user_secrets = UserSecretsClient()
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
    os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
    print("âœ… API Key configured successfully!")
except:
    print("âš ï¸� Error: Could not find or configure GOOGLE_API_KEY.")
    print("Please ensure you have added it to Kaggle Secrets.")


from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.models import Gemini

# 1. Define the Model and Tool
#    --- THIS IS THE FIX (from your Day 3 code) ---
gemini_model_name = 'gemini-2.5-flash-lite'
google_search_tool = google_search

# 2. Define the "Worker" Agents
course_agent = LlmAgent(
    model=Gemini(model=gemini_model_name),
    tools=[google_search_tool],
    name="Course_Finder",
    instruction="""You are an expert academic advisor. Your single job is to find the
    one or two best, top-rated, and free online courses or video playlists
    for the user's topic. Return only the course name and its URL."""
)

code_agent = LlmAgent(
    model=Gemini(model=gemini_model_name),
    tools=[google_search_tool],
    name="Code_Finder",
    instruction="""You are an expert developer. Your single job is to find the top 2-3
    most-shared or highest-star GitHub repositories for the user's topic.
    Only search GitHub. Return only the repository name and its URL."""
)

expert_agent = LlmAgent(
    model=Gemini(model=gemini_model_name),
    tools=[google_search_tool],
    name="Expert_Finder",
    instruction="""You are a community manager. Your single job is to find 2-3 of the
    most-followed experts, X (Twitter) accounts, or popular blogs on the
    user's topic. Return only the expert/blog name and their URL."""
)

# 3. Define the "Synthesizer" Agent
synthesizer_agent = LlmAgent(
    model=Gemini(model=gemini_model_name),
    name="Synthesizer",
    instruction="""You are a helpful assistant. The user asked for a learning plan.
    You will receive a messy JSON blob containing all the data found by
    other agents. Your one and only job is to re-format this data into a
    clean, easy-to-read markdown report.

    Use the following format:

    Here is your 360Â° Learning Path for [TOPIC]!

    ## ğŸ�“ Top Courses & Videos
    * [Course Name]: [URL]
    * [Course Name]: [URL]

    ## ğŸ’» GitHub Repositories
    * [Repo Name]: [URL]
    * [Repo Name]: [URL]

    ## ğŸ—£ï¸� Experts & Blogs to Follow
    * [Expert Name]: [URL]
    * [Expert Name]: [URL]

    Do not add any other commentary. Just produce the report.
    """
)

# 4. Define the Workflow
parallel_search_step = ParallelAgent(
    sub_agents=[
        course_agent,
        code_agent,
        expert_agent
    ],
    name="Parallel_Search_Workflow"
)

root_workflow_agent = SequentialAgent(
    sub_agents=[
        parallel_search_step,
        synthesizer_agent
    ],
    name="Root_Learning_Orchestrator"
)

print("âœ… All Agents & Workflow Defined (Workshop Fix): Using 'gemini-2.5-flash-lite'")


import asyncio
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.genai.types import Content, Part

# 1. Create a Session Service
session_service = InMemorySessionService()

# 2. Create the Runner
runner = Runner(
    agent=root_workflow_agent,
    app_name="360_learning_app",
    session_service=session_service,
)

# 3. Define an async function to run our agent
async def run_agent_query(topic: str):
    print(f"ğŸš€ Running 360Â° Learning Path Agent for: '{topic}'...")
    
    # 4. Create a new session
    session = await session_service.create_session(
        app_name="360_learning_app",
        user_id="kaggle-user"
    )
    
    # 5. Prepare the message
    user_query = f"Please find a 360 learning plan for the topic: {topic}"
    message = Content(
        role="user",
        parts=[Part(text=user_query)]
    )
    
    print("\n--- Processing (this may take 30-60 seconds)... ---\n")
    
    # 6. Collect events
    event_count = 0
    max_events = 100
    final_output = []
    
    try:
        async for event in runner.run_async(
            user_id="kaggle-user",
            session_id=session.id,
            new_message=message
        ):
            event_count += 1
            print(f"Event {event_count}: {type(event).__name__}")
            
            # Capture output
            if hasattr(event, 'content'):
                content_str = str(event.content)
                if len(content_str) > 200:  # Only print long content
                    print(f"  â†’ {content_str[:200]}...")
                    final_output.append(content_str)
            
            if event_count >= max_events:
                print(f"\nâš ï¸� Reached {max_events} events")
                break
                
    except Exception as e:
        print(f"\nâ�Œ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"âœ… Completed with {event_count} events")
    print(f"{'='*60}\n")
    
    if final_output:
        print("ğŸ“‹ FINAL LEARNING PATH:\n")
        print(final_output[-1])  # Print the last (final) output
    
    return final_output

# --- Run with timeout ---
user_topic = "Prompt Engineering"

try:
    result = await asyncio.wait_for(run_agent_query(user_topic), timeout=120.0)
    print("\nâœ… Agent completed successfully!")
except asyncio.TimeoutError:
    print("â�° Operation timed out after 120 seconds")


# Create a new project folder named 'learning-path-agent'
!adk create learning-path-agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY


%%writefile learning-path-agent/agent.py

from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.models import Gemini

# --- CONFIGURATION ---
# Using the workshop-compatible model
MODEL_NAME = 'gemini-2.5-flash-lite'

# --- WORKER AGENTS ---

# 1. Course Finder
course_agent = LlmAgent(
    model=Gemini(model=MODEL_NAME),
    tools=[google_search],
    name="Course_Finder",
    instruction="""You are an expert academic advisor. Your single job is to find the
    one or two best, top-rated, and free online courses or video playlists
    for the user's topic. Return only the course name and its URL."""
)

# 2. Code Finder
code_agent = LlmAgent(
    model=Gemini(model=MODEL_NAME),
    tools=[google_search],
    name="Code_Finder",
    instruction="""You are an expert developer. Your single job is to find the top 2-3
    most-shared or highest-star GitHub repositories for the user's topic.
    Only search GitHub. Return only the repository name and its URL."""
)

# 3. Expert Finder
expert_agent = LlmAgent(
    model=Gemini(model=MODEL_NAME),
    tools=[google_search],
    name="Expert_Finder",
    instruction="""You are a community manager. Your single job is to find 2-3 of the
    most-followed experts, X (Twitter) accounts, or popular blogs on the
    user's topic. Return only the expert/blog name and their URL."""
)

# --- SYNTHESIZER AGENT ---
synthesizer_agent = LlmAgent(
    model=Gemini(model=MODEL_NAME),
    name="Synthesizer",
    instruction="""You are a helpful assistant. The user asked for a learning plan.
    You will receive a messy JSON blob containing all the data found by
    other agents. Your one and only job is to re-format this data into a
    clean, easy-to-read markdown report.

    Use the following format:

    Here is your 360Â° Learning Path for [TOPIC]!

    ## ğŸ�“ Top Courses & Videos
    * [Course Name]: [URL]
    * [Course Name]: [URL]

    ## ğŸ’» GitHub Repositories
    * [Repo Name]: [URL]
    * [Repo Name]: [URL]

    ## ğŸ—£ï¸� Experts & Blogs to Follow
    * [Expert Name]: [URL]
    * [Expert Name]: [URL]

    Do not add any other commentary. Just produce the report.
    """
)

# --- WORKFLOW ---

# Parallel Step
parallel_search_step = ParallelAgent(
    sub_agents=[course_agent, code_agent, expert_agent],
    name="Parallel_Search_Workflow"
)

# Root Sequential Agent
# The ADK UI looks for a variable named 'agent' by default
agent = SequentialAgent(
    sub_agents=[parallel_search_step, synthesizer_agent],
    name="Root_Learning_Orchestrator"
)


import asyncio
from IPython.display import display, Markdown, clear_output
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

# 1. Setup (Same as before)
session_service = InMemorySessionService()
runner = Runner(
    agent=root_workflow_agent, 
    app_name="360_learning_app",
    session_service=session_service,
)

# 2. Create a session
session = await session_service.create_session(
    app_name="360_learning_app",
    user_id="kaggle_judge"
)

print("âœ… Agent Ready! Type 'exit' to stop.\n")

# 3. The Interactive Loop
while True:
    print("-" * 40)
    user_topic = input("Enter a topic to learn (or 'exit'): ")
    
    if user_topic.lower() in ["exit", "quit"]:
        print("ğŸ‘‹ Exiting chat.")
        break
    
    if not user_topic.strip():
        continue

    print(f"\nğŸ”� Searching for '{user_topic}'... (Please wait 30-60s)\n")
    
    # B. Prepare the input
    input_content = Content(
        role="user",
        parts=[Part(text=f"Please find a 360 learning plan for the topic: {user_topic}")]
    )
    
    final_output_text = ""

    try:
        # C. Run the Agent (Iterate over the stream)
        async for event in runner.run_async(
            new_message=input_content,
            session_id=session.id,
            user_id="kaggle_judge"
        ):
            # We capture the text from the events. 
            # The last event will contain the final synthesized report.
            if hasattr(event, 'content') and event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        final_output_text = part.text

        # D. Display the Result
        if final_output_text:
            print("\nâœ¨ RESULT âœ¨\n")
            display(Markdown(final_output_text))
        else:
            print("âš ï¸� Agent finished but returned no text.")
        
    except Exception as e:
        print(f"â�Œ Error: {e}")
        import traceback
        traceback.print_exc()

