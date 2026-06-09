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
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


import google.generativeai as genai
from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.genai import types
from kaggle_secrets import UserSecretsClient
import os
import pandas as pd
import logging
from IPython.display import display, Markdown


# Retry Configuration for Stability
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=2,
    initial_delay=1,
    http_status_codes=[429, 500, 503],
)

# Common Model Config
model_config = Gemini(
    model="gemini-2.5-flash-lite",
    retry_options=retry_config
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True
)


# --- (Topic 1: Multi-agent System) ---

# 1. Story Architect Agent
# Takes the user's initial idea and creates a structured outline.
story_architect = Agent(
    name="StoryArchitect",
    model=model_config,
    instruction="""You are an expert Story Architect for short films. 
Your goal is to take a simple user idea and expand it into a solid story outline.
The outline must include:
- Title
- Logline
- Character List (with brief descriptions)
- 3-Act Structure Outline (Setup, Confrontation, Resolution)
Keep it suitable for a 3-5 minute short film.""",
    output_key="story_outline", # Stores the result in the session state
)

# 2. Script Writer Agent
# Takes the outline and writes the actual script.
script_writer = Agent(
    name="ScriptWriter",
    model=model_config,
    instruction="""You are a professional Screenwriter.
Your goal is to write a complete screenplay based on the provided story outline.
Story Outline: {story_outline}

Format it as a standard screenplay (Scene Headings, Action, Character Name, Dialogue).
Ensure the pacing fits a 3-5 minute film (approx 3-5 pages of text).
Focus on visual storytelling and natural dialogue.""",
    output_key="script_draft",
)

# 3. Evaluation Agent (Topic 2: LLM-as-Judge)
# Critiques the generated script.
evaluator = Agent(
    name="Evaluator",
    model=model_config,
    instruction="""You are a strict Film Critic and Script Doctor.
Your goal is to evaluate the provided screenplay.
Screenplay: {script_draft}

You must provide:
1. A Score out of 10.
2. Strengths (bullet points).
3. Weaknesses (bullet points).
4. A final verdict: "GREENLIGHT" or "REWRITE".

Be critical but constructive. Focus on structure, dialogue, and feasibility.""",
    output_key="evaluation_report",
)

# --- SEQUENTIAL WORKFLOW ---
# Chains the agents together in a fixed order.
creative_studio = SequentialAgent(
    name="CreativeStudio",
    sub_agents=[story_architect, script_writer, evaluator],
)

print("âœ… Agents and Workflow defined.")


# --- (Topic 3) ---

async def run_studio(user_idea):
    print(f"ğŸ�¬ Starting Creative Studio with idea: '{user_idea}'\n" + "="*50)
    
    # Initialize the runner with our sequential agent AND the LoggingPlugin
    # This explicitly fulfills the Observability requirement using the ADK standard way
    runner = InMemoryRunner(
        agent=creative_studio,
        plugins=[LoggingPlugin()]
    )
    
    # Run the agent system
    # run_debug prints traces to stdout. Combined with LoggingPlugin, we get robust observability.
    response = await runner.run_debug(user_idea)
    
    print("\n" + "="*50 + "\nâœ… Workflow Complete!")
    return runner

# Define the user prompt
USER_PROMPT = "A lonely astronaut on Mars finds a flower growing in the dust."

if GOOGLE_API_KEY:
    # Run the async function
    runner_instance = await run_studio(USER_PROMPT)
else:
    print("â�Œ API Key not set. Please configure GOOGLE_API_KEY.")

