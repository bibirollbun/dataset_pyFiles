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


pip install google-adk


from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types

print("âœ… ADK components imported successfully.")


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)


# Quest Outline Agent: Creates the quest structure
quest_outline_agent = Agent(
    name="QuestOutlineAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    Create a quest outline for a game based on the user's request.

    The outline must include:
    1. Quest Title
    2. Quest Hook (1â€“2 lines)
    3. Objectives (3â€“5 bullet points)
    4. Key NPCs (names + short role)
    5. Rewards
    """,
    output_key="quest_outline",
)

print("âœ… quest_outline_agent created.")


# Quest Writer Agent: Expands the outline into a full quest narrative
quest_writer_agent = Agent(
    name="QuestWriterAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    Using this outline: {quest_outline}
    Write a 200â€“300 word game quest scenario.
    Make it engaging, descriptive, and game-friendly.
    """,
    output_key="quest_draft",
)

print("âœ… quest_writer_agent created.")



# Quest Polisher Agent: Edits and enhances the quest
quest_editor_agent = Agent(
    name="QuestEditorAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    Edit this draft: {quest_draft}
    Improve flow, clarity, game immersion, grammar, and narrative polish.
    Maintain the tone and style of a game quest.
    """,
    output_key="final_quest",
)

print("âœ… quest_editor_agent created.")


# Sequential Pipeline (just like in Kaggle)
root_agent = SequentialAgent(
    name="GameQuestPipeline",
    sub_agents=[quest_outline_agent, quest_writer_agent, quest_editor_agent],
)

print("âœ… Sequential Agent created.")


# Runner
runner = InMemoryRunner(agent=root_agent)

response = await runner.run_debug(
    "Create a fantasy RPG quest involving a cursed forest and a missing villager."
)

response

