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


from google.genai import types
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools import google_search 

APP_NAME = "insightbot"
USER_ID = "demo_user"
SESSION_ID = "session_001"
GEMINI_MODEL = "gemini-2.5-flash-lite"


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)



GEMINI_MODEL = "gemini-2.5-flash-lite"
research_agent = LlmAgent(
    name = "ResearchAgent",
    model = GEMINI_MODEL, 
    instruction = (
        "You are a focused Research Agent."
        "Use google search to gather 2-4 key facts and numbers."
        "or quotes about the user research topic."
        "Return your answers as clear bullet points with inline citations"
        "like [source]. Keep it under 250 words."
        
    ),
    description = "Finds web-based evidence and returns structured research notes.",
    tools = [google_search],
    output_key = "research_findings",
)
print("Research Agent Created")

summary_agent = LlmAgent(
    name = "SummarizerAgent",
    model = GEMINI_MODEL, 
    instruction = (
        "You are an expert Summarizer.\n\n"
        "You are given research notes in {research_findings}.\n"
        "Write a concise, user-friendly summary:\n"
        "- 3-5 bullet points\n"
        "- Focus on the most decision relevant insights\n"
        "- Avoid repeating citations; keep them only where helpful\n"
        "- Max ~120 words total."
    ),
    description = "Summarizes research notes into a compact insight report.",
    output_key = "summary_bullets",
)
print("Summarizer Agent Created")

reviewer_agent = LlmAgent(
    name = "ReviewerAgent",
    model = GEMINI_MODEL,
    instruction = (
        "You are strict but helpful reviewer.\n\n"
        "You receieve:\n"
        "- Research Notes: {research_findings}\n"
        "- Summary Draft: {summary_bullets}\n\n"
        "Your job:\n\n"
        "1. Give a 1-5 score for factual coverage."
        "2. Give a 1-5 score for clarity and structure."
        "3. Give 2-3 specific suggestions for summary.\n\n"
        "Return your answer in this format:\n"
        "Coverage: X/5\n"
        "Clarity: Y/5\n"
        "Suggestions:\n"
        "- ...\n"
        "- ..."
    ),
    description = "Reviews summary and provides simple evaluation metrics.",
    output_key = "review_notes",
)
print("Review Agent Created")


insight_bot = SequentialAgent(
    name = "InsightBotSequential",
    sub_agents = [research_agent, summary_agent, reviewer_agent],
)
print("InsighBot Sequential Created")


!pip install -q nest_asyncio



def run_insightbot(user_query: str):
    content = types.Content(
        role="user",
        parts=[types.Part(text=user_query)],
    )

    print(f"\nğŸ§‘â€�ğŸ’» User Query: {user_query}\n")

    final_text = None

    events = runner.run(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=content,
    )

    for event in events:
        if hasattr(event, "author") and event.author:
            print(f"[event] from agent: {event.author}")

        if event.is_final_response():
            final_text = event.content.parts[0].text

    print("\nğŸ“Œ Final assistant response (from ReviewerAgent):\n")
    print(final_text)

    return final_text



_ = run_insightbot(
    "What are the main benefits and risks of using AI agents in healthcare?"
)


