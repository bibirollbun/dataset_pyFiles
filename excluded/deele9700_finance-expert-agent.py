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


!pip install google-adk google-generativeai google-api-python-client google-auth


import os
from kaggle_secrets import UserSecretsClient
GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
print("âœ… Gemini API key configured.")


from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.adk.tools import google_search

initial_search_agent = LlmAgent(
    name="InitialSearchAgent",
    model=Gemini(model="gemini-2.5-pro"),
    instruction=(
        "Use the google_search tool to research the userâ€™s question. "
        "Return a concise answer formatted as markdown with clear headings and bullet points. "
        "Highlight the most important trends or facts, and keep the narrative short."
    ),
    tools=[google_search],
    output_key="current_answer"
)




critic_agent = LlmAgent(
    name="CriticAgent",
    model=Gemini(model="gemini-2.5-flash"),
    instruction=(
        "Review the answer. If it is concise, wellâ€‘structured (headings + bullet points), "
        "and covers the key points, reply only with 'APPROVED'. Otherwise, suggest specific "
        "improvements to shorten and organise the content."
    ),
    output_key="critique"
)


from google.adk.tools import FunctionTool

def exit_loop():
    return "EXIT"

iterate_agent = LlmAgent(
    name="IterateAgent",
    model=Gemini(model="gemini-2.5-flash"),
    instruction=(
        "Rewrite the answer incorporating the criticâ€™s feedback. "
        "Ensure the result is concise, uses markdown headings and bullet points, "
        "and clearly emphasises the most important information. "
        "If the critique is 'APPROVED', call exit_loop()."
    ),
    tools=[FunctionTool(exit_loop)],
    output_key="current_answer"
)


from google.adk.agents import LoopAgent

answer_loop = LoopAgent(
    name="AnswerRefinementLoop",
    sub_agents=[critic_agent, iterate_agent],
    max_iterations=3
)


from google.adk.agents import SequentialAgent

root_agent = SequentialAgent(
    name="AnswerPipeline",
    sub_agents=[initial_search_agent, answer_loop]
)


from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService

session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()

runner = Runner(
    agent=root_agent,
    session_service=session_service,
    memory_service=memory_service,
    app_name="fp&a-agent"
)


from google.genai.types import Content, Part

app_name = "fp&a-agent"
user_id = "user1"
session_id = "fp&a-session-001"
query_text = "Summarize the latest automation trends in FP&A."
await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)
user_msg = Content(parts=[Part(text=query_text)], role="user")
print(f"\n--- Query 1: {query_text} ---\n")
for event in runner.run(user_id=user_id, session_id=session_id, new_message=user_msg):
    if event.is_final_response():
        print(event.content.parts[0].text)
        break

