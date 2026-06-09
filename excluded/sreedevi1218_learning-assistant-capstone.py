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


# 1.1 Attach and verify your Kaggle secret before running this cell
import os
from kaggle_secrets import UserSecretsClient
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret('GOOGLE_API_KEY')
    os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
    print('âœ… Gemini API key setup complete.')
except Exception as e:
    print('ğŸ”‘ Authentication Error: Please attach GOOGLE_API_KEY in Add-ons -> Secrets. Error:', e)


# 1.2 Install / confirm packages (Kaggle usually has adk preinstalled)
!pip show google-adk


import os
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("GOOGLE_API_KEY")

os.environ["GOOGLE_API_KEY"] = api_key

print("âœ… API Key loaded!")


from kaggle_secrets import UserSecretsClient
import os

user_secrets = UserSecretsClient()

api_key = user_secrets.get_secret("GOOGLE_API_KEY")   # Load secret properly

os.environ["GOOGLE_API_KEY"] = api_key                # Set for Gemini

print("âœ… API Key loaded!")


print(os.getenv("GOOGLE_API_KEY"))


print(os.environ["GOOGLE_API_KEY"][:5] + "****")


# âœ… Working imports for Kaggle ADK v1.18 (tested)

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent

from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner

# Tools (correct modules for v1.18)
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.google_search_tool import GoogleSearchTool   # âœ” This is the correct one!

import asyncio
import json

print("âœ… ADK v1.18 imports successful â€” ready to build agents!")


# â�Œ REMOVE InMemorySessionService (not available in v1.18)

# Helper: pretty print agent responses
def show_response(resp):
    try:
        print('\n'.join([str(resp)]))
    except Exception:
        print(resp)

print("âœ… Session handling skipped (v1.18 has no InMemorySessionService)")


# Simple task extractor: parse notes into tasks
def extract_tasks(notes: str) -> str:
    """Return a semicolon-separated list of tasks extracted from notes."""
    parts = [p.strip() for p in notes.split('.') if p.strip()]
    tasks = []
    for p in parts:
        # basic rule-based extraction; you can expand with regex or LLM
        tasks.append(p)
    return json.dumps({'tasks': tasks})

# Wrap with FunctionTool (Kaggle ADK uses FunctionTool(func) signature)
task_tool = FunctionTool(extract_tasks)
print('âœ… Task extraction tool created')


# MCQ generator uses LLM via agent, but we'll also provide a small evaluator tool
# Evaluator: compare user answers with correct ones and return score

def evaluate_mcq(mcq_list_json: str, user_answers_json: str) -> str:
    """mcq_list_json: JSON list of {q, options, correct_index}
       user_answers_json: JSON list of indices submitted by user
       returns: JSON with score and feedback
    """
    mcq_list = json.loads(mcq_list_json)
    user_answers = json.loads(user_answers_json)
    correct = 0
    feedback = []
    for i, q in enumerate(mcq_list):
        correct_index = q.get('correct_index')
        user_i = user_answers[i] if i < len(user_answers) else None
        ok = (user_i == correct_index)
        if ok:
            correct += 1
            feedback.append({'q_index': i, 'ok': True})
        else:
            feedback.append({'q_index': i, 'ok': False, 'correct_index': correct_index})
    return json.dumps({'score': correct, 'total': len(mcq_list), 'feedback': feedback})

mcq_evaluator_tool = FunctionTool(evaluate_mcq)
print('âœ… MCQ evaluator tool created')


# This async function simulates a long-running operation requiring approval.
async def add_to_schedule(task: str, date: str, confirm: bool = False) -> str:
    """If confirm==False, return a pause request message. If confirm==True, perform 'save'."""
    if not confirm:
        # Return a structured pause message â€” ADK will show this in the trace
        return json.dumps({'status': 'PAUSE', 'message': f"Please confirm adding '{task}' on {date}.", 'task': task, 'date': date})
    # Simulate work
    await asyncio.sleep(1)
    # Save to an imaginary calendar (here we'll store in a file or memory)
    # For demo, we return success message.
    return json.dumps({'status': 'SAVED', 'task': task, 'date': date})

schedule_tool = LongRunningFunctionTool(add_to_schedule)
print('âœ… Long-running schedule tool created (MCP)')


# Create the Google Search tool instance
search_tool = GoogleSearchTool()

# Create the explain agent (fixed for v1.18)
explain_agent = LlmAgent(
    name='explain_agent',
    model=Gemini(model='gemini-1.5-flash'),   # â†� use model= instead of llm=
    description='Explain technical concepts simply using Google Search for fresh info.',
    instruction=(
        'You are a friendly tutor. '
        'Give simple explanations with examples and one exercise at the end. '
        'Use GoogleSearchTool for up-to-date info.'
    ),
    tools=[search_tool]
)

print("âœ… explain_agent created successfully!")


model=Gemini(model="gemini-2.0-flash-lite-preview")


model=Gemini(model="gemini-2.0-pro-exp")


# Create search tool
search_tool = GoogleSearchTool()

# FIX: Use correct model name supported by Kaggle
explain_agent = LlmAgent(
    name='explain_agent',
    model=Gemini(model="gemini-2.0-flash-lite-preview"),
    description='Explain technical concepts simply using Google Search for fresh info.',
    instruction=(
        'You are a friendly tutor. '
        'Give simple explanations with examples and a practice question.'
    ),
    tools=[search_tool]
)

print("âœ… explain_agent created with supported model!")


runner = InMemoryRunner(explain_agent)

response = await runner.run_debug("Explain what is cloud computing")
print(response)


# Create search tool
search_tool = GoogleSearchTool()

# Create explain_agent (clean, working)
explain_agent = LlmAgent(
    name='explain_agent',
    model=Gemini(model="gemini-2.0-flash-lite-preview"),
    description="Explains concepts in simple terms.",
    instruction=(
        "Explain topics clearly with examples and a practice question. "
        "Use GoogleSearchTool if needed."
    ),
    tools=[search_tool]
)

print("âœ… explain_agent created!")


mcq_agent = LlmAgent(
    name="mcq_agent",
    model=Gemini(model="gemini-2.0-flash-lite-preview"),
    description="Creates quiz questions.",
    instruction="Generate 5 MCQs with answers for any topic."
)

print("âœ… mcq_agent created!")


planner_agent = LlmAgent(
    name="planner_agent",
    model=Gemini(model="gemini-2.0-flash-lite-preview"),
    description="Creates study plans.",
    instruction="Create a simple day-wise study plan."
)

print("âœ… planner_agent created!")


import inspect
from google.adk.agents.sequential_agent import SequentialAgent

print(inspect.getsource(SequentialAgent))


# Create the root sequential agent correctly for ADK v1.18

root_agent = SequentialAgent(
    name='learning_assistant',
    sub_agents=[explain_agent, mcq_agent, planner_agent],   # <<< correct field
    description='Coordinator agent: routes user requests to correct sub-agent.'
)

print("âœ… root_agent created successfully!")


runner = InMemoryRunner(root_agent)

response = await runner.run_debug(
    "Explain cloud computing and create 2 MCQs"
)

print(response)


runner = InMemoryRunner(root_agent)
print("Runner ready!")


runner = InMemoryRunner(explain_agent)
print("Runner ready!")


resp = await runner.run_debug(
    "Explain virtualization and give a simple example for beginners."
)

print(resp)


runner = InMemoryRunner(root_agent)
print("Runner ready!")


resp = await runner.run_debug(
    "Generate 5 MCQs on Python lists. Return JSON with questions, options and correct_index."
)

print(resp)


runner = InMemoryRunner(root_agent)
print("Runner ready!")


resp = await runner.run_debug(
    "Generate 5 MCQs on Python lists. Return JSON with questions, options and correct_index."
)

print(resp)


# Example: pretend mcqs_json and user_answers_json are available
mcqs_json = '[{"q":"1+1","options":["1","2","3","4"],"correct_index":1}]'
user_answers_json = '[1]'
resp = await runner.run_debug(f'Evaluate these MCQs | mcqs:{mcqs_json} | answers:{user_answers_json}')
show_response(resp)


# Step 1: ask to add a study task
resp = await runner.run_debug("Add to schedule: 'Study Numerical Methods' on 2025-12-02")
show_response(resp)

# The agent should return a pause message asking for confirmation
# Step 2: Confirm
resp2 = await runner.run_debug("Yes, confirm=True")
show_response(resp2)


# Simple custom session memory (replacement for missing ADK session)
class SimpleSessionMemory:
    def __init__(self):
        self.store = {}

    def create_session(self, session_id):
        if session_id not in self.store:
            self.store[session_id] = {}
        return self

    def get_state(self):
        return self.store

    def save_state(self, state):
        self.store = state

# Create memory
session_service = SimpleSessionMemory()
print("âœ… Simple session memory ready!")


session = session_service.create_session('student_1')

state = session.get_state()
state['profile'] = {
    'name': 'Student A',
    'weak_subjects': ['Numerical Methods'],
    'preferred_study_hours': 2
}

session.save_state(state)
print("âœ… Saved profile to session state")


User
  |
  v
Root SequentialAgent (learni


# Custom function tool
def create_study_plan(hours: int, topic: str) -> str:
    """
    Create a 3-step study plan based on total hours.
    """
    per_topic = hours / 3
    return (
        f"ğŸ“˜ Study Plan for {topic}\n"
        f"- {per_topic:.1f} hours: Understanding theory\n"
        f"- {per_topic:.1f} hours: Practice problems\n"
        f"- {per_topic:.1f} hours: Revision + summary notes\n"
    )

# Tool must be created using ONLY the function
study_plan_tool = FunctionTool(create_study_plan)

# Google Search tool
search_tool = GoogleSearchTool()

print("âœ… Tools ready: Custom tool + Google Search tool")

