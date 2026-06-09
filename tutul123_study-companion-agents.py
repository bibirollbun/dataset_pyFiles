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


from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search, AgentTool, ToolContext
from google.adk.code_executors import BuiltInCodeExecutor


print("âœ… ADK components imported successfully.")


import logging
import os

# Clean up any previous logs
for log_file in ["logger.log", "web.log", "tunnel.log"]:
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"ğŸ§¹ Cleaned up {log_file}")

# Configure logging with DEBUG log level.
logging.basicConfig(
    filename="logger.log",
    level=logging.DEBUG,
    format="%(filename)s:%(lineno)s %(levelname)s:%(message)s",
)

print("âœ… Logging configured")


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

print('âœ…Retry options created successfully.')


search_agent = Agent(
    name="SearchAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="A simple agent that will search and create an outline study materials foe an engineer with itermediate technology knowledge for a topic that was asked by the user.",
    instruction="You are a helpful assistant. Use Google Search for current info or if unsure.",
    tools=[google_search],
    output_key = "search_findings",  # The result of this agent will be stored in the session state with this key.
)

print("âœ… Search Agent defined that will search study materials based on the requested topic.")


# Content CreatorAgent: Its job is to create detailed content based on the text it receives.
creator_agent = Agent(
    name="CreatorAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    # The instruction is modified to request a bulleted list for a clear output format.
    instruction=""" Read the provided detailed content for the requested topic: {search_findings}.
    Create detailed content for study material for each of the sub-topics that should include the concept, 
    what the subtopic achieves, detailed theory with examples and where relevant, required tools. .""",
    tools = [google_search],
    output_key="detailed_content",
)

print("âœ… creator_agent created.")


# Study PlanerAgent: Its job is to create detailed study plan for a required duration based on the content it receives previously.
planner_agent = Agent(
    name="PlannerAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    # The instruction is modified to request a bulleted list for a clear output format.
    instruction="""Read the provided detailed content for the requested topic: {detailed_content}.
    Create a Study plan for 7 days to gain maximum konwledge and expertise on the topic based on the detiled content created. .""",
    tools = [google_search],
    output_key="study_plan",
)

print("âœ… planner_agent created.")


# PreparationAgent: Its job is to create outline questions with options asa list of dictionaries based on the detailed text it receives.
preparation_agent = Agent(
    name="PreparationAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    # The instruction is modified to request a bulleted list for a clear output format.
    instruction=""" Read the provided detailed content for the requested topic: {detailed_content}.
    Create a python list of 10 dictionaries, where each dictionary represents a question and has the following structure. However, no need to show the output:
                        {
                            "question_text": "What is the capital of France?",
                            "options": ["London", "Paris", "Berlin", "Rome"],
                            "answer": "Paris"
                        }""",
                        
    output_key="questions_data", # This is the final output of the pipeline.
)

print("âœ… preparation_agent created.")


from typing import List, Dict, Any, Optional

def create_mcq_quiz(mcq_list: List[Dict[str, Any]], user_answers: Optional[List[int]] = None) -> None:
    print("\nWelcome to the MCQ Quiz!\n")
    score = 0
    total = 0

    for idx, item in enumerate(mcq_list, start=1):
        try:
            question, options, answer_raw = normalize_mcq_item(item, idx)
            correct_index = resolve_correct_answer(options, answer_raw, idx)
        except ValueError as e:
            print(f"Skipping Q{idx}: {e}\n")
            continue

        total += 1
        print(f"Q{idx}. {question}")
        for i, opt in enumerate(options, start=1):
            print(f"   {i}. {opt}")

        if user_answers is not None:
            if idx-1 < len(user_answers):
                choice_num = user_answers[idx-1]
            else:
                print("âš ï¸� No answer supplied for this question, marking incorrect.\n")
                choice_num = -1
        else:
            try:
                choice_raw = input("Enter your choice (number): ").strip()
                choice_num = int(choice_raw)
            except Exception:
                print("âš ï¸� Input not available, marking incorrect.\n")
                choice_num = -1

        is_correct = (choice_num - 1) == correct_index
        if is_correct:
            print("âœ… Correct!\n")
            score += 1
        else:
            print(f"â�Œ Wrong! Correct answer: {correct_index + 1}. {options[correct_index]}\n")

    print(f"Quiz Completed! Your Score: {score}/{total} ({(score/total*100) if total else 0:.0f}%)")

print('Function created successfully!')


# MCQAgent: Its job is to create detailed content based on the text it receives.
mcq_agent = LlmAgent(
    name="McqAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    You are an expert in creating multiple-choice questions. 
    Your task is to generate 10 multiple-choice questions based on: {questions_data}
    Each question should have one correct answer and three plausible incorrect options.
    Format the output clearly, indicating the correct answer for each question."
    """,
    tools=[create_mcq_quiz],
    output_key = "mcq_questionset",
)

print("âœ… mcq_agent created.")


# Root Coordinator: Orchestrates the workflow by calling the sub-agents as tools.
root_agent = SequentialAgent(
    name= "Pipeline",
    sub_agents= [ search_agent, creator_agent, planner_agent, preparation_agent, mcq_agent],
    
)
# print("âœ… Sequential Agent created.")

print("âœ… root_agent created.")


runner = InMemoryRunner(agent=root_agent)

print("âœ… Runner created.")


response = await runner.run_debug(
    "What is AWS Machine Learning Engineer?"
)


!adk create search-agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY


%%writefile search-agent/agent.py

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.google_search_tool import google_search

from google.genai import types
from typing import List

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

# ---- Intentionally pass incorrect datatype - `str` instead of `List[str]` ----
def count_papers(papers: List[str]):
    """
    This function counts the number of papers in a list of strings.
    Args:
      papers: A list of strings, where each string is a research paper.
    Returns:
      The number of papers in the list.
    """
    return len(papers)


# Google Search agent
search_agent = LlmAgent(
    name="search_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="Searches for information using Google search",
    instruction="""Use the google_search tool to find information on the given topic. Return the raw search results.
    If the user asks for a list of papers, then give them the list of search papers you found and not the summary.""",
    tools=[google_search]
)


# Root agent
root_agent = LlmAgent(
    name="search_paper_finder_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""Your task is to find research papers and count them. 

    You MUST ALWAYS follow these steps:
    1) Find research papers on the user provided topic using the 'google_search_agent'. 
    2) Then, pass the papers to 'count_papers' tool to count the number of papers returned.
    3) Return both the list of research papers and the total number of papers.
    """,
    tools=[AgentTool(agent=search_agent), count_papers]
)


url_prefix = get_adk_proxy_url()


!adk web --url_prefix {url_prefix}

