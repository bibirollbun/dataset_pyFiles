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


This project introduces a multi-agent, AI-driven career coaching platform built using Google Gemini and the Agent Developer Kit (ADK). Unlike traditional chatbots that provide one-off answers, this system delivers continuous, personalized, and structured career guidance by coordinating multiple specialized agents, tools, and persistent memory.

Purpose & Audience

The system is designed for students and early-career professionals who struggle to translate courses, projects, and part-time experience into real job opportunities. It guides users through role exploration, skill development, resume improvement, and interview practice in a coherent, long-term coaching workflow.

Problem

Students often face barriers to career advancement:

Difficulty converting academic experiences into professional assets

Uncertainty about suitable job roles or how to qualify for them

Overloaded career centers offering limited personalized support

Online platforms listing roles but offering no structured coaching

Generic LLM chat lacking memory, follow-through, and process discipline

This results in trial-and-error applications, unclear progress, and limited strategic planning.

Solution

The system reframes Gemini from a simple Q&A model to the reasoning core of a multi-agent coaching architecture. It divides the coaching process into specialized modulesâ€”resume parsing, role matching, skill-gap analysis, resume critique, mock interviews, and progress trackingâ€”each handled by a dedicated LlmAgent with its own instructions and tools.

A central Planner Agent orchestrates these components, deciding which specialists to invoke and merging their outputs into a unified response. The ADK's Runner and SQLite-backed Session Service maintain conversation history so users can return anytime and continue seamlessly.

Architecture Highlights

Gemini Model Layer: Configured with retry logic for reliable performance

Specialist Agents: Each focused on a single capability (e.g., resume critique)

Planner Agent: Interprets user intent and coordinates specialists

Persistent Memory: Conversation history stored via DatabaseSessionService

Event Compaction: Summaries keep historical context manageable

Python Tools: Structured resume parsing, skill-gap analysis, and job matching

Web UI: ADKâ€™s built-in planner interface, exposed through Kaggle proxy

Key Capabilities:

Parse resumes into structured data

Identify realistic job roles based on skill profiles

Diagnose missing or underdeveloped skills

Recommend tailored learning plans

Deliver a targeted resume critique

Generate mock interview questions aligned with chosen roles

Track user goals and progress over multiple sessions

Outcomes

The project demonstrates how to transform LLM components into a stateful, observable, and extensible application. Students benefit from personalized, ongoing guidance, while developers gain transparency into the underlying interactions between the agent and tools. The systemâ€™s modular design makes it easy to debug, extend, and evaluate.

Limitations & Future Improvements

Current role data is static, and the user interface is minimal. All agents share a single Gemini configuration, and formal evaluation is pending. Future enhancements include connecting to live job-market APIs, expanding portfolio and application-tracking tools, upgrading the UI, and refining memory strategies to emphasize long-term goals.

add Codeadd Markdown
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
add Codeadd Markdown
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
âœ… Gemini API key setup complete.
add Codeadd Markdown
#>> Gemini API key setup complete.****

add Codeadd Markdown
from google.adk.agents import Agent,LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search, AgentTool, ToolContext
from google.adk.code_executors import BuiltInCodeExecutor


print("âœ… ADK components imported successfully.")
âœ… ADK components imported successfully.
add Codeadd Markdown
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
âœ… Logging configured
add Codeadd Markdown
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
âœ… Helper functions defined.
add Codeadd Markdown
retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)
add Codeadd Markdown
!rm -rf /kaggle/working/sample-agent
add Codeadd Markdown
!adk create sample-agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY

Agent created in /kaggle/working/sample-agent:
- .env
- __init__.py
- agent.py

add Codeadd Markdown


root_agent = Agent(
    name="career_coach",     # must be a valid identifier (no spaces)
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ), 
    description="""Agentic coach that integrates resume review, career exploration, 
    and skill-gap analysis continuously and adaptively, like a real coach.

    This agent supports dynamic, dialog-based interaction (not just form-filling), 
    provides strategic career guidance, identifies strengths and weaknesses, 
    recommends learning paths, and helps users navigate job transitions.
    """,
    instruction="you are a career instructor",
    tools=[google_search],

    
    )

print("âœ… Root Agent defined.")

âœ… Root Agent defined.
add Codeadd Markdown
runner = InMemoryRunner(agent=root_agent)

print("âœ… Runner created.")
âœ… Runner created.
add Codeadd Markdown
response = await runner.run_debug(
    "What careers are in demand?"
)

 ### Created new session: debug_session_id

User > What careers are in demand?
career_coach > The job market in 2025 is dynamic, with significant growth anticipated in several key sectors. The healthcare and technology industries, in particular, are experiencing robust expansion and are expected to offer numerous opportunities.

Here are some of the careers in high demand:

**Healthcare:**
*   **Nurse Practitioners:** Projected to grow by 40% by 2034, driven by an aging population and increased prevalence of chronic conditions.
*   **Physician Assistants:** Expected to see 20% growth by 2034.
*   **Medical and Health Services Managers:** With a projected growth of 23%, these roles are crucial for managing healthcare facilities.
*   **Physical Therapist Assistants:** Anticipated to grow by 22%.
*   **Mental Health Professionals:** Demand is surging due to increased awareness and need for services.

**Technology and Data Science:**
*   **Data Scientists:** Essential for analyzing data to predict trends and inform business decisions.
*   **AI and Machine Learning Specialists:** With the rapid advancement of AI, these roles are in high demand for creating and refining AI systems.
*   **Information Security Analysts/Cybersecurity Specialists:** Crucial for protecting data against increasing cyber threats.
*   **Software and Application Developers:** Continued strong growth is fueled by digital transformation.
*   **Operations Research Analysts:** Expected to grow by 21%.

**Renewable Energy:**
*   **Wind Turbine Service Technicians:** Among the fastest-growing occupations.
*   **Solar Photovoltaic Installers:** Demand is high due to the growth in solar power installations.
*   **Environmental Engineers:** Driven by climate change mitigation efforts.

**Other Growing Sectors:**
*   **Transportation and Warehousing:** Fueled by the growth of e-commerce and the need for logistics.
*   **Digital Marketing and E-commerce:** With expanding online markets, roles like e-commerce specialists and digital marketing professionals are in demand.
*   **Accountants:** Despite some leaving the profession, there's a continuous high demand for accountants, offering plentiful job opportunities.

Key skills that are highly valued in the 2025 job market include analytical thinking, resilience, flexibility, agility, leadership, social influence, AI and big data proficiency, cybersecurity, and technological literacy.
add Codeadd Markdown
url_prefix = get_adk_proxy_url()
add Codeadd Markdown
!adk web --url_prefix {url_prefix}
add Codeadd Markdown
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
âœ… Helper functions defined.
add Codeadd Markdown
retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)
add Codeadd Markdown
#Tools created as Functions
from typing import List, TypedDict

# ----------------------------------------------------
# 1. TypedDict definitions (must come BEFORE functions)
# ----------------------------------------------------

class ResumeData(TypedDict):
    skills: List[str]
    experience_years: int
    education: str

class SkillGapData(TypedDict):
    target_role: str
    missing_skills: List[str]

class JobMatch(TypedDict):
    title: str
    fit: int
def parse_resume(resume_text: str) -> ResumeData:
    return {
        "skills": ["Python", "SQL", "Data Analysis"],
        "experience_years": 3,
        "education": "BS Computer Science"
    }

def analyze_skill_gap(user_skills: List[str], target_role: str) -> SkillGapData:
    role_requirements = {
        "data_scientist": ["Python", "ML", "Statistics", "SQL"],
        "data_analyst": ["Excel", "SQL", "Python"],
    }
    required = set(role_requirements.get(target_role, []))
    missing = list(required - set(user_skills))
    return {
        "target_role": target_role,
        "missing_skills": missing
    }

def match_jobs(skills: List[str]) -> List[JobMatch]:
    return [
        {"title": "Data Analyst", "fit": 90},
        {"title": "ML Intern", "fit": 77},
    ]
print ("parse_resume function created")
print ("analyze_skill_gap function created")
print ("match_jobs function created")
print (f"Test:{match_jobs('ML Intern')}")
add Codeadd Markdown
â€œorchestratorâ€� / planner agent that routes between multiple sub-agents.********
add Codeadd Markdown
!rm -rf /kaggle/working/planner_agent
add Codeadd Markdown
!adk create planner_agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY
add Codeadd Markdown
%%writefile planner_agent/agent.py
skill_assessment_agent = LlmAgent(
    name="skill_assessment_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="Analyses user skills vs target roles and identifies gaps.",
    instruction=(
        "You analyse the user's skills versus a target role. "
        "Use analyze_skill_gap to compute missing skills. "
        "Use match_jobs to suggest suitable roles based on the user skills. "
        "Explain clearly: current skills, missing skills, and a concrete learning plan."
    ),
    tools=[analyze_skill_gap, match_jobs],
)

mock_interview_agent = LlmAgent(
    name="mock_interview_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="Runs mock interviews and gives feedback.",
    instruction=(
        "You are a mock interview coach.\n"
        "- Ask one question at a time (behavioural and role-specific).\n"
        "- Wait for the user's answer before asking the next.\n"
        "- After each answer, briefly rate it and suggest improvements.\n"
        "- Keep answers concise and practical."
    ),
)

progress_tracker_agent = LlmAgent(
    name="progress_tracker_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="Tracks user progress and keeps a simple evolving plan.",
    instruction=(
        "You help the user track progress toward their career goals.\n"
        "- Summarize what the user has done so far (skills, learning, applications, interviews).\n"
        "- Suggest next 3â€“5 concrete actions.\n"
        "- Be consistent across calls: assume you are reading the same session history."
    ),
)

# 2. Wrap specialist agents as tools so the planner can call them
#    (AgentTool is the standard ADK pattern for 'agent-as-a-tool'). :contentReference[oaicite:0]{index=0}

resume_tool   = AgentTool(agent=resume_critique_agent)
skills_tool   = AgentTool(agent=skill_assessment_agent)
mock_tool     = AgentTool(agent=mock_interview_agent)
progress_tool = AgentTool(agent=progress_tracker_agent)

# 3. Planner / orchestrator agent
#    This is the top-level agent your UI / runner should talk to.

planner_agent = LlmAgent(
    name="planner_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description=(
        "Top-level career coach that orchestrates specialist agents for "
        "resume review, skill assessment, mock interviews, and progress tracking."
    ),
    instruction=(
        "You are the orchestration / planner agent in a career-coaching system.\n\n"
        "Your job:\n"
        "1. Read the user's request and decide which specialist tools to call:\n"
        "   - resume_critique_agent (for resume/LinkedIn review)\n"
        "   - skill_assessment_agent (for skills, gaps, and roles)\n"
        "   - mock_interview_agent (for interview practice)\n"
        "   - progress_tracker_agent (for tracking and planning)\n"
        "2. Call one or more of these tools as needed (you may chain them).\n"
        "3. Combine their results into a clear final answer for the user.\n"
        "4. Briefly explain which specialists you used and why.\n"
    ),
    tools=[
        resume_tool,
        skills_tool,
        mock_tool,
        progress_tool,
        google_search,  # optional: let the planner still use web search if needed
    ],
)



planner_runner = InMemoryRunner(agent=planner_agent)


add Codeadd Markdown
url_prefix = get_adk_proxy_url()
add Codeadd Markdown
!adk web --log_level DEBUG --url_prefix {url_prefix}
add Codeadd Markdown
#Stateful Agent Creation#
add Codeadd Markdown
#In ADK, tools are passed as a list of Python functions to the Agent constructor.

career_agent = LlmAgent( name="career_agent", # must be valid identifier (no spaces) description="Agentic career coach with resume analysis, skill-gap analysis, and job matching", tools=[parse_resume, analyze_skill_gap, match_jobs], model=Gemini( model="gemini-2.5-flash-lite", retry_options=retry_config ) )

career_runner = InMemoryRunner(agent=career_agent)

events = await career_runner.run_debug( "Hello my name is Jeff and here is my resume text: I know Python and SQL. I have 3 years of experience in data analysis. " "Based on this, what roles fit me?" )

Suppose career_agent has a 'match_jobs' function
response = await career_runner.run_debug({ "resume_text": "I know Python and SQL and have 3 years of experience in data analysis." })

#final_text = response.get_final_text() #print(final_text)

collect all text parts
#final_text = "" #for event in events: #for part in getattr(event.content, "parts", []): #if hasattr(part, "text"): #final_text += part.text

#print(final_text)

add Codeadd Markdown
#checking Agents' forgetfulness**
add Codeadd Markdown
add Codeadd Markdown
response = await career_runner.run_debug({
    "What did I ask you earlier and what is my name?"
})

add Codeadd Markdown
# Define helper functions that will be reused throughout the notebook

from google.adk.runners import Runner

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
add Codeadd Markdown
Persistent Memory Session Created
add Codeadd Markdown

from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
MODEL_NAME = "career_coach_agent"

root_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="career_agent",
    description="A career_agent with persistent memory",
)

APP_NAME = "career_coach_app"
USER_ID = "user_123"

# Step 1: Use SQLite for persistent sessions
db_url = "sqlite:///career_agent_memory.db"
session_service = DatabaseSessionService(db_url=db_url)

# Step 2: Create runner with persistent memory
runner = Runner(
    agent=root_agent,   # <-- use your existing career agent
    app_name=APP_NAME,
    session_service=session_service
)

print("âœ… Persistent memory enabled!")
print("   - Database:", "career_agent_memory.db")
print("   - Sessions survive kernel restarts and machine reboots!")

add Codeadd Markdown
await run_session(
    runner,["My name is Jeff and here is my resume:Python,SQL,1 year experience."],"test-db-session-01")




add Codeadd Markdown

await run_session(
    runner,
   [
       "Here is my resume: Python, SQL, 1 year experience."
        "What is my name?"],
    "test-db-session-01",
)

add Codeadd Markdown
await run_session(
    runner,
   [
       "Here is my resume: Python, SQL, 1 year experience."
        "What is my name?"],
    "test-db-session-02",
)
add Codeadd Markdown
##Context Engineering(Context Compaction)###
add Codeadd Markdown
# Re-define our app with Events Compaction enabled
from google.adk.apps.app import App, EventsCompactionConfig
research_app_compacting = App(
    name="research_app_compacting",
    root_agent=root_agent,

    events_compaction_config=EventsCompactionConfig(
        compaction_interval=2,  # Trigger compaction every 2 invocations
        overlap_size=1,  # Keep 1 previous turn for context
    ),
)

db_url = "sqlite:///my_agent_data.db"  # Local SQLite file
session_service = DatabaseSessionService(db_url=db_url)

# Create a new runner for the upgraded app
research_runner_compacting = Runner(
    app=research_app_compacting, session_service=session_service
)


print("âœ… Research App upgraded with Events Compaction!")
add Codeadd Markdown
# Turn 1
await run_session(
    research_runner_compacting,
    "What is the latest news about AI jobs in healthcare?",
    "compaction_demo",
)

# Turn 2
await run_session(
    research_runner_compacting,
    "Are there any new developments in AI jobs?",
    "compaction_demo",
)

# Turn 3 - Compaction should trigger after this turn!
await run_session(
    research_runner_compacting,
    "Tell me more about the second development you found.",
    "compaction_demo",
)


add Codeadd Markdown
# Get the final session state
final_session = await session_service.get_session(
    app_name=research_runner_compacting.app_name,
    user_id=USER_ID,
    session_id="compaction_demo",
)

print("--- Searching for Compaction Summary Event ---")
found_summary = False
for event in final_session.events:
    # Compaction events have a 'compaction' attribute
    if event.actions and event.actions.compaction:
        print("\nâœ… SUCCESS! Found the Compaction Event:")
        print(f"  Author: {event.author}")
        print(f"\n Compacted information: {event}")
        found_summary = True
        break

if not found_summary:
    print(
        "\nâ�Œ No compaction event found. Try increasing the number of turns in the demo."
    )
add Codeadd Markdown
##Agent Obsevability with built-in plugins###
add Codeadd Markdown
print("----- EXAMPLE PLUGIN - DOES NOTHING ----- ")

import logging
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.plugins.base_plugin import BasePlugin


# Applies to all agent and model calls
class CountInvocationPlugin(BasePlugin):
    """A custom plugin that counts agent and tool invocations."""

    def __init__(self) -> None:
        """Initialize the plugin with counters."""
        super().__init__(name="count_invocation")
        self.agent_count: int = 0
        self.tool_count: int = 0
        self.llm_request_count: int = 0

    # Callback 1: Runs before an agent is called. You can add any custom logic here.
    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> None:
        """Count agent runs."""
        self.agent_count += 1
        logging.info(f"[Plugin] Agent run count: {self.agent_count}")

    # Callback 2: Runs before a model is called. You can add any custom logic here.
    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        """Count LLM requests."""
        self.llm_request_count += 1
        logging.info(f"[Plugin] LLM request count: {self.llm_request_count}")
add Codeadd Markdown
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.runners import InMemoryRunner # Already imported in cell [21]

# 1. You no longer need to define 'planner_agent_with_plugin'.
# 2. Define a NEW runner that takes the original agent and the plugin.
from google.genai import types
#import asyncio

runner = InMemoryRunner(
    agent=planner_agent, # Use the original agent
    plugins=[
        LoggingPlugin() # Pass the plugin into the runner
    ]
)
print("âœ… Runner configured")
add Codeadd Markdown
print("ğŸš€ Running agent with LoggingPlugin...")
print("ğŸ“Š Watch the comprehensive logging output below:\n")

response = await runner.run_debug("Tell me more about the second development you found.")
add Codeadd Markdown

