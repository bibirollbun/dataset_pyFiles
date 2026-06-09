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


 This Python 3 environment comes with many helpful analytics libraries installed
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
âœ… Gemini API key setup complete.
#>> Gemini API key setup complete.**

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
retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)
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
runner = InMemoryRunner(agent=root_agent)

print("âœ… Runner created.")
âœ… Runner created.
response = await runner.run_debug(
    "What careers are in demand?"
)
 ### Created new session: debug_session_id

User > What careers are in demand?
career_coach > The job market in 2025 is experiencing significant shifts driven by technological advancements, particularly in AI and data science, alongside evolving demographic and economic factors. Many of the most in-demand careers are concentrated in technology, healthcare, and roles requiring specialized analytical and creative skills.

Here are some of the careers in high demand for 2025:

**Technology and Data Science:**
*   **AI and Machine Learning Specialists:** Professionals who develop, train, and deploy AI and machine learning solutions are crucial across various sectors. The field is experiencing immense growth, with projections indicating a significant increase in job opportunities driven by AI.
*   **Data Scientists and Analysts:** These roles are vital for analyzing raw data to predict future trends, find innovative solutions, and enable data-driven decision-making.
*   **Software Engineers and Developers:** With nearly every industry undergoing digital transformation, the demand for coders, app developers, and cloud engineers continues to surge.
*   **Cybersecurity Analysts and Engineers:** Increased cyber threats and regulatory environments are driving a strong demand for professionals who can protect data and systems.
*   **Computer and Information Research Scientists:** These professionals explore new technological solutions and work with cutting-edge technologies like AI and quantum computing.
*   **Computer Network Architects:** Designing, maintaining, and optimizing data communication systems remains critical for organizations.
*   **DevOps Engineers:** These professionals bridge the gap between development and operations, ensuring smooth progress and efficiency.

**Healthcare and Care Economy:**
*   **Registered Nurses, Nurse Practitioners, and Health Technicians:** An aging population and persistent staff shortages are fueling a high demand in healthcare.
*   **Medical and Health Services Managers:** These roles are projected to see substantial job growth.
*   **Mental Health Professionals:** Demand is increasing significantly due to growing awareness.

**Business and Finance:**
*   **Sales Representatives and Managers:** Businesses are focusing on growth, making these roles critical, and they are increasingly requiring data fluency and tech adoption.
*   **Financial Managers:** These positions remain in demand as companies navigate complex financial landscapes.
*   **Accountants and Bookkeepers:** While bookkeeping is seeing some automation, opportunities still exist, and accountants remain in demand for their financial expertise.
*   **Business Development Professionals:** These roles are essential for driving expansion and capitalizing on new growth avenues.

**Other In-Demand Fields:**
*   **Digital Marketing Professionals:** With expanding digital markets, demand for specialists in this area is increasing.
*   **Project Management:** Companies are navigating complex work models and global teams, making project managers highly sought after.
*   **Skilled Trades:** Careers such as electricians, plumbers, and heavy equipment operators are seeing renewed interest, especially in infrastructure, logistics, and clean energy.
*   **Content Creators and Marketers:** The digital content creation market is a significant force, with demand for these roles continuing to grow.

Key skills that will be highly valued across many of these roles include analytical thinking, problem-solving, creativity, digital literacy, adaptability, and a commitment to lifelong learning. Many sources emphasize the growing importance of AI and big data skills.
!adk create sample-agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY

Agent created in /kaggle/working/sample-agent:
- .env
- __init__.py
- agent.py

url_prefix = get_adk_proxy_url()
âš ï¸� IMPORTANT: Action Required
The ADK web UI is not running yet. You must start it in the next cell.
Run the next cell (the one with !adk web ...) to start the ADK web UI.
Wait for that cell to show it is "Running" (it will not "complete").
Once it's running, return to this button and click it to open the UI.
(If you click the button before running the next cell, you will get a 500 error.)
Open ADK Web UI (after running cell below) â†—
!adk web --url_prefix {url_prefix}
/usr/local/lib/python3.11/dist-packages/google/adk/cli/fast_api.py:130: UserWarning: [EXPERIMENTAL] InMemoryCredentialService: This feature is experimental and may change or be removed in future versions without notice. It may introduce breaking changes at any time.
  credential_service = InMemoryCredentialService()
/usr/local/lib/python3.11/dist-packages/google/adk/auth/credential_service/in_memory_credential_service.py:33: UserWarning: [EXPERIMENTAL] BaseCredentialService: This feature is experimental and may change or be removed in future versions without notice. It may introduce breaking changes at any time.
  super().__init__()
INFO:     Started server process [95]
INFO:     Waiting for application startup.

+-----------------------------------------------------------------------------+
| ADK Web Server started                                                      |
|                                                                             |
| For local testing, access at http://127.0.0.1:8000.                         |
+-----------------------------------------------------------------------------+

INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
^C
INFO:     Shutting down
INFO:     Waiting for application shutdown.

+-----------------------------------------------------------------------------+
| ADK Web Server shutting down...                                             |
+-----------------------------------------------------------------------------+

INFO:     Application shutdown complete.
INFO:     Finished server process [95]

Aborted!
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
retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)
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
parse_resume function created
analyze_skill_gap function created
match_jobs function created
Test:[{'title': 'Data Analyst', 'fit': 90}, {'title': 'ML Intern', 'fit': 77}]
â€œorchestratorâ€� / planner agent that routes between multiple sub-agents.**
!adk create planner_agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY

Agent created in /kaggle/working/planner_agent:
- .env
- __init__.py
- agent.py

# === Planner + Specialist Agents for Career Coach ===

from google.adk.agents import LlmAgent
# Robust import for AgentTool (depends on ADK version)
try:
    from google.adk.tools.agent_tool import AgentTool
except ImportError:
    from google.adk.tools import AgentTool  # falls back if re-exported

# 1. Specialist agents (they focus on one thing each)

resume_critique_agent = LlmAgent(
    name="resume_critique_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="Reviews and critiques resumes and LinkedIn-style profiles.",
    instruction=(
        "You are a strict but constructive resume reviewer. "
        "When the user provides raw resume text, first call the parse_resume tool "
        "to extract structured information. Then: "
        "1) Highlight strengths, 2) Point out weaknesses, 3) Suggest concrete edits, "
        "and 4) Optionally rewrite key bullet points or summary."
    ),
    tools=[parse_resume],
)

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

# 4. Simple runner for the planner agent (in-memory; no DB)
planner_runner = InMemoryRunner(agent=planner_agent)

print("âœ… Planner agent and specialist agents created.")
âœ… Planner agent and specialist agents created.
url_prefix = get_adk_proxy_url()
âš ï¸� IMPORTANT: Action Required
The ADK web UI is not running yet. You must start it in the next cell.
Run the next cell (the one with !adk web ...) to start the ADK web UI.
Wait for that cell to show it is "Running" (it will not "complete").
Once it's running, return to this button and click it to open the UI.
(If you click the button before running the next cell, you will get a 500 error.)
Open ADK Web UI (after running cell below) â†—
!adk web --log_level DEBUG --url_prefix {url_prefix}
/usr/local/lib/python3.11/dist-packages/google/adk/cli/fast_api.py:130: UserWarning: [EXPERIMENTAL] InMemoryCredentialService: This feature is experimental and may change or be removed in future versions without notice. It may introduce breaking changes at any time.
  credential_service = InMemoryCredentialService()
/usr/local/lib/python3.11/dist-packages/google/adk/auth/credential_service/in_memory_credential_service.py:33: UserWarning: [EXPERIMENTAL] BaseCredentialService: This feature is experimental and may change or be removed in future versions without notice. It may introduce breaking changes at any time.
  super().__init__()
INFO:     Started server process [111]
INFO:     Waiting for application startup.

+-----------------------------------------------------------------------------+
| ADK Web Server started                                                      |
|                                                                             |
| For local testing, access at http://127.0.0.1:8000.                         |
+-----------------------------------------------------------------------------+

INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     35.191.73.201:0 - "GET / HTTP/1.1" 307 Temporary Redirect
INFO:     35.191.73.200:0 - "GET /dev-ui/ HTTP/1.1" 200 OK
INFO:     35.191.73.201:0 - "GET /dev-ui/chunk-2WH2EVR6.js HTTP/1.1" 200 OK
INFO:     35.191.73.200:0 - "GET /dev-ui/main-OS2OH2S3.js HTTP/1.1" 200 OK
INFO:     35.191.73.202:0 - "GET /dev-ui/styles-EVMPSV3U.css HTTP/1.1" 200 OK
INFO:     35.191.73.203:0 - "GET /dev-ui/polyfills-B6TNHZQ6.js HTTP/1.1" 200 OK
INFO:     35.191.73.201:0 - "GET /dev-ui/assets/config/runtime-config.json HTTP/1.1" 200 OK
INFO:     35.191.73.203:0 - "GET /list-apps?relative_path=./ HTTP/1.1" 200 OK
INFO:     35.191.73.201:0 - "GET /dev-ui/assets/ADK-512-color.svg HTTP/1.1" 200 OK
INFO:     35.191.73.203:0 - "GET /dev-ui/adk_favicon.svg HTTP/1.1" 200 OK
INFO:google_adk.google.adk.cli.adk_web_server:New session created: 0b4b3b86-47ed-4cf0-8f6b-aaa482337ff8
INFO:     35.191.73.202:0 - "POST /apps/planner_agent/users/user/sessions HTTP/1.1" 200 OK
INFO:     35.191.73.201:0 - "GET /builder/app/planner_agent?ts=1764574750994 HTTP/1.1" 200 OK
INFO:     35.191.73.202:0 - "GET /apps/planner_agent/users/user/sessions HTTP/1.1" 200 OK
INFO:     35.191.73.200:0 - "GET /apps/planner_agent/users/user/sessions/0b4b3b86-47ed-4cf0-8f6b-aaa482337ff8 HTTP/1.1" 200 OK
INFO:     35.191.73.202:0 - "GET /debug/trace/session/0b4b3b86-47ed-4cf0-8f6b-aaa482337ff8 HTTP/1.1" 200 OK
INFO:     35.191.73.202:0 - "GET /apps/planner_agent/users/user/sessions HTTP/1.1" 200 OK
/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'alias' attribute with value 'appName' was provided to the `Field()` function, which has no effect in the context it was used. 'alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'validation_alias' attribute with value 'appName' was provided to the `Field()` function, which has no effect in the context it was used. 'validation_alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'serialization_alias' attribute with value 'appName' was provided to the `Field()` function, which has no effect in the context it was used. 'serialization_alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'alias' attribute with value 'userId' was provided to the `Field()` function, which has no effect in the context it was used. 'alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'validation_alias' attribute with value 'userId' was provided to the `Field()` function, which has no effect in the context it was used. 'validation_alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'serialization_alias' attribute with value 'userId' was provided to the `Field()` function, which has no effect in the context it was used. 'serialization_alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'alias' attribute with value 'sessionId' was provided to the `Field()` function, which has no effect in the context it was used. 'alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'validation_alias' attribute with value 'sessionId' was provided to the `Field()` function, which has no effect in the context it was used. 'validation_alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'serialization_alias' attribute with value 'sessionId' was provided to the `Field()` function, which has no effect in the context it was used. 'serialization_alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'alias' attribute with value 'newMessage' was provided to the `Field()` function, which has no effect in the context it was used. 'alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'validation_alias' attribute with value 'newMessage' was provided to the `Field()` function, which has no effect in the context it was used. 'validation_alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'serialization_alias' attribute with value 'newMessage' was provided to the `Field()` function, which has no effect in the context it was used. 'serialization_alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'alias' attribute with value 'streaming' was provided to the `Field()` function, which has no effect in the context it was used. 'alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'validation_alias' attribute with value 'streaming' was provided to the `Field()` function, which has no effect in the context it was used. 'validation_alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'serialization_alias' attribute with value 'streaming' was provided to the `Field()` function, which has no effect in the context it was used. 'serialization_alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'alias' attribute with value 'stateDelta' was provided to the `Field()` function, which has no effect in the context it was used. 'alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'validation_alias' attribute with value 'stateDelta' was provided to the `Field()` function, which has no effect in the context it was used. 'validation_alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'serialization_alias' attribute with value 'stateDelta' was provided to the `Field()` function, which has no effect in the context it was used. 'serialization_alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'alias' attribute with value 'invocationId' was provided to the `Field()` function, which has no effect in the context it was used. 'alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'validation_alias' attribute with value 'invocationId' was provided to the `Field()` function, which has no effect in the context it was used. 'validation_alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'serialization_alias' attribute with value 'invocationId' was provided to the `Field()` function, which has no effect in the context it was used. 'serialization_alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
INFO:     35.191.73.171:0 - "POST /run_sse HTTP/1.1" 200 OK
DEBUG:google_adk.google.adk.cli.utils.agent_loader:Loading agent planner_agent - not in cache.
DEBUG:google_adk.google.adk.cli.utils.agent_loader:Loading .env for agent planner_agent from /kaggle/working
DEBUG:google_adk.google.adk.cli.utils.agent_loader:Module planner_agent has no root_agent. Trying next pattern.
INFO:google_adk.google.adk.cli.utils.agent_loader:Found root_agent in planner_agent.agent
INFO:google_adk.google.adk.models.google_llm:Sending out request, model: gemini-2.5-flash-lite, backend: GoogleLLMVariant.GEMINI_API, stream: False
DEBUG:google_adk.google.adk.models.google_llm:
LLM Request:
-----------------------------------------------------------
System Instruction:
Answer user questions to the best of your knowledge

You are an agent. Your internal name is "root_agent". The description about you is "A helpful assistant for user questions.".
-----------------------------------------------------------
Config:
{}
-----------------------------------------------------------
Contents:
{"parts":[{"text":"Help me choose a career path. I havea MSBA degree."}],"role":"user"}
-----------------------------------------------------------
Functions:

-----------------------------------------------------------

INFO:google_adk.google.adk.models.google_llm:Response received from the model.
DEBUG:google_adk.google.adk.models.google_llm:
LLM Response:
-----------------------------------------------------------
Text:
That's fantastic! An MSBA (Master of Science in Business Analytics) degree opens up a wide range of exciting career paths. To help you choose, let's explore some options based on your interests and strengths.

First, tell me a little more about yourself:

**1. What aspects of your MSBA program did you enjoy the most?**
*   Were you more drawn to the technical side (programming, statistical modeling, machine learning)?
*   Did you prefer the business application and strategy side (understanding business problems, communicating insights, stakeholder management)?
*   Did you enjoy data visualization and storytelling?
*   Was it the problem-solving and analytical thinking involved?

**2. What kind of work environment are you looking for?**
*   Large corporation vs. startup?
*   Fast-paced and dynamic vs. more structured and stable?
*   Remote, hybrid, or in-office?
*   Industry preference (e.g., tech, finance, healthcare, retail, consulting)?

**3. What are your long-term career aspirations?**
*   Do you want to be a hands-on individual contributor, a team lead, or a manager?
*   Are you aiming for a specialized technical role or a more strategic, business-focused role?
*   What kind of impact do you want to make?

**4. What are your existing skills and experiences (even outside your MSBA)?**
*   Any programming languages you're proficient in (Python, R, SQL)?
*   Experience with specific analytics tools (Tableau, Power BI, Excel, etc.)?
*   Domain knowledge from previous work or internships?
*   Soft skills like communication, leadership, or problem-solving?

**In the meantime, here are some common and high-demand career paths for MSBA graduates:**

**A. Data & Analytics Focused Roles:**

*   **Data Scientist:** This is a broad role that often involves building predictive models, machine learning algorithms, and conducting complex statistical analysis to solve business problems. Requires strong programming and statistical skills.
*   **Data Analyst:** Focuses on collecting, cleaning, analyzing, and interpreting data to identify trends, patterns, and insights. Often involves creating reports and dashboards.
*   **Business Intelligence (BI) Analyst/Developer:** Specializes in developing and managing BI tools and dashboards to help businesses make data-driven decisions. Strong focus on data visualization and reporting.
*   **Machine Learning Engineer:** Focuses on building, deploying, and maintaining machine learning models in production environments. Requires strong software engineering and ML skills.
*   **AI Engineer:** Similar to ML Engineer but with a broader focus on artificial intelligence technologies.

**B. Business Strategy & Operations Focused Roles:**

*   **Business Analyst:** Bridges the gap between business needs and technical solutions. They gather requirements, analyze processes, and recommend improvements, often using data to support their recommendations.
*   **Product Analyst:** Analyzes product usage data to understand customer behavior, identify areas for improvement, and inform product development strategies.
*   **Marketing Analyst:** Uses data to measure the effectiveness of marketing campaigns, identify target audiences, and optimize marketing spend.
*   **Financial Analyst (with an analytics focus):** Uses data to forecast financial performance, analyze investments, and identify cost-saving opportunities.
*   **Operations Analyst:** Analyzes operational data to improve efficiency, reduce costs, and streamline processes.
*   **Strategy Analyst/Consultant:** Uses data and analytical frameworks to help organizations develop and implement business strategies.

**C. Specialized Roles:**

*   **Data Engineer:** Focuses on building and maintaining the infrastructure and pipelines that collect, store, and process data, making it accessible for analysts and scientists.
*   **Risk Analyst:** Uses data to assess and mitigate financial, operational, or credit risks.
*   **Supply Chain Analyst:** Analyzes data related to supply chains to optimize inventory, logistics, and production.

**Once you provide more information about your preferences, I can give you more tailored advice!**
-----------------------------------------------------------
Function calls:

-----------------------------------------------------------
Raw response:
{"sdk_http_response":{"headers":{"Content-Type":"application/json; charset=UTF-8","Vary":"Origin, X-Origin, Referer","Content-Encoding":"gzip","Date":"Mon, 01 Dec 2025 07:40:05 GMT","Server":"scaffolding on HTTPServer2","X-XSS-Protection":"0","X-Frame-Options":"SAMEORIGIN","X-Content-Type-Options":"nosniff","Server-Timing":"gfet4t7; dur=5427","Transfer-Encoding":"chunked"}},"candidates":[{"content":{"parts":[{"text":"That's fantastic! An MSBA (Master of Science in Business Analytics) degree opens up a wide range of exciting career paths. To help you choose, let's explore some options based on your interests and strengths.\n\nFirst, tell me a little more about yourself:\n\n**1. What aspects of your MSBA program did you enjoy the most?**\n*   Were you more drawn to the technical side (programming, statistical modeling, machine learning)?\n*   Did you prefer the business application and strategy side (understanding business problems, communicating insights, stakeholder management)?\n*   Did you enjoy data visualization and storytelling?\n*   Was it the problem-solving and analytical thinking involved?\n\n**2. What kind of work environment are you looking for?**\n*   Large corporation vs. startup?\n*   Fast-paced and dynamic vs. more structured and stable?\n*   Remote, hybrid, or in-office?\n*   Industry preference (e.g., tech, finance, healthcare, retail, consulting)?\n\n**3. What are your long-term career aspirations?**\n*   Do you want to be a hands-on individual contributor, a team lead, or a manager?\n*   Are you aiming for a specialized technical role or a more strategic, business-focused role?\n*   What kind of impact do you want to make?\n\n**4. What are your existing skills and experiences (even outside your MSBA)?**\n*   Any programming languages you're proficient in (Python, R, SQL)?\n*   Experience with specific analytics tools (Tableau, Power BI, Excel, etc.)?\n*   Domain knowledge from previous work or internships?\n*   Soft skills like communication, leadership, or problem-solving?\n\n**In the meantime, here are some common and high-demand career paths for MSBA graduates:**\n\n**A. Data & Analytics Focused Roles:**\n\n*   **Data Scientist:** This is a broad role that often involves building predictive models, machine learning algorithms, and conducting complex statistical analysis to solve business problems. Requires strong programming and statistical skills.\n*   **Data Analyst:** Focuses on collecting, cleaning, analyzing, and interpreting data to identify trends, patterns, and insights. Often involves creating reports and dashboards.\n*   **Business Intelligence (BI) Analyst/Developer:** Specializes in developing and managing BI tools and dashboards to help businesses make data-driven decisions. Strong focus on data visualization and reporting.\n*   **Machine Learning Engineer:** Focuses on building, deploying, and maintaining machine learning models in production environments. Requires strong software engineering and ML skills.\n*   **AI Engineer:** Similar to ML Engineer but with a broader focus on artificial intelligence technologies.\n\n**B. Business Strategy & Operations Focused Roles:**\n\n*   **Business Analyst:** Bridges the gap between business needs and technical solutions. They gather requirements, analyze processes, and recommend improvements, often using data to support their recommendations.\n*   **Product Analyst:** Analyzes product usage data to understand customer behavior, identify areas for improvement, and inform product development strategies.\n*   **Marketing Analyst:** Uses data to measure the effectiveness of marketing campaigns, identify target audiences, and optimize marketing spend.\n*   **Financial Analyst (with an analytics focus):** Uses data to forecast financial performance, analyze investments, and identify cost-saving opportunities.\n*   **Operations Analyst:** Analyzes operational data to improve efficiency, reduce costs, and streamline processes.\n*   **Strategy Analyst/Consultant:** Uses data and analytical frameworks to help organizations develop and implement business strategies.\n\n**C. Specialized Roles:**\n\n*   **Data Engineer:** Focuses on building and maintaining the infrastructure and pipelines that collect, store, and process data, making it accessible for analysts and scientists.\n*   **Risk Analyst:** Uses data to assess and mitigate financial, operational, or credit risks.\n*   **Supply Chain Analyst:** Analyzes data related to supply chains to optimize inventory, logistics, and production.\n\n**Once you provide more information about your preferences, I can give you more tailored advice!**"}],"role":"model"},"finish_reason":"STOP","index":0}],"model_version":"gemini-2.5-flash-lite","response_id":"VUYtaZ2tBsusz7IP3-vx4A4","usage_metadata":{"candidates_token_count":849,"prompt_token_count":53,"prompt_tokens_details":[{"modality":"TEXT","token_count":53}],"total_token_count":902},"automatic_function_calling_history":[]}
-----------------------------------------------------------

DEBUG:google_adk.google.adk.cli.adk_web_server:Generated event in agent run streaming: {"modelVersion":"gemini-2.5-flash-lite","content":{"parts":[{"text":"That's fantastic! An MSBA (Master of Science in Business Analytics) degree opens up a wide range of exciting career paths. To help you choose, let's explore some options based on your interests and strengths.\n\nFirst, tell me a little more about yourself:\n\n**1. What aspects of your MSBA program did you enjoy the most?**\n*   Were you more drawn to the technical side (programming, statistical modeling, machine learning)?\n*   Did you prefer the business application and strategy side (understanding business problems, communicating insights, stakeholder management)?\n*   Did you enjoy data visualization and storytelling?\n*   Was it the problem-solving and analytical thinking involved?\n\n**2. What kind of work environment are you looking for?**\n*   Large corporation vs. startup?\n*   Fast-paced and dynamic vs. more structured and stable?\n*   Remote, hybrid, or in-office?\n*   Industry preference (e.g., tech, finance, healthcare, retail, consulting)?\n\n**3. What are your long-term career aspirations?**\n*   Do you want to be a hands-on individual contributor, a team lead, or a manager?\n*   Are you aiming for a specialized technical role or a more strategic, business-focused role?\n*   What kind of impact do you want to make?\n\n**4. What are your existing skills and experiences (even outside your MSBA)?**\n*   Any programming languages you're proficient in (Python, R, SQL)?\n*   Experience with specific analytics tools (Tableau, Power BI, Excel, etc.)?\n*   Domain knowledge from previous work or internships?\n*   Soft skills like communication, leadership, or problem-solving?\n\n**In the meantime, here are some common and high-demand career paths for MSBA graduates:**\n\n**A. Data & Analytics Focused Roles:**\n\n*   **Data Scientist:** This is a broad role that often involves building predictive models, machine learning algorithms, and conducting complex statistical analysis to solve business problems. Requires strong programming and statistical skills.\n*   **Data Analyst:** Focuses on collecting, cleaning, analyzing, and interpreting data to identify trends, patterns, and insights. Often involves creating reports and dashboards.\n*   **Business Intelligence (BI) Analyst/Developer:** Specializes in developing and managing BI tools and dashboards to help businesses make data-driven decisions. Strong focus on data visualization and reporting.\n*   **Machine Learning Engineer:** Focuses on building, deploying, and maintaining machine learning models in production environments. Requires strong software engineering and ML skills.\n*   **AI Engineer:** Similar to ML Engineer but with a broader focus on artificial intelligence technologies.\n\n**B. Business Strategy & Operations Focused Roles:**\n\n*   **Business Analyst:** Bridges the gap between business needs and technical solutions. They gather requirements, analyze processes, and recommend improvements, often using data to support their recommendations.\n*   **Product Analyst:** Analyzes product usage data to understand customer behavior, identify areas for improvement, and inform product development strategies.\n*   **Marketing Analyst:** Uses data to measure the effectiveness of marketing campaigns, identify target audiences, and optimize marketing spend.\n*   **Financial Analyst (with an analytics focus):** Uses data to forecast financial performance, analyze investments, and identify cost-saving opportunities.\n*   **Operations Analyst:** Analyzes operational data to improve efficiency, reduce costs, and streamline processes.\n*   **Strategy Analyst/Consultant:** Uses data and analytical frameworks to help organizations develop and implement business strategies.\n\n**C. Specialized Roles:**\n\n*   **Data Engineer:** Focuses on building and maintaining the infrastructure and pipelines that collect, store, and process data, making it accessible for analysts and scientists.\n*   **Risk Analyst:** Uses data to assess and mitigate financial, operational, or credit risks.\n*   **Supply Chain Analyst:** Analyzes data related to supply chains to optimize inventory, logistics, and production.\n\n**Once you provide more information about your preferences, I can give you more tailored advice!**"}],"role":"model"},"finishReason":"STOP","usageMetadata":{"candidatesTokenCount":849,"promptTokenCount":53,"promptTokensDetails":[{"modality":"TEXT","tokenCount":53}],"totalTokenCount":902},"invocationId":"e-7f445caa-ad0b-41e6-bdde-ed59bfc2df56","author":"root_agent","actions":{"stateDelta":{},"artifactDelta":{},"requestedAuthConfigs":{},"requestedToolConfirmations":{}},"id":"2b684b7c-b446-4445-abc2-97c49822d6f3","timestamp":1764574799.512052}
INFO:     35.191.73.168:0 - "GET /apps/planner_agent/users/user/sessions/0b4b3b86-47ed-4cf0-8f6b-aaa482337ff8 HTTP/1.1" 200 OK
INFO:     35.191.73.170:0 - "GET /debug/trace/session/0b4b3b86-47ed-4cf0-8f6b-aaa482337ff8 HTTP/1.1" 200 OK
INFO:     35.191.73.171:0 - "GET /debug/trace/session/0b4b3b86-47ed-4cf0-8f6b-aaa482337ff8 HTTP/1.1" 200 OK
INFO:     35.191.73.170:0 - "GET /debug/trace/2b684b7c-b446-4445-abc2-97c49822d6f3 HTTP/1.1" 200 OK
DEBUG:google_adk.google.adk.cli.utils.agent_loader:Returning cached agent for planner_agent (async)
INFO:     35.191.73.169:0 - "GET /apps/planner_agent/users/user/sessions/0b4b3b86-47ed-4cf0-8f6b-aaa482337ff8/events/2b684b7c-b446-4445-abc2-97c49822d6f3/graph HTTP/1.1" 200 OK
^C
INFO:     Shutting down
INFO:     Waiting for application shutdown.

+-----------------------------------------------------------------------------+
| ADK Web Server shutting down...                                             |
+-----------------------------------------------------------------------------+

INFO:     Application shutdown complete.
INFO:     Finished server process [111]

Aborted!
#Stateful Agent Creation
#In ADK, tools are passed as a list of Python functions to the Agent constructor.


career_agent = LlmAgent(
    name="career_agent",  # must be valid identifier (no spaces)
    description="Agentic career coach with resume analysis, skill-gap analysis, and job matching",
    tools=[parse_resume, analyze_skill_gap, match_jobs],
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    )
)

career_runner = InMemoryRunner(agent=career_agent)

events = await career_runner.run_debug(
    "Hello my name is Jeff and here is my resume text: I know Python and SQL. I have 3 years of experience in data analysis. "
    "Based on this, what roles fit me?"
)

# Suppose career_agent has a 'match_jobs' function
response = await career_runner.run_debug({
    "resume_text": "I know Python and SQL and have 3 years of experience in data analysis."
})



#final_text = response.get_final_text()
#print(final_text)

# collect all text parts
#final_text = ""
#for event in events:
    #for part in getattr(event.content, "parts", []):
        #if hasattr(part, "text"):
            #final_text += part.text

#print(final_text)
 ### Created new session: debug_session_id

User > Hello my name is Jeff and here is my resume text: I know Python and SQL. I have 3 years of experience in data analysis. Based on this, what roles fit me?
WARNING:google_genai.types:Warning: there are non-text parts in the response: ['function_call'], returning concatenated text result from text parts. Check the full candidates.content.parts accessor to get the full model response.
WARNING:google_genai.types:Warning: there are non-text parts in the response: ['function_call'], returning concatenated text result from text parts. Check the full candidates.content.parts accessor to get the full model response.
career_agent > I've analyzed your resume and found that the roles of "Data Analyst" (90% fit) and "ML Intern" (77% fit) are a good match for your skills in Python, SQL, and data analysis.

 ### Continue session: debug_session_id

User > resume_text
career_agent > Please provide me with your resume text so I can analyze it and suggest suitable job roles.
#checking Agents' forgetfulness**
response = await career_runner.run_debug({
    "What did I ask you earlier and what is my name?"
})
 ### Continue session: debug_session_id

User > What did I ask you earlier and what is my name?
career_agent > You asked me earlier what roles fit you based on your resume text, stating that you know Python and SQL and have 3 years of experience in data analysis. You also mentioned that your name is Jeff.
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
âœ… Helper functions defined.
Persistent Memory Session Created
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
âœ… Persistent memory enabled!
   - Database: career_agent_memory.db
   - Sessions survive kernel restarts and machine reboots!
await run_session(
    runner,["My name is Jeff and here is my resume:Python,SQL,1 year experience."],"test-db-session-01")
 ### Session: test-db-session-01

User > My name is Jeff and here is my resume:Python,SQL,1 year experience.
career_coach_agent >  Hi Jeff! It's great to meet you. I've noted your resume highlights: **Python** and **SQL** with **1 year of experience**.

How can I help you with your career today? Are you looking for job opportunities, resume feedback, interview preparation, or something else?
await run_session(
    runner,
   [
       "Here is my resume: Python, SQL, 1 year experience."
        "What is my name?"],
    "test-db-session-01",
)
 ### Session: test-db-session-01

User > Here is my resume: Python, SQL, 1 year experience.What is my name?
career_coach_agent >  Your name is Jeff.
await run_session(
    runner,
   [
       "Here is my resume: Python, SQL, 1 year experience."
        "What is my name?"],
    "test-db-session-02",
)
##Context Engineering(Context Compaction)##
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
âœ… Research App upgraded with Events Compaction!
/tmp/ipykernel_47/3342429766.py:7: UserWarning: [EXPERIMENTAL] EventsCompactionConfig: This feature is experimental and may change or be removed in future versions without notice. It may introduce breaking changes at any time.
  events_compaction_config=EventsCompactionConfig(
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
 ### Session: compaction_demo

User > What is the latest news about AI jobs in healthcare?
career_coach_agent >  Here's a summary of the latest news and trends in AI jobs within the healthcare sector, drawing on recent reports and industry insights:

**Key Trends and Growth Areas:**

*   **Explosion in AI-Powered Diagnostics and Imaging:** This remains a major driver of AI job growth. Companies are actively hiring AI/ML engineers, data scientists, and computer vision specialists to develop and refine algorithms for:
    *   **Radiology:** Detecting anomalies in X-rays, CT scans, and MRIs (e.g., identifying cancerous tumors, early signs of stroke, diabetic retinopathy).
    *   **Pathology:** Analyzing digital pathology slides for more accurate and faster cancer diagnosis.
    *   **Dermatology:** AI tools for identifying skin lesions and potential melanoma.
*   **Personalized Medicine and Drug Discovery:** AI is revolutionizing how treatments are developed and tailored to individual patients. This is creating demand for:
    *   **Bioinformaticians and Computational Biologists:** To analyze vast genomic and proteomic datasets.
    *   **Data Scientists and ML Engineers:** To build predictive models for drug efficacy, patient response, and identifying new drug targets.
    *   **AI Researchers:** Focusing on areas like protein folding prediction (e.g., AlphaFold's impact) and molecular simulation.
*   **Operational Efficiency and Administrative Automation:** Healthcare systems are increasingly using AI to streamline operations and reduce administrative burdens. This means jobs in:
    *   **Natural Language Processing (NLP) Specialists:** For tasks like clinical note summarization, automated coding for billing, and improving patient communication platforms.
    *   **ML Engineers and Data Scientists:** To develop predictive models for patient flow, resource allocation, and fraud detection.
    *   **AI Product Managers:** To bridge the gap between technical development and clinical needs.
*   **Remote Patient Monitoring and Telehealth:** AI is enhancing the capabilities of remote monitoring devices and telehealth platforms, leading to jobs in:
    *   **Data Engineers:** To manage and process data from wearable devices and IoT sensors.
    *   **AI/ML Engineers:** To develop algorithms that can detect early warning signs of health deterioration or predict patient readmissions.
*   **Robotics in Healthcare:** While not solely AI, AI plays a crucial role in enabling advanced robotics for surgery, rehabilitation, and logistics within hospitals. This requires AI engineers with expertise in robotics and control systems.

**Specific Job Roles in Demand:**

*   **AI/ML Engineers:** Core to building, deploying, and maintaining AI models.
*   **Data Scientists:** Analyzing healthcare data to extract insights and inform model development.
*   **Computer Vision Engineers:** Specializing in image analysis for diagnostics.
*   **NLP Engineers/Specialists:** Working with text-based clinical data and patient interactions.
*   **Bioinformaticians/Computational Biologists:** Focusing on biological and genomic data for drug discovery and personalized medicine.
*   **AI Product Managers:** Translating clinical needs into AI solutions.
*   **AI Ethics and Governance Specialists:** As AI adoption grows, ensuring responsible and ethical deployment is becoming critical.
*   **Clinical Informaticists with AI Expertise:** Bridging the gap between clinical practice and AI technology.

**Recent Developments and News Highlights:**

*   **Increased Investment:** Venture capital funding continues to pour into health AI startups, fueling innovation and job creation. Major tech companies are also heavily investing in their healthcare AI divisions.
*   **FDA Approvals for AI Medical Devices:** Regulatory bodies like the FDA are approving more AI-powered medical devices, signaling maturity and growing acceptance of these technologies in clinical settings. This translates to more opportunities for developers and researchers.
*   **Focus on Explainable AI (XAI):** As AI becomes more integrated into critical healthcare decisions, there's a growing emphasis on developing AI models that are transparent and understandable to clinicians. This could lead to specialized roles in XAI development and validation.
*   **AI for Mental Health:** AI is being explored for applications in mental health, including early detection of conditions, personalized therapy recommendations, and AI-powered chatbots for support.
*   **The Rise of Generative AI in Healthcare:** While still nascent, generative AI (like LLMs) is showing promise in areas like drafting clinical documentation, generating synthetic data for research, and aiding in medical education. This will likely create new roles focused on fine-tuning and applying these models in healthcare contexts.

**Challenges and Considerations:**

*   **Data Privacy and Security:** Handling sensitive patient data requires robust security measures and compliance with regulations like HIPAA.
*   **Regulatory Hurdles:** Navigating the complex regulatory landscape for medical devices and AI applications can be challenging.
*   **Integration into Clinical Workflows:** Successfully integrating AI tools into existing hospital systems and clinical practices requires careful planning and change management.
*   **Talent Shortage:** There's a high demand for skilled AI professionals, and healthcare often competes with other industries for top talent.

**Where to Look for Opportunities:**

*   **Health Tech Startups:** Companies specifically focused on AI in healthcare.
*   **Large Technology Companies:** Google, Microsoft, Amazon, IBM, etc., all have significant healthcare AI divisions.
*   **Pharmaceutical and Biotech Companies:** For drug discovery and personalized medicine roles.
*   **Hospitals and Healthcare Systems:** Increasingly hiring in-house AI/data science teams for operational and clinical improvements.
*   **Research Institutions and Universities:** For cutting-edge research and development roles.

In summary, the field of AI jobs in healthcare is dynamic and rapidly expanding. The demand is strong for individuals with specialized AI/ML skills, data science expertise, and a good understanding of the healthcare domain.

 ### Session: compaction_demo

User > Are there any new developments in AI jobs?
career_coach_agent >  Yes, absolutely! The landscape of AI jobs is constantly evolving. Here are some of the most significant **new developments and emerging trends** in AI jobs beyond the general growth areas I mentioned previously:

**1. Generative AI Specialists (LLMs, Diffusion Models, etc.):**

*   **The Big Story:** The explosion of generative AI, particularly Large Language Models (LLMs) like GPT-3/4, Claude, and Llama, and image generation models (e.g., DALL-E, Midjourney), has created a massive demand for new roles.
*   **New Job Titles:**
    *   **Prompt Engineers:** Crafting effective prompts to get desired outputs from generative AI models.
    *   **AI Content Strategists/Creators:** Leveraging generative AI for marketing, content creation, and personalized communication.
    *   **LLM Fine-Tuning Specialists:** Adapting pre-trained LLMs for specific industry applications (e.g., legal, medical, coding).
    *   **Generative AI Researchers/Developers:** Pushing the boundaries of generative model capabilities.
    *   **AI Ethics and Safety Officers for Generative AI:** Focusing on mitigating risks like bias, misinformation, and misuse.
*   **Impact:** These roles are appearing across almost every industry, from tech and media to finance and healthcare.

**2. AI Ethics, Governance, and Responsible AI:**

*   **Growing Importance:** As AI becomes more pervasive, concerns about bias, fairness, transparency, privacy, and accountability are paramount. This is driving the creation of specialized roles.
*   **New Job Titles:**
    *   **AI Ethicist:** Developing ethical frameworks and guidelines for AI development and deployment.
    *   **AI Governance Manager:** Establishing policies and procedures to ensure compliance and responsible AI use.
    *   **Responsible AI Lead/Specialist:** Implementing strategies to build and deploy AI systems that are fair, transparent, and robust.
    *   **AI Bias Auditor:** Identifying and mitigating bias in AI algorithms and datasets.
*   **Context:** Companies are realizing that failing to address these issues can lead to reputational damage, legal challenges, and lack of public trust.

**3. AI for Cybersecurity:**

*   **Evolving Threat Landscape:** AI is being used both by cybercriminals and by defenders. This is creating a race for talent.
*   **New Job Titles:**
    *   **AI-Powered Threat Detection Engineers:** Developing AI systems to identify and neutralize cyber threats in real-time.
    *   **AI Security Analysts:** Using AI tools to analyze security logs and predict vulnerabilities.
    *   **Adversarial ML Researchers:** Studying how AI models can be attacked and developing defenses against such attacks.
*   **Demand:** The increasing sophistication of cyberattacks is making AI a critical component of modern cybersecurity strategies.

**4. AI in Robotics and Automation (Advanced Levels):**

*   **Beyond Basic Automation:** AI is enabling more sophisticated robots that can learn, adapt, and interact with complex environments.
*   **New Job Titles:**
    *   **Robotics Perception Engineers:** Developing AI systems for robots to "see" and understand their surroundings.
    *   **Reinforcement Learning Engineers for Robotics:** Training robots to perform complex tasks through trial and error.
    *   **Human-Robot Interaction Designers:** Creating intuitive and safe ways for humans to collaborate with robots.
*   **Applications:** Manufacturing, logistics, autonomous vehicles, and even elder care are seeing advancements.

**5. Specialized AI Model Optimization and Deployment:**

*   **Beyond Training:** While training models is crucial, getting them to run efficiently and reliably in real-world applications is another challenge.
*   **New Job Titles:**
    *   **MLOps (Machine Learning Operations) Engineers:** Streamlining the deployment, monitoring, and management of ML models in production. This field has matured significantly.
    *   **AI Performance Engineers:** Optimizing AI models for speed, memory usage, and power consumption, especially for edge devices.
    *   **Edge AI Engineers:** Developing and deploying AI models on devices with limited computing power (smartphones, IoT devices).

**6. AI for Scientific Discovery and Research Acceleration:**

*   **Beyond Healthcare:** While healthcare is a major area, AI is also transforming fundamental scientific research.
*   **New Job Titles:**
    *   **AI for Materials Science Engineers:** Using AI to discover new materials with specific properties.
    *   **AI for Climate Modeling Specialists:** Applying AI to understand and predict climate change.
    *   **AI for Astronomy/Physics Researchers:** Analyzing vast datasets from telescopes and experiments.
*   **Focus:** These roles often require a strong background in a specific scientific domain combined with AI expertise.

**Key Takeaways on New AI Job Developments:**

*   **Generative AI is the current disruptor, creating entirely new categories of roles.**
*   **Responsibility and ethics are no longer afterthoughts but are becoming core job functions.**
*   **Deployment and efficiency are gaining as much attention as model development.**
*   **AI is permeating more niche scientific and technical fields.**

This list isn't exhaustive, but it highlights some of the most significant shifts in the AI job market happening right now. The field is incredibly dynamic, so staying curious and adaptable is key!

 ### Session: compaction_demo

User > Tell me more about the second development you found.
career_coach_agent >  You're interested in the **AI Ethics, Governance, and Responsible AI** development! That's an excellent area to explore, as it's becoming increasingly critical for the sustainable and beneficial integration of AI.

Let me elaborate on why this is a significant development and what it entails:

**Why is AI Ethics, Governance, and Responsible AI a Major Development?**

Historically, the focus in AI was heavily on technical innovation â€“ making models smarter, faster, and more capable. However, as AI systems have moved from research labs into critical real-world applications (finance, hiring, criminal justice, healthcare, autonomous systems, content generation), the potential for negative societal impacts has become undeniable. These impacts include:

*   **Bias and Discrimination:** AI models can inadvertently perpetuate and even amplify existing societal biases present in the data they are trained on. This can lead to unfair outcomes in loan applications, job candidate screening, or even facial recognition systems.
*   **Lack of Transparency and Explainability:** "Black box" AI models can make decisions that are impossible for humans to understand or audit. This is problematic when life-altering decisions are being made.
*   **Privacy Violations:** AI systems often require vast amounts of data, raising concerns about how that data is collected, stored, used, and protected.
*   **Security Vulnerabilities:** AI systems themselves can be targets for malicious attacks, leading to data breaches or system failures.
*   **Misinformation and Manipulation:** Generative AI, in particular, can be used to create convincing fake content, leading to the spread of misinformation and manipulation.
*   **Job Displacement and Economic Inequality:** The automation driven by AI raises questions about its impact on employment and the distribution of wealth.
*   **Accountability:** When an AI system makes a mistake, who is responsible? The developer, the deployer, the user? Establishing clear lines of accountability is crucial.

These concerns are not just theoretical; they have real-world consequences. As a result, governments, regulatory bodies, industry leaders, and the public are demanding that AI be developed and deployed in a manner that is **safe, fair, and beneficial to society.** This demand is directly driving the creation and growth of jobs focused on these aspects.

**What Does This Development Mean in Terms of Jobs?**

It means a shift from purely technical roles to hybrid roles that require both technical understanding and a deep appreciation for humanistic principles, law, and societal impact.

Here's a deeper dive into the job titles and responsibilities:

*   **AI Ethicist:**
    *   **Focus:** Developing and advocating for ethical principles in AI design and deployment.
    *   **Responsibilities:**
        *   Conducting ethical risk assessments for new AI projects.
        *   Advising development teams on how to mitigate ethical risks.
        *   Staying abreast of evolving ethical guidelines and best practices.
        *   Facilitating discussions and training on AI ethics within an organization.
        *   Often requires a background in philosophy, ethics, law, or social sciences, coupled with an understanding of AI.

*   **AI Governance Manager/Specialist:**
    *   **Focus:** Establishing and enforcing policies, standards, and processes for AI development and use within an organization.
    *   **Responsibilities:**
        *   Developing AI policies that align with legal requirements (e.g., GDPR, emerging AI regulations) and company values.
        *   Creating frameworks for AI risk management and compliance.
        *   Ensuring that AI systems are documented and auditable.
        *   Working with legal, compliance, and IT departments.
        *   May have a background in law, compliance, risk management, or IT policy.

*   **Responsible AI Lead/Specialist:**
    *   **Focus:** The practical implementation of ethical principles and governance frameworks into the AI development lifecycle. This role is often more hands-on than a pure ethicist or governance manager.
    *   **Responsibilities:**
        *   Collaborating with data scientists and engineers to build fairness and transparency into models.
        *   Developing and implementing testing procedures to identify and mitigate bias.
        *   Ensuring that AI systems meet criteria for explainability and robustness.
        *   Overseeing the process of auditing AI systems.
        *   Requires a strong technical understanding of AI/ML, along with an ethical mindset.

*   **AI Bias Auditor:**
    *   **Focus:** Specifically identifying and quantifying biases within AI models and the data used to train them.
    *   **Responsibilities:**
        *   Designing and executing tests to detect disparate impact across different demographic groups.
        *   Analyzing model outputs for unfair patterns.
        *   Recommending strategies for bias mitigation, such as data rebalancing or algorithmic adjustments.
        *   Often requires strong statistical and analytical skills, in addition to AI knowledge.

*   **AI Safety Engineer/Researcher:**
    *   **Focus:** Ensuring that AI systems behave in a predictable and safe manner, especially in safety-critical applications (e.g., autonomous vehicles, robotics).
    *   **Responsibilities:**
        *   Developing techniques to prevent AI systems from causing unintended harm.
        *   Creating frameworks for verifying AI system safety.
        *   Researching methods to make AI systems more robust against adversarial attacks.
        *   Often a highly technical role requiring advanced ML and systems engineering skills.

**Where are these jobs being created?**

*   **Large Tech Companies:** Companies developing AI platforms and services (Google, Microsoft, Meta, Amazon, IBM) are heavily investing in these roles.
*   **Companies Deploying AI Extensively:** Financial institutions, major retailers, automotive manufacturers, and healthcare organizations are also building out these teams.
*   **Consulting Firms:** Many firms are offering AI ethics and responsible AI advisory services.
*   **Regulatory Bodies and Government Agencies:** As new AI regulations emerge, these bodies will need experts to enforce them.
*   **Research Labs and Academia:** Continuing to push the boundaries of what's possible in AI safety and fairness.

This development signifies a maturity in the AI field, acknowledging that technical prowess alone is insufficient. The future of AI will be shaped not just by its intelligence, but by its integrity.
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
--- Searching for Compaction Summary Event ---

âœ… SUCCESS! Found the Compaction Event:
  Author: user

 Compacted information: model_version=None content=None grounding_metadata=None partial=None turn_complete=None finish_reason=None error_code=None error_message=None interrupted=None custom_metadata=None usage_metadata=None live_session_resumption_update=None input_transcription=None output_transcription=None avg_logprobs=None logprobs_result=None cache_metadata=None citation_metadata=None invocation_id='6d6a4182-72d6-447e-b3b2-2293886bfd10' author='user' actions=EventActions(skip_summarization=None, state_delta={}, artifact_delta={}, transfer_to_agent=None, escalate=None, requested_auth_configs={}, requested_tool_confirmations={}, compaction={'start_timestamp': 1764574923.780925, 'end_timestamp': 1764574928.935069, 'compacted_content': {'parts': [{'function_call': None, 'code_execution_result': None, 'executable_code': None, 'file_data': None, 'function_response': None, 'inline_data': None, 'text': 'The conversation between the user and the AI agent has focused on the latest developments in AI jobs, particularly within the healthcare sector and more broadly across industries.\n\n**Key Information and Decisions:**\n\n*   **Healthcare AI Jobs:** The agent provided a comprehensive overview of AI job trends in healthcare, highlighting growth areas like AI-powered diagnostics/imaging, personalized medicine/drug discovery, operational efficiency, remote monitoring/telehealth, and robotics. Key roles in demand include AI/ML Engineers, Data Scientists, Computer Vision Engineers, NLP Specialists, Bioinformaticians, AI Product Managers, AI Ethics Specialists, and Clinical Informaticists with AI expertise. Recent developments like increased investment, FDA approvals, the focus on Explainable AI (XAI), AI for mental health, and the rise of Generative AI in healthcare were also discussed. Challenges such as data privacy, regulatory hurdles, integration issues, and talent shortages were identified.\n*   **Broader AI Job Developments:** The agent then detailed emerging trends in AI jobs across various industries. The most significant new development is the **explosion of Generative AI**, leading to new roles like Prompt Engineers, AI Content Strategists, LLM Fine-Tuning Specialists, and Generative AI Researchers. **AI Ethics, Governance, and Responsible AI** are also becoming critical, creating roles such as AI Ethicists and AI Governance Managers. The use of **AI in Cybersecurity** is expanding, with demand for AI-Powered Threat Detection Engineers and AI Security Analysts. **Advanced AI in Robotics and Automation** is creating roles for Robotics Perception Engineers and Reinforcement Learning Engineers. Furthermore, there\'s a growing emphasis on **Specialized AI Model Optimization and Deployment**, leading to more MLOps Engineers and AI Performance Engineers. Finally, **AI for Scientific Discovery** is creating specialized roles in fields like materials science and climate modeling.\n\n**Unresolved Questions or Tasks:**\n\nThere are no explicit unresolved questions or outstanding tasks from the user in this conversation history. The user\'s initial query was answered, and the subsequent question about "new developments in AI jobs" was also addressed comprehensively.', 'thought': None, 'thought_signature': None, 'video_metadata': None}], 'role': 'model'}}, end_of_agent=None, agent_state=None, rewind_before_invocation_id=None) long_running_tool_ids=set() branch=None id='714128fa-c730-470d-83cc-ea21314175cf' timestamp=1764574937.385732
##Agent Obsevability with built-in plugins###
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
----- EXAMPLE PLUGIN - DOES NOTHING ----- 
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
âœ… Runner configured
print("ğŸš€ Running agent with LoggingPlugin...")
print("ğŸ“Š Watch the comprehensive logging output below:\n")

response = await runner.run_debug("Tell me more about the second development you found.")
ğŸš€ Running agent with LoggingPlugin...
ğŸ“Š Watch the comprehensive logging output below:


 ### Created new session: debug_session_id

User > Tell me more about the second development you found.
[logging_plugin] ğŸš€ USER MESSAGE RECEIVED
[logging_plugin]    Invocation ID: e-689d59eb-3d5e-476f-81ce-6d6835ccfe75
[logging_plugin]    Session ID: debug_session_id
[logging_plugin]    User ID: debug_user_id
[logging_plugin]    App Name: InMemoryRunner
[logging_plugin]    Root Agent: planner_agent
[logging_plugin]    User Content: text: 'Tell me more about the second development you found.'
[logging_plugin] ğŸ�ƒ INVOCATION STARTING
[logging_plugin]    Invocation ID: e-689d59eb-3d5e-476f-81ce-6d6835ccfe75
[logging_plugin]    Starting Agent: planner_agent
[logging_plugin] ğŸ¤– AGENT STARTING
[logging_plugin]    Agent Name: planner_agent
[logging_plugin]    Invocation ID: e-689d59eb-3d5e-476f-81ce-6d6835ccfe75
[logging_plugin] ğŸ§  LLM REQUEST
[logging_plugin]    Model: gemini-2.5-flash-lite
[logging_plugin]    Agent: planner_agent
[logging_plugin]    System Instruction: 'You are the orchestration / planner agent in a career-coaching system.

Your job:
1. Read the user's request and decide which specialist tools to call:
   - resume_critique_agent (for resume/LinkedIn ...'
[logging_plugin]    Available Tools: ['resume_critique_agent', 'skill_assessment_agent', 'mock_interview_agent', 'progress_tracker_agent']
WARNING:google_genai.models:Tools at indices [0] are not compatible with automatic function calling (AFC). AFC is disabled. If AFC is intended, please include python callables in the tool list, and do not include function declaration in the tool list.
[logging_plugin] ğŸ§  LLM ERROR
[logging_plugin]    Agent: planner_agent
[logging_plugin]    Error: 400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': 'Tool use with function calling is unsupported', 'status': 'INVALID_ARGUMENT'}}
---------------------------------------------------------------------------
ClientError                               Traceback (most recent call last)
/tmp/ipykernel_47/2128565886.py in <cell line: 1>()
      2 print("ğŸ“Š Watch the comprehensive logging output below:\n")
      3 
----> 4 response = await runner.run_debug("Tell me more about the second development you found.")

/usr/local/lib/python3.11/dist-packages/google/adk/runners.py in run_debug(self, user_messages, user_id, session_id, run_config, quiet, verbose)
   1021         print(f'\nUser > {message}')
   1022 
-> 1023       async for event in self.run_async(
   1024           user_id=user_id,
   1025           session_id=session.id,

/usr/local/lib/python3.11/dist-packages/google/adk/runners.py in run_async(self, user_id, session_id, invocation_id, new_message, state_delta, run_config)
    441 
    442     async with Aclosing(_run_with_trace(new_message, invocation_id)) as agen:
--> 443       async for event in agen:
    444         yield event
    445 

/usr/local/lib/python3.11/dist-packages/google/adk/runners.py in _run_with_trace(new_message, invocation_id)
    425             )
    426         ) as agen:
--> 427           async for event in agen:
    428             yield event
    429         # Run compaction after all events are yielded from the agent.

/usr/local/lib/python3.11/dist-packages/google/adk/runners.py in _exec_with_plugin(self, invocation_context, session, execute_fn, is_live_call)
    651       # Step 2: Otherwise continue with normal execution
    652       async with Aclosing(execute_fn(invocation_context)) as agen:
--> 653         async for event in agen:
    654           if not event.partial:
    655             if self._should_append_event(event, is_live_call):

/usr/local/lib/python3.11/dist-packages/google/adk/runners.py in execute(ctx)
    414         async def execute(ctx: InvocationContext) -> AsyncGenerator[Event]:
    415           async with Aclosing(ctx.agent.run_async(ctx)) as agen:
--> 416             async for event in agen:
    417               yield event
    418 

/usr/local/lib/python3.11/dist-packages/google/adk/agents/base_agent.py in run_async(self, parent_context)
    292 
    293       async with Aclosing(self._run_async_impl(ctx)) as agen:
--> 294         async for event in agen:
    295           yield event
    296 

/usr/local/lib/python3.11/dist-packages/google/adk/agents/llm_agent.py in _run_async_impl(self, ctx)
    433     should_pause = False
    434     async with Aclosing(self._llm_flow.run_async(ctx)) as agen:
--> 435       async for event in agen:
    436         self.__maybe_save_output_to_state(event)
    437         yield event

/usr/local/lib/python3.11/dist-packages/google/adk/flows/llm_flows/base_llm_flow.py in run_async(self, invocation_context)
    354  

