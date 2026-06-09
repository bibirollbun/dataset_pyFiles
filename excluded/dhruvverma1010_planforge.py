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


from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types

print("âœ… ADK components imported successfully.")


from IPython.core.display import display, HTML
from jupyter_server.serverapp import list_running_servers

def get_adk_proxy_url():
    PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
    ADK_PORT = "8000"

    servers = list(list_running_servers())
    if not servers:
        raise Exception("No running Jupyter servers found.")

    baseURL = servers[0]["base_url"]
    parts = baseURL.split("/")
    kernel = parts[2]
    token = parts[3]

    url_prefix = f"/k/{kernel}/{token}/proxy/proxy/{ADK_PORT}"
    url = f"{PROXY_HOST}{url_prefix}"

    display(HTML(
        f"""
        <div style='padding:15px;border:2px solid #FFD966;border-radius:8px;background:#FFF8E1'>
            <h3>âš ï¸� Start ADK Web UI first!</h3>
            <p>Run the <b>!adk web</b> command after this cell.<br>
            Once it's running, click below:</p>
            <a href='{url}' target='_blank'
            style='padding:10px 20px;background:#1A73E8;color:white;border-radius:20px;text-decoration:none'>
                Open ADK Web UI â†—
            </a>
        </div>
        """
    ))
    return url_prefix

print("âœ… Helper functions loaded.")



retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504]
)



from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools import google_search
from google.adk.runners import InMemoryRunner

study_planner_agent = Agent(
    name="study_planner",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="An agent that generates structured study plans.",
    instruction="""
You are StudyBuddy, an intelligent academic planner AI.

Goals:
- Generate structured study plans
- Use reasoning + Google Search when needed
- Output clean Markdown plans with tasks, pomodoro blocks, and time allocation
""",
    tools=[google_search],
)

print("âœ… StudyPlanner Agent defined.")



planner_runner = InMemoryRunner(agent=study_planner_agent)
print("âœ… Planner Runner created.")



async def planner_plan(subjects, hours, level="beginner"):
    prompt = f"""
Create a detailed study plan.

Subjects: {subjects}
Hours Available: {hours}
Level: {level}

Requirements:
- Break schedule into Pomodoro blocks (25/5)
- List micro-tasks under each subject
- Mark difficulty (Easy/Medium/Hard)
- Include To-Do list
- Include Motivation Line
- Format in Markdown
"""

    result = await planner_runner.run_debug(prompt)

    # CASE 1: result is a list â†’ extract event with "response"
    if isinstance(result, list):
        for ev in result:
            if hasattr(ev, "response"):
                return ev.response
        return "No response found!"

    # CASE 2: result is a single Event object
    if hasattr(result, "response"):
        return result.response

    return "Unknown response format."



subjects = ["DSA", "Math for ML", "EMFT", "Python"]
hours = 5

output = await planner_plan(subjects, hours, "intermediate")
print(output)



!adk create studybuddy_agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY

