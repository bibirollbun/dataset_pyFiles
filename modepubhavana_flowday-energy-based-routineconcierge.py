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
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")


from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search 
from google.genai import types

from typing import List, Dict

print("âœ… ADK components imported successfully.")



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

    baseURL = servers[0]['base_url']

    try:
        path_parts = baseURL.split('/')
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


# ==== FlowDay tools: energy classification, task selection, preferences ====

user_prefs = {}

def classify_energy(sleep_hours, stress, pain, mood):
    """
    Returns an energy level High / Medium / Low based on quick check-in.
    """
    score = sleep_hours + (6 - stress) + (6 - pain) + mood
    if score >= 16:
        level = "High"
    elif score >= 12:
        level = "Medium"
    else:
        level = "Low"
    return {"energy_level": level, "score": score}


def suggest_tasks_for_energy(energy_level, tasks):
    """
    tasks: list of strings like
      "deep: Prepare for BIDS interview"
      "routine: Clean kitchen"
      "light: Sort emails"
    """
    mapping = {
        "High": {"deep", "routine"},
        "Medium": {"routine"},
        "Low": {"light"},
    }
    allowed = mapping.get(energy_level, {"routine"})

    selected = []
    for t in tasks:
        parts = t.split(":", 1)
        if len(parts) == 2:
            t_type = parts[0].strip().lower()
            name = parts[1].strip()
        else:
            t_type = "routine"
            name = t.strip()

        if t_type in allowed:
            selected.append({"type": t_type, "name": name})

    return {"energy_level": energy_level, "selected_tasks": selected}


def save_preferences(user_id, prefer_deep_morning, latest_bedtime):
    """
    Store simple user preferences for later sessions.
    """
    user_prefs[user_id] = {
        "prefer_deep_morning": bool(prefer_deep_morning),
        "latest_bedtime": latest_bedtime,
    }
    return {"status": "saved", "prefs": user_prefs[user_id]}



flowday_agent = Agent(
    name="flowday_energy_concierge",
    model="gemini-2.5-flash-lite",
    description="A concierge agent that designs daily routines based on the user's energy level.",
    instruction=(
        "You are FlowDay, an energy-aware day planner. "
        "Use the details the user gives (sleep hours, stress 1-5, pain 1-5, mood 1-5, free hours, and task list) "
        "to: (1) estimate whether today is a High, Medium, or Low energy day; "
        "(2) choose appropriate tasks (deep work mainly on High days, routine work on Medium days, "
        "and light or self-care tasks on Low days); "
        "(3) create a realistic schedule for morning / afternoon / evening; and "
        "(4) briefly explain your reasoning in kind, supportive language. "
        "If you need general wellbeing tips, you may call the google_search tool."
    ),
    tools=[google_search],  # ğŸ‘ˆ add this
)

print("âœ… FlowDay agent defined.")



runner = InMemoryRunner(agent=flowday_agent)
print("âœ… Runner created.")



prompt = """
You are FlowDay, my energy-aware day planner.

Today details:
- I slept 5 hours
- Stress level: 4 (1=low, 5=high)
- Pain/discomfort: 3 (1=low, 5=high)
- Mood: 2 (1=low, 5=great)
- I have 3 free hours today.

My task backlog:
- deep: Prepare for BIDS interview
- deep: Work on Kaggle capstone writeup
- routine: Clean the kitchen
- routine: Sort important emails
- light: Watch a comfort show
- light: Scroll Instagram

Please:
1) Decide if today is a High, Medium, or Low energy day.
2) Choose which tasks I should focus on today.
3) Build a realistic schedule for morning / afternoon / evening using those tasks.
4) Explain briefly why you chose these tasks based on my energy.
"""

response = await runner.run_debug(prompt)
response



prompt_base = """
You are FlowDay, my energy-aware day planner.

Today details:
- I slept 5 hours
- Stress level: 4 (1=low, 5=high)
- Pain/discomfort: 3 (1=low, 5=high)
- Mood: 3
- I have 3 free hours today.

My task backlog:
- deep: Prepare for BIDS interview
- deep: Work on Kaggle capstone writeup
- routine: Clean the kitchen
- routine: Sort important emails
- light: Watch a comfort show
- light: Scroll Instagram

Please:
1) Decide if today is High / Medium / Low energy.
2) Pick tasks for today.
3) Build a realistic morning / afternoon / evening schedule.
4) Explain your choices briefly.
"""

resp_base = await runner.run_debug(prompt_base)
resp_base



prompt_high = """
You are FlowDay, my energy-aware day planner.

Today details:
- I slept 8 hours
- Stress level: 2
- Pain/discomfort: 1
- Mood: 4
- I have 5 free hours today.

My task backlog:
- deep: Build Power BI dashboard
- deep: Study SQL window functions
- routine: Clean my room
- light: Watch YouTube shorts

Please:
1) Decide if today is High / Medium / Low energy.
2) Pick tasks for today.
3) Build a morning / afternoon / evening schedule.
4) Explain your choices.
"""

resp_high = await runner.run_debug(prompt_high)
resp_high



prompt_low = """
You are FlowDay, my energy-aware day planner.

Today details:
- I slept 4 hours
- Stress level: 5
- Pain/discomfort: 4
- Mood: 2
- I have 2 free hours today.

My task backlog:
- deep: Work on machine learning project
- routine: Do laundry
- light: Prepare simple dinner
- light: Watch a comfort show

Please:
1) Decide if today is High / Medium / Low energy.
2) Pick gentle tasks for today.
3) Build a short schedule.
4) Explain your choices and include at least one self-care suggestion.
"""

resp_low = await runner.run_debug(prompt_low)
resp_low

# Extract and print just the model's text for the low-energy scenario
for e in resp_low:
    if getattr(e, "content", None) and e.content.parts:
        print(e.content.parts[0].text)
        break



def simple_energy_eval(response_text: str, expected_energy: str) -> float:
    """
    Very rough evaluation:
    +0.5 if the expected energy label appears in the text
    +0.5 if at least one self-care / light task is mentioned for Low,
         or deep work is mentioned for High.
    Returns score between 0 and 1.
    """
    text = response_text.lower()
    score = 0.0

    if expected_energy.lower() in text:
        score += 0.5

    if expected_energy.lower() == "high":
        if "prepare for bids interview" in text or "kaggle capstone" in text:
            score += 0.5
    elif expected_energy.lower() == "low":
        if "comfort show" in text or "rest" in text or "self-care" in text:
            score += 0.5

    return score



high_text = str(resp_high)
low_text = str(resp_low)
base_text = str(resp_base)

print("Baseline scenario score:", simple_energy_eval(base_text, "High"))

print("High-energy scenario score:", simple_energy_eval(high_text, "High"))
print("Low-energy scenario score:", simple_energy_eval(low_text, "Low"))





