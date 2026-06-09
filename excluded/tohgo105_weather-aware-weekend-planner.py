# Load Gemini API key from Kaggle Secrets and set it as an environment variable.

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


# --- ADK / Gemini / Utility Imports ---

# ADK core agent + runner
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner

# ADK built-in toolsï¼ˆWeather Intelligence Agentã�§Google Searchã‚’ä½¿ã�†ï¼‰
from google.adk.tools import google_search

# Gemini types (JSON mode / response schemas)
from google.genai import types

# JSON formatting
import json
from datetime import datetime

# Session Service
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

print("âœ… ADK / Gemini imports loaded successfully.")


# --- Shared configuration for all agents ---

# Model to use (lightweight and fast: gemini-2.5-flash-lite)
MODEL_NAME = "gemini-2.5-flash-lite"

# Helper function to enforce JSON output schema for all agents
def as_json_schema(description: str):
    return types.ResponseSchema(
        type=types.Type.OBJECT,
        description=description,
    )

# Return today's date in ISO format (used for Weather Profile)
def today_iso():
    return datetime.now().strftime("%Y-%m-%d")

print("âœ… Shared configuration ready.")


# --- Weather Intelligence Agent ---

weather_agent = Agent(
    name="weather_agent",
    model=MODEL_NAME,
    description=(
        "Fetches a one-day weather forecast for a given location and target date "
        "and returns a structured profile for weekend planning."
    ),
    instruction="""
You are the Weather Intelligence Agent for Alex, a software engineer living in
Mountain View, CA, on the US West Coast.

Your job is to:
- Use the Google Search tool to look up the weather forecast for the given
  location AND a specific target date (typically an upcoming Saturday).
- Extract a single-day weather profile with the following fields:
  - location (string)
  - date (ISO string, e.g. "2025-11-29")
  - temp_c (approximate daytime temperature in Celsius)
  - humidity (approximate relative humidity in %)
  - uv_index (max UV index for the day, integer)
  - wind_mps (typical daytime wind speed in m/s)
  - precip_prob (approximate precipitation probability in %)
  - air_quality (qualitative description: "good", "moderate", or "unhealthy")

Important:
- The caller may say things like:
    - "Get the weather for next Saturday in Mountain View, CA"
    - or give you an explicit ISO date in Pacific Time (America/Los_Angeles).
- In both cases, you MUST focus on the forecast for that specific date,
  not "today" in some other timezone.
- Prefer 7â€“10 day forecast data that clearly corresponds to the target date.
- If only Fahrenheit is given, convert it approximately to Celsius.

Output format:
- Always return a single JSON object with exactly these keys:
  { "location", "date", "temp_c", "humidity", "uv_index",
    "wind_mps", "precip_prob", "air_quality" }
- Do NOT add any extra text before or after the JSON.
- If you are uncertain, make a reasonable approximation based on the forecast
  summary for that date, and keep the values consistent with the description.
""",
    tools=[google_search],
)

# Simple runner just for this agent (stateless, good for prototyping)
weather_runner = InMemoryRunner(agent=weather_agent)

print("âœ… Weather agent initialized.")


# Helper function:
# Computes the date of â€œnext Saturdayâ€� based on the U.S. West Coast timezone.
# This ensures the planner always uses Alexâ€™s upcoming weekend.

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

PACIFIC_TZ = ZoneInfo("America/Los_Angeles")

def next_saturday_pacific_iso() -> str:
    """Return the next Saturday date in America/Los_Angeles as ISO string."""
    today = datetime.now(PACIFIC_TZ).date()
    # Monday=0 ... Sunday=6
    days_ahead = (5 - today.weekday()) % 7  # 5 = Saturday

    target = today + timedelta(days=days_ahead)
    return target.isoformat()


# Helper function:
# Extracts clean JSON from the Event list returned by run_debug().
# LLMs sometimes wrap JSON in ```json ... ``` fences â€” this removes them
# and safely parses the result into a Python dict.

import re
import json

def extract_json_from_events(events):
    """
    Take the list of Event objects returned by run_debug(...)
    and extract the last model text as JSON.
    Assumes the model output is either:
      - pure JSON, or
      - wrapped in ```json ... ``` fences.
    """
    # Get the last event from the list
    last_event = events[-1]
    text = last_event.content.parts[0].text

    # Try to strip ```json ... ``` fences if present
    match = re.search(r"```json\s*(\{.*\})\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = text

    return json.loads(json_str)


# --- Fixed weekend tasks for Alex ---

WEEKEND_TASKS = [
    {
        "id": "run",
        "name": "Bay Trail Running",
        "category": "outdoor",
        "default_preference": "morning",
        "duration_hours": 1.0,
    },
    {
        "id": "farmers_market",
        "name": "Farmers Market Visit",
        "category": "outdoor",
        "default_preference": "late_morning",
        "duration_hours": 1.5,
        "requires_driving": True,
    },
    {
        "id": "grocery",
        "name": "Grocery Shopping",
        "category": "hybrid",  # mostly indoor but requires going outside
        "default_preference": "afternoon",
        "duration_hours": 1.0,
    },
    {
        "id": "car_wash",
        "name": "Car Wash",
        "category": "outdoor",
        "default_preference": "afternoon",
        "duration_hours": 1.0,
    },
    {
        "id": "laundry",
        "name": "Laundry",
        "category": "indoor",
        "default_preference": "flexible",
        "duration_hours": 1.0,
    },
    {
        "id": "cleaning",
        "name": "House Cleaning",
        "category": "indoor",
        "default_preference": "flexible",
        "duration_hours": 2.0,
    },
    {
        "id": "cafe",
        "name": "CafÃ© Reading / Coding",
        "category": "indoor",
        "default_preference": "afternoon",
        "duration_hours": 2.0,
    },
    {
        "id": "meal_prep",
        "name": "Meal Prep",
        "category": "indoor",
        "default_preference": "evening",
        "duration_hours": 1.5,
    },
]

print("âœ… Weekend tasks defined.")


# --- Task Sensitivity Agent ---

task_sensitivity_agent = Agent(
    name="task_sensitivity_agent",
    model=MODEL_NAME,
    description="Evaluates how weather conditions affect each weekend task.",
    instruction="""
You are the Task Sensitivity Agent.

You receive:
- A single-day weather profile with keys like:
  - location, date, temp_c, humidity, uv_index, wind_mps, precip_prob, air_quality
- A list of tasks. Each task has:
  - id, name, category ("outdoor", "indoor", or "hybrid"),
    default_preference, duration_hours

Your job is to:
- For each task, evaluate how suitable it is to perform today given the weather.
- Assign:
  - risk_score: an integer from 0 to 100
    (0 = no risk / fully comfortable, 100 = extremely risky or strongly discouraged)
  - risk_label: one of "very_low", "low", "medium", "high", "very_high"
  - recommendation: 1â€“2 sentences explaining whether and when the task
    is recommended today.

Guidelines:
- Outdoor tasks are sensitive to UV, wind, precipitation, and air quality.
  - High UV, strong wind, high precipitation, or poor air quality should increase risk.
- Indoor tasks are usually very_low risk.
  - They are good candidates for time windows when outdoor conditions are uncomfortable.
- Hybrid tasks (like grocery shopping) are partly affected by weather
  but less than pure outdoor activities.

You will be given a JSON object with the following structure:

{
  "weather_profile": { ... },
  "tasks": [ ... ]
}

Use this input to compute risks for each task.

Return a JSON object with the following structure:

{
  "tasks": [
    {
      "id": "...",
      "name": "...",
      "category": "...",
      "risk_score": 42,
      "risk_label": "medium",
      "recommendation": "..."
    },
    ...
  ]
}

Important:
- Output ONLY the JSON object.
- Do NOT include any text before or after the JSON.
""",
)

task_sensitivity_runner = InMemoryRunner(agent=task_sensitivity_agent)

print("âœ… Task Sensitivity Agent initialized.")


# --- Planner Agent ---

planner_agent = Agent(
    name="planner_agent",
    model=MODEL_NAME,
    description="Builds a one-day schedule based on weather-aware task risks.",
    instruction="""
You are the Planner Agent.

You receive:
- user_profile with:
  - day_start (e.g. "09:00")
  - day_end (e.g. "20:00")
  - max_tasks_per_day (e.g. 6)
- weather_profile for the day
- a list of tasks, each with:
  - id, name, category, default_preference, duration_hours
  - risk_score, risk_label, recommendation

Your goal:
- Create a non-overlapping schedule for the day between day_start and day_end.
- Assign start_time and end_time for a subset of tasks.

Constraints:
- Avoid scheduling tasks with risk_label "very_high".
- Be cautious with "high" risk outdoor tasks:
  - Only schedule them if they can be done in safer periods (e.g. early morning).
  - Otherwise, skip them for today.
- Prefer outdoor/hybrid tasks in time windows that are generally cooler and with lower UV:
  - Morning: 09:00â€“11:00
  - Late morning to early afternoon: 11:00â€“14:00
  - Afternoon: 14:00â€“17:00
  - Evening: 17:00â€“20:00
- Place indoor tasks at times when outdoor conditions are less comfortable
  (midday, or when overall outdoor risk is high).
- Do not exceed max_tasks_per_day.
- Respect default_preference (morning / late_morning / afternoon / evening)
  when possible, but override it if the weather is clearly unfavorable.
- Use duration_hours to determine how long each task takes.
  Assume tasks are scheduled in 1-hour blocks (you can round 1.5h to 1.5h blocks).

Input format (JSON object):

{
  "user_profile": {
    "day_start": "09:00",
    "day_end": "20:00",
    "max_tasks_per_day": 6
  },
  "weather_profile": { ... },
  "tasks": [
    {
      "id": "...",
      "name": "...",
      "category": "...",
      "default_preference": "...",
      "duration_hours": 1.0,
      "risk_score": 35,
      "risk_label": "medium",
      "recommendation": "..."
    },
    ...
  ]
}

Output format (JSON object):

{
  "plan": [
    {
      "task_id": "run",
      "task_name": "Bay Trail Running",
      "start_time": "09:00",
      "end_time": "10:00"
    },
    ...
  ],
  "notes": [
    "Explain briefly why the most important scheduling decisions were made."
  ]
}

Important:
- Output ONLY the JSON object, with keys 'plan' and 'notes'.
- Do NOT include any extra text before or after the JSON.
""",
)

planner_runner = InMemoryRunner(agent=planner_agent)

print("âœ… Planner Agent initialized.")


# --- Advisor Agent ---

advisor_agent = Agent(
    name="advisor_agent",
    model=MODEL_NAME,
    description="Explains the weather-aware schedule to the user in natural language.",
    instruction="""
You are the Advisor Agent.

You receive:
- weather_profile for the day
- task_risk_result: a list of tasks with risk_score, risk_label, recommendation
- planner_result: the final daily plan with time slots

Your job is to:
- Explain the schedule to the user in clear, friendly English.
- Make the reasoning transparent:
  - How did the weather (UV, temperature, wind, precipitation, air quality)
    influence the placement or exclusion of tasks?
  - Why are outdoor tasks scheduled at certain times?
  - Why are some tasks skipped or suggested for another day?

Guidelines:
- Refer to tasks by name and time range (e.g., "09:00â€“10:00 Bay Trail Running").
- Briefly mention the key weather drivers (e.g. "moderate UV", "low precipitation").
- Highlight at least:
  - 1â€“2 outdoor tasks and why they are placed where they are.
  - 1â€“2 indoor tasks and how they help use less comfortable weather periods.
  - Any tasks that were not scheduled and why they are easy to postpone.

Input will be provided as a JSON object like:

{
  "weather_profile": { ... },
  "task_risk_result": { "tasks": [ ... ] },
  "planner_result": {
    "plan": [ ... ],
    "notes": [ ... ]
  }
}

Output:
- Write a short explanation in natural language (a few paragraphs).
- Use bullet points where helpful.
- Do NOT output JSON. Respond in plain text.
""",
)

advisor_runner = InMemoryRunner(agent=advisor_agent)

print("âœ… Advisor Agent initialized.")


async def run_weather_aware_planner(
    location: str = "Mountain View, CA",
    target_date: str | None = None,
    max_tasks_per_day: int = 6,
):
    """
    Run the full Weather-Aware Weekend Planner pipeline:
    1. Get weather forecast for a specific date (next Saturday in Pacific by default)
    2. Evaluate weather risk for each weekend task
    3. Build a one-day schedule
    4. Generate a natural-language explanation for the user
    """

    # 0. Decide which date to plan for (default: next Saturday in Pacific time)
    if target_date is None:
        target_date = next_saturday_pacific_iso()

    # 1. Weather (use forecast for the target_date, not â€œtodayâ€�)
    weather_prompt = (
        f"Get the weather forecast for {location} on {target_date} as JSON. "
        "Use the forecast for that specific date in Mountain View, "
        "not today's weather in another timezone. "
        "Return ONLY a single JSON object with the required fields."
    )
    weather_events = await weather_runner.run_debug(weather_prompt)
    weather_profile = extract_json_from_events(weather_events)

    # 2. Task Sensitivity
    payload = {
        "weather_profile": weather_profile,
        "tasks": WEEKEND_TASKS,
    }
    task_prompt = (
        "You are given the following JSON as input. "
        "Use it to evaluate the weather risk for each task and return the result as JSON.\n\n"
        + json.dumps(payload)
    )
    task_events = await task_sensitivity_runner.run_debug(task_prompt)
    task_risk_result = extract_json_from_events(task_events)

    # 3. Planner
    user_profile = {
        "day_start": "09:00",
        "day_end": "20:00",
        "max_tasks_per_day": max_tasks_per_day,
    }
    planner_payload = {
        "user_profile": user_profile,
        "weather_profile": weather_profile,
        "tasks": task_risk_result["tasks"],
    }
    planner_prompt = (
        "You are given the following JSON as input. "
        "Use it to build a one-day schedule and return the result as JSON.\n\n"
        + json.dumps(planner_payload)
    )
    planner_events = await planner_runner.run_debug(planner_prompt)
    planner_result = extract_json_from_events(planner_events)

    # 4. Advisor
    advisor_payload = {
        "weather_profile": weather_profile,
        "task_risk_result": task_risk_result,
        "planner_result": planner_result,
    }
    advisor_prompt = (
        "You are given the following JSON with weather, risk evaluation, and the final plan. "
        "Explain the schedule to the user based on this JSON.\n\n"
        + json.dumps(advisor_payload)
    )
    advisor_events = await advisor_runner.run_debug(advisor_prompt)
    advisor_text = advisor_events[-1].content.parts[0].text

    return {
        "location": location,
        "target_date": target_date,
        "weather_profile": weather_profile,
        "task_risk_result": task_risk_result,
        "planner_result": planner_result,
        "advisor_text": advisor_text,
    }


import json

def pretty_print_plan(result: dict):
    """Nicely format the pipeline result for humans."""
    print("ğŸ“� Location:", result["location"])
    print("ğŸ“… Target date:", result["target_date"])
    print()

    # --- Weather ---
    print("=== Weather profile ===")
    print(json.dumps(result["weather_profile"], indent=2))
    print()

    # --- Risk tableï¼ˆç°¡æ˜“ï¼‰---
    print("=== Task risk overview ===")
    for t in result["task_risk_result"]["tasks"]:
        print(
            f"- {t['name']:<25} | "
            f"{t['category']:<7} | "
            f"score={t['risk_score']:>3} | "
            f"label={t['risk_label']}"
        )
    print()

    # --- Planner schedule ---
    print("=== Planned schedule ===")
    for slot in result["planner_result"]["plan"]:
        print(f"{slot['start_time']}â€“{slot['end_time']}  {slot['task_name']}")
    print()

    # --- Advisor explanation ---
    print("=== Advisor explanation ===\n")
    print(result["advisor_text"])


# Run the full pipeline (with debug output from each agent)

result = await run_weather_aware_planner()


# Display the final, human-friendly summary of the plan

pretty_print_plan(result)


# --- Session-aware Agent for Alex (grounded in WEEKEND_TASKS) ---

session_aware_agent = Agent(
    name="session_aware_advisor",
    model=MODEL_NAME,
    description="Chat agent that remembers Alex's preferences within a session and only recommends from a fixed task catalog.",
    instruction="""
You are a session-aware weekend planning assistant for Alex.

You will ALWAYS receive input as a JSON object with:
{
  "tasks_catalog": [ ... ],  // list of available tasks
  "message": "..."           // the latest user message
}

The tasks_catalog contains objects like:
{
  "id": "run",
  "name": "Bay Trail Running",
  "category": "outdoor",
  "default_preference": "morning",
  "duration_hours": 1.0
}

Your responsibilities:
- Within a session, remember what Alex said about:
  - preferences (e.g., "focus on outdoor activities")
  - constraints (e.g., "avoid driving", "no long activities today")
- On each turn, read BOTH:
  - the full conversation history in this session
  - the current tasks_catalog from the JSON payload
- When Alex asks for recommendations:
  - ONLY recommend tasks from tasks_catalog.
  - NEVER invent new activities that are not in the catalog
    (e.g., do NOT mention generic 'picnic', 'bike ride', 'random hike' etc.).
  - Instead, map preferences to the closest tasks in the catalog and
    refer to them by their 'name'.
- You may:
  - Suggest 2â€“4 tasks from the catalog that best match the preferences.
  - Briefly justify why each suggested task fits Alex's preferences.

Example behavior:
- If Alex first says:
  "This Saturday I'd like to focus on outdoor activities and avoid driving."
  and later asks:
  "Given that, what would you recommend for the afternoon?",
  you might answer:
  - "Farmers Market Visit" (outdoor, likely reachable without a long drive)
  - "Bay Trail Running" if it fits the time window
  - or a hybrid/indoor task if outdoor risk is too high.

Output:
- Respond in clear English.
- Use bullet points when listing recommended tasks.
- Do NOT output JSON.
"""
)


# --- Session service & Runner setup ---

APP_NAME = "weather_weekend_planner"
USER_ID = "alex"

session_service = InMemorySessionService()

session_runner = Runner(
    agent=session_aware_agent,
    app_name=APP_NAME,
    session_service=session_service,
)

print("âœ… Session-aware runner initialized.")


# First interaction: store the user's preferences into the session state
# (Only summarize & confirm preferences, no task recommendations yet)

from copy import deepcopy

SESSION_ID = "alex_weekend_session_1"

first_payload = {
    "tasks_catalog": deepcopy(WEEKEND_TASKS),
    "message": "This Saturday I'd like to focus on outdoor activities and avoid driving if possible."
}

first_prompt = (
    "You are given the following JSON with a task catalog and the user's latest message.\n"
    "In this turn, the user is only expressing preferences, not asking for recommendations.\n"
    "Your job now is ONLY to:\n"
    "- briefly summarize the preferences you understood, and\n"
    "- confirm you will remember them for this session.\n"
    "Do NOT list or recommend any tasks yet.\n\n"
    + json.dumps(first_payload)
)

events_1 = await session_runner.run_debug(
    first_prompt,
    user_id=USER_ID,
    session_id=SESSION_ID,
)

print("=== First turn (set preferences only) ===")
print(events_1[-1].content.parts[0].text)


# Second turn: use the stored session preferences to generate recommendations

second_payload = {
    "tasks_catalog": deepcopy(WEEKEND_TASKS),
    "message": "Given that, what would you recommend for the afternoon?"
}

second_prompt = (
    "You are given the following JSON with the SAME task catalog and the user's latest message.\n"
    "Use BOTH the catalog and the stored session preferences to answer.\n\n"
    + json.dumps(second_payload)
)

events_2 = await session_runner.run_debug(
    second_prompt,
    user_id=USER_ID,
    session_id=SESSION_ID,
)

print("\n=== Second turn (uses remembered preferences, grounded in tasks) ===")
print(events_2[-1].content.parts[0].text)

