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


from google import genai
from google.genai import types
from google.colab import userdata

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search

import textwrap
import json

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
print("âœ… ADK & Gemini imports successful.")


profile_agent = Agent(
    name="profile_and_goals_agent",
    model="gemini-2.5-flash-lite",
    instruction="""
You are the Profile & Goals Agent in the MyLife Concierge system.

Your job:
- Extract or infer the user's constraints, preferences, and goals in major areas:
  - Work / study
  - Health & fitness
  - Learning / skills
  - Personal life / hobbies
- Produce a concise JSON object describing the profile and goals.

Return STRICTLY a JSON object with this shape:

{
  "time_budget_hours_per_day": {
    "weekday": <number>,
    "weekend": <number>
  },
  "preferred_workout_days": [<string>],
  "sleep_window": "<string like '23:00-07:00'>",
  "learning_goals": [ "<text>" ],
  "fitness_goals": [ "<text>" ],
  "career_goals": [ "<text>" ],
  "other_goals": [ "<text>" ],
  "constraints": [ "<text>" ],
  "preferences": [ "<text>" ]
}

If information is missing, make a reasonable assumption and clearly mark it in the JSON value.
Never include any explanation outside the JSON.
""",
)

print("âœ… Profile & Goals Agent defined.")


research_agent = Agent(
    name="research_agent",
    model="gemini-2.5-flash-lite",
    tools=[google_search],
    instruction="""
You are the Research Agent in the MyLife Concierge system.

Input:
- A JSON profile of the user and their goals.

Task:
- For each goal area (learning, fitness, career, other), use the google_search tool
  when needed to find 2â€“3 high-quality, beginner-friendly resources:
  - Online courses
  - YouTube playlists
  - Blog posts
  - Simple workout plans, etc.

Output:
Return strictly a JSON object:

{
  "learning_resources": [ { "goal": "<goal>", "title": "<title>", "url": "<url>", "notes": "<short why>" } ],
  "fitness_resources": [ ... ],
  "career_resources": [ ... ],
  "other_resources": [ ... ]
}

Use google_search selectively. Do NOT hallucinate URLs; use search when unsure.
Never output anything outside the JSON.
""",
)

print("âœ… Research Agent defined.")


planner_agent = Agent(
    name="planner_agent",
    model="gemini-2.5-flash-lite",
    instruction="""
You are the Planner Agent in the MyLife Concierge system.

Inputs:
- User profile JSON.
- Research results JSON.

Task:
- Create a realistic WEEKLY PLAN that balances:
  - Learning goals
  - Fitness goals
  - Career goals
  - Personal life / hobbies
- Respect the user's time budget, preferred workout days, and constraints.
- Don't overload any single day.
- Assume a 7-day week (Mondayâ€“Sunday).

Output:
Return STRICTLY a JSON object with this structure:

{
  "week_plan": {
    "Monday": [
      {"time": "07:00-07:30", "category": "fitness", "activity": "<text>", "resource": "<optional url or note>"},
      {"time": "18:00-18:45", "category": "learning", "activity": "<text>", "resource": "<optional url>"}
    ],
    "Tuesday": [ ... ],
    ...
    "Sunday": [ ... ]
  }
}

Do not add explanations outside this JSON.
""",
)

print("âœ… Planner Agent defined.")


writer_agent = Agent(
    name="writer_agent",
    model="gemini-2.5-flash-lite",
    instruction="""
You are the Writer Agent in the MyLife Concierge system.

Input:
- The structured weekly plan JSON.

Task:
- Turn this into a friendly, motivational, and easy-to-follow weekly summary
  for the user.

Output:
- A markdown-formatted text that:
  - Starts with a short overview.
  - Then lists each day with bullet points and short, clear instructions.
  - Uses simple language and avoids jargon.

Do not change the plan logic, just present it nicely.
""",
)

print("âœ… Writer Agent defined.")


critic_agent = Agent(
    name="critic_agent",
    model="gemini-2.5-flash-lite",
    instruction="""
You are the Critic Agent in the MyLife Concierge system.

Input:
- User profile JSON.
- Weekly plan JSON.

Task:
- Evaluate whether the plan is realistic and balanced:
  - Is the time per day roughly within the user's time budget?
  - Are the goals covered across the week?
  - Are there obvious overloads or conflicts?

Output:
Return a JSON object:

{
  "overall_score": <number from 1 to 10>,
  "issues": [ "<text description of problem>" ],
  "suggested_changes": [ "<high-level suggestions>" ],
  "is_acceptable": <true or false>
}

If the plan is already good, still fill these fields and set is_acceptable to true.
Never output anything outside the JSON.
""",
)

print("âœ… Critic Agent defined.")


profile_runner = InMemoryRunner(agent=profile_agent)
research_runner = InMemoryRunner(agent=research_agent)
planner_runner = InMemoryRunner(agent=planner_agent)
critic_runner = InMemoryRunner(agent=critic_agent)
writer_runner = InMemoryRunner(agent=writer_agent)

print("âœ… Runners for all sub-agents created.")


def get_final_text_from_events(events):
    """
    Given the list of Events returned by runner.run_debug(...),
    return the last non-empty text part.
    """
    last_text = ""
    for e in events:
        if e.content and e.content.parts:
            for p in e.content.parts:
                if getattr(p, "text", None):
                    last_text = p.text
    return last_text


# Simple in-notebook "memory" for user profiles.
# In a real system, this could be a DB or Memory Bank.

user_profiles_memory = {}

def save_profile(user_id: str, profile_json_str: str):
    """Store the profile JSON string for a given user_id."""
    user_profiles_memory[user_id] = profile_json_str

def get_profile(user_id: str):
    """Retrieve the stored profile JSON string for a given user_id, or None."""
    return user_profiles_memory.get(user_id)


async def run_mylife_concierge(
    user_id: str,
    user_request: str,
    force_new_profile: bool = False,
    max_plan_iterations: int = 2,
) -> str:
    # 1) Profile & goals (with memory)
    stored_profile = None if force_new_profile else get_profile(user_id)

    if stored_profile:
        profile_clean = stored_profile
    else:
        profile_events = await profile_runner.run_debug(user_request)
        profile_text = get_final_text_from_events(profile_events)

        profile_clean = profile_text.strip()
        if profile_clean.startswith("```"):
            profile_clean = profile_clean.strip("`")
            profile_clean = profile_clean.replace("json\n", "").replace("json\r\n", "")

        save_profile(user_id, profile_clean)

    # 2) Research (optional; uses google_search inside the agent)
    research_prompt = f"""
You are the Research Agent.

User profile and goals (JSON):

{profile_clean}

Use this profile as INPUT and do your research.
Return the JSON result as specified in your instructions.
"""
    research_prompt = textwrap.dedent(research_prompt).strip()

    research_events = await research_runner.run_debug(research_prompt)
    research_text = get_final_text_from_events(research_events)

    research_clean = research_text.strip()
    if research_clean.startswith("```"):
        research_clean = research_clean.strip("`")
        research_clean = research_clean.replace("json\n", "").replace("json\r\n", "")

    # 3) Planning + Critic loop
    planner_clean = None
    critic_data = None

    for attempt in range(max_plan_iterations):
        planner_prompt = f"""
You are the Planner Agent.

User profile JSON:
{profile_clean}

Research results JSON:
{research_clean}

Using these, create the weekly plan in the JSON format specified in your instructions.
"""
        if critic_data and attempt > 0:
            planner_prompt += f"""

Critic feedback to consider when updating the plan:

{json.dumps(critic_data, indent=2)}
"""
        planner_prompt = textwrap.dedent(planner_prompt).strip()

        planner_events = await planner_runner.run_debug(planner_prompt)
        planner_text = get_final_text_from_events(planner_events)

        planner_clean = planner_text.strip()
        if planner_clean.startswith("```"):
            planner_clean = planner_clean.strip("`")
            planner_clean = planner_clean.replace("json\n", "").replace("json\r\n", "")

        # Critic stage
        critic_prompt = f"""
You are the Critic Agent.

User profile JSON:
{profile_clean}

Weekly plan JSON:
{planner_clean}

Evaluate the plan and return the JSON described in your instructions.
"""
        critic_prompt = textwrap.dedent(critic_prompt).strip()

        critic_events = await critic_runner.run_debug(critic_prompt)
        critic_text = get_final_text_from_events(critic_events)

        critic_clean = critic_text.strip()
        if critic_clean.startswith("```"):
            critic_clean = critic_clean.strip("`")
            critic_clean = critic_clean.replace("json\n", "").replace("json\r\n", "")

        try:
            critic_data = json.loads(critic_clean)
        except Exception:
            critic_data = None

        # Stop if we have no critic data or it's acceptable, or we've reached max iterations
        if not critic_data or critic_data.get("is_acceptable", True) or attempt == max_plan_iterations - 1:
            break

    # 4) Writer: final human-friendly summary
    writer_prompt = f"""
You are the Writer Agent.

Here is the WEEK PLAN JSON:

{planner_clean}

Write a friendly, markdown-formatted weekly schedule for the user.
"""
    writer_prompt = textwrap.dedent(writer_prompt).strip()

    writer_events = await writer_runner.run_debug(writer_prompt)
    final_text = get_final_text_from_events(writer_events)

    # Optionally append critic summary
    if critic_data:
        issues = critic_data.get("issues", [])
        score = critic_data.get("overall_score", None)
        final_text += "\n\n---\n\n**Critic Summary**\n\n"
        if score is not None:
            final_text += f"- Overall score: **{score}/10**\n"
        if issues:
            final_text += "- Issues noted:\n"
            for issue in issues:
                final_text += f"  - {issue}\n"

    return final_text


# First run â€” creates a new profile and generates a complete week plan
user_id = "user-001"

user_request = """
I work a 9â€“6 job Monday to Friday.
I want to:
- Get fitter
- Learn AI / machine learning
- Spend some time on a personal side project

I can give 1.5 hours per weekday and 3 hours on weekends.
Please create a weekly plan for me.
"""

first_plan = await run_mylife_concierge(user_id, user_request)
print(first_plan)


# Second run â€” uses stored profile, updates plan based on new request
user_request_update = """
Update my weekly plan, but keep my preferences.
Shift a bit more time toward my personal side project.
"""

updated_plan = await run_mylife_concierge(user_id, user_request_update)
print(updated_plan)


from google.genai.errors import ClientError

test_cases = [
    {
        "id": "case_1",
        "user_id": "eval-user-1",
        "prompt": """
My goals:
- Learn AI
- Improve strength
I have 1 hour everyday.
""",
        "must_contain": ["AI", "strength"],
    },
    {
        "id": "case_2",
        "user_id": "eval-user-2",
        "prompt": "Focus only on fitness this week.",
        "must_contain": ["fitness", "workout"],
    },
    {
        "id": "case_3",
        "user_id": "eval-user-3",
        "prompt": "I want to work on my startup idea and do light workouts.",
        "must_contain": ["startup", "workout"],
    },
]

eval_results = []

for case in test_cases:
    try:
        output = await run_mylife_concierge(case["user_id"], case["prompt"])
        passed_keywords = all(term.lower() in output.lower() for term in case["must_contain"])
        eval_results.append({
            "test_id": case["id"],
            "status": "ok",
            "passed_keywords": passed_keywords,
            "output_length": len(output),
            "error": None,
        })
    except ClientError as e:
        # Handles 429 RESOURCE_EXHAUSTED (and other client errors) gracefully
        eval_results.append({
            "test_id": case["id"],
            "status": "error",
            "passed_keywords": False,
            "output_length": 0,
            "error": str(e),
        })

eval_results

