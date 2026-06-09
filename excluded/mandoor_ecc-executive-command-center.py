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


!pip install -U google-adk google-genai



#2. Configure Gemini API Key


import os
from kaggle_secrets import UserSecretsClient
from google import genai

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )

MODEL_NAME = "gemini-2.0-flash"  # fast + good

client = genai.Client()
model = client.models.generate_content


#3. Helper to call model + safe JSON parsing


import json
from typing import Dict, Any


def call_model(prompt: str) -> str:
    """
    Simple wrapper around Gemini text generation.
    """
    resp = model(
        model=MODEL_NAME,
        contents=prompt,
    )
    # Gemini responses can contain multiple candidates; we take the first text part
    text = ""
    for cand in resp.candidates:
        for part in cand.content.parts:
            if part.text:
                text += part.text
    return text.strip()


def parse_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract the first JSON object from a text response and parse it.
    """
    # Find first '{' and last '}' to isolate JSON
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not find JSON object in text:\n{text}")
    json_str = text[start:end+1]
    return json.loads(json_str)



#4. Strategic goals (A / C / D)


from dataclasses import dataclass
from enum import Enum
from typing import Dict


class GoalID(str, Enum):
    A = "A"      # AI consulting / revenue engine
    C = "C"      # ECC product
    D = "D"      # Discipline / personal OS
    OTHER = "other"


@dataclass
class StrategicGoal:
    id: GoalID
    name: str
    description: str
    time_horizon: str


LONG_TERM_GOALS: Dict[GoalID, StrategicGoal] = {
    GoalID.A: StrategicGoal(
        id=GoalID.A,
        name="AI Consulting Revenue Engine",
        description="Build and scale an AI consulting + agent-building business.",
        time_horizon="12â€“24 months",
    ),
    GoalID.C: StrategicGoal(
        id=GoalID.C,
        name="Executive Command Center (ECC) Product",
        description="Turn ECC into a repeatable, sellable strategic assistant product.",
        time_horizon="12â€“24 months",
    ),
    GoalID.D: StrategicGoal(
        id=GoalID.D,
        name="Discipline & Personal Operating System",
        description="Maintain elite discipline in health, learning, and execution.",
        time_horizon="ongoing",
    ),
}



#5. Agent 1: Knowledge Capture


def knowledge_capture_agent(raw_context: str) -> Dict[str, Any]:
    """
    Agent 1: takes messy weekly context and produces structured JSON.
    """
    goals_text = "\n".join(
        f"- {gid.value}: {g.description}"
        for gid, g in LONG_TERM_GOALS.items()
    )

    prompt = f"""
You are an executive knowledge capture agent.

User context (weekly brain dump):
\"\"\" 
{raw_context}
\"\"\"

Long-term goals:
{goals_text}

Your job:
- Extract ONLY useful information for planning and execution.

Return STRICT JSON ONLY (no explanation), with this schema:

{{
  "goals": [ {{"id": "A|C|D|other", "summary": "..."}} ],
  "decisions": [ {{"summary": "...", "area": "A|C|D|other"}} ],
  "action_items": [
     {{"title": "...", "area": "A|C|D|other", "due_hint": "today|this_week|this_month|later"}}
  ],
  "risks": [ {{"summary": "...", "area": "A|C|D|other"}} ],
  "notes": ["..."]
}}

Rules:
- Ignore irrelevant chatter.
- If something is unclear, put it into notes instead of guessing.
- Do NOT wrap the JSON in backticks or any other formatting.
"""

    text = call_model(prompt)
    return parse_json_from_text(text)



#6. Agent 2: Strategic Planner


def strategic_planner_agent(ecc_knowledge: Dict[str, Any]) -> Dict[str, Any]:
    """
    Agent 2: builds a one-week strategic plan.
    """
    goals_text = "\n".join(
        f"- {gid.value}: {g.description}"
        for gid, g in LONG_TERM_GOALS.items()
    )

    prompt = f"""
You are a PHD-level Chief Strategy Officer for a solo founder.

Long-term goals:
{goals_text}

Structured weekly intel (ecc_knowledge):
{json.dumps(ecc_knowledge, indent=2)}

Task:
Design a focused 1-week execution plan that moves the user 5 steps ahead.
Balance:
- A = AI consulting / revenue
- C = ECC product
- D = Discipline / routines

Return STRICT JSON ONLY:

{{
  "week_theme": "one sharp sentence",
  "non_negotiables": [
    {{"goal_id": "A|C|D", "title": "...", "why_now": "..."}}
  ],
  "plan_by_day": {{
    "monday":    [{{"title": "...", "goal_id": "A|C|D|other", "block_hint": "morning|afternoon|evening"}}],
    "tuesday":   [],
    "wednesday": [],
    "thursday":  [],
    "friday":    []
  }}
}}

Constraints:
- Assume 3â€“5 hours deep work per weekday.
- Prefer fewer, high-impact tasks over long lists.
- JSON only. No explanation text.
"""

    text = call_model(prompt)
    return parse_json_from_text(text)



#7. Agent 3: Priority Agent


def priority_agent(ecc_plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Agent 3: assigns P1/P2/P3, time estimates, rationale.
    """
    prompt = f"""
You are an execution and operations specialist.

Here is the 1-week plan (ecc_plan):

{json.dumps(ecc_plan, indent=2)}

For EACH item in plan_by_day, produce an enriched task object:

{{
  "title": "...",
  "goal_id": "A|C|D|other",
  "day": "monday|tuesday|wednesday|thursday|friday",
  "block_hint": "morning|afternoon|evening",
  "priority": "P1|P2|P3",
  "time_estimate_hours": 0.5,
  "rationale": "why this matters",
  "area": "Consulting|ECC|Discipline|Other"
}}

Priority rules:
- P1 = moves a strategic needle THIS week.
- P2 = important but can slip.
- P3 = low impact / optional.

Return STRICT JSON ONLY:

{{
  "prioritized_tasks": [ ... ]
}}

No explanation text.
"""

    text = call_model(prompt)
    return parse_json_from_text(text)



#8. Agent 4: Accountability / Briefing


def accountability_agent(ecc_prioritized_plan: Dict[str, Any]) -> str:
    """
    Agent 4: turns the prioritized plan into a human-facing ECC briefing (Markdown).
    """
    prompt = f"""
You are an executive coach + chief of staff.

Here is the prioritized plan for the week:

{json.dumps(ecc_prioritized_plan, indent=2)}

Create a concise Markdown briefing with the following sections:

1. **Week Theme** â€“ 1 sentence.
2. **Top 3 P1 Tasks for Today** â€“ bullet list, each labeled with goal (A/C/D/other).
3. **Key Risks / Bottlenecks** â€“ 3â€“5 bullets.
4. **Discipline Check** â€“ 2â€“4 bullets about focus, routines, and energy.
5. **One Hard Question** â€“ a direct, uncomfortable question the user must answer.

Tone:
- Direct, professional, strategic.
- No fluff, no motivational clichÃ©s.
"""

    text = call_model(prompt)
    return text.strip()



#9. ECC pipeline + demo


def ecc_pipeline(raw_context: str) -> str:
    """
    Full ECC flow: context -> knowledge -> plan -> priorities -> briefing.
    """
    # 1) capture knowledge
    ecc_knowledge = knowledge_capture_agent(raw_context)

    # 2) plan week
    ecc_plan = strategic_planner_agent(ecc_knowledge)

    # 3) prioritize tasks
    ecc_prioritized_plan = priority_agent(ecc_plan)

    # 4) final briefing
    briefing = accountability_agent(ecc_prioritized_plan)

    return briefing


def ecc_demo_input() -> str:
    return """
    This week I must:
    - Finish the ECC Kaggle capstone and GitHub repo (non-negotiable).
    - Design the first version of my AI consulting offer with 3 pricing tiers.
    - Keep morning workouts and at least 2h deep work on ECC for 4 days.
    - I keep getting distracted in the evenings by social media and YouTube.
    - One potential client is asking about an AI agent for operations.
    - I feel a bit overloaded, so I need clarity on what is truly P1.
    """


briefing = ecc_pipeline(ecc_demo_input())
print(briefing)


