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


!pip install google-generativeai



import os
import json
import logging
import textwrap
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import google.generativeai as genai

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("Pathmate")



os.environ["GEMINI_API_KEY"] = input("Enter your Gemini API key: ").strip()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

DEFAULT_MODEL = "gemini-2.0-flash"



# def call_llm(system_prompt, user_prompt, model=DEFAULT_MODEL):
#     logger.info("Calling LLM model=%s", model)
#     logger.debug("System prompt: %s", textwrap.shorten(system_prompt, 300))
#     logger.debug("User prompt: %s", textwrap.shorten(user_prompt, 300))

#     try:
#         resp = genai.GenerativeModel(model).generate_content(
#             [
#                 {"role": "system", "parts": [system_prompt]},
#                 {"role": "user", "parts": [user_prompt]}
#             ]
#         )
#         text = resp.text if hasattr(resp, "text") else str(resp)
#         logger.debug("Response: %s", textwrap.shorten(text, 300))
#         return text
#     except Exception as e:
#         logger.error("LLM error: %s", str(e))
#         return "{}"

def call_llm(system_prompt, user_prompt, model=DEFAULT_MODEL):
    logger.info("Calling LLM model=%s", model)
    logger.debug("System prompt: %s", textwrap.shorten(system_prompt, 300))
    logger.debug("User prompt: %s", textwrap.shorten(user_prompt, 300))

    try:
        # Gemini 2.x does NOT support explicit "system" role content in lists.
        # So we concatenate system + user prompt into a single string.
        full_prompt = system_prompt.strip() + "\n\n" + user_prompt.strip()

        resp = genai.GenerativeModel(model).generate_content(full_prompt)
        text = resp.text if hasattr(resp, "text") else str(resp)

        logger.debug("Response: %s", textwrap.shorten(text, 300))
        return text
    except Exception as e:
        logger.error("LLM error in call_llm: %s", str(e))
        # Return empty JSON so agents can fall back
        return "{}"



@dataclass
class SessionState:
    user_id: str
    goal_text: str = ""
    profile: Dict[str, Any] = None
    roadmap: Dict[str, Any] = None
    skill_gaps: Dict[str, Any] = None
    plan: Dict[str, Any] = None
    history: List[Dict[str, Any]] = None

class SessionManager:
    def __init__(self):
        self.sessions = {}

    def get_or_create(self, session_id, user_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState(
                user_id=user_id,
                profile={},
                roadmap={},
                skill_gaps={},
                plan={},
                history=[]
            )
        return self.sessions[session_id]

    def add_history(self, session_id, role, content):
        st = self.sessions[session_id]
        st.history.append({"role": role, "content": content})
        st.history = st.history[-20:]



def estimate_time_commitment(months, hours_per_week):
    return months * 4 * hours_per_week



@dataclass
class UserProfile:
    name: str
    background: str
    experience_years: int
    current_role: str
    hours_per_week: int
    location: str = "Remote"



import json, re

def extract_json(raw: str):
    """
    Try to extract the first JSON object from the LLM response.
    Falls back to {} if nothing works.
    """
    if not raw:
        return {}

    # Try direct load first
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Try to find a {...} block
    match = re.search(r"\{.*\}", raw, re.S)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            return {}
    return {}



from dataclasses import dataclass, asdict
from typing import Dict, Any, List

@dataclass
class UserProfile:
    name: str
    background: str
    experience_years: int
    current_role: str
    hours_per_week: int
    location: str = "Remote"


class GoalStrategyAgent:
    def run(self, state, profile: UserProfile):
        system_prompt = """
You are the Goal Strategy Agent ("Architect").
Your ONLY job is to output a single valid JSON object.
NO explanations, NO markdown, NO backticks.

The JSON MUST look like:

{
  "smart_goal": "string",
  "phases": [
    { "name": "string", "description": "string", "order_index": 1 }
  ],
  "milestones": [
    { "phase_name": "string", "milestone": "string", "success_criteria": "string" }
  ],
  "timeline_months": 6
}
"""
        user_prompt = f"""
User profile:
{json.dumps(asdict(profile), indent=2)}

User goal:
{state.goal_text}

1. Rewrite the goal as a SMART goal (time-bound, specific).
2. Create 3â€“6 phases.
3. Suggest timeline_months.
4. Fill milestones with at least 1 milestone per phase.
Return ONLY the JSON object.
"""
        raw = call_llm(system_prompt, user_prompt)
        roadmap = extract_json(raw)

        if not roadmap or "smart_goal" not in roadmap:
            # fallback
            roadmap = {
                "smart_goal": f"Advance goal: {state.goal_text}",
                "phases": [
                    {"name": "Foundations", "description": "Learn basics.", "order_index": 1},
                    {"name": "Practice", "description": "Apply in small projects.", "order_index": 2},
                    {"name": "Execution", "description": "Launch / apply / ship.", "order_index": 3},
                ],
                "milestones": [],
                "timeline_months": 3,
            }

        state.roadmap = roadmap
        return roadmap


class SkillGapAgent:
    def run(self, state, profile: UserProfile, self_skills: List[str]):
        system_prompt = """
You are the Skill Gap Agent.
Return ONLY a valid JSON object. NO text before or after it.

Schema:
{
  "required_skills": ["..."],
  "user_skills": ["..."],
  "missing_skills": ["..."],
  "priority_skills": ["..."]
}
"""
        user_prompt = f"""
Goal: {state.goal_text}

User self-reported skills:
{json.dumps(self_skills, indent=2)}

Step 1: Infer the key skills needed for this goal.
Step 2: Compare them with user skills.
Step 3: Fill the JSON fields accordingly.
Return ONLY the JSON.
"""
        raw = call_llm(system_prompt, user_prompt)
        gaps = extract_json(raw)

        if not gaps:
            gaps = {
                "required_skills": [],
                "user_skills": self_skills,
                "missing_skills": [],
                "priority_skills": [],
            }

        state.skill_gaps = gaps
        return gaps


class PlannerAgent:
    def run(self, state, profile: UserProfile, horizon_days: int = 7):
        system_prompt = """
You are the Planner Agent.
Return ONLY a valid JSON object.

Schema:
{
  "weekly_plan": [
    {
      "day": "Day 1",
      "items": [
        { "title": "Task", "type": "learning/practice/execution", "est_minutes": 30 }
      ]
    }
  ],
  "today_plan": [
    { "title": "Task", "type": "learning", "est_minutes": 25 }
  ]
}
"""
        user_prompt = f"""
Profile:
{json.dumps(asdict(profile), indent=2)}

Roadmap:
{json.dumps(state.roadmap, indent=2)}

Skill gaps:
{json.dumps(state.skill_gaps, indent=2)}

User can spend about {profile.hours_per_week} hours per week.

Create a realistic {horizon_days}-day plan and a short today_plan.
Return ONLY the JSON.
"""
        raw = call_llm(system_prompt, user_prompt)
        plan = extract_json(raw)

        if not plan:
            plan = {"weekly_plan": [], "today_plan": []}

        state.plan = plan
        return plan



# class GoalStrategyAgent:
#     def run(self, state: SessionState, profile: UserProfile):
#         system_prompt = """
# You are the Goal Strategy Agent ("Architect").
# Convert ANY goal (business, career, fitness, creativity)
# into a SMART goal and roadmap.

# STRICT JSON:
# {
#  "smart_goal": "",
#  "phases": [{"name":"", "description":"", "order_index":1}],
#  "milestones": [{"phase_name":"", "milestone":"", "success_criteria":""}],
#  "timeline_months": 0
# }
# """
#         user_prompt = f"""
# User: {json.dumps(asdict(profile), indent=2)}
# Goal: {state.goal_text}
# """
#         raw = call_llm(system_prompt, user_prompt)
#         try:
#             roadmap = json.loads(raw)
#         except:
#             roadmap = {
#                 "smart_goal": f"Advance goal: {state.goal_text}",
#                 "phases": [
#                     {"name":"Foundations","description":"Learn basics","order_index":1},
#                     {"name":"Practice","description":"Apply skills","order_index":2},
#                     {"name":"Execution","description":"Launch or apply","order_index":3},
#                 ],
#                 "milestones": [],
#                 "timeline_months": 6
#             }
#         state.roadmap = roadmap
#         return roadmap



# class SkillGapAgent:
#     def run(self, state, profile, self_skills):
#         system_prompt = """
# Infer required skills for ANY goal and compare to user's skills.

# STRICT JSON:
# {
#  "required_skills": [],
#  "user_skills": [],
#  "missing_skills": [],
#  "priority_skills": []
# }
# """
#         raw = call_llm(system_prompt, f"Goal: {state.goal_text}\nUser skills: {self_skills}")
#         try:
#             gaps = json.loads(raw)
#         except:
#             gaps = {
#                 "required_skills": [],
#                 "user_skills": self_skills,
#                 "missing_skills": [],
#                 "priority_skills": []
#             }
#         state.skill_gaps = gaps
#         return gaps



# class PlannerAgent:
#     def run(self, state: SessionState, profile: UserProfile, horizon_days=7):
#         system_prompt = """
# Create a realistic 7-day plan based on roadmap + skill gaps + time.

# STRICT JSON:
# {
#  "weekly_plan": [],
#  "today_plan": []
# }
# """
#         user_prompt = f"""
# Profile: {json.dumps(asdict(profile), indent=2)}
# Roadmap: {json.dumps(state.roadmap, indent=2)}
# Skill gaps: {json.dumps(state.skill_gaps, indent=2)}
# """
#         raw = call_llm(system_prompt, user_prompt)
#         try:
#             plan = json.loads(raw)
#         except:
#             plan = {"weekly_plan": [], "today_plan": []}
#         state.plan = plan
#         return plan



class TutorAgent:
    def run(self, topic, level="beginner"):
        system_prompt = """
Explain topic + give example + quiz.

STRICT JSON:
{
 "explanation": "",
 "example": "",
 "quiz_questions": []
}
"""
        raw = call_llm(system_prompt, f"Topic: {topic}, Level: {level}")
        try:
            data = json.loads(raw)
        except:
            data = {
                "explanation": f"Explanation of {topic}",
                "example": f"Example for {topic}",
                "quiz_questions": ["Q1","Q2","Q3"]
            }
        return data



class MindsetAgent:
    def run(self, message, goal=""):
        system_prompt = """
You are the Mindset Agent.
Provide reflection, reframing, and 1â€“3 actionable steps.

STRICT JSON:
{
 "reflection": "",
 "reframed_view": "",
 "next_steps": []
}
"""
        raw = call_llm(system_prompt, f"User message: {message}\nGoal: {goal}")
        try:
            return json.loads(raw)
        except:
            return {
                "reflection": "It's okay to feel this way.",
                "reframed_view": "Feeling stuck means you're aware and trying.",
                "next_steps": ["Do one small task.", "Clarify what feels heavy."]
            }



session_manager = SessionManager()

strategy = GoalStrategyAgent()
skills = SkillGapAgent()
planner = PlannerAgent()
tutor_agent = TutorAgent()
coach = MindsetAgent()

def run_pipeline(session_id, user_id, goal, profile, user_skills):
    state = session_manager.get_or_create(session_id, user_id)
    state.goal_text = goal
    state.profile = profile.__dict__

    roadmap = strategy.run(state, profile)
    gaps = skills.run(state, profile, user_skills)
    plan = planner.run(state, profile)

    return {
        "roadmap": roadmap,
        "skill_gaps": gaps,
        "plan": plan
    }

def tutor(topic):
    return tutor_agent.run(topic)

def mindset(message, session_id):
    state = session_manager.get_or_create(session_id, "user")
    summary = state.roadmap.get("smart_goal", "")
    return coach.run(message, summary)



def interactive_chat():
    print("ðŸš€ Welcome to Pathmate â€” AI Goal Execution OS")
    print("You can type ANY goal. Commands:")
    print(" - /tutor <topic>")
    print(" - /mindset <message>")
    print(" - quit\n")

    name = input("Name: ")
    role = input("Current role: ")
    exp = int(input("Years of experience: ") or 0)
    hours = int(input("Hours/week for goal: ") or 5)

    profile = UserProfile(
        name=name,
        background=f"{role} with {exp} years experience",
        experience_years=exp,
        current_role=role,
        hours_per_week=hours
    )

    skills_raw = input("List a few skills (comma separated): ")
    user_skills = [s.strip() for s in skills_raw.split(",")]

    session = "user-session"

    while True:
        msg = input("\nYou: ").strip()

        if msg.lower() == "quit":
            print("Goodbye! Stay consistent ðŸŒ±")
            break

        if msg.startswith("/tutor"):
            topic = msg.replace("/tutor", "").strip()
            data = tutor(topic)
            print("\nðŸ“˜ Tutor:")
            print("Explanation:", data["explanation"])
            print("Example:", data["example"])
            print("Quiz:", data["quiz_questions"])
            continue

        if msg.startswith("/mindset"):
            thought = msg.replace("/mindset", "").strip()
            data = mindset(thought, session)
            print("\nðŸ§  Coach:")
            print("Reflection:", data["reflection"])
            print("Reframe:", data["reframed_view"])
            print("Next steps:")
            for s in data["next_steps"]:
                print("-", s)
            continue

        print("\nðŸ”„ Building your roadmap...")
        out = run_pipeline(session, name, msg, profile, user_skills)

        print("\nðŸŽ¯ SMART Goal:", out["roadmap"]["smart_goal"])
        print("\nðŸ›¤ Phases:")
        for p in out["roadmap"]["phases"]:
            print("-", p["name"])

        print("\nðŸ“š Skill Gaps:")
        for s in out["skill_gaps"]["missing_skills"]:
            print("-", s)

        print("\nðŸ—“ Todayâ€™s Plan:")
        for t in out["plan"].get("today_plan", []):
            print("-", t.get("title"))



interactive_chat()


