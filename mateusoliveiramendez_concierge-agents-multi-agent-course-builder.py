# Section 1 – Setup and Authentication

import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    # Force Gemini API (not Vertex) in this notebook
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("✅ Gemini API key setup complete.")
except Exception as e:
    print(
        "❌ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' "
        "to your Kaggle Secrets (Add-ons ➜ Secrets)."
    )



# Section 1.2 – Import ADK and supporting libraries

from typing import List, Dict, Any
import logging

from google.genai import types

from google.adk.models.google_llm import Gemini
from google.adk.agents import (
    LlmAgent,
)
from google.adk.tools import (
    google_search,
    AgentTool,
    ToolContext,
)
from google.adk.tools.function_tool import FunctionTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService

print("✅ ADK components imported successfully.")



from google.genai import types

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

APP_NAME = "CourseCraftConciergeApp"
USER_ID = "demo_user"

print("Retry config loaded.")



# Section 4 – Custom helper tools

def normalize_duration(weeks: int, hours_per_week: float) -> Dict[str, Any]:
    """
    Simple helper to compute total hours and recommended lesson length.
    """
    total_hours = weeks * hours_per_week
    # naive heuristic: 1.5 hours per lesson
    lessons_estimate = max(1, round(total_hours / 1.5))
    return {
        "weeks": weeks,
        "hours_per_week": hours_per_week,
        "total_hours": total_hours,
        "estimated_lessons": lessons_estimate,
    }


def bloom_level_suggestions(level: str) -> List[str]:
    """
    Suggest action verbs for a given Bloom level.
    """
    level = level.lower().strip()
    mapping = {
        "remember": ["define", "list", "recall"],
        "understand": ["explain", "describe", "summarize"],
        "apply": ["use", "implement", "demonstrate"],
        "analyze": ["compare", "contrast", "differentiate"],
        "evaluate": ["justify", "critique", "defend"],
        "create": ["design", "build", "compose"],
    }
    return mapping.get(level, ["explain", "describe"])


def save_preference(tool_context: ToolContext, key: str, value: str) -> Dict[str, Any]:
    """
    Stub: In this ADK version ToolContext does not expose session_state,
    so we just echo back the preference instead of storing it.
    This still works as a custom tool for the agent.
    """
    state_key = f"user:{key}"
    return {"saved_key": state_key, "saved_value": value}


def load_preferences(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Stub: Return an empty preference dict. Kept to avoid errors and
    to demonstrate a read-style tool, even if this runtime doesn't
    give us session_state on ToolContext.
    """
    return {}



# Wrap Python functions as ADK tools.
# In this ADK version you just call FunctionTool(func), no .from_fn().

normalize_duration_tool = FunctionTool(
    normalize_duration,
)

bloom_level_tool = FunctionTool(
    bloom_level_suggestions,
)

save_preference_tool = FunctionTool(
    save_preference,
)

load_preferences_tool = FunctionTool(
    load_preferences,
)

print("✅ Custom helper tools created.")



# Section 5 – Specialized sub-agents

model = Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config)

requirements_agent = LlmAgent(
    model=model,
    name="RequirementsAgent",
    instruction=(
        "You are a requirements gathering specialist for course design. "
        "Your job is to turn a vague request into a precise course spec. "
        "If something is missing, explicitly list open questions. "
        "Output a JSON-like sketch of the final spec at the end under 'spec'."
    ),
    tools=[
        save_preference_tool,
        load_preferences_tool,
        normalize_duration_tool,
    ],
)

curriculum_agent = LlmAgent(
    model=model,
    name="CurriculumAgent",
    instruction=(
        "You are a curriculum design expert. "
        "Given a course spec, design modules and lessons with:\n"
        "- Learning objectives (with Bloom level and associated verb)\n"
        "- Lesson titles and brief descriptions\n"
        "- Approx duration per lesson\n\n"
        "Be very structured and explicit. Use the helper tools when appropriate."
    ),
    tools=[        
        bloom_level_tool,
        load_preferences_tool,
    ],
)

assessment_agent = LlmAgent(
    model=model,
    name="AssessmentAgent",
    instruction=(
        "You design assessments that align with learning objectives. "
        "Given modules and objectives, propose:\n"
        "- Formative checks per module (quizzes, quick tasks)\n"
        "- Summative projects or exams\n"
        "Explain how each assessment maps to objectives."
    ),
    tools=[
        load_preferences_tool,
    ],
)

reviewer_agent = LlmAgent(
    model=model,
    name="ReviewerAgent",
    instruction=(
        "You are a critical course reviewer. "
        "Evaluate the overall course plan for:\n"
        "- Alignment between objectives, content, and assessments\n"
        "- Level progression (from easier to harder)\n"
        "- Time and workload balance\n"
        "Suggest specific improvements. When the plan is acceptable, "
        "summarize it clearly at the end under 'final_course_plan'."
    ),
    tools=[
        load_preferences_tool,
    ],
)
# Search agent that ONLY uses google_search (no FunctionTools)

from google.adk.tools import google_search
from google.adk.agents import LlmAgent

search_agent = LlmAgent(
    model=model,
    name="SearchAgent",
    instruction=(
        "You perform web searches using google_search. "
        "Extract concise, useful facts. "
        "Do not generate course designs; only surface information relevant to the topic."
    ),
    tools=[google_search],   # IMPORTANT: ONLY built-in tool
)


print("✅ Sub-agents defined.")



# Section 6 – Sessions, Memory, and Runners for each agent

from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.runners import Runner

# Shared session + memory services
session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()

# One runner per sub-agent (all share same services)
requirements_runner = Runner(
    agent=requirements_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service,
)

curriculum_runner = Runner(
    agent=curriculum_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service,
)

assessment_runner = Runner(
    agent=assessment_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service,
)

reviewer_runner = Runner(
    agent=reviewer_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service,
)

search_runner = Runner(
    agent=search_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service,
)


print("✅ Runners for all sub-agents are ready.")



# Section 7 – Session helper and generic call_runner

import asyncio

async def ensure_session(session_id: str):
    existing = await session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
    )
    if existing is None:
        print(f"Creating new session: {session_id}")
        await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=session_id,
        )
    else:
        print(f"Reusing existing session: {session_id}")


async def call_runner(runner: Runner, prompt: str, session_id: str) -> str:
    await ensure_session(session_id)

    print(f"\n=== {runner.agent.name} ===")
    content = types.Content(
        role="user",
        parts=[types.Part(text=prompt)],
    )

    full_text = ""

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=content,
    ):
        # In this ADK version, text is on event.content
        if event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    text = part.text
                    print(text, end="", flush=True)
                    full_text += text

    print("\n" + "-" * 80)
    return full_text


print("✅ Session helpers ready.")



# Section 8 – Full course builder workflow (sequential multi-agent)

async def run_full_course_builder(user_prompt: str, session_id: str = "python_beginners"):
    
    # STEP 1 – Requirements
    # STEP 0 – Search (optional)
    search_prompt = (
        "Search the web for introductory resources, common topics, and typical skills "
        "for the following topic:\n" + user_prompt
    )

    search_text = await call_runner(search_runner, search_prompt, session_id)

# STEP 1 – Requirements
    req_prompt = (
        "Here is some external context that may help:\n"
        f"{search_text}\n\n"
        "Now clarify the course requirements:\n"
        f"{user_prompt}"
    )

    req_text = await call_runner(requirements_runner, req_prompt, session_id)

    req_text = await call_runner(requirements_runner, user_prompt, session_id)

    # STEP 2 – Curriculum from requirements
    cur_prompt = (
        "Using the clarified course requirements below, design a course curriculum "
        "with modules, lessons, and learning objectives.\n\n"
        "=== REQUIREMENTS ===\n"
        f"{req_text}\n"
    )
    cur_text = await call_runner(curriculum_runner, cur_prompt, session_id)

    # STEP 3 – Assessments from curriculum
    ass_prompt = (
        "Using the curriculum below, design assessments and projects that align with "
        "the learning objectives. Include formative and summative assessments.\n\n"
        "=== CURRICULUM ===\n"
        f"{cur_text}\n"
    )
    ass_text = await call_runner(assessment_runner, ass_prompt, session_id)

    # STEP 4 – Review everything
    rev_prompt = (
        "Review the following course design for alignment, progression, and workload. "
        "Then present a final improved course plan.\n\n"
        "=== REQUIREMENTS ===\n"
        f"{req_text}\n\n"
        "=== CURRICULUM ===\n"
        f"{cur_text}\n\n"
        "=== ASSESSMENTS ===\n"
        f"{ass_text}\n"
    )
    final_text = await call_runner(reviewer_runner, rev_prompt, session_id)

    return {
        "requirements": req_text,
        "curriculum": cur_text,
        "assessments": ass_text,
        "final_plan": final_text,
    }



# Section 9 – Demo: Design a course in one go

test_prompt = (
    "I want a 4-week beginner Python course for complete beginners. "
    "We have 4 hours per week, remote live sessions, and I prefer project-based learning. "
    "The goal is that learners can write simple scripts to automate small tasks at work."
)

result = await run_full_course_builder(
    test_prompt,
    session_id="python_beginners",
)


