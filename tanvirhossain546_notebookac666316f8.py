# Agent MedicAI - Text Chat (Kaggle Capstone)
# Single-file implementation suitable for a Kaggle Notebook or local script.
# Demonstrates: multi-agent orchestration, tools, memory, observability, evaluation.

import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional


# Optional: load Google API key from Kaggle Secrets
# NOTE: No API keys are hard-coded; this reads from the environment-safe secret store.
try:
    from kaggle_secrets import UserSecretsClient
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
except Exception:
    # If not on Kaggle or secret is missing, we simply skip setting the key here.
    pass


# Core ADK imports: Agent abstraction, in-memory runner, and AgentTool wrapper.
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools.agent_tool import AgentTool

# Types from Gemini client for streaming content and parts.
from google.genai import types

# ---- App-level constants ----
APP_NAME = "agent_medicai"             # Logical name of this agentic application
USER_ID = "demo_user"                  # Simple single-user demo identifier
MODEL_NAME = "gemini-2.5-flash"        # LLM used for all agents
USERS_FILE = "users_memory.json"       # File backing long-term user memory
CURRICULUM_FILE = "curriculum_memory.json"  # File backing curriculum graph


# ---- Logging & Observability ----

def setup_logging():
    """Configure root logger for the Agent MedicAI app."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("AgentMedicAI")

LOGGER = setup_logging()

# Simple in-memory metrics accumulator for observability.
METRICS = {
    "tool_calls": 0,        # Total number of tool invocations
    "lessons_served": 0,    # Number of lessons returned to the user
    "quizzes_served": 0,    # Number of quiz questions served
    "quizzes_correct": 0,   # Number of correct quiz answers
}

def log_tool_call(name: str, payload: Dict[str, Any]):
    """
    Helper to log and count tool calls.
    - Increments METRICS["tool_calls"].
    - Writes a structured log entry to the logger.
    """
    METRICS["tool_calls"] += 1
    LOGGER.info(f"[ToolCall] {name} | payload={payload}")


# ---- MemoryBank: long-term user + curriculum storage ----

class MemoryBank:
    """
    JSON-backed persistence layer for:
    - User profiles (age band, mastery, last lesson)
    - Curriculum graph (tracks, lessons, facts, analogies, quizzes)

    This implements "long-term memory" for the agent.
    """

    def __init__(self, users_path: str, curriculum_path: str):
        self.users_path = users_path
        self.curriculum_path = curriculum_path
        self._ensure_files()  # Make sure backing JSON files exist

    def _ensure_files(self):
        """Create empty users file and sample curriculum file if they don't exist yet."""
        if not os.path.exists(self.users_path):
            with open(self.users_path, "w") as f:
                json.dump({}, f)

        if not os.path.exists(self.curriculum_path):
            sample_curriculum = self._build_sample_curriculum()
            with open(self.curriculum_path, "w") as f:
                json.dump(sample_curriculum, f, indent=2)

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Retrieve the user profile for a given user_id.
        If the user does not exist yet, create a default profile with:
        - age_band: default to G3_5
        - mastery: 0.0 for each track
        - last_track / last_lesson_id: None
        """
        data = self._read_json(self.users_path)
        if user_id not in data:
            data[user_id] = {
                "age_band": "G3_5",
                "mastery": {
                    "anatomy": 0.0,
                    "circulation": 0.0,
                    "immunity": 0.0,
                    "cells": 0.0,
                    "careers_ethics": 0.0,
                },
                "last_track": None,
                "last_lesson_id": None,
            }
            self._write_json(self.users_path, data)
        return data[user_id]

    def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a partial update to the user profile and persist to disk.
        This is used by tools such as set_age_band_tool.
        """
        data = self._read_json(self.users_path)
        profile = data.get(user_id, self.get_user_profile(user_id))
        profile.update(updates)
        data[user_id] = profile
        self._write_json(self.users_path, data)
        LOGGER.info(f"[Memory] Updated user profile for {user_id}: {updates}")
        return profile

    def update_mastery(self, user_id: str, track: str, delta: float) -> Dict[str, Any]:
        """
        Incrementally adjust mastery for a given track based on quiz performance.
        - Ensures the mastery score stays within [0.0, 1.0].
        """
        data = self._read_json(self.users_path)
        profile = data.get(user_id, self.get_user_profile(user_id))
        mastery = profile.get("mastery", {})
        old_val = mastery.get(track, 0.0)
        new_val = max(0.0, min(1.0, old_val + delta))
        mastery[track] = new_val
        profile["mastery"] = mastery
        data[user_id] = profile
        self._write_json(self.users_path, data)
        return profile

    def get_curriculum(self) -> Dict[str, Any]:
        """Load the full curriculum graph from disk."""
        return self._read_json(self.curriculum_path)

    def get_track_lessons(self, track: str, age_band: str) -> List[Dict[str, Any]]:
        """
        Retrieve all lessons for a given track and age band from the curriculum.
        For example: track="circulation", age_band="G3_5".
        """
        curriculum = self.get_curriculum()
        return curriculum.get(track, {}).get(age_band, [])

    def get_next_lesson(self, user_id: str, track: str, age_band: str) -> Optional[Dict[str, Any]]:
        """
        Compute the "next" lesson for a user based on their last_lesson_id.
        - If no lessons exist, returns None.
        - If user has no history, returns the first lesson.
        - Otherwise returns the lesson after last_lesson_id, clamped to the final lesson.
        """
        profile = self.get_user_profile(user_id)
        lessons = self.get_track_lessons(track, age_band)
        if not lessons:
            return None

        last_lesson_id = profile.get("last_lesson_id")
        next_idx = 0

        if last_lesson_id is not None:
            for i, lesson in enumerate(lessons):
                if lesson["lesson_id"] == last_lesson_id:
                    next_idx = min(i + 1, len(lessons) - 1)
                    break

        next_lesson = lessons[next_idx]

        # Persist updated last track / last lesson to user profile
        self.update_user_profile(user_id, {
            "last_track": track,
            "last_lesson_id": next_lesson["lesson_id"],
        })
        return next_lesson

    def _read_json(self, path: str) -> Dict[str, Any]:
        """Utility to safely read JSON from disk, returning {} on error."""
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _write_json(self, path: str, data: Dict[str, Any]):
        """Write JSON data to disk with indentation for readability."""
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _build_sample_curriculum(self) -> Dict[str, Any]:
        """
        Build a minimal sample curriculum.
        - Circulation track, G3_5 age band.
        - Contains objectives, facts, analogy, and a single quiz question.
        This keeps the example small but extensible.
        """
        return {
            "circulation": {
                "G3_5": [
                    {
                        "lesson_id": "circ_1_heart_basics",
                        "title": "What Does the Heart Do?",
                        "objectives": ["Understand that the heart is a pump for blood"],
                        "facts": [
                            "Your heart is a strong muscle that pumps blood all around your body.",
                            "Blood carries oxygen and food to your body's cells.",
                        ],
                        "analogy": "Think of your heart like a water pump that sends water through pipes.",
                        "quiz": [{
                            "q": "What does your heart pump?",
                            "choices": ["Water", "Blood", "Air"],
                            "answer": "Blood",
                        }],
                    }
                ]
            }
        }


# Global MemoryBank instance used by all tools / agents.
MEMORY = MemoryBank(USERS_FILE, CURRICULUM_FILE)


# ---- Tool Functions (exposed to agents) ----

def get_user_profile_tool() -> Dict[str, Any]:
    """
    Tool: Return the current user's long-term profile (age_band, mastery, last lesson, etc.).
    Used by TutorCoordinator and CurriculumAgent.
    """
    log_tool_call("get_user_profile_tool", {"user_id": USER_ID})
    return MEMORY.get_user_profile(USER_ID)


def set_age_band_tool(age_band: str) -> Dict[str, Any]:
    """
    Tool: Update the learner's age band (e.g., K2, G3_5, G6_8, HS).
    This lets the user or tutor adjust reading level explicitly.
    """
    log_tool_call("set_age_band_tool", {"user_id": USER_ID, "age_band": age_band})
    return MEMORY.update_user_profile(USER_ID, {"age_band": age_band})


def get_next_lesson_tool(track: str) -> Dict[str, Any]:
    """
    Tool: Retrieve the next lesson for the current user in a given track.
    - Respects stored age_band.
    - Updates lessons_served metric.
    """
    profile = MEMORY.get_user_profile(USER_ID)
    age_band = profile.get("age_band", "G3_5")
    log_tool_call("get_next_lesson_tool", {
        "user_id": USER_ID,
        "track": track,
        "age_band": age_band,
    })
    lesson = MEMORY.get_next_lesson(USER_ID, track, age_band)
    if lesson is None:
        return {"status": "error", "message": f"No lessons for {track}/{age_band}"}
    METRICS["lessons_served"] += 1
    return {"status": "ok", "track": track, "age_band": age_band, "lesson": lesson}


def fetch_fact_tool(topic: str) -> Dict[str, Any]:
    """
    Tool: Perform a simple keyword scan across the curriculum to find matching lessons.
    - Used to answer topical questions like "Tell me about the heart".
    - Returns up to 2 matching lesson snippets with facts and analogy.
    """
    profile = MEMORY.get_user_profile(USER_ID)
    age_band = profile.get("age_band", "G3_5")
    log_tool_call("fetch_fact_tool", {
        "user_id": USER_ID,
        "topic": topic,
        "age_band": age_band,
    })

    curriculum = MEMORY.get_curriculum()
    matches: List[Dict[str, Any]] = []

    for track, by_age in curriculum.items():
        for band, lessons in by_age.items():
            for lesson in lessons:
                text_blob = " ".join(
                    [lesson.get("title", "")] +
                    lesson.get("facts", []) +
                    [lesson.get("analogy", "")]
                )
                if topic.lower() in text_blob.lower():
                    matches.append({
                        "track": track,
                        "age_band": band,
                        "lesson_id": lesson["lesson_id"],
                        "title": lesson["title"],
                        "facts": lesson.get("facts", [])[:3],
                        "analogy": lesson.get("analogy"),
                    })

    if not matches:
        return {"status": "not_found", "topic": topic, "results": []}

    # Limit to the first 2 matches to keep responses compact.
    return {"status": "ok", "topic": topic, "results": matches[:2]}


def quiz_item_tool(track: str) -> Dict[str, Any]:
    """
    Tool: Retrieve a quiz question for the next lesson in a given track.
    - Relies on get_next_lesson() to stay aligned with current progress.
    - Increments quizzes_served metric.
    """
    profile = MEMORY.get_user_profile(USER_ID)
    age_band = profile.get("age_band", "G3_5")
    log_tool_call("quiz_item_tool", {
        "user_id": USER_ID,
        "track": track,
        "age_band": age_band,
    })

    lesson = MEMORY.get_next_lesson(USER_ID, track, age_band)
    if not lesson or "quiz" not in lesson:
        return {
            "status": "error",
            "message": f"No quiz available for track={track}, age_band={age_band}",
        }

    quizzes = lesson["quiz"]
    if not quizzes:
        return {"status": "error", "message": "No quiz questions defined"}

    q = quizzes[0]
    METRICS["quizzes_served"] += 1

    return {
        "status": "ok",
        "track": track,
        "age_band": age_band,
        "lesson_id": lesson["lesson_id"],
        "question": q["q"],
        "choices": q["choices"],
        "answer": q["answer"],  # Returned so QuizAgent/Tutor can grade user response.
    }


def grade_quiz_tool(track: str, user_answer: str, correct_answer: str) -> Dict[str, Any]:
    """
    Tool: Grade a quiz answer and update mastery.
    - Compares user_answer vs correct_answer (case-insensitive).
    - Adjusts mastery slightly up (correct) or down (incorrect).
    """
    log_tool_call("grade_quiz_tool", {
        "user_id": USER_ID,
        "track": track,
        "user_answer": user_answer,
        "correct_answer": correct_answer,
    })

    is_correct = user_answer.strip().lower() == correct_answer.strip().lower()
    METRICS["quizzes_correct"] += int(is_correct)

    # Small positive bump on correct, small negative bump on incorrect.
    delta = 0.05 if is_correct else -0.02
    profile = MEMORY.update_mastery(USER_ID, track, delta)

    return {
        "status": "ok",
        "correct": is_correct,
        "new_mastery": profile["mastery"].get(track, 0.0),
    }


# ---- Multi-Agent Definitions ----

# CurriculumAgent: responsible for retrieving and structuring curriculum content.
curriculum_instruction = """
You are the CurriculumAgent for Agent MedicAI. Retrieve lessons and facts,
and prepare short, age-appropriate teaching content for the TutorCoordinator.
"""
curriculum_agent = Agent(
    name="CurriculumAgent",
    model=MODEL_NAME,
    description="Retrieves and structures curriculum content.",
    instruction=curriculum_instruction,
    tools=[get_next_lesson_tool, fetch_fact_tool, get_user_profile_tool],
)

# QuizAgent: responsible for quiz retrieval and grading.
quiz_instruction = """
You are the QuizAgent for Agent MedicAI. Retrieve quiz questions and grade answers.
"""
quiz_agent = Agent(
    name="QuizAgent",
    model=MODEL_NAME,
    description="Provides quiz questions and grading.",
    instruction=quiz_instruction,
    tools=[quiz_item_tool, grade_quiz_tool],
)

# TutorCoordinator: root orchestrator and user-facing tutor.
tutor_instruction = """
You are TutorCoordinator for Agent MedicAI. You teach medical science to kids and teens.
Adapt your explanations based on age_band from get_user_profile_tool.
Never provide medical advice or dosing; you only teach concepts.
"""
tutor_agent = Agent(
    name="TutorCoordinator",
    model=MODEL_NAME,
    description="Coordinates teaching, curriculum, and quizzes.",
    instruction=tutor_instruction,
    tools=[
        # Wrap sub-agents so TutorCoordinator can delegate to them as tools.
        AgentTool(curriculum_agent),
        AgentTool(quiz_agent),

        # Direct access to tools for memory and curriculum.
        get_user_profile_tool,
        set_age_band_tool,
        get_next_lesson_tool,
        fetch_fact_tool,
        quiz_item_tool,
        grade_quiz_tool,
    ],
)

# Root agent for the app is the TutorCoordinator.
root_agent = tutor_agent

# In-memory runner handles sessions, state, and streaming events.
runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)


# ---- Runner helpers for sessions and turns ----

async def create_session(user_id: str = USER_ID):
    """
    Create a new chat session for a given user.
    - Uses InMemoryRunner's session_service for ephemeral, in-memory sessions.
    """
    session = await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id=user_id,
    )
    return session


async def run_turn(session_id: str, user_text: str, user_id: str = USER_ID) -> str:
    """
    Execute one conversational turn:
    - Wraps user_text into a Content object.
    - Streams events from runner.run_async.
    - Collects all assistant text parts into a single string response.
    """
    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_text)],
    )

    parts: List[str] = []

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
    ):
        event_content = getattr(event, "content", None)
        if event_content and event_content.parts:
            for p in event_content.parts:
                if p.text:
                    parts.append(p.text)

    # Join all streamed text parts into one final answer string.
    return "\n".join(parts).strip()


async def interactive_chat():
    """
    Simple interactive CLI loop for local testing.
    - Creates a session once.
    - Reads user input until 'exit' or 'quit'.
    - Prints responses from Agent MedicAI.
    """
    session = await create_session(USER_ID)
    while True:
        text = input("You: ").strip()
        if text.lower() in ["exit", "quit"]:
            break
        answer = await run_turn(session.id, text)
        print("MedicAI:", answer)



#Demo: 
async def demo_interaction():
    # 1. Create a new session
    session = await create_session(USER_ID)
    print("Session created:", session.id)
    print("")

    # 2. User asks to learn about the heart (TutorCoordinator)
    print("User: Teach me about the heart.")
    reply1 = await run_turn(session.id, "Teach me about the heart.")
    print("MedicAI:", reply1)
    print("")

    # 3. User asks for a quiz (QuizAgent triggered by TutorCoordinator)
    print("User: Yes, quiz me.")
    reply2 = await run_turn(session.id, "Yes, quiz me.")
    print("MedicAI:", reply2)
    print("")

    # 4. User answers the quiz (graded via grade_quiz_tool)
    print("User: Blood")
    reply3 = await run_turn(session.id, "Blood")
    print("MedicAI:", reply3)
    print("")

# ðŸ‘‡ Run the demo in a notebook cell
await demo_interaction()

