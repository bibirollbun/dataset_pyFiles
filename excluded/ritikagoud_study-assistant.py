# Cell 1: Setup & detect ADK availability
try:
    import google
    from google.adk.agents import Agent
    from google.adk.models.google_llm import Gemini
    from google.adk.runners import InMemoryRunner
    from google.adk.tools import tool
    from google.adk.tools import google_search
    from google.genai import types
    ADK_AVAILABLE = True
except Exception:
    ADK_AVAILABLE = False

import os, json, random, textwrap
print('ADK available:', ADK_AVAILABLE)


# Cell 2: Utilities (fallback implementations)
import json, os

def make_study_plan(topic: str, hours: int = 2):
    parts = ["Overview", "Core Concepts", "Practice & Revision"]
    per_part = max(1, hours // len(parts))
    plan = {p: f"Spend ~{per_part}h on {p} of {topic}. Key actions: read summary, make notes, do 3 questions." for p in parts}
    return {"topic": topic, "hours": hours, "plan": plan}

def generate_quiz(topic: str, n: int = 3):
    sample_qs = [
        {"q": f"What is the main idea of {topic}?", "options": ["A: Summary", "B: Detail", "C: Example", "D: None"], "answer": "A"},
        {"q": f"Which method is used in {topic} for analysis?", "options": ["A", "B", "C", "D"], "answer": "B"},
        {"q": f"Pick true statement about {topic}", "options": ["True", "False"], "answer": "True"}
    ]
    return sample_qs[:n]

def save_memory(path: str, memory: dict):
    with open(path, 'w') as f:
        json.dump(memory, f, indent=2)

def load_memory(path: str):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"history": []}

MEMORY_PATH = 'study_memory.json'
memory = load_memory(MEMORY_PATH)
print('Utilities loaded. Memory items:', len(memory.get('history', [])))



# Cell 3: ADK Agent (optional) - only run if ADK_AVAILABLE and GOOGLE_API_KEY set
if ADK_AVAILABLE:
    from google.adk.tools import tool
    @tool
    def planner_tool(prompt_text: str) -> dict:
        return make_study_plan(prompt_text, hours=2)

    @tool
    def quiz_tool(prompt_text: str) -> list:
        return generate_quiz(prompt_text, n=3)

    @tool
    def memory_tool(action_json: str) -> str:
        try:
            payload = json.loads(action_json)
            if payload.get('action') == 'save':
                mem = load_memory(MEMORY_PATH)
                mem.setdefault('history', []).append(payload.get('record'))
                save_memory(MEMORY_PATH, mem)
                return 'saved'
        except Exception as e:
            return f'error: {e}'
        return 'no-op'

    root_agent = Agent(
        name='smart_study_agent',
        model=Gemini(model='gemini-2.5-flash-lite', retry_options=types.HttpRetryOptions(attempts=3, initial_delay=1, exp_base=2)),
        description='Agent to help create study plans, quizzes and store memory',
        instruction='You are a study assistant. Use planner_tool, quiz_tool, and memory_tool when asked to produce structured outputs.',
        tools=[planner_tool, quiz_tool, memory_tool]
    )
    runner = InMemoryRunner(agent=root_agent)
    print('ADK agent prepared. Use await runner.run_debug("...") in a notebook cell to test.')
else:
    print('ADK not available - skip ADK agent cells.')



# Cell 4: Fallback Agent - pure Python
class FallbackAgent:
    def __init__(self, memory_path=MEMORY_PATH):
        self.memory_path = memory_path
        self.memory = load_memory(memory_path)

    def plan(self, topic: str, hours: int = 2):
        return make_study_plan(topic, hours)

    def quiz(self, topic: str, n: int = 3):
        return generate_quiz(topic, n)

    def save_progress(self, topic: str, note: str):
        rec = {"topic": topic, "note": note}
        self.memory.setdefault('history', []).append(rec)
        save_memory(self.memory_path, self.memory)
        return {'status': 'saved', 'record': rec}

    def summary(self, topic: str):
        return f"Short summary for {topic}: key ideas, formulae, and example questions to practice."

agent = FallbackAgent()
print('Fallback agent ready.')



# Cell 5: Demo - run this to generate outputs and create study_memory.json
TOPIC = 'Hypothesis Testing'
print('--- PLAN ---')
plan = agent.plan(TOPIC, hours=2)
print(plan)
print('\n--- QUIZ ---')
quiz = agent.quiz(TOPIC, n=3)
print(quiz)
print('\n--- SAVE PROGRESS ---')
save = agent.save_progress(TOPIC, 'Read summary and solved 5 sample questions')
print(save)
print('\n--- MEMORY CONTENTS ---')
print(load_memory(MEMORY_PATH))


