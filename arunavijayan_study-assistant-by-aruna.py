


"""
LLM_Multi-Agent Study Assistant
File: LLM_Multi-Agent_Study_Assistant.py

A self-contained Python implementation of an LLM-powered
Multi-Agent Study Assistant designed for the Kaggle capstone in the Study/Enterprise
Agents style. Demonstrates:
- Multi-agent orchestration (GoalAgent, PlannerAgent, QuizAgent, ReviewAgent, Coordinator)
- Tools: ProgressStore (custom tool)
- Sessions & Memory: SessionService and MemoryBank
- Observability: logging and metrics
- Agent evaluation: simple user-simulation tests

"""

from __future__ import annotations
import json
import os
import re
import time
import uuid
import logging
from typing import Dict, Any, List, Optional

# ---------------------------
# Observability
# ---------------------------
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger('multiagent-study')

metrics = {
    'total_requests': 0,
    'quizzes_generated': 0,
    'reviews_scheduled': 0,
}

# ---------------------------
# Simple LLM interface and MockLLM
# ---------------------------
class LLM:
    def call(self, prompt: str, system: Optional[str] = None) -> str:
        raise NotImplementedError

class MockLLM(LLM):
    def call(self, prompt: str, system: Optional[str] = None) -> str:
        txt = (system or '') + '\n' + prompt
        txt = txt.lower()
        if 'analyze goals' in (system or '').lower() or 'what are the study goals' in prompt.lower():
            topic = 'general'
            days = 7
            match = re.search(r'exam|exam in (\d+)', prompt.lower())
            if 'final' in txt:
                topic = 'finals'
                days = 14
            num_match = re.search(r'(\d+) days', prompt.lower())
            if num_match:
                days = int(num_match.group(1))
            return json.dumps({'topic': topic, 'days_until_exam': days, 'priority': 'high' if days < 7 else 'medium'})
        if 'create plan' in (system or '').lower() or 'generate schedule' in prompt.lower():
            plan = []
            for i in range(1,6):
                plan.append({'day': i, 'sessions': [f'Read chapter {i}', f'Practice problems {i}']})
            return json.dumps({'plan': plan})
        if 'generate quiz' in (system or '').lower() or 'make a quiz' in prompt.lower():
            metrics['quizzes_generated'] += 1
            q = [
                {'q': 'What is the definition of X?', 'a': 'Definition of X'},
                {'q': 'List 3 examples of Y', 'a': 'example1; example2; example3'}
            ]
            return json.dumps({'quiz': q})
        if 'create flashcards' in (system or '').lower() or 'flashcards' in prompt.lower():
            return json.dumps({'cards': [{'front': 'Term A', 'back': 'Definition A'}, {'front': 'Term B', 'back': 'Definition B'}]})
        if 'schedule review' in (system or '').lower() or 'when should they review' in prompt.lower():
            metrics['reviews_scheduled'] += 1
            return json.dumps({'next_review_days': 2})
        return 'I do not understand.'

# ---------------------------
# Sessions & Memory
# ---------------------------
class SessionService:
    def __init__(self):
        self.sessions: Dict[str, List[Dict[str, Any]]] = {}

    def new_session(self) -> str:
        sid = str(uuid.uuid4())
        self.sessions[sid] = []
        logger.debug(f'New session: {sid}')
        return sid

    def append_message(self, session_id: str, role: str, content: str):
        if session_id not in self.sessions:
            raise KeyError('Unknown session')
        self.sessions[session_id].append({'role': role, 'content': content, 'ts': time.time()})

    def get_session(self, session_id: str) -> List[Dict[str, Any]]:
        return self.sessions.get(session_id, [])

class MemoryBank:
    def __init__(self, path: Optional[str] = None):
        self.mem: Dict[str, Dict[str, Any]] = {}
        self.path = path
        if path and os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    self.mem = json.load(f)
                    logger.info('MemoryBank loaded')
            except Exception as e:
                logger.warning('Failed to load memory: ' + str(e))

    def get(self, student_id: str) -> Dict[str, Any]:
        return self.mem.get(student_id, {})

    def update(self, student_id: str, data: Dict[str, Any]):
        existing = self.mem.get(student_id, {})
        existing.update(data)
        self.mem[student_id] = existing
        if self.path:
            with open(self.path, 'w') as f:
                json.dump(self.mem, f, indent=2)

# ---------------------------
# Tools
# ---------------------------
class ProgressStore:
    def __init__(self, path: str = 'progress.json'):
        self.path = path
        if not os.path.exists(self.path):
            with open(self.path, 'w') as f:
                json.dump([], f)

    def save_progress(self, record: Dict[str, Any]) -> str:
        with open(self.path, 'r') as f:
            data = json.load(f)
        rec_id = str(uuid.uuid4())
        record['id'] = rec_id
        data.append(record)
        with open(self.path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f'Progress saved: {rec_id}')
        return rec_id

# ---------------------------
# Agents
# ---------------------------
class GoalAgent:
    def __init__(self, llm: LLM):
        self.llm = llm

    def analyze(self, prompt: str) -> Dict[str, Any]:
        system = 'Analyze goals: identify topic, days_until_exam, and priority. Return JSON.'
        raw = self.llm.call(prompt, system=system)
        try:
            res = json.loads(raw)
            logger.info(f'GoalAgent: {res}')
            return res
        except Exception:
            return {'topic': 'general', 'days_until_exam': 7, 'priority': 'medium'}

class PlannerAgent:
    def __init__(self, llm: LLM):
        self.llm = llm

    def create_plan(self, topic: str, days: int) -> Dict[str, Any]:
        system = 'Create plan: generate a study plan JSON distributing topics across days.'
        prompt = f'Topic: {topic}\nDays: {days}\nGenerate schedule plan as JSON.'
        raw = self.llm.call(prompt, system=system)
        try:
            plan = json.loads(raw)
            logger.info('PlannerAgent created plan')
            return plan
        except Exception:
            return {'plan': [{'day': 1, 'sessions': ['Intro', 'Practice']}]}

class QuizAgent:
    def __init__(self, llm: LLM):
        self.llm = llm

    def generate_quiz(self, topic: str, difficulty: str = 'medium') -> Dict[str, Any]:
        system = 'Generate quiz: produce JSON with a list of Q/A pairs.'
        prompt = f'Create a short quiz for {topic} at {difficulty} difficulty.'
        raw = self.llm.call(prompt, system=system)
        try:
            quiz = json.loads(raw)
            logger.info('QuizAgent generated quiz')
            return quiz
        except Exception:
            return {'quiz': [{'q': 'Sample Q?', 'a': 'Sample A'}]}

class ReviewAgent:
    def __init__(self, llm: LLM):
        self.llm = llm

    def schedule_review(self, performance: Dict[str, Any]) -> Dict[str, Any]:
        system = 'Schedule review: return JSON {next_review_days: int}.'
        prompt = f'Performance: {json.dumps(performance)}\nWhen should they review?'
        raw = self.llm.call(prompt, system=system)
        try:
            res = json.loads(raw)
            logger.info('ReviewAgent scheduled review')
            return res
        except Exception:
            return {'next_review_days': 2}

# ---------------------------
# Coordinator
# ---------------------------
class Coordinator:
    def __init__(self, llm: LLM, progress_store: ProgressStore, memory_bank: MemoryBank, sessions: SessionService):
        self.goal_agent = GoalAgent(llm)
        self.planner = PlannerAgent(llm)
        self.quiz_agent = QuizAgent(llm)
        self.review_agent = ReviewAgent(llm)
        self.progress_store = progress_store
        self.memory_bank = memory_bank
        self.sessions = sessions

    def handle_request(self, user_prompt: str, student_id: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        metrics['total_requests'] += 1
        if session_id is None:
            session_id = self.sessions.new_session()
        self.sessions.append_message(session_id, 'user', user_prompt)

        goal = self.goal_agent.analyze(user_prompt)
        topic = goal.get('topic', 'general')
        days = int(goal.get('days_until_exam', 7))

        plan = self.planner.create_plan(topic, days)
        quiz = self.quiz_agent.generate_quiz(topic)
        flashcards = {'cards': [{'front': 'Term X', 'back': 'Definition X'}]}

        record = {
            'student_id': student_id,
            'topic': topic,
            'plan_summary': plan,
            'quiz_summary': quiz,
            'ts': time.time(),
        }
        rec_id = self.progress_store.save_progress(record)

        if student_id:
            existing = self.memory_bank.get(student_id)
            contacts = existing.get('sessions', 0) + 1
            self.memory_bank.update(student_id, {'last_plan_id': rec_id, 'sessions': contacts, 'topic': topic})

        return {
            'session_id': session_id,
            'student_id': student_id,
            'goal': goal,
            'plan': plan,
            'quiz': quiz,
            'flashcards': flashcards,
            'progress_record_id': rec_id,
        }

# ---------------------------
# Evaluation harness & demo
# ---------------------------

def evaluate_system(coord: Coordinator, tests: List[Dict[str, Any]]) -> Dict[str, Any]:
    results = []
    for t in tests:
        out = coord.handle_request(t['prompt'], student_id=t.get('student_id'))
        results.append({'prompt': t['prompt'], 'goal': out['goal'], 'plan_exists': bool(out['plan'])})
    return {'results': results}


def demo_mode(llm: LLM):
    progress = ProgressStore('demo_progress.json')
    memory = MemoryBank('demo_study_memory.json')
    sessions = SessionService()
    coord = Coordinator(llm, progress, memory, sessions)

    samples = [
        {'prompt': 'I have a math final in 10 days, help me study derivatives', 'student_id': 's1'},
        {'prompt': 'I need to prepare for a history exam in 5 days covering WW2', 'student_id': 's2'},
    ]

    eval_res = evaluate_system(coord, samples)
    print('\n=== Evaluation ===')
    print(json.dumps(eval_res, indent=2))
    print('\n=== Metrics ===')
    print(json.dumps(metrics, indent=2))
    print('\n=== Memory ===')
    print(json.dumps(memory.mem, indent=2))

# ---------------------------
# OpenAIStub placeholder
# ---------------------------
class OpenAIStub(LLM):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')

    def call(self, prompt: str, system: Optional[str] = None) -> str:
        raise RuntimeError('Replace OpenAIStub.call with your provider API code.')

# ---------------------------
# Main
# ---------------------------
if __name__ == '__main__':
    print('Starting LLM Multi-Agent Study Assistant demo (MOCK LLM)')
    llm = MockLLM()
    demo_mode(llm)
    print('\nDemo complete. To use a real LLM, implement OpenAIStub.call or another LLM wrapper and re-run.')





