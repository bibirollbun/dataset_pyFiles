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
import json
import datetime
import uuid
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional

import google.generativeai as genai

# Kaggle secrets
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY")
except Exception as e:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Gemini API key not found. Set Kaggle secret 'GEMINI_API_KEY'.")

genai.configure(api_key=GEMINI_API_KEY)

# Choose a model â€“ adjust if your course specifies a different one
GEMINI_MODEL_NAME = "gemini-2.0-flash"

def make_model(system_instruction: str):
    """Factory to create a configured Gemini model with a system instruction."""
    return genai.GenerativeModel(
        model_name=GEMINI_MODEL_NAME,
        system_instruction=system_instruction,
    )



# HYBRID JSON EXTRACTION TOOLSET  


import re
import json

def extract_json_hybrid(output: str) -> Optional[str]:
    """
    Hybrid JSON extraction:
    
    1. STRICT MODE â†’ Extract <json>...</json>
    2. FLEXIBLE MODE â†’ Extract first {...} block
    3. Clean output:
       - Remove backticks
       - Remove markdown fences
       - Remove trailing commas
    """

    # STRICT MODE 
    strict_match = re.search(r"<json>(.*?)</json>", output, re.DOTALL)
    if strict_match:
        cleaned = strict_match.group(1).strip()
        cleaned = clean_json_text(cleaned)
        return cleaned

    # FLEXIBLE MODE 
    brace_match = re.search(r"\{[\s\S]*\}", output)
    if brace_match:
        cleaned = brace_match.group(0).strip()
        cleaned = clean_json_text(cleaned)
        return cleaned

    return None


def clean_json_text(text: str) -> str:
    """
    Clean common issues in Gemini JSON:
    - Remove ```json and ``` blocks
    - Remove trailing commas before } or ]
    - Strip whitespace
    """

    # Remove ```
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)

    # Remove trailing commas
    text = re.sub(r",\s*([\]}])", r"\1", text)

    return text.strip()


def parse_json_safely(text: str) -> Optional[Any]:
    """
    Try json.loads safely.
    """
    try:
        return json.loads(text)
    except Exception:
        return None



from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
import datetime
import uuid


# Data models


@dataclass
class Assignment:
    id: str
    raw_text: str
    title: str
    subject: str
    due_date: Optional[str]  
    key_requirements: List[str]


@dataclass
class TaskStep:
    id: str
    description: str
    estimated_minutes: int
    prerequisite_step_ids: List[str] = field(default_factory=list)


@dataclass
class StudyPlan:
    assignment_id: str
    steps: List[TaskStep]
    daily_schedule: Dict[str, List[str]]  


@dataclass
class AssignmentInsights:
    """
    High-level understanding of the assignment:
    - difficulty
    - clarity
    - missing info
    - teacher questions
    - suggested working strategy
    """
    difficulty_level: str            
    clarity_score: int               
    main_objective: str
    key_challenges: List[str]
    missing_information_questions: List[str]
    suggested_questions_for_teacher: List[str]
    suggested_strategy: str         


@dataclass
class ResearchSource:
    title: str
    type: str              
    description: str
    how_to_use: str


@dataclass
class ResearchPack:
    assignment_id: str
    topic_summary: str
    suggested_keywords: List[str]
    sources: List[ResearchSource]


@dataclass
class SessionState:
    session_id: str
    student_id: str
    assignment: Assignment
    study_plan: StudyPlan
    completed_steps: List[str] = field(default_factory=list)
    assignment_insights: Optional[AssignmentInsights] = None
    research_pack: Optional[ResearchPack] = None
    created_at: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())


@dataclass
class Reflection:
    session_id: str
    student_id: str
    date: str
    easy: List[str]
    hard: List[str]
    suggestions: str




# Simple JSON "Memory Bank"


class JSONMemoryStore:
    """
    Very simple long-term memory:
    - sessions.json: list of SessionState
    - reflections.json: list of Reflection
    - student_profile.json: dict keyed by student_id with preferences
    """
    def __init__(self, base_dir: str = "./memory_store"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self._init_file("sessions.json", [])
        self._init_file("reflections.json", [])
        self._init_file("student_profile.json", {})

    def _init_file(self, name: str, default):
        path = os.path.join(self.base_dir, name)
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump(default, f)

    def _read(self, name: str):
        path = os.path.join(self.base_dir, name)
        with open(path, "r") as f:
            return json.load(f)

    def _write(self, name: str, data):
        path = os.path.join(self.base_dir, name)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # Sessions 
    def save_session(self, session: SessionState):
        sessions = self._read("sessions.json")
        sessions = [s for s in sessions if s["session_id"] != session.session_id]
        sessions.append(asdict(session))
        self._write("sessions.json", sessions)

    def load_session(self, session_id: str) -> Optional[SessionState]:
        sessions = self._read("sessions.json")
        for s in sessions:
            if s["session_id"] == session_id:
                return SessionState(**s)
        return None

    # Reflections 
    def save_reflection(self, reflection: Reflection):
        reflections = self._read("reflections.json")
        reflections.append(asdict(reflection))
        self._write("reflections.json", reflections)

    def get_reflections_for_student(self, student_id: str) -> List[Reflection]:
        reflections = self._read("reflections.json")
        return [Reflection(**r) for r in reflections if r["student_id"] == student_id]

    # Student profile (long-term memory / preferences)
    def get_student_profile(self, student_id: str) -> Dict[str, Any]:
        profiles = self._read("student_profile.json")
        return profiles.get(student_id, {})

    def update_student_profile(self, student_id: str, updates: Dict[str, Any]):
        profiles = self._read("student_profile.json")
        base = profiles.get(student_id, {})
        base.update(updates)
        profiles[student_id] = base
        self._write("student_profile.json", profiles)


memory_store = JSONMemoryStore()



# Helper utilities
from typing import List, Dict, Any
import datetime

def call_gemini(model, prompt: str, temperature: float = 0.4):
    """
    Wrapper around Gemini that:
    - Calls model
    - Extracts JSON using hybrid method
    - If JSON invalid â†’ ask Gemini to fix it
    """
    try:
        raw = model.generate_content(
            prompt,
            generation_config={"temperature": temperature}
        )
        output = raw.text
        return output
    except Exception as e:
        print("Gemini error:", e)
        return None


def call_gemini_json(model, prompt: str, fallback_prompt=None, temperature: float = 0.3):
    """
    Special method for JSON-returning agents.
    Steps:
    1. Generate response
    2. Extract JSON via hybrid extractor
    3. Try parsing
    4. If parsing fails â†’ retry with 'fix my JSON'
    """
    raw_output = call_gemini(model, prompt, temperature)
    if not raw_output:
        return None

    extracted = extract_json_hybrid(raw_output)
    if extracted:
        parsed = parse_json_safely(extracted)
        if parsed:
            return parsed

    fix_prompt = f"""
    The following output was supposed to be valid JSON but is not:

    {raw_output}

    FIX IT.
    Return ONLY corrected JSON inside <json>...</json>.
    """

    fixed_raw = call_gemini(model, fix_prompt, temperature=0.2)
    if fixed_raw:
        extracted = extract_json_hybrid(fixed_raw)
        if extracted:
            parsed = parse_json_safely(extracted)
            if parsed:
                return parsed

    return None


def compact_context(history: List[Dict[str, Any]], max_items: int = 5) -> str:
    """
    Very simple context compaction:
    - only keep the last `max_items` events
    - summarise them into a short paragraph with Gemini
    """
    if not history:
        return "No prior history."

    trimmed = history[-max_items:]
    text = "\n".join([f"- {h['type']}: {h['summary']}" for h in trimmed])

    model = make_model(
        "You summarise previous study sessions into a short profile that helps future planning."
    )

    prompt = (
        "Summarise the following student history into a short paragraph (max 120 words). "
        "Focus on patterns about what is easy/hard, preferred times of day, and task types.\n\n"
        f"{text}"
    )

    summary = call_gemini(model, prompt)
    return summary.strip()


def log_event(event_type: str, detail: str):
    """Simple console logger."""
    timestamp = datetime.datetime.utcnow().isoformat()
    print(f"[{timestamp}] {event_type}: {detail}")



# Tools


class AssignmentExtractorTool:
    """
    Converts raw pasted assignment text into a structured Assignment.
    For simplicity, we ask Gemini to extract: title, subject, due date, key requirements.
    """
    def __init__(self):
        self.model = make_model(
            "You are an assistant that extracts structured fields from school assignment descriptions."
        )

    def extract(self, raw_text: str) -> Assignment:
        prompt = f"""
        You will be given a student's assignment description.

        Extract the following fields in JSON format:
        - title: short 5-10 word title
        - subject: subject or course name, or "General"
        - due_date: ISO date (YYYY-MM-DD) if mentioned; otherwise null
        - key_requirements: list of 3-8 short bullet requirements

        Return ONLY valid JSON, no extra text.

        Assignment description:
        {raw_text}
        """
        response = call_gemini(self.model, prompt)
        try:
            data = json.loads(response)
        except Exception:
           
            data = {
                "title": raw_text[:50],
                "subject": "General",
                "due_date":None,
                "key_requirements": [raw_text[:200]],
            }
        assignment = Assignment(
            id=str(uuid.uuid4()),
            raw_text=raw_text,
            title=data.get("title", "Untitled Assignment"),
            subject=data.get("subject", "General"),
            due_date=data.get("due_date"),
            key_requirements=data.get("key_requirements", []),
        )
        return assignment


class PlanEvaluatorTool:
    """
    Basic automatic evaluation for your agent system.
    Checks whether a plan:
    - covers requirements
    - has deadlines mapped to dates
    - has reasonably sized steps
    """
    def evaluate(self, assignment: Assignment, plan: StudyPlan) -> Dict[str, Any]:
        issues = []
        if not plan.steps:
            issues.append("No steps generated.")

        # Checks if each requirement appears in some step description.
        missing_reqs = []
        for req in assignment.key_requirements:
            if not any(req_part.lower() in step.description.lower()
                       for step in plan.steps
                       for req_part in req.split()[:2]):
                missing_reqs.append(req)

        if missing_reqs:
            issues.append(f"Some requirements may not be covered: {missing_reqs}")

        # Checks estimated minutes
        long_steps = [s.description for s in plan.steps if s.estimated_minutes > 90]
        if long_steps:
            issues.append("Some steps are very long; consider breaking them down further.")

        ok = len(issues) == 0
        return {
            "ok": ok,
            "issues": issues,
            "total_steps": len(plan.steps),
            "total_estimated_minutes": sum(s.estimated_minutes for s in plan.steps),
        }



# Base Agent


class BaseAgent:
    def __init__(self, name: str, system_instruction: str):
        self.name = name
        self.model = make_model(system_instruction)

    def run(self, **kwargs):
        raise NotImplementedError("Subclasses must implement .run()")



# Agent 1: Input Normaliser


class InputNormalizerAgent(BaseAgent):
    """
    Uses AssignmentExtractorTool to turn messy raw text into Assignment.
    """
    def __init__(self, extractor: AssignmentExtractorTool):
        super().__init__(
            "InputNormalizer",
            "You orchestrate extraction of structured assignment details for dyslexic students."
        )
        self.extractor = extractor

    def run(self, raw_text: str) -> Assignment:
        log_event(self.name, "Starting extraction.")
        assignment = self.extractor.extract(raw_text)
        log_event(
            self.name,
            f"Extracted assignment: title='{assignment.title}', due_date={assignment.due_date}",
        )
        return assignment



# Agent 2: Task Decomposer 


class TaskDecomposerAgent(BaseAgent):
    """
    Breaks an assignment into dyslexia-friendly micro-steps.
    Guarantees a multi-step plan (min 6 steps) using:
    - A strong prompt
    - A fallback heuristic if the model under-decomposes
    """
    def __init__(self):
        super().__init__(
            "TaskDecomposer",
            (
                "You are a study coach for dyslexic students. "
                "You ALWAYS break one assignment into multiple small, clear, concrete steps "
                "in simple English. Each step should be small enough to do in 15â€“45 minutes. "
                "Avoid long paragraphs and complex wording."
            ),
        )

    def _default_steps(self, assignment: Assignment) -> List[TaskStep]:
        """
        Fallback plan if the model returns too few steps.
        Generic but sensible for most school assignments.
        """
        base_desc = assignment.title.strip() or "this assignment"
        templates = [
            f"Read the assignment instructions for {base_desc} slowly and highlight key words.",
            f"Brainstorm main ideas and write down 3â€“5 points you want to include in {base_desc}.",
            f"Do focused research and collect 3â€“5 reliable sources for {base_desc}.",
            f"Create a simple outline for {base_desc} with an introduction, main points, and conclusion.",
            f"Write the first draft of {base_desc} using your outline.",
            f"Edit and proofread your draft of {base_desc}, checking spelling and clarity.",
        ]
        steps = []
        for i, desc in enumerate(templates, start=1):
            steps.append(
                TaskStep(
                    id=f"step{i}",
                    description=desc,
                    estimated_minutes=30 if i not in (5, 6) else 45,
                    prerequisite_step_ids=[f"step{i-1}"] if i > 1 else [],
                )
            )
        return steps

    def run(self, assignment: Assignment) -> List[TaskStep]:
        reqs_text = "\n".join(f"- {r}" for r in assignment.key_requirements) or "No specific requirements listed."

        prompt = f"""
        Break this assignment into MULTIPLE clear, small, dyslexia-friendly steps.
        
        Assignment:
        Title: {assignment.title}
        Requirements:
        {reqs_text}
        
        Rules:
        1. Return between 6 and 16 steps.
        2. Each step takes 15â€“45 minutes.
        3. Simple English only.
        
        Return ONLY this JSON inside <json></json>:

        <json>
        [
        {{
        "id": "step1",
        "description": "string",
        "estimated_minutes": 30,
        "prerequisite_step_ids": []
        }},
        {{
        "id": "step2",
        "description": "string",
        "estimated_minutes": 30,
        "prerequisite_step_ids": ["step1"]
        }}
        ]
        </json>
        """


        data = call_gemini_json(self.model, prompt)
        if not data or len(data) < 6:
            log_event(self.name, "JSON parse failed or too few steps â†’ fallback multi-step plan.")
            return self._default_steps(assignment)


        # Transform into TaskStep list
        steps: List[TaskStep] = []
        for item in data:
            steps.append(
                TaskStep(
                    id=item.get("id", f"step{len(steps)+1}"),
                    description=item.get("description", "Work on assignment"),
                    estimated_minutes=int(item.get("estimated_minutes", 30)),
                    prerequisite_step_ids=item.get("prerequisite_step_ids", []),
                )
            )

        # Fallback
        if len(steps) < 6:
            log_event(
                self.name,
                f"Model returned only {len(steps)} steps; replacing with default multi-step plan.",
            )
            return self._default_steps(assignment)

        log_event(self.name, f"Generated {len(steps)} steps.")
        return steps



# Agent 3: Study Planner


class StudyPlannerAgent(BaseAgent):
    """
    Maps steps into a realistic daily schedule based on student availability and preferences.
    For the demo, we will ask the user for:
    - number of days until due date they want to study
    - preferred daily study minutes
    """
    def __init__(self):
        super().__init__(
            "StudyPlanner",
            (
                "You are a planning assistant for dyslexic students. "
                "You distribute study steps across days, respecting limits on daily minutes. "
                "Keep days balanced and leave some buffer before the due date."
            ),
        )

    def run(
        self,
        assignment: Assignment,
        steps: List[TaskStep],
        start_date: datetime.date,
        preferred_daily_minutes: int,
    ) -> StudyPlan:

        # Duedate
        if assignment.due_date:
            due = datetime.date.fromisoformat(assignment.due_date)
            last_date = max(start_date, due - datetime.timedelta(days=1))
        else:
            last_date = start_date + datetime.timedelta(days=6)

        # Build a list of dates
        num_days = (last_date - start_date).days + 1
        dates = [start_date + datetime.timedelta(days=i) for i in range(num_days)]

        # Simple greedy allocation
        schedule: Dict[str, List[str]] = {d.isoformat(): [] for d in dates}
        remaining_minutes = {d.isoformat(): preferred_daily_minutes for d in dates}

        for step in steps:
            placed = False
            for d in dates:
                d_str = d.isoformat()
                if remaining_minutes[d_str] >= step.estimated_minutes:
                    schedule[d_str].append(step.id)
                    remaining_minutes[d_str] -= step.estimated_minutes
                    placed = True
                    break
            if not placed:
                # Put it on the last day even if it overflows
                schedule[dates[-1].isoformat()].append(step.id)

        plan = StudyPlan(
            assignment_id=assignment.id,
            steps=steps,
            daily_schedule=schedule,
        )

        log_event(
            self.name,
            f"Created schedule from {dates[0]} to {dates[-1]} "
            f"for {len(steps)} steps.",
        )
        return plan



# Agent: Micro-Task Coach


class MicroTaskCoachAgent(BaseAgent):
    """
    Gives short, concrete guidance and tips for the current step.
    """

    def __init__(self):
        super().__init__(
            "MicroTaskCoach",
            (
                "You are a friendly micro-task coach for a dyslexic student. "
                "You explain only the CURRENT step in simple English, "
                "and give 2â€“4 practical tips. Keep it short and clear."
            ),
        )

    def run(self, step: TaskStep, session: SessionState) -> str:
        completed = len(session.completed_steps)
        total = len(session.study_plan.steps)

        prompt = f"""
        The student is working on this step:

        Step description:
        {step.description}

        Progress:
        - Completed steps: {completed}
        - Total steps: {total}

        Task:

        1. Briefly explain what to do in this step using very simple words.
        2. Give 2â€“4 practical tips that help with focus, memory, and reading/writing.
        3. Keep everything under 130 words.

        Use bullets only where helpful. Avoid long paragraphs.
        """

        response = call_gemini(self.model, prompt, temperature=0.5)
        return response.strip()



# Agent: Check-In Agent


class CheckInAgent(BaseAgent):
    """
    Asks reflective questions and checks understanding and confidence.
    """

    def __init__(self):
        super().__init__(
            "CheckInAgent",
            (
                "You ask short, simple questions to help a dyslexic student reflect "
                "on their understanding and confidence with the current step."
            ),
        )

    def run(self, step: TaskStep, session: SessionState) -> str:
        prompt = f"""
        The student has just seen this step:

        {step.description}

        Your job:

        1. Ask 1â€“3 very short questions to check:
           - Do they understand what to do?
           - How confident they feel (0-10).
        2. Offer ONE short sentence of encouragement.

        Output format (no bullets needed, just short lines):

        - Question 1: ...
        - Question 2: ...
        - (optional) Question 3: ...
        - Encouragement: ...
        """

        response = call_gemini(self.model, prompt, temperature=0.4)
        return response.strip()



# Agent 5: Reflection Agent


class ReflectionAgent(BaseAgent):
    """
    Takes student's short feedback (what was easy/hard) and updates long-term profile.
    """
    def __init__(self):
        super().__init__(
            "ReflectionAgent",
            (
                "You analyse short reflections from a dyslexic student and extract patterns "
                "about difficulty, strengths, and preferred working conditions."
            ),
        )

    def run(
        self,
        student_id: str,
        session: SessionState,
        easy: List[str],
        hard: List[str],
    ) -> Reflection:
        date_str = datetime.date.today().isoformat()

        # Generates suggestion string using Gemini
        history = memory_store.get_reflections_for_student(student_id)
        history_events = [
            {"type": "past_reflection", "summary": r.suggestions}
            for r in history
        ]
        compacted = compact_context(history_events, max_items=5)

        prompt = f"""
        Student ID: {student_id}

        Today's session:
        - Assignment title: {session.assignment.title}
        - Completed steps: {len(session.completed_steps)} / {len(session.study_plan.steps)}

        Student said these were EASY:
        {easy}

        Student said these were HARD:
        {hard}

        Summary of past patterns:
        {compacted}

        Task:
        1. In 3â€“6 sentences, suggest what the student can try next time
           (timing, environment, breaking tasks, supports).
        2. Focus on being positive, practical, and non-judgmental.
        """
        suggestions = call_gemini(self.model, prompt, temperature=0.5)

        reflection = Reflection(
            session_id=session.session_id,
            student_id=student_id,
            date=date_str,
            easy=easy,
            hard=hard,
            suggestions=suggestions.strip(),
        )

        # Updates long-term profile
        profile_updates = {
            "last_suggestions": suggestions.strip(),
            "last_reflection_date": date_str,
        }
        memory_store.update_student_profile(student_id, profile_updates)

        log_event(self.name, "Saved reflection and updated profile.")
        return reflection




# Agent: Assignment Understanding 

class AssignmentUnderstandingAgent(BaseAgent):
    """
    Reads the assignment deeply and produces:
    - difficulty estimate
    - clarity score
    - missing information
    - helpful questions to ask the teacher
    - a short suggested strategy
    """

    def __init__(self):
        super().__init__(
            "AssignmentUnderstandingAgent",
            (
                "You analyse school assignments for a dyslexic student. "
                "Your job is to understand what the task REALLY is, how hard it may feel, "
                "and what might be confusing or missing. "
                "Always be practical, supportive, and clear."
            ),
        )

    def run(self, assignment: Assignment) -> AssignmentInsights:
        reqs_text = "\n".join(f"- {r}" for r in assignment.key_requirements) or "No specific requirements listed."

        # Prompt with escaped JSON 
        prompt = f"""
        You are helping a dyslexic student understand a school assignment.

        Assignment details:
        Title: {assignment.title}
        Subject: {assignment.subject}
        Due date: {assignment.due_date}
        Requirements:
        {reqs_text}

        Your job is to deeply understand the assignment.

        Return ALL results as JSON inside <json></json>.
        The JSON object MUST follow this exact structure:

        <json>
        {{
          "difficulty_level": "easy | medium | hard",
          "clarity_score": 1,
          "main_objective": "string",
          "key_challenges": ["string", "string", "..."],
          "missing_information_questions": ["string", "string"],
          "suggested_questions_for_teacher": ["string", "string"],
          "suggested_strategy": "string"
        }}
        </json>
        """

        #Calls Gemini with hybrid JSON extractor 
        data = call_gemini_json(self.model, prompt, temperature=0.4)

        
        if not data:
            log_event(self.name, "JSON parsing failed â†’ fallback insights.")
            data = {
                "difficulty_level": "medium",
                "clarity_score": 6,
                "main_objective": f"Understand and complete: {assignment.title}",
                "key_challenges": [
                    "Understanding what the teacher expects",
                    "Planning work over multiple days",
                    "Staying focused while reading",
                ],
                "missing_information_questions": [],
                "suggested_questions_for_teacher": [
                    "Can you show an example of a good answer for this type of assignment?"
                ],
                "suggested_strategy": "Read instructions twice, make a short plan, and work a bit each day.",
            }

        
        insights = AssignmentInsights(
            difficulty_level=data.get("difficulty_level", "medium"),
            clarity_score=int(data.get("clarity_score", 5)),
            main_objective=data.get("main_objective", ""),
            key_challenges=data.get("key_challenges", []),
            missing_information_questions=data.get("missing_information_questions", []),
            suggested_questions_for_teacher=data.get("suggested_questions_for_teacher", []),
            suggested_strategy=data.get("suggested_strategy", ""),
        )

        log_event(
            self.name,
            f"Insights: difficulty={insights.difficulty_level}, clarity={insights.clarity_score}",
        )

        return insights




# Agent: Research Agent


class ResearchAgent(BaseAgent):
    """
    Suggests useful research directions and resources:
    - a short topic summary
    - search keywords
    - a small set of suggested sources with 'how to use' tips
    """

    def __init__(self):
        super().__init__(
            "ResearchAgent",
            (
                "You help a dyslexic student find useful resources for a school assignment. "
                "You DO NOT need to know the live web. Instead, suggest realistic, typical resources, "
                "good search keywords, and explain how they can be used."
            ),
        )

    def run(self, assignment: Assignment, insights: AssignmentInsights) -> ResearchPack:
        reqs_text = "\n".join(f"- {r}" for r in assignment.key_requirements) or "No specific requirements listed."

        prompt = f"""
        You are a research helper for a dyslexic student.
        Assignment:
        - Title: {assignment.title}
        - Subject: {assignment.subject}
        - Requirements:
        {reqs_text}
        Understanding Summary:
        - Main objective: {insights.main_objective}
        - Key challenges: {insights.key_challenges}
        
        Return ALL results as JSON inside <json></json>.
        
        The JSON MUST follow this exact structure:
        <json>
        {{
        "topic_summary": "string",
        "suggested_keywords": ["string", "string", ...],
        "sources": [
        {{
        "title": "string",
        "type": "website | book | article | video | report",
        "description": "string",
        "how_to_use": "string"
        }}
        ]
        }}
        </json>
        """

        data = call_gemini_json(self.model, prompt)
        if not data:
            log_event(self.name, "JSON parse failed â†’ fallback research pack.")
            data = {
                "topic_summary": f"This assignment is about: {assignment.title}",
                "suggested_keywords": [
                    assignment.title,
                    f"intro to {assignment.subject}",
                    "beginner explanation"
                ],
                "sources": [
                    {
                        "title": "General introduction article",
                        "type": "website",
                        "description": "A simple overview.",
                        "how_to_use": "Read first to build basic understanding."
                    }
                ]
            }


        sources = []
        for s in data.get("sources", []):
            sources.append(
                ResearchSource(
                    title=s.get("title", "Untitled source"),
                    type=s.get("type", "website"),
                    description=s.get("description", ""),
                    how_to_use=s.get("how_to_use", ""),
                )
            )

        pack = ResearchPack(
            assignment_id=assignment.id,
            topic_summary=data.get("topic_summary", ""),
            suggested_keywords=data.get("suggested_keywords", []),
            sources=sources,
        )

        log_event(self.name, f"Research pack generated with {len(pack.sources)} sources.")
        return pack



# Orchestrator 


class NeuroNavigatorOrchestrator:
    def __init__(self):
        # Tools
        self.extractor_tool = AssignmentExtractorTool()
        self.plan_evaluator = PlanEvaluatorTool()

        # Agents
        self.input_normalizer = InputNormalizerAgent(self.extractor_tool)
        self.assignment_understanding = AssignmentUnderstandingAgent()
        self.research_agent = ResearchAgent()
        self.task_decomposer = TaskDecomposerAgent()
        self.study_planner = StudyPlannerAgent()
        self.micro_task_coach = MicroTaskCoachAgent()
        self.checkin_agent = CheckInAgent()
        self.reflection_agent = ReflectionAgent()

    def create_session(
        self,
        student_id: str,
        raw_assignment_text: str,
        preferred_daily_minutes: int = 60,
    ) -> SessionState:
        # 1. Normalise input
        assignment = self.input_normalizer.run(raw_assignment_text)

        # 2. Understand assignment at high level
        insights = self.assignment_understanding.run(assignment)

        # 3. Research support
        research_pack = self.research_agent.run(assignment, insights)

        # 4. Decompose into steps
        steps = self.task_decomposer.run(assignment)

        # 5. Create schedule
        today = datetime.date.today()
        plan = self.study_planner.run(
            assignment,
            steps,
            start_date=today,
            preferred_daily_minutes=preferred_daily_minutes,
        )

        # 6. Evaluation
        eval_result = self.plan_evaluator.evaluate(assignment, plan)
        log_event("PlanEvaluation", json.dumps(eval_result, indent=2))

        # 7. Build session
        session = SessionState(
            session_id=str(uuid.uuid4()),
            student_id=student_id,
            assignment=assignment,
            study_plan=plan,
            assignment_insights=insights,
            research_pack=research_pack,
        )

        memory_store.save_session(session)
        return session

    def reload_session(self, session_id: str) -> SessionState:
        session = memory_store.load_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found.")
        return session

    def get_next_step(self, session: SessionState) -> Optional[TaskStep]:
        completed = set(session.completed_steps)
        
        for step in session.study_plan.steps:
            if step.id in completed:
                continue
            if all(pr in completed for pr in step.prerequisite_step_ids):
                return step
        return None

    def run_study_loop(self, session: SessionState):
        """
        Simple text-based loop (for local / interactive use, not Kaggle input()).
        Here we just demonstrate how the two agents would be called together.
        """
        while True:
            next_step = self.get_next_step(session)
            if not next_step:
                print("\nðŸŽ‰ All steps completed for this assignment!")
                break

            print("\n==============================")
            print(f"Next Step: {next_step.description}")
            print(f"Estimated time: {next_step.estimated_minutes} minutes")

            guidance = self.micro_task_coach.run(next_step, session)
            checkin = self.checkin_agent.run(next_step, session)

            print("\nMicro-Task Coach:\n")
            print(guidance)
            print("\nCheck-In:\n")
            print(checkin)

            
            print("\n(Interactive loop disabled in Kaggle demo.)")
            break  

    def collect_reflection(self, session: SessionState):
        
        easy = ["Breaking down the task into small steps"]
        hard = ["Managing time and staying focused"]

        reflection = self.reflection_agent.run(
            student_id=session.student_id,
            session=session,
            easy=easy,
            hard=hard,
        )
        memory_store.save_reflection(reflection)
        print("\n=== REFLECTION SUGGESTIONS ===\n")
        print(reflection.suggestions)




# Demo run 


orchestrator = NeuroNavigatorOrchestrator()

student_id = "student_1"

raw_assignment_text = """
For your final Digital Innovation & Sustainability assessment, you must produce a 2,500-word analytical report evaluating how a real-world company has implemented AI-driven sustainability practices. 

Your report MUST include:
1. A clear explanation of the chosen company's sustainability challenges.
2. An evaluation of at least two AI systems or digital tools they currently use (or plan to use).
3. A section discussing the ethical risks, biases, and data governance issues.
4. At least one visual (graph, chart, or infographic) created by you.
5. A comparison with one competitor.
6. At least FIVE academic sources (peer-reviewed), plus THREE industry reports.
7. A concluding section proposing actionable recommendations for the next 3â€“5 years.

Formatting Requirements:
- Written in a formal academic tone.
- Harvard referencing format.
- Submit as PDF via TurnItIn.
- Max 10% AI-generated content allowed (must declare usage).

Due Date: December 18, 2025, 23:59 (Italian Time)
Course: Digital Innovation and Sustainability Management (ECON408)

Bonus marks if:
- You include a short appendix describing the AI tools you used to support your research process.
- You evaluate carbon footprint of AI tools used (basic estimation acceptable).
"""

session = orchestrator.create_session(
    student_id=student_id,
    raw_assignment_text=raw_assignment_text,
    preferred_daily_minutes=60,
)

print("=== SESSION CREATED ===")
print("Session ID:", session.session_id)
print("Assignment title:", session.assignment.title)
print("Due date:", session.assignment.due_date)
print("Steps generated:", len(session.study_plan.steps))

print("\n=== ASSIGNMENT INSIGHTS ===")
ai = session.assignment_insights
print("Difficulty level:", ai.difficulty_level)
print("Clarity score:", ai.clarity_score)
print("Main objective:", ai.main_objective)
print("Key challenges:", ai.key_challenges)
print("Suggested questions for teacher:", ai.suggested_questions_for_teacher)

print("\n=== RESEARCH PACK SUMMARY ===")
rp = session.research_pack
print("Topic summary:", rp.topic_summary)
print("Suggested keywords:", rp.suggested_keywords)
print("Sources:")
for src in rp.sources[:3]:
    print(f"- {src.title} ({src.type})")
    print("  Description:", src.description)
    print("  How to use:", src.how_to_use)

print("\n=== DAILY SCHEDULE ===")
for day, step_ids in session.study_plan.daily_schedule.items():
    print(f"{day}: {step_ids}")

# Show next step + both study agents
next_step = orchestrator.get_next_step(session)
if next_step:
    print("\n=== NEXT STUDY STEP ===")
    print("Description:", next_step.description)
    print("Estimated minutes:", next_step.estimated_minutes)

    guidance = orchestrator.micro_task_coach.run(next_step, session)
    checkin = orchestrator.checkin_agent.run(next_step, session)

    print("\n=== MICRO-TASK COACH OUTPUT ===\n")
    print(guidance)

    print("\n=== CHECK-IN AGENT OUTPUT ===\n")
    print(checkin)

# Reflection (simulated)
orchestrator.collect_reflection(session)





