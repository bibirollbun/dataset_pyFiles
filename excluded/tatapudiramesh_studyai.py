import os
from kaggle_secrets import UserSecretsClient

# Google Bootcamp Recommended API Key Setup
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please add 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )



!pip install -q google-genai

import os
import json
import datetime
from typing import List, Dict, Any
from dataclasses import dataclass, field
from urllib.parse import quote_plus

from google import genai

# Create Gemini client
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

MODEL_NAME = "gemini-2.5-flash"



@dataclass
class AgentMemory:
    study_plans: List[Dict[str, Any]] = field(default_factory=list)
    doubts: List[Dict[str, Any]] = field(default_factory=list)
    flashcards: List[Dict[str, Any]] = field(default_factory=list)
    quizzes: List[Dict[str, Any]] = field(default_factory=list)
    youtube_cache: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)


def call_llm(prompt: str, model: str = MODEL_NAME) -> str:
    resp = client.models.generate_content(model=model, contents=prompt)
    return resp.text


def extract_json_maybe(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        idx = stripped.find("{")
        if idx == -1:
            idx = stripped.find("[")
        if idx != -1:
            stripped = stripped[idx:]
    try:
        return json.loads(stripped)
    except:
        return stripped


# ------------------- PLANNER AGENT -------------------

class PlannerAgent:
    def __init__(self, memory: AgentMemory):
        self.memory = memory

    def generate_plan(self, subjects: List[str], exam_date: str, hours_per_day: int) -> str:
        today = datetime.date.today()
        prompt = f"""
You are PlannerAgent.

Generate a realistic day-wise study plan from today ({today}) to exam date {exam_date}.
Subjects: {subjects}
Daily study hours: {hours_per_day}

Output format: A Markdown table with columns:
Date | Subject | Topic | DurationHours
"""
        plan = call_llm(prompt)
        self.memory.study_plans.append(
            {
                "created_at": str(today),
                "subjects": subjects,
                "hours_per_day": hours_per_day,
                "exam_date": exam_date,
                "plan_markdown": plan,
            }
        )
        return plan

    def get_latest_plan(self):
        if not self.memory.study_plans:
            return "No study plan created yet."
        return self.memory.study_plans[-1]["plan_markdown"]


# ------------------- TUTOR AGENT -------------------

class TutorAgent:
    def __init__(self, memory: AgentMemory):
        self.memory = memory

    def explain_topic(self, subject: str, topic: str):
        prompt = f"""
Explain topic: {topic}
Subject: {subject}

Sections:
## Explanation
## Intuition
## Practice Questions (3)
"""
        explanation = call_llm(prompt)
        self.memory.doubts.append(
            {
                "timestamp": str(datetime.datetime.now()),
                "subject": subject,
                "topic": topic,
                "content": explanation,
            }
        )
        return explanation

    def answer_doubt(self, subject: str, topic: str, question: str):
        prompt = f"""
Student doubt: {question}
Explain step-by-step in simple language.
"""
        answer = call_llm(prompt)
        self.memory.doubts.append(
            {
                "timestamp": str(datetime.datetime.now()),
                "subject": subject,
                "topic": topic,
                "question": question,
                "answer": answer,
            }
        )
        return answer

    def generate_flashcards(self, topic: str, num_cards: int = 6):
        prompt = f"""
Generate {num_cards} flashcards for topic: {topic}.
Return ONLY JSON:
[
  {{"question": "...", "answer": "..."}},
  ...
]
"""
        raw = call_llm(prompt)
        data = extract_json_maybe(raw)

        if not isinstance(data, list):
            data = [
                {
                    "question": f"What is {topic}?",
                    "answer": "Flashcard generation fallback.",
                }
            ]

        for card in data:
            card["topic"] = topic
            card["created_at"] = str(datetime.datetime.now())

        self.memory.flashcards.extend(data)
        return data

    def suggest_youtube_links(self, topic: str, max_links: int = 3):
        if topic in self.memory.youtube_cache:
            return self.memory.youtube_cache[topic]

        query = quote_plus(f"{topic} tutorial for beginners")
        base_url = "https://www.youtube.com/results?search_query=" + query

        links = [
            {
                "title": f"{topic} Tutorial (Search Result)",
                "url": base_url,
                "note": "Open search link & pick video.",
            }
            for _ in range(max_links)
        ]

        self.memory.youtube_cache[topic] = links
        return links


# ------------------- QUIZ AGENT -------------------

class QuizAgent:
    def __init__(self, memory: AgentMemory):
        self.memory = memory

    def generate_quiz(self, topic: str, num_questions: int = 5):
        prompt = f"""
Generate a {num_questions}-question quiz for topic {topic}.
Return JSON:
{{
  "topic": "{topic}",
  "questions": [
     {{
        "id": 1,
        "type": "mcq" or "short",
        "question": "...",
        "options": ["A","B","C","D"] or [],
        "answer": "..."
     }},
     ...
  ]
}}
"""
        raw = call_llm(prompt)
        quiz_data = extract_json_maybe(raw)

        if not isinstance(quiz_data, dict):
            quiz_data = {
                "topic": topic,
                "questions": [
                    {
                        "id": 1,
                        "type": "short",
                        "question": f"What is {topic}?",
                        "options": [],
                        "answer": f"Core idea of {topic}.",
                    }
                ],
            }

        quiz_data["created_at"] = str(datetime.datetime.now())
        self.memory.quizzes.append(quiz_data)
        return quiz_data


# ------------------- COACH AGENT -------------------

class CoachAgent:
    def __init__(self, memory: AgentMemory):
        self.memory = memory

    def daily_summary(self):
        today = str(datetime.date.today())
        doubts_today = [d for d in self.memory.doubts if d["timestamp"].startswith(today)]

        prompt = f"""
Write a daily study summary.
Recent doubts: {doubts_today}
Recent quiz topics: {[q['topic'] for q in self.memory.quizzes[-5:]]}
Recent flashcards topics: {[c['topic'] for c in self.memory.flashcards[-5:]]}

Include:
1. What student focused on
2. Weak areas
3. Recommendations for tomorrow
"""
        return call_llm(prompt)


class StudyAgentSystem:
    def __init__(self):
        self.memory = AgentMemory()
        self.planner = PlannerAgent(self.memory)
        self.tutor = TutorAgent(self.memory)
        self.quiz = QuizAgent(self.memory)
        self.coach = CoachAgent(self.memory)

    # Direct tool-style methods (you already used these)
    def create_study_plan(self, subjects, exam_date, hours):
        return self.planner.generate_plan(subjects, exam_date, hours)

    def view_plan(self):
        return self.planner.get_latest_plan()

    def explain(self, subject, topic):
        return self.tutor.explain_topic(subject, topic)

    def doubt(self, subject, topic, question):
        return self.tutor.answer_doubt(subject, topic, question)

    def flashcards(self, topic, num_cards=6):
        return self.tutor.generate_flashcards(topic, num_cards)

    def youtube(self, topic, max_links=3):
        return self.tutor.suggest_youtube_links(topic, max_links)

    def make_quiz(self, topic, num=5):
        return self.quiz.generate_quiz(topic, num)

    def summary(self):
        return self.coach.daily_summary()

    # -------- NEW: free-form chat method --------
    def chat(self, user_message: str) -> str:
        """
        Free-form chat. User can say anything related to study.
        This method uses the LLM to decide which internal agent/tool to call.
        """

        router_prompt = f"""
You are an orchestrator for a multi-agent study assistant.

You have these actions:

1) "create_study_plan":
   params: {{"subjects": [list of subjects], "exam_date": "YYYY-MM-DD", "hours_per_day": int}}

2) "explain_topic":
   params: {{"subject": "subject name", "topic": "topic name"}}

3) "answer_doubt":
   params: {{"subject": "subject name", "topic": "topic name", "question": "full user doubt"}}

4) "generate_flashcards":
   params: {{"topic": "topic name", "num_cards": int}}

5) "youtube_links":
   params: {{"topic": "topic name"}}

6) "generate_quiz":
   params: {{"topic": "topic name", "num_questions": int}}

7) "daily_summary":
   params: {{}}

8) "chitchat":
   params: {{}}
   Use this when the user is just greeting, talking generally, or the request is not about the tools.

Given the user's message:
\"\"\"{user_message}\"\"\" 

Choose the SINGLE most appropriate action and extract parameters from the message.
If you are not sure of exam_date or hours_per_day, you may guess a reasonable value.

Return ONLY valid JSON in this format:
{{
  "action": "one of the above",
  "params": {{ ... }}
}}
No extra commentary.
"""
        routing_raw = call_llm(router_prompt)
        routing = extract_json_maybe(routing_raw)

        # Fallback if parsing failed
        if not isinstance(routing, dict) or "action" not in routing:
            # Just treat as generic tutoring chat
            generic_prompt = f"You are a helpful study assistant. Answer this: {user_message}"
            return call_llm(generic_prompt)

        action = routing.get("action", "chitchat")
        params = routing.get("params", {})

        # ---- Route to the right internal method ----
        try:
            if action == "create_study_plan":
                subjects = params.get("subjects", [])
                exam_date = params.get("exam_date", str(datetime.date.today()))
                hours = int(params.get("hours_per_day", 3))
                plan = self.create_study_plan(subjects, exam_date, hours)
                return (
                    f"Here is your personalized study plan:\n\n{plan}"
                )

            elif action == "explain_topic":
                subject = params.get("subject", "General")
                topic = params.get("topic", "")
                if not topic:
                    generic_prompt = f"You are a helpful tutor. Answer this: {user_message}"
                    return call_llm(generic_prompt)
                return self.explain(subject, topic)

            elif action == "answer_doubt":
                subject = params.get("subject", "General")
                topic = params.get("topic", "")
                question = params.get("question", user_message)
                return self.doubt(subject, topic or "General", question)

            elif action == "generate_flashcards":
                topic = params.get("topic", "")
                num_cards = int(params.get("num_cards", 6))
                cards = self.flashcards(topic, num_cards=num_cards)
                out_lines = [f"Generated {len(cards)} flashcards for {topic}:\n"]
                for i, c in enumerate(cards, 1):
                    out_lines.append(f"{i}. Q: {c['question']}\n   A: {c['answer']}\n")
                return "\n".join(out_lines)

            elif action == "youtube_links":
                topic = params.get("topic", "")
                links = self.youtube(topic)
                out_lines = [f"Here are some YouTube search links for {topic}:\n"]
                for i, l in enumerate(links, 1):
                    out_lines.append(f"{i}. {l['title']}\n   {l['url']}")
                return "\n".join(out_lines)

            elif action == "generate_quiz":
                topic = params.get("topic", "")
                num_q = int(params.get("num_questions", 5))
                quiz = self.make_quiz(topic, num=num_q)
                out_lines = [f"Here is a {num_q}-question quiz on {topic}:\n"]
                for q in quiz["questions"]:
                    out_lines.append(f"Q{q['id']}: {q['question']}")
                    if q.get("options"):
                        for opt in q["options"]:
                            out_lines.append(f"   - {opt}")
                    out_lines.append("")  # blank line
                return "\n".join(out_lines)

            elif action == "daily_summary":
                return self.summary()

            else:  # chitchat or unknown
                generic_prompt = f"You are a friendly study assistant. Answer naturally: {user_message}"
                return call_llm(generic_prompt)

        except Exception as e:
            # Fallback if some routing/tool fails
            fallback_prompt = f"""
You are a helpful study assistant.

The user said:
\"\"\"{user_message}\"\"\" 

An internal error occurred: {e}

Ignore the error and just help the user directly with an explanation, suggestion or guidance.
"""
            return call_llm(fallback_prompt)


# Re-instantiate (or reuse) the system
agent = StudyAgentSystem()



def chat_loop():
    print("ðŸŽ“ Study Agent Chat Mode (type 'exit' to stop)\n")
    while True:
        user = input("You: ").strip()
        if user.lower() in ("exit", "quit", "bye"):
            print("Agent: All the best for your studies! ðŸ‘‹")
            break
        reply = agent.chat(user)
        print("\nAgent:\n", reply, "\n")

chat_loop()


