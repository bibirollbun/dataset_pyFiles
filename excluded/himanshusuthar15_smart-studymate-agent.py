# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np   # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create
# a version using "Save & Run All".
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session.

import json
import re
import random
from datetime import datetime, timedelta

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

# ============================================================================
# S1 â€” Utilities & Persistence
# ============================================================================

def now_time():
    """Return current time as HH:MM string."""
    return datetime.now().strftime("%H:%M")

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

STATE_PATH = "studymate_state.json"

print("S1 loaded â€” Utilities & persistence ready.")

# ============================================================================
# S2 â€” High-level Idea & Features (printed description)
# ============================================================================

intro_text = """
ğŸ�“ Smart StudyMate Agent â€” Kaggle Notebook

This notebook implements a **prototype version** of the Smart StudyMate Agent:
an AI-inspired personal study companion that automates common academic workflows.

Core Goals:
- Provide a **unified interface** for notes, quizzes, doubts, and study planning.
- Simulate a **multi-agent architecture** using clean Python classes.
- Show how a **tool-routing engine** can orchestrate different study workflows.
- Be fully runnable inside a **Kaggle Notebook** with no external APIs.

âœ¨ Feature Overview (Rule-based / Heuristic Prototype)

1) ğŸ“š AI Notes Generator (Rule-based)
   - Summarizes raw text into short bullets.
   - Extracts key concepts and formulas (heuristics).
   - Builds simple â€œflashcard-styleâ€� Q&A pairs.

2) â�“ Doubt Solver (Template + Math Helper)
   - Detects simple math expressions and evaluates them.
   - For conceptual questions, returns a structured explanation template:
     â€¢ What the concept is
     â€¢ Why it is important
     â€¢ How to remember it
   - Adds follow-up prompts for deeper revision.

3) ğŸ“� Quiz & Practice Generator
   - Generates practice questions from input text:
     â€¢ MCQs
     â€¢ 1-mark short questions
     â€¢ Long-answer prompts
   - Simple heuristic difficulty tags: easy / medium / hard.

4) ğŸ“† Smart Study Planner
   - Takes:
     â€¢ Exam date (as days from now)
     â€¢ List of topics
   - Builds:
     â€¢ Daily topic allocation
     â€¢ Built-in revision days
     â€¢ Light vs heavy days.

5) ğŸ§© Multi-Agent Orchestration
   - InputRouter decides:
     â€¢ notes â†’ NotesAgent
     â€¢ quiz â†’ QuizAgent
     â€¢ doubt â†’ DoubtAgent
     â€¢ plan / schedule â†’ PlannerAgent
   - Everything is wrapped in a small, interactive chat-style UI.

NOTE:
This is a **demo prototype** for the Kaggle â€œAgents Intensive â€” Capstone Projectâ€�.
Real production version would plug in:
- LLMs (for deep reasoning, content generation)
- Vision OCR models (for PDF / image notes)
- Calendar APIs (for actual reminders and time blocking)
"""

print(intro_text)

# ============================================================================
# S3 â€” State & Profile Management
# ============================================================================

class StudyProfile:
    """
    Simple per-user study profile:
    - name
    - branch / semester
    - subjects
    - preferences (e.g., prefers MCQs, hates long theory, etc.)
    """
    def __init__(self, data=None):
        data = data or {}
        self.name = data.get("name", None)
        self.branch = data.get("branch", None)
        self.semester = data.get("semester", None)
        self.subjects = data.get("subjects", [])
        self.preferences = data.get("preferences", {})  # e.g., {"quiz_style":"mcq-heavy"}

    def to_dict(self):
        return {
            "name": self.name,
            "branch": self.branch,
            "semester": self.semester,
            "subjects": self.subjects,
            "preferences": self.preferences,
        }


class StudyMateState:
    """
    Global persistent state for the notebook:
    - profile
    - history of tasks (notes/quizzes/doubts/plans)
    """
    def __init__(self, path=STATE_PATH):
        self.path = path
        raw = load_json(path) or {}
        self.profile = StudyProfile(raw.get("profile"))
        self.history = raw.get("history", [])

    def add_event(self, kind, payload):
        event = {
            "type": kind,
            "time": now_time(),
            "payload": payload,
        }
        self.history.append(event)
        self.save()

    def save(self):
        data = {
            "profile": self.profile.to_dict(),
            "history": self.history,
        }
        save_json(self.path, data)

    def recent(self, n=10):
        return list(self.history[-n:])

state = StudyMateState()
print("S3 loaded â€” StudyMateState ready. History size:", len(state.history))

# ============================================================================
# S4 â€” NotesAgent
# ============================================================================

class NotesAgent:
    """
    NotesAgent:
    - Summarizes long text into bullet points.
    - Finds key terms (naive keyword & capitalized words).
    - Detects formula-like patterns.
    - Builds basic flashcards from key sentences.
    """

    def _split_sentences(self, text):
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [p.strip() for p in parts if p.strip()]

    def _summarize(self, text, max_points=6):
        sents = self._split_sentences(text)
        if not sents:
            return []
        # Simple heuristic: choose first N and some central ones
        summary = []
        if sents:
            summary.append(sents[0])
        if len(sents) > 2:
            mid = len(sents) // 2
            summary.append(sents[mid])
        for s in sents[1:]:
            if len(summary) >= max_points:
                break
            if len(s) < 150:  # prefer shorter sentences
                summary.append(s)
        # deduplicate
        final = []
        seen = set()
        for s in summary:
            if s not in seen:
                final.append(s)
                seen.add(s)
        return final

    def _extract_keywords(self, text, max_terms=10):
        words = re.findall(r'\b[A-Za-z][A-Za-z0-9_]+\b', text)
        freq = {}
        for w in words:
            wl = w.lower()
            if len(wl) <= 3:
                continue
            freq[wl] = freq.get(wl, 0) + 1
        sorted_terms = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [t[0] for t in sorted_terms[:max_terms]]

    def _extract_formulas(self, text, max_formulas=10):
        # extremely naive â€œformula-likeâ€� detection
        pattern = r'[\w\s\+\-\*/\^\=\(\)]+=[\w\s\+\-\*/\^\(\)]+'
        found = re.findall(pattern, text)
        cleaned = []
        for f in found:
            f = f.strip()
            if len(f) < 5 or len(f) > 80:
                continue
            cleaned.append(f)
        # deduplicate
        uniq = []
        seen = set()
        for f in cleaned:
            if f not in seen:
                uniq.append(f)
                seen.add(f)
        return uniq[:max_formulas]

    def _flashcards(self, text, keywords, max_cards=6):
        sents = self._split_sentences(text)
        cards = []
        for k in keywords:
            for s in sents:
                if k.lower() in s.lower():
                    q = f"What is {k}?"
                    a = s
                    cards.append({"question": q, "answer": a})
                    break
            if len(cards) >= max_cards:
                break
        return cards

    def generate_notes(self, raw_text):
        if not raw_text.strip():
            return {"summary": [], "keywords": [], "formulas": [], "flashcards": []}

        summary = self._summarize(raw_text)
        keywords = self._extract_keywords(raw_text)
        formulas = self._extract_formulas(raw_text)
        flashcards = self._flashcards(raw_text, keywords)

        result = {
            "summary": summary,
            "keywords": keywords,
            "formulas": formulas,
            "flashcards": flashcards,
        }
        state.add_event("notes", result)
        return result

notes_agent = NotesAgent()
print("S4 loaded â€” NotesAgent ready.")

# ============================================================================
# S5 â€” QuizAgent
# ============================================================================

class QuizAgent:
    """
    QuizAgent:
    - Creates MCQs, 1-mark, and long-answer questions from given text.
    - Difficulty is heuristic: based on length and presence of formulas.
    """

    def _split_sentences(self, text):
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [p.strip() for p in parts if p.strip()]

    def _pick_terms(self, text, n=8):
        words = re.findall(r'\b[A-Za-z][A-Za-z0-9_]+\b', text)
        freq = {}
        for w in words:
            wl = w.lower()
            if len(wl) <= 3:
                continue
            freq[wl] = freq.get(wl, 0) + 1
        sorted_terms = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [t[0] for t in sorted_terms[:n]]

    def _difficulty_tag(self, sentence):
        f_count = sentence.count("=") + sentence.count("^")
        length = len(sentence)
        score = f_count + (length / 120)
        if score < 1.2:
            return "easy"
        elif score < 2.0:
            return "medium"
        else:
            return "hard"

    def generate_quiz(self, topic_text, n_mcq=5, n_short=5, n_long=3):
        sentences = self._split_sentences(topic_text)
        if not sentences:
            return {"mcq": [], "short": [], "long": []}

        terms = self._pick_terms(topic_text, n=10)
        mcqs = []
        for s in sentences:
            if len(mcqs) >= n_mcq:
                break
            candidates = [t for t in terms if t.lower() in s.lower()]
            if not candidates:
                continue
            key_term = random.choice(candidates)
            stem = re.sub(key_term, "_____", s, flags=re.IGNORECASE)
            wrong_options = [t for t in terms if t != key_term]
            random.shuffle(wrong_options)
            options = [key_term] + wrong_options[:3]
            random.shuffle(options)

            mcqs.append({
                "question": stem,
                "options": options,
                "answer": key_term,
                "difficulty": self._difficulty_tag(s),
            })

        short_qs = []
        for s in sentences:
            if len(short_qs) >= n_short:
                break
            short_qs.append({
                "question": f"Explain briefly: {s}",
                "difficulty": self._difficulty_tag(s),
            })

        long_qs = []
        for s in sentences[: max(n_long, 3)]:
            long_qs.append({
                "question": f"Write a detailed note on: {s}",
                "difficulty": self._difficulty_tag(s),
            })

        quiz = {"mcq": mcqs, "short": short_qs, "long": long_qs}
        state.add_event("quiz", {"counts": {"mcq": len(mcqs), "short": len(short_qs), "long": len(long_qs)}})
        return quiz

quiz_agent = QuizAgent()
print("S5 loaded â€” QuizAgent ready.")

# ============================================================================
# S6 â€” DoubtSolverAgent
# ============================================================================

class DoubtSolverAgent:
    """
    DoubtSolverAgent:
    - Detects and solves simple arithmetic expressions.
    - For general conceptual doubts, returns a structured explanation template.
    """

    def _extract_math(self, text):
        # Replace common symbols
        expr = text.replace("Ã—", "*").replace("Ã·", "/").replace("^", "**")
        # Very naive pattern: just keep math-like chars
        candidate = "".join(ch for ch in expr if ch in "0123456789.+-*/() ")
        candidate = candidate.strip()
        if candidate and any(ch.isdigit() for ch in candidate):
            return candidate
        return None

    def _solve_math(self, expr):
        try:
            val = eval(expr, {"__builtins__": {}})
            return f"Computation: `{expr}` = **{val}**"
        except Exception as e:
            return f"I tried to compute `{expr}` but got an error: {e}"

    def answer_doubt(self, text):
        math_expr = self._extract_math(text)
        math_part = None
        if math_expr:
            math_part = self._solve_math(math_expr)

        # Very simple extraction of main topic (first noun-ish word)
        tokens = re.findall(r'\b[A-Za-z][A-Za-z0-9_]+\b', text)
        topic = tokens[0] if tokens else "this concept"

        explanation = f"""
ğŸ“˜ Concept Focus: **{topic}**

1ï¸�âƒ£ What it is (high-level)
- This is a placeholder explanation for **{topic}**.
- In a full Smart StudyMate pipeline, this would be expanded using an LLM.
- You can plug in your favourite model here to return a real explanation.

2ï¸�âƒ£ Why it is important
- {topic} often appears in exams and interviews.
- Understanding the intuition helps you apply it in different problems.

3ï¸�âƒ£ How to remember
- Try to connect **{topic}** with:
  - a small visual diagram
  - a real-world analogy
  - 2â€“3 typical exam questions.

4ï¸�âƒ£ Next steps
- Create 3 practice questions related to **{topic}**.
- Attempt to solve them without looking at the solution.
"""

        combined = explanation
        if math_part:
            combined = math_part + "\n\n" + explanation

        state.add_event("doubt", {"question": text, "has_math": bool(math_expr)})
        return combined.strip()

doubt_agent = DoubtSolverAgent()
print("S6 loaded â€” DoubtSolverAgent ready.")

# ============================================================================
# S7 â€” StudyPlannerAgent
# ============================================================================

class StudyPlannerAgent:
    """
    StudyPlannerAgent:
    - Takes exam date (as days-from-today or "in N days" style) and topic list.
    - Splits topics across days.
    - Adds revision days.
    """

    def _parse_days(self, text):
        # try to detect something like "in 10 days" or "10 days"
        match = re.search(r'(\d+)\s*day', text.lower())
        if match:
            return max(1, int(match.group(1)))
        return 7  # default 1-week plan

    def _parse_topics(self, text):
        # topics separated by commas or newlines
        raw = re.split(r'[,;\n]', text)
        topics = [r.strip() for r in raw if r.strip()]
        return topics or ["General Revision"]

    def build_plan(self, days_text, topics_text):
        n_days = self._parse_days(days_text)
        topics = self._parse_topics(topics_text)

        today = datetime.today().date()
        daily_plan = []

        base_per_day = max(1, len(topics) // max(1, n_days - 2))
        idx = 0
        for d in range(n_days):
            date = today + timedelta(days=d)
            day_topics = []
            if d in [n_days - 2, n_days - 1]:
                # last 2 days reserved for revision
                day_topics.append("Full syllabus revision + previous mistakes review")
            else:
                for _ in range(base_per_day):
                    if idx < len(topics):
                        day_topics.append(topics[idx])
                        idx += 1
            if not day_topics and topics:
                day_topics.append(random.choice(topics))

            daily_plan.append({
                "day": d + 1,
                "date": str(date),
                "tasks": day_topics,
            })

        plan = {"days": n_days, "topics": topics, "schedule": daily_plan}
        state.add_event("plan", {"days": n_days, "topics_count": len(topics)})
        return plan

planner_agent = StudyPlannerAgent()
print("S7 loaded â€” StudyPlannerAgent ready.")

# ============================================================================
# S8 â€” RouterAgent (Brain of Smart StudyMate)
# ============================================================================

class RouterAgent:
    """
    RouterAgent:
    - Classifies input mode and routes to specific agents.
    - Modes: notes, quiz, doubt, plan.
    """

    def __init__(self, notes_agent, quiz_agent, doubt_agent, planner_agent):
        self.notes_agent = notes_agent
        self.quiz_agent = quiz_agent
        self.doubt_agent = doubt_agent
        self.planner_agent = planner_agent

    def detect_mode(self, mode_text, user_text):
        mt = (mode_text or "").lower().strip()
        ut = (user_text or "").lower()

        # explicit dropdown selection is trusted
        if mt in ["notes", "quiz", "doubt", "plan"]:
            return mt

        # fallback: keyword-based detection
        if any(k in ut for k in ["summarize", "notes", "revise", "revision"]):
            return "notes"
        if any(k in ut for k in ["quiz", "mcq", "practice", "test"]):
            return "quiz"
        if any(k in ut for k in ["why", "what is", "how to", "explain"]):
            return "doubt"
        if any(k in ut for k in ["plan", "schedule", "timetable", "exam"]):
            return "plan"
        return "notes"

    def handle(self, mode_text, user_text, extra_text=""):
        mode = self.detect_mode(mode_text, user_text)
        if mode == "notes":
            notes = self.notes_agent.generate_notes(user_text)
            return "notes", notes
        elif mode == "quiz":
            quiz = self.quiz_agent.generate_quiz(user_text)
            return "quiz", quiz
        elif mode == "doubt":
            ans = self.doubt_agent.answer_doubt(user_text)
            return "doubt", ans
        elif mode == "plan":
            plan = self.planner_agent.build_plan(days_text=extra_text or "7 days", topics_text=user_text)
            return "plan", plan
        else:
            return "error", {"error": "Unknown mode"}

router = RouterAgent(notes_agent, quiz_agent, doubt_agent, planner_agent)
print("S8 loaded â€” RouterAgent ready.")

# ============================================================================
# S9 â€” UI: Smart StudyMate Console (ipywidgets)
# ============================================================================

ui_messages = []

console_out = widgets.Output(
    layout=widgets.Layout(border="1px solid #333", height="420px", overflow="auto", padding="8px")
)

mode_dropdown = widgets.Dropdown(
    options=[("Notes (Summarize)", "notes"),
             ("Quiz Generator", "quiz"),
             ("Doubt Solver", "doubt"),
             ("Study Planner", "plan")],
    value="notes",
    description="Mode:",
    layout=widgets.Layout(width="60%")
)

main_input = widgets.Textarea(
    placeholder="Paste chapter text, write a doubt, or list topics...",
    layout=widgets.Layout(width="100%", height="120px")
)

extra_input = widgets.Text(
    placeholder="For planner: e.g., 'in 10 days' (optional)",
    description="Extra:",
    layout=widgets.Layout(width="60%")
)

run_btn = widgets.Button(description="Run Smart StudyMate", button_style="primary")
clear_btn = widgets.Button(description="Clear Output", button_style="warning")

def render_block(content_html):
    with console_out:
        display(HTML(content_html))

def render_history():
    events = state.recent(8)
    if not events:
        return "<i>No recent activity yet.</i>"
    rows = []
    for e in reversed(events):
        rows.append(f"<li><b>{e['time']}</b> â€” {e['type'].upper()}</li>")
    return "<ul>" + "\n".join(rows) + "</ul>"

def handle_run(_=None):
    text = main_input.value.strip()
    extra = extra_input.value.strip()
    if not text:
        return

    mode = mode_dropdown.value
    kind, payload = router.handle(mode, text, extra_text=extra)

    if kind == "notes":
        summary = payload["summary"]
        keywords = payload["keywords"]
        formulas = payload["formulas"]
        cards = payload["flashcards"]

        html = "<h3>ğŸ“š Generated Revision Notes</h3>"
        if summary:
            html += "<h4>Summary</h4><ul>" + "".join(f"<li>{s}</li>" for s in summary) + "</ul>"
        if keywords:
            html += "<h4>Key Terms</h4><p>" + ", ".join(f"<code>{k}</code>" for k in keywords) + "</p>"
        if formulas:
            html += "<h4>Formulas (Heuristic)</h4><ul>" + "".join(f"<li><code>{f}</code></li>" for f in formulas) + "</ul>"
        if cards:
            html += "<h4>Flashcards</h4>"
            for c in cards:
                html += f"<p><b>Q:</b> {c['question']}<br><b>A:</b> {c['answer']}</p>"

    elif kind == "quiz":
        mcq = payload["mcq"]
        short = payload["short"]
        long_q = payload["long"]

        html = "<h3>ğŸ“� Quiz Package</h3>"

        if mcq:
            html += "<h4>MCQs</h4>"
            for i, q in enumerate(mcq, 1):
                html += f"<p><b>Q{i} ({q['difficulty']}):</b> {q['question']}<br>"
                for idx, opt in enumerate(q['options']):
                    html += f"&nbsp;&nbsp;({chr(ord('A')+idx)}) {opt}<br>"
                html += f"<i>Ans:</i> {q['answer']}<br></p>"

        if short:
            html += "<h4>1-Mark / Short Questions</h4><ol>"
            for q in short:
                html += f"<li>({q['difficulty']}) {q['question']}</li>"
            html += "</ol>"

        if long_q:
            html += "<h4>Long Answer Questions</h4><ol>"
            for q in long_q:
                html += f"<li>({q['difficulty']}) {q['question']}</li>"
            html += "</ol>"

    elif kind == "doubt":
        html = "<h3>â�“ Doubt Solution (Prototype)</h3>"
        html += "<pre style='white-space:pre-wrap;'>" + payload + "</pre>"

    elif kind == "plan":
        schedule = payload["schedule"]
        html = "<h3>ğŸ“† Study Plan</h3>"
        html += f"<p>Total days: <b>{payload['days']}</b> | Topics: <b>{len(payload['topics'])}</b></p>"
        html += "<table border='1' cellspacing='0' cellpadding='4'>"
        html += "<tr><th>Day</th><th>Date</th><th>Tasks</th></tr>"
        for day in schedule:
            tasks = "<br>".join(day["tasks"])
            html += f"<tr><td>{day['day']}</td><td>{day['date']}</td><td>{tasks}</td></tr>"
        html += "</table>"

    else:
        html = "<p>Error routing request.</p>"

    hist_html = "<h4>Recent Activity</h4>" + render_history()
    final_html = html + "<hr>" + hist_html

    console_out.clear_output()
    render_block(final_html)

def handle_clear(_=None):
    console_out.clear_output()
    with console_out:
        display(HTML("<i>Output cleared. Run Smart StudyMate again with new input.</i>"))

run_btn.on_click(handle_run)
clear_btn.on_click(handle_clear)

ui = widgets.VBox([
    widgets.HTML("<h2>ğŸ¤– Smart StudyMate Agent â€” Interactive Console</h2>"
                 "<p>Select a mode, paste your content, and click <b>Run Smart StudyMate</b>.</p>"),
    mode_dropdown,
    main_input,
    extra_input,
    widgets.HBox([run_btn, clear_btn]),
    console_out
])

display(ui)

with console_out:
    display(HTML("""
    <b>Welcome!</b><br>
    - <b>Notes:</b> Paste chapter text to get summary, key terms, formulas, and flashcards.<br>
    - <b>Quiz:</b> Paste topic notes to get MCQs and theory questions.<br>
    - <b>Doubt:</b> Ask any conceptual question or small math expression.<br>
    - <b>Plan:</b> List topics in the main box and write e.g. <code>in 10 days</code> in Extra box.
    """))

print("S9 loaded â€” Smart StudyMate UI ready.")

# ============================================================================
# S10 â€” Architecture Diagram (Printed)
# ============================================================================

arch = r"""
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�
â”‚              Smart StudyMate UI              â”‚
â”‚     (Kaggle Notebook + ipywidgets)           â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                â”‚  user input
                                â–¼
                      â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�
                      â”‚      RouterAgent      â”‚
                      â”‚  - mode detection     â”‚
                      â”‚  - task routing       â”‚
                      â””â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
       â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�
       â–¼                      â–¼                         â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�       â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�           â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�
â”‚  NotesAgent â”‚       â”‚  QuizAgent  â”‚           â”‚ DoubtAgent     â”‚
â”‚ - summary   â”‚       â”‚ - MCQs      â”‚           â”‚ - math helper  â”‚
â”‚ - keywords  â”‚       â”‚ - 1-mark    â”‚           â”‚ - explanation  â”‚
â”‚ - formulas  â”‚       â”‚ - long ans. â”‚           â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
â”‚ - flashcard â”‚       â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
â””â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”˜
       â”‚
       â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�
â”‚ PlannerAgent  â”‚
â”‚ - days        â”‚
â”‚ - topic split â”‚
â”‚ - revision    â”‚
â””â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”˜
       â”‚
       â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�
â”‚  StudyMateState    â”‚
â”‚ - profile          â”‚
â”‚ - history log      â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
"""

print(arch)
print("S10 loaded â€” Architecture diagram printed.")


