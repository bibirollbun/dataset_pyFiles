# -------------------------
#  Imports & Model Stub
# -------------------------

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import time
import uuid
import os

# PDF loader
import pypdf  

class SimpleLLM:
    """
    A stub LLM that just echoes the prompt
    or does a basic reversal. 
    This is for demonstration in offline environment.
    """
    def ask(self, prompt: str) -> str:
        # Very basic fake answer
        return "This is a stubbed answer for: " + prompt[:100] + "... (real LLM not available in this environment)"

# Instantiate
llm = SimpleLLM()

print("LLM stub ready.")



# -------------------------
# Data Classes
# -------------------------

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Output for the summarizer agent(to summarize the query)
@dataclass
class SummaryOutput:
    tldr: str
    bullets: List[str]
    study_notes: List[str]

# Output for the QA agent (answers + sources)
@dataclass
class QAOutput:
    answer: str
    sources: List[str]

# Output for the productivity agent (study plan)
@dataclass
class StudyPlan:
    topic: str
    days: Dict[str, List[str]]

# Session state structure
@dataclass
class SessionState:
    pdf_text: str = ""
    last_query: str = ""
    study_preferences: Optional[Dict[str, Any]] = None



# -------------------------
# In-Memory Session Store
# -------------------------

class InMemorySessionService:
    """
    Very simple in-memory session store.
    Each session has:
      - user_id
      - created_at
      - metadata (optional)
      - state (dict to store pdf text, last query, etc.)
    """

    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(self, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "user_id": user_id,
            "created_at": time.time(),
            "metadata": metadata or {},
            "state": {}
        }
        return session_id

    def get(self, session_id: str) -> Dict[str, Any]:
        return self.sessions.get(session_id, {})

    def set_state(self, session_id: str, key: str, value: Any):
        if session_id not in self.sessions:
            raise KeyError("Session not found")
        self.sessions[session_id]["state"][key] = value

    def get_state(self, session_id: str, key: str, default=None):
        return self.sessions.get(session_id, {}).get("state", {}).get(key, default)



# -------------------------
#  PDF Loader
# -------------------------

class PDFLoader:
    def load(self, file_path: str) -> str:
        text = ""
        try:
            # Try pypdf first
            import pypdf
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n\n"
                    except:
                        pass

        except ImportError:
            # Fallback: basic binary read (least reliable, but safe fallback)
            try:
                with open(file_path, "rb") as f:
                    raw = f.read().decode(errors="ignore")
                    text = raw
            except:
                print("PDF parsing failed: no PDF library available.")
                return ""

        return text.strip()



# -------------------------
# Offline LLM Wrapper
# -------------------------

class LLM:
    """
    Simple offline-language-model wrapper.
    Uses the KaggleHub llama3.1-8b model loaded in Cell 1.
    
    Assumes 'model' variable is already initialized:
        model = kagglehub.model_client("kaggle/llama3.1-8b")
    """

    def ask(self, prompt: str, max_tokens: int = 800) -> str:
        try:
            output = model.generate(
                prompt=prompt,
                max_length=max_tokens
            )
            return output
        except Exception as e:
            print("LLM Error:", e)
            return "Sorry, I could not process that request."



# -------------------------
# Summarizer Agent
# -------------------------

class SummarizerAgent:
    """
    Summarizes long PDF text into:
      - TLDR (1 sentence)
      - Key points (3-5 bullets)
      - Study notes (easy explanation)
    """

    def __init__(self, llm: LLM):
        self.llm = llm

    def summarize(self, text: str) -> SummaryOutput:
        prompt = f"""
You are a helpful study assistant for students.

Summarize the following text into THREE clear sections:

1. TLDR (1 sentence).
2. Key Points (3 to 5 bullet points).
3. Study Notes (simple explanation for students).

Text to summarize:
{text}

Format your answer EXACTLY like:
TLDR: ...
Key Points:
- ...
- ...
Study Notes:
* ...
* ...
"""

        raw_output = self.llm.ask(prompt, max_tokens=700)

        # ---- Simple extraction parsing ----
        tldr = ""
        bullets = []
        study_notes = []

        lines = raw_output.split("\n")

        mode = None
        for line in lines:
            line_strip = line.strip()

            # detect sections
            if line_strip.lower().startswith("tldr"):
                mode = "tldr"
                continue
            if "key points" in line_strip.lower():
                mode = "key"
                continue
            if "study notes" in line_strip.lower():
                mode = "notes"
                continue

            # fill sections
            if mode == "tldr" and line_strip:
                tldr = line_strip
            elif mode == "key" and line_strip.startswith("-"):
                bullets.append(line_strip)
            elif mode == "notes" and line_strip.startswith("*"):
                study_notes.append(line_strip)

        # fallback safety
        if not tldr:
            tldr = raw_output[:150]

        return SummaryOutput(
            tldr=tldr,
            bullets=bullets,
            study_notes=study_notes,
        )



# -------------------------
# QA Agent
# -------------------------

class QAAgent:
    """
    Answers student questions using ONLY the PDF text.
    Since Kaggle doesn't allow internet, we simulate retrieval
    by selecting the most relevant PDF chunks.
    """

    def __init__(self, llm: LLM):
        self.llm = llm

    def _simple_retrieval(self, query: str, pdf_text: str, window: int = 500) -> List[str]:
        """
        VERY lightweight retrieval:
        - Splits the PDF text into chunks
        - Picks chunks containing important query words
        """
        chunks = []
        words = pdf_text.split()

        # chunk the PDF text
        for i in range(0, len(words), window):
            chunk = " ".join(words[i:i+window])
            chunks.append(chunk)

        # pick chunks matching query keywords
        query_words = [w.lower() for w in query.split() if len(w) > 3]
        matched = []

        for chunk in chunks:
            if any(qw in chunk.lower() for qw in query_words):
                matched.append(chunk)

        # fallback if no match
        if not matched:
            matched = chunks[:2]

        return matched[:3]   # return top 3 relevant chunks

    def answer(self, query: str, pdf_text: str) -> QAOutput:

        # Step 1: get relevant PDF chunks
        sources = self._simple_retrieval(query, pdf_text)
        context = "\n\n---\n\n".join(sources)

        # Step 2: ask offline LLM
        prompt = f"""
You are a helpful study assistant.

Using ONLY the context below, answer the student's question clearly.

QUESTION:
{query}

CONTEXT:
{context}

Write your answer in simple language for students. 
Then provide 1-2 bullet points about which parts of the PDF helped your answer (these are your 'sources').
"""

        raw_output = self.llm.ask(prompt, max_tokens=700)

        return QAOutput(
            answer=raw_output,
            sources=sources
        )



# -------------------------
# Productivity Agent
# -------------------------

class ProductivityAgent:
    """
    Generates a simple multi-day study plan for a student.
    Works fully offline using the LLM.
    Produces:
      - daily tasks
      - checkpoints
      - simple schedule
    """

    def __init__(self, llm: LLM):
        self.llm = llm

    def create_study_plan(self, topic: str, days: int = 7) -> StudyPlan:
        prompt = f"""
You are a student productivity coach.

Create a {days}-day study plan for the topic: "{topic}".

For EACH day give:
- 2 to 3 small tasks
- simple explanation
- 1 short checkpoint to review progress

Format EXACTLY like this:
Day 1:
- task 1
- task 2
checkpoint: ...

Day 2:
- task 1
- task 2
checkpoint: ...

(continue until Day {days})
"""

        raw = self.llm.ask(prompt, max_tokens=900)

        # ---- Simple parsing ----
        plan_dict = {}
        current_day = None

        for line in raw.split("\n"):
            line_strip = line.strip()

            # Detect "Day X"
            if line_strip.lower().startswith("day"):
                current_day = line_strip.replace(":", "")
                plan_dict[current_day] = []
                continue

            # Daily tasks
            if current_day and (line_strip.startswith("-") or line_strip.startswith("checkpoint")):
                plan_dict[current_day].append(line_strip)

        # Safety fallback
        if not plan_dict:
            plan_dict = {"Day 1": ["- Review basics", "- Watch intro video", "checkpoint: recap key terms"]}

        return StudyPlan(topic=topic, days=plan_dict)



# -------------------------
# Coordinator Agent
# -------------------------

class Coordinator:
    """
    Central controller that connects:
      - PDF loader
      - Summarizer agent
      - QA agent
      - Productivity agent

    Uses the in-memory session store to keep data.
    """

    def __init__(self, session_service, summarizer, qa_agent, productivity_agent):
        self.session_service = session_service
        self.summarizer = summarizer
        self.qa_agent = qa_agent
        self.productivity = productivity_agent

    # Load PDF into session
    def load_pdf(self, session_id: str, file_path: str):
        loader = PDFLoader()
        pdf_text = loader.load(file_path)

        # Save to session
        self.session_service.set_state(session_id, "pdf_text", pdf_text)
        return "PDF loaded successfully."

    # Summarize the uploaded PDF
    def summarize_pdf(self, session_id: str) -> SummaryOutput:
        pdf_text = self.session_service.get_state(session_id, "pdf_text")

        if not pdf_text:
            return SummaryOutput(
                tldr="No PDF loaded.",
                bullets=["Please upload a PDF first."],
                study_notes=[]
            )

        return self.summarizer.summarize(pdf_text)

    # Ask questions about the PDF
    def ask_question(self, session_id: str, question: str) -> QAOutput:
        pdf_text = self.session_service.get_state(session_id, "pdf_text")

        if not pdf_text:
            return QAOutput(
                answer="No PDF loaded. Upload a PDF first.",
                sources=[]
            )

        # Store last query (optional, for follow-ups)
        self.session_service.set_state(session_id, "last_query", question)

        return self.qa_agent.answer(question, pdf_text)

    # Generate a study plan
    def generate_study_plan(self, session_id: str, topic: str, days: int = 7) -> StudyPlan:
        return self.productivity.create_study_plan(topic, days=days)



# -------------------------
# Initialize Agents + Session
# -------------------------

# 1. LLM wrapper (model already loaded in Cell 1)
llm = LLM()

# 2. Initialize agents
summarizer_agent = SummarizerAgent(llm)
qa_agent = QAAgent(llm)
productivity_agent = ProductivityAgent(llm)

# 3. Create session manager
session_service = InMemorySessionService()

# 4. Create coordinator
coordinator = Coordinator(
    session_service=session_service,
    summarizer=summarizer_agent,
    qa_agent=qa_agent,
    productivity_agent=productivity_agent
)

# 5. Start a new session for the demo
session_id = session_service.create_session(user_id="student_001")

print("All agents initialized successfully!")
print("Session ID:", session_id)



# -------------------------
# Demo: Ask Questions About the PDF
# -------------------------

import ipywidgets as widgets
from IPython.display import display, clear_output

# Text box for user to enter question
question_box = widgets.Text(
    value='',
    placeholder='Ask anything about the PDF...',
    description='Question:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='80%')
)

# Button to submit question
ask_btn = widgets.Button(
    description='Ask',
    button_style='info',
    tooltip='Ask question'
)

# Output area
output_area = widgets.Output()

def on_ask_clicked(b):
    with output_area:
        clear_output()
        
        user_q = question_box.value.strip()
        if not user_q:
            print("Please enter a question.")
            return
        
        print("Processing your question...")
        
        # Call QA agent through coordinator
        qa_result = coordinator.ask_question(session_id, user_q)
        
        print("\n========== ANSWER ==========")
        print(qa_result.answer)
        
        print("\n========== SOURCES (PDF EXTRACTS) ==========")
        for i, src in enumerate(qa_result.sources, 1):
            print(f"[Chunk {i}] {src[:300]}...")  # show first 300 chars

# Bind button event
ask_btn.on_click(on_ask_clicked)

# Display widgets
display(question_box)
display(ask_btn)
display(output_area)



# -------------------------
# Demo: Study Plan Generator
# -------------------------

import ipywidgets as widgets
from IPython.display import display, clear_output

# Topic input widget
topic_box = widgets.Text(
    value='',
    placeholder='Enter the study topic (e.g., Deep Learning)',
    description='Study Topic:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='80%')
)

# Days input widget
days_box = widgets.IntText(
    value=7,
    description='Days:',
    style={'description_width': 'initial'}
)

# Button to generate study plan
plan_button = widgets.Button(
    description='Generate Study Plan',
    button_style='success',
    tooltip='Generate plan'
)

# Output area
plan_output = widgets.Output()

def on_plan_click(b):
    with plan_output:
        clear_output()
        
        topic = topic_box.value.strip()
        days = days_box.value
        
        if not topic:
            print("Please enter a topic.")
            return
        
        print(f"Generating a {days}-day study plan for: {topic} ...\n")
        
        study_plan = coordinator.generate_study_plan(session_id, topic, days=days)
        
        # Display formatted output
        print("========== STUDY PLAN ==========\n")

        for day, tasks in study_plan.days.items():
            print(day)
            for t in tasks:
                print(" ", t)
            print()

# Bind event
plan_button.on_click(on_plan_click)

# Display UI
display(topic_box)
display(days_box)
display(plan_button)
display(plan_output)


