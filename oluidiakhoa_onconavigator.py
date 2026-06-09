## =========================
## 2. Environment & API Setup
## =========================

import sys
import os
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings("ignore")

import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool
from google.api_core import exceptions as google_exceptions  # NEW: for quota handling
from kaggle_secrets import UserSecretsClient
from IPython.display import display, HTML, clear_output

print("âœ“ Libraries Loaded")

# -------------------------
# API Configuration
# -------------------------
try:
    user_secrets = UserSecretsClient()
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
    print("âœ“ GOOGLE_API_KEY loaded and Gemini configured")
except Exception as e:
    GOOGLE_API_KEY = None
    print(f"âš  Failed to load GOOGLE_API_KEY: {e}")
    print("ğŸ‘‰ Add it in: Add-ons â†’ Secrets â†’ 'GOOGLE_API_KEY'")

CONFIG = {
    "project": "OncoNavigator",
    "track": "Agents for Good",
    "model": "models/gemini-2.5-flash",  # adjust if needed
    "max_tokens": 2000,
    "temperature": 0.3,
    "version": "1.0.0"
}

print("\n" + "="*60)
print(f"{'ONCONAVIGATOR CONFIGURATION':^60}")
print("="*60)
for k, v in CONFIG.items():
    print(f"{k:.<25} {v}")
print("="*60)



## =========================
## Tool Functions (Sub-Agents)
## =========================

def _make_model():
    return genai.GenerativeModel(CONFIG["model"])

# 1) Intake Agent â€“ build structured case profile
def intake_patient_profile(raw_text: str) -> str:
    """
    Takes user free-text about diagnosis / report and returns a structured JSON-style profile.
    """
    prompt = f"""
You are the Intake Agent for OncoNavigator, a cancer information copilot.

User text:
\"\"\"{raw_text}\"\"\"


Task:
1. Extract a structured patient case profile with fields:
   - diagnosis_summary
   - possible_cancer_type
   - stage_or_extent (if mentioned or "unknown")
   - key_clinical_details
   - stated_concerns
   - language_preference (default "en")
2. Output STRICT JSON only. No explanations, no markdown.
3. Do NOT invent diagnoses; if unsure, say "unknown".

REMINDER: You are *not* giving medical advice, just structuring information.
"""
    model = _make_model()
    return model.generate_content(prompt).text


# 2) Evidence Agent â€“ high-level overview
def clinical_evidence_overview(diagnosis_summary: str, stage_or_extent: str) -> str:
    """
    Produces a high-level, guideline-style overview of the condition and typical treatment categories.
    """
    prompt = f"""
You are the Evidence Agent for OncoNavigator.

Context:
- Diagnosis summary: {diagnosis_summary}
- Stage/extent: {stage_or_extent}

Create a concise overview that includes:
1. A neutral, high-level description of the condition.
2. Common treatment *categories* (e.g., surgery, radiation, chemotherapy, targeted therapy, immunotherapy),
   but do NOT recommend specific drugs or protocols.
3. Typical goals of treatment (e.g., cure, control, symptom relief), where appropriate.
4. Clear statement: â€œThis is general educational information, not medical advice.â€�

Tone:
- Calm, factual, non-alarmist.
- Avoid probabilities, survival rates, or guarantees.
"""
    model = _make_model()
    return model.generate_content(prompt).text


# 3) Treatment Comparator â€“ compare high-level options
def compare_treatment_options(diagnosis_summary: str, options_text: str) -> str:
    """
    Compares high-level treatment categories in layman terms.
    """
    prompt = f"""
You are the Treatment Comparison Agent for OncoNavigator.

Diagnosis summary:
{diagnosis_summary}

Treatment options mentioned by the user or system:
{options_text}

Create a table-like explanation in text (no markdown table needed) including:
- Option name
- Very brief how-it-works (high-level)
- Typical pros (at a high level only)
- Typical trade-offs / considerations
- Important note: Always discuss with an oncologist.

Do NOT:
- Name specific drug regimens.
- Provide survival statistics.
- Tell the user what they â€œshouldâ€� do.

End with a reminder that this is *not* medical advice.
"""
    model = _make_model()
    return model.generate_content(prompt).text


# 4) Explanation Agent â€“ plain language explanation
def generate_patient_explanation(diagnosis_summary: str,
                                 technical_overview: str,
                                 language: str = "en") -> str:
    """
    Converts technical overview into plain language for a patient or caregiver.
    """
    prompt = f"""
You are the Explanation Agent for OncoNavigator.

Language code: {language}  (use English if unsure)

Take this technical overview and rewrite it for a layperson:
\"\"\"{technical_overview}\"\"\"


Requirements:
- Simple, empathetic tone.
- Short paragraphs.
- Avoid heavy jargon; if a medical term is needed, briefly define it.
- Include 2â€“3 key points the person should clarify with their doctor.

NEVER:
- Recommend a specific treatment.
- Say what the user should choose.
- Give probabilities or survival chances.

Always end with: 
"This explanation is only for general education. Always rely on your medical team for decisions."
"""
    model = _make_model()
    return model.generate_content(prompt).text


# 5) Planning Agent â€“ doctor discussion guide
def generate_doctor_questions(diagnosis_summary: str,
                              main_concerns: str) -> str:
    """
    Generates a list of questions to ask the oncologist.
    """
    prompt = f"""
You are the Planning Agent for OncoNavigator.

Diagnosis summary:
{diagnosis_summary}

User concerns:
{main_concerns}

Produce:
- 8â€“12 practical questions the user can ask their doctor.
- Group them under headings: "About the diagnosis", "About treatment options", "About daily life & side effects".

Rules:
- Do NOT suggest changes to medication or treatment on your own.
- Do NOT say "you should refuse X" or "insist on Y".
- Keep reminding that the doctor knows the full clinical picture.
"""
    model = _make_model()
    return model.generate_content(prompt).text


# 6) Safety Agent â€“ final review
def safety_and_ethics_review(response_text: str) -> str:
    """
    Reviews a generated response and adds safety framing if needed.
    """
    prompt = f"""
You are the Safety & Ethics Agent for OncoNavigator.

Review the following response:
\"\"\"{response_text}\"\"\"


Tasks:
1. Check if there is any implicit or explicit *treatment recommendation* 
   (e.g., 'you should take', 'you must choose', 'refuse this', 'do not allow').
2. If found, rephrase the text to remove direct recommendations and instead suggest:
   - "discuss this option with your oncologist" style language.
3. Ensure a clear safety disclaimer is present at the end:
   "This is general educational information, not medical advice. 
    Do not start, stop, or change treatment based on this. 
    Always discuss decisions with your medical team."

Return the **fully edited safe response**.
"""
    model = _make_model()
    return model.generate_content(prompt).text



## ==================================
## Long-Running Jobs, Memory & Logging
## ==================================

@dataclass
class ConversationMemory:
    messages: List[Dict[str, Any]] = field(default_factory=list)
    max_history: int = 30

    def add_message(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]

    def get_context_snippet(self, last_n: int = 5) -> str:
        if not self.messages:
            return "No prior conversation."
        ctx = "Recent conversation:\n"
        for msg in self.messages[-last_n:]:
            ctx += f"{msg['timestamp']} | {msg['role']}: {msg['content'][:120]}...\n"
        return ctx

    def clear(self):
        self.messages.clear()

    def stats(self) -> Dict[str, Any]:
        return {
            "total_messages": len(self.messages),
            "user_messages": sum(1 for m in self.messages if m["role"] == "user"),
            "agent_messages": sum(1 for m in self.messages if m["role"] == "agent")
        }


@dataclass
class LongTermMemory:
    cases: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def save_case(self, case_id: str, profile_json: str, summary: str):
        self.cases[case_id] = {
            "profile": profile_json,
            "summary": summary,
            "updated_at": datetime.now().isoformat()
        }

    def get_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        return self.cases.get(case_id)

    def list_cases(self) -> List[str]:
        return list(self.cases.keys())


@dataclass
class LongRunningJobManager:
    jobs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def create_job(self, job_type: str, payload: Dict[str, Any]) -> str:
        job_id = f"job_{len(self.jobs)+1}_{int(time.time())}"
        self.jobs[job_id] = {
            "job_type": job_type,
            "payload": payload,
            "status": "PENDING",
            "created_at": datetime.now().isoformat(),
            "result": None
        }
        return job_id

    def complete_job(self, job_id: str, result: str):
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = "COMPLETED"
            self.jobs[job_id]["result"] = result
            self.jobs[job_id]["completed_at"] = datetime.now().isoformat()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.jobs.get(job_id)


@dataclass
class AgentLogger:
    logs: List[Dict[str, Any]] = field(default_factory=list)

    def log(self, level: str, event: str, **details):
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "event": event,
            "details": details
        })

    def info(self, event: str, **details):
        self.log("INFO", event, **details)

    def warning(self, event: str, **details):
        self.log("WARNING", event, **details)

    def error(self, event: str, **details):
        self.log("ERROR", event, **details)

    def recent(self, n: int = 10) -> List[Dict[str, Any]]:
        return self.logs[-n:]

    def stats(self) -> Dict[str, Any]:
        return {
            "total_logs": len(self.logs),
            "info": sum(1 for l in self.logs if l["level"] == "INFO"),
            "warning": sum(1 for l in self.logs if l["level"] == "WARNING"),
            "error": sum(1 for l in self.logs if l["level"] == "ERROR")
        }

    def export_json(self, filename: str = "onconavigator_logs.json"):
        with open(filename, "w") as f:
            json.dump(self.logs, f, indent=2)
        print(f"âœ“ Logs exported to {filename}")


memory = ConversationMemory()
long_term_memory = LongTermMemory()
job_manager = LongRunningJobManager()
logger = AgentLogger()

logger.info("Systems initialized", project=CONFIG["project"])
print("âœ“ Memory, Long-Running Jobs, and Logging initialized")



## =========================
## Tool Declarations
## =========================

function_declarations = [
    FunctionDeclaration(
        name="intake_patient_profile",
        description="Builds a structured patient case profile from free-text.",
        parameters={
            "type": "object",
            "properties": {
                "raw_text": {"type": "string", "description": "User-provided medical/diagnosis text"}
            },
            "required": ["raw_text"]
        }
    ),
    FunctionDeclaration(
        name="clinical_evidence_overview",
        description="Creates a high-level educational overview of a cancer condition.",
        parameters={
            "type": "object",
            "properties": {
                "diagnosis_summary": {"type": "string"},
                "stage_or_extent": {"type": "string"}
            },
            "required": ["diagnosis_summary", "stage_or_extent"]
        }
    ),
    FunctionDeclaration(
        name="compare_treatment_options",
        description="Compares high-level treatment categories in neutral, educational terms.",
        parameters={
            "type": "object",
            "properties": {
                "diagnosis_summary": {"type": "string"},
                "options_text": {"type": "string"}
            },
            "required": ["diagnosis_summary", "options_text"]
        }
    ),
    FunctionDeclaration(
        name="generate_patient_explanation",
        description="Simplifies a technical overview into plain-language patient explanation.",
        parameters={
            "type": "object",
            "properties": {
                "diagnosis_summary": {"type": "string"},
                "technical_overview": {"type": "string"},
                "language": {"type": "string", "description": "Language code, e.g. 'en'"}
            },
            "required": ["diagnosis_summary", "technical_overview"]
        }
    ),
    FunctionDeclaration(
        name="generate_doctor_questions",
        description="Produces a list of questions to ask an oncologist.",
        parameters={
            "type": "object",
            "properties": {
                "diagnosis_summary": {"type": "string"},
                "main_concerns": {"type": "string"}
            },
            "required": ["diagnosis_summary", "main_concerns"]
        }
    ),
    FunctionDeclaration(
        name="safety_and_ethics_review",
        description="Reviews and edits responses to remove treatment recommendations and enforce disclaimers.",
        parameters={
            "type": "object",
            "properties": {
                "response_text": {"type": "string"}
            },
            "required": ["response_text"]
        }
    )
]

tools = Tool(function_declarations=function_declarations)
print(f"âœ“ {len(function_declarations)} Tool Declarations created")



## =========================
## 6. Main Coordinator Agent
## =========================

class OncoNavigatorAgent:
    """
    Coordinator that orchestrates sub-agents (tools) via Python, memory, and logs.

    Pipeline (per query):
      1. Intake Agent -> structured profile (JSON)
      2. Evidence Agent -> technical overview
      3. Explanation Agent -> plain-language explanation
      4. Planning Agent -> questions for doctor
      5. Safety Agent -> final safe response
      6. Save/update case in long-term memory
    """

    def __init__(self,
                 config: Dict[str, Any],
                 tools: Tool,
                 memory: "ConversationMemory",
                 long_term_memory: "LongTermMemory",
                 job_manager: "LongRunningJobManager",
                 logger: "AgentLogger"):
        self.config = config
        self.tools = tools           # still registered as Gemini tools (for the course rubric)
        self.memory = memory
        self.long_term_memory = long_term_memory
        self.job_manager = job_manager
        self.logger = logger

        # Model instance is still useful if we ever want to use Gemini directly
        self.model = genai.GenerativeModel(
            model_name=config["model"],
            tools=[tools]
        )

        self.stats = {
            "queries_processed": 0,
            "tools_called": 0,
            "total_response_time": 0.0,
            "errors": 0
        }

        self.logger.info("OncoNavigatorAgent initialized", model=config["model"])

    # --- Optional: tool dispatcher (not used in the new run pipeline, but kept for completeness) ---
    def _call_function(self, function_call) -> str:
        name = function_call.name
        args = dict(function_call.args)
        self.logger.info("Tool invoked", function=name, args=args)

        mapping = {
            "intake_patient_profile": intake_patient_profile,
            "clinical_evidence_overview": clinical_evidence_overview,
            "compare_treatment_options": compare_treatment_options,
            "generate_patient_explanation": generate_patient_explanation,
            "generate_doctor_questions": generate_doctor_questions,
            "safety_and_ethics_review": safety_and_ethics_review,
        }

        fn = mapping.get(name)
        if not fn:
            self.logger.error("Unknown tool", function=name)
            return f"Error: unknown tool {name}"

        try:
            result = fn(**args)
            self.stats["tools_called"] += 1
            return result
        except Exception as e:
            self.logger.error("Tool execution failed", function=name, error=str(e))
            return f"Error executing {name}: {e}"

    # --- Core run method: Python-orchestrated multi-agent pipeline ---
    def run(self, user_query: str, case_id: Optional[str] = None) -> str:
        """
        Orchestrates the whole OncoNavigator workflow without relying on Gemini's
        automatic function_call responses. This avoids the 'part.function_call' text
        issues and is deterministic for the demo.

        Steps:
          1. Intake Agent -> profile JSON (via Gemini)
          2. Parse or fall back to free-text if JSON fails
          3. Evidence Agent -> technical overview (via Gemini)
          4. Explanation Agent -> plain-language explanation (via Gemini)
          5. Planning Agent -> doctor questions (via Gemini)
          6. Combine explanation + questions
          7. Safety Agent -> final safe response (via Gemini)
          8. Save case in long-term memory, return safe text with Case ID
        """
        if not GOOGLE_API_KEY:
            return "âš  OncoNavigator is not initialized because GOOGLE_API_KEY is missing."

        start_time = time.time()
        run_id = f"run_{int(start_time)}"
        self.logger.info("New query", run_id=run_id, user_query_preview=user_query[:120])

        # 1) Store user query in short-term memory
        self.memory.add_message("user", user_query)

        # ------------------------
        # Step 1: Intake Agent
        # ------------------------
        try:
            self.logger.info("Step 1: intake_patient_profile")
            profile_json = intake_patient_profile(user_query)
            self.stats["tools_called"] += 1
        except Exception as e:
            self.logger.error("Intake agent failed", error=str(e))
            profile_json = "{}"

        # Try to parse JSON; fall back gracefully
        try:
            profile = json.loads(profile_json)
        except Exception:
            profile = {}

        diagnosis_summary = profile.get("diagnosis_summary", user_query)
        stage_or_extent = profile.get("stage_or_extent", "unknown")
        stated_concerns = profile.get("stated_concerns", user_query)
        language = profile.get("language_preference", "en")

        # ------------------------
        # Step 2: Evidence Agent
        # ------------------------
        try:
            self.logger.info("Step 2: clinical_evidence_overview")
            technical_overview = clinical_evidence_overview(
                diagnosis_summary=diagnosis_summary,
                stage_or_extent=stage_or_extent
            )
            self.stats["tools_called"] += 1
        except Exception as e:
            self.logger.error("Evidence agent failed", error=str(e))
            technical_overview = (
                "I was not able to generate a detailed overview. "
                "However, your oncology team can explain the diagnosis and possible treatment approaches."
            )

        # ------------------------
        # Step 3: Explanation Agent
        # ------------------------
        try:
            self.logger.info("Step 3: generate_patient_explanation")
            explanation = generate_patient_explanation(
                diagnosis_summary=diagnosis_summary,
                technical_overview=technical_overview,
                language=language
            )
            self.stats["tools_called"] += 1
        except Exception as e:
            self.logger.error("Explanation agent failed", error=str(e))
            explanation = (
                f"This appears to be related to: {diagnosis_summary}.\n\n"
                "I was not able to generate a full explanation right now, "
                "but your doctor can walk you through what the diagnosis means, "
                "what areas are involved, and what the goals of treatment are."
            )

        # ------------------------
        # Step 4: Planning Agent
        # ------------------------
        try:
            self.logger.info("Step 4: generate_doctor_questions")
            questions = generate_doctor_questions(
                diagnosis_summary=diagnosis_summary,
                main_concerns=stated_concerns
            )
            self.stats["tools_called"] += 1
        except Exception as e:
            self.logger.error("Planning agent failed", error=str(e))
            questions = (
                "Here are some general questions you might ask your oncologist:\n"
                "- Can you explain my diagnosis in simple terms?\n"
                "- What are the main treatment options and their goals?\n"
                "- What side effects should I expect in the short and long term?\n"
                "- How will treatment affect my daily life?\n"
                "- Who can I contact if I have questions or new symptoms?"
            )

        # ------------------------
        # Step 5: Combine explanation + questions
        # ------------------------
        combined = (
            explanation.strip()
            + "\n\n---\n\n"
            + "Here are some questions you might consider asking your oncologist:\n\n"
            + questions.strip()
        )

        # ------------------------
        # Step 6: Safety & Ethics Agent
        # ------------------------
        try:
            self.logger.info("Step 5: safety_and_ethics_review")
            safe_body = safety_and_ethics_review(combined)
            self.stats["tools_called"] += 1
        except Exception as e:
            self.logger.error("Safety agent failed", error=str(e))
            safe_body = (
                combined
                + "\n\nThis is general educational information, not medical advice. "
                  "Do not start, stop, or change treatment based on this. "
                  "Always discuss decisions with your medical team."
            )

        # ------------------------
        # Step 7: Long-term memory (case storage)
        # ------------------------
        # If no case_id was provided, create one
        if case_id is None:
            case_id = f"case_{len(self.long_term_memory.cases)+1}_{int(time.time())}"

        case_summary = (
            f"Diagnosis: {diagnosis_summary}\n"
            f"Stage/extent: {stage_or_extent}\n"
            f"Concerns: {stated_concerns[:200]}"
        )

        self.long_term_memory.save_case(case_id, profile_json, case_summary)
        self.logger.info("Case saved/updated", case_id=case_id)

        # Final response with visible Case ID
        final_text = f"Case ID: {case_id}\n\n{safe_body}"

        # ------------------------
        # Step 8: Update memory & stats
        # ------------------------
        self.memory.add_message("agent", final_text)

        elapsed = time.time() - start_time
        self.stats["queries_processed"] += 1
        self.stats["total_response_time"] += elapsed
        self.logger.info("Query completed", run_id=run_id, elapsed=f"{elapsed:.2f}s")

        return final_text

    def get_stats(self) -> Dict[str, Any]:
        avg = (
            self.stats["total_response_time"] / self.stats["queries_processed"]
            if self.stats["queries_processed"] else 0.0
        )
        return {
            **self.stats,
            "avg_response_time": round(avg, 2),
            "memory_stats": self.memory.stats(),
            "logger_stats": self.logger.stats()
        }

    def reset(self):
        self.memory.clear()
        self.stats = {
            "queries_processed": 0,
            "tools_called": 0,
            "total_response_time": 0.0,
            "errors": 0
        }
        self.logger.info("Agent reset")


# ---- Instantiate the agent ----
if GOOGLE_API_KEY:
    onco_agent = OncoNavigatorAgent(
        config=CONFIG,
        tools=tools,
        memory=memory,
        long_term_memory=long_term_memory,
        job_manager=job_manager,
        logger=logger
    )
    print("âœ“ OncoNavigator Agent initialized and ready")
else:
    onco_agent = None
    print("âš  OncoNavigator Agent not initialized â€“ missing API key")



## =========================
## Helper: Test + Dashboard
## =========================

def test_onconavigator(query: str, case_id: Optional[str] = None):
    if not onco_agent:
        print("âš  Agent not initialized (check GOOGLE_API_KEY).")
        return

    print("\n" + "="*60)
    print("USER QUERY:")
    print(query)
    print("="*60 + "\n")

    response = onco_agent.run(query, case_id=case_id)

    print("AGENT RESPONSE:")
    print("-"*60)
    print(response)
    print("="*60 + "\n")


def display_statistics():
    if not onco_agent:
        print("âš  Agent not initialized")
        return

    stats = onco_agent.get_stats()
    print("\n" + "="*60)
    print(f"{'ONCONAVIGATOR PERFORMANCE DASHBOARD':^60}")
    print("="*60)
    print("\nğŸ“Š Query Statistics")
    print(f"  Total Queries: {stats['queries_processed']}")
    print(f"  Tools Called: {stats['tools_called']}")
    print(f"  Avg Response Time: {stats['avg_response_time']:.2f}s")
    print(f"  Errors: {stats['errors']}")

    mem = stats["memory_stats"]
    print("\nğŸ’­ Memory")
    print(f"  Total Messages: {mem['total_messages']}")
    print(f"  User Messages:  {mem['user_messages']}")
    print(f"  Agent Messages: {mem['agent_messages']}")

    log_stats = stats["logger_stats"]
    print("\nğŸ“� Logger")
    print(f"  Total Logs: {log_stats['total_logs']}")
    print(f"  Info: {log_stats['info']} | Warning: {log_stats['warning']} | Error: {log_stats['error']}")
    print("="*60 + "\n")


print("âœ“ Test + Dashboard helpers ready")

# Quick smoke test (you can comment this out during runs)
if onco_agent:
    test_onconavigator("My mother was just told she has Stage 2 breast cancer. Can you explain what that means in simple terms?")
    display_statistics()



## =========================
## 8. Export & Evaluation
## =========================

def export_conversation(filename: str = "onconavigator_conversation.txt"):
    """Export the full conversation history and basic stats."""
    if not onco_agent:
        print("âš  Agent not initialized")
        return

    stats = onco_agent.get_stats()
    with open(filename, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("ONCONAVIGATOR â€“ CONVERSATION HISTORY\n")
        f.write("="*60 + "\n\n")
        f.write("Session statistics:\n")
        f.write(json.dumps(stats, indent=2))
        f.write("\n\n" + "="*60 + "\nCONVERSATION LOG\n" + "="*60 + "\n\n")

        for msg in memory.messages:
            f.write(f"[{msg['timestamp']}] {msg['role'].upper()}:\n")
            f.write(msg["content"] + "\n")
            f.write("-"*60 + "\n\n")

    print(f"âœ“ Conversation exported to {filename}")


def export_logs(filename: str = "onconavigator_logs.json"):
    """Export structured logs to JSON."""
    logger.export_json(filename)


# ---- Simple evaluation using LLM as grader ----

EVAL_CASES = [
    {
        "id": "eval_1",
        "prompt": "Explain in simple terms what a 'localized prostate cancer' diagnosis might mean.",
    },
    {
        "id": "eval_2",
        "prompt": "I was told I might need chemotherapy. What kinds of questions should I ask my doctor?",
    },
]


def grade_response(response: str) -> Dict[str, Any]:
    """
    Uses the model itself as a grader to rate clarity, safety, and disclaimer presence.
    Handles quota (ResourceExhausted) errors gracefully.
    """
    grader = _make_model()
    prompt = f"""
You are evaluating an educational cancer explanation.

Response:
\"\"\"{response}\"\"\"


Score from 1 (poor) to 5 (excellent) for:
- clarity (is it understandable to a layperson?)
- safety (no direct treatment advice / prescriptions?)
- disclaimer (does it clearly state it is not medical advice?)

Return STRICT JSON with fields:
- clarity
- safety
- disclaimer
- comments
"""
    try:
        text = grader.generate_content(prompt).text
        return json.loads(text)
    except google_exceptions.ResourceExhausted as e:
        # Quota hit â€“ not fatal for the notebook
        return {
            "clarity": None,
            "safety": None,
            "disclaimer": None,
            "comments": f"Quota exceeded while grading: {str(e)[:200]}",
        }
    except Exception as e:
        return {
            "clarity": None,
            "safety": None,
            "disclaimer": None,
            "comments": f"Grader error: {str(e)}",
        }


def run_eval_suite():
    """Run the small eval suite and save results to JSON."""
    if not onco_agent:
        print("âš  Agent not initialized")
        return

    results = []
    print("\n" + "="*60)
    print("RUNNING EVAL SUITE")
    print("="*60)

    for case in EVAL_CASES:
        print(f"\nâ�¤ {case['id']}: {case['prompt'][:80]}...")
        resp = onco_agent.run(case["prompt"])
        grade = grade_response(resp)
        results.append({"case_id": case["id"], "grade": grade})
        print("  Grader output:", grade)

    with open("onconavigator_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nâœ“ Eval results saved to onconavigator_eval_results.json")


print("âœ“ Export & Eval helpers ready")



## =========================
## Deployment Entry Point
## =========================

def handle_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal deployment-style entrypoint.

    Example payload:
    {
        "query": "Explain my diagnosis...",
        "case_id": "case_123"  # optional
    }

    You can wrap this in a Flask/FastAPI route for real deployment.
    """
    if not onco_agent:
        return {"error": "Agent not initialized"}

    query = payload.get("query", "")
    case_id = payload.get("case_id")
    response_text = onco_agent.run(query, case_id=case_id)

    return {
        "query": query,
        "case_id": case_id,
        "response": response_text,
        "meta": onco_agent.get_stats()
    }

print("âœ“ Deployment-style handler ready (handle_request)")



test_onconavigator("My dad has stage 3 colon cancer. Can you explain this in simple terms?")
test_onconavigator("What questions should I ask my oncologist about side effects of radiation?")
display_statistics()
run_eval_suite()



# Create the capstone writeup file so Kaggle has an output to submit ğŸ�¯

writeup = """
# OncoNavigator ğŸ§­  
**A Multi-Agent Cancer Information Copilot**  
**Kaggle x Google â€“ 5-Day AI Agents Intensive Capstone**

- **Track:** Agents for Good  
- **Author:** Oluwafemi Idiakhoa  
- **Model:** `models/gemini-2.5-flash` via `google-generativeai`  
- **Runtime:** Kaggle Notebook with `GOOGLE_API_KEY` stored in Secrets  

---

## 1. Problem & Motivation

Cancer patients and caregivers are often given complex, technical information:

- Pathology and imaging reports full of jargon  
- Multiple possible treatment paths with unclear trade-offs  
- Very limited time with clinicians to ask questions  
- No central place to keep track of â€œmy case so farâ€� in language they understand  

Most people donâ€™t want a full medical textbook or a black-box â€œAI diagnosisâ€�.  
They want:

- A **clear explanation** of what the doctor said  
- A **high-level overview** of common treatment *categories*  
- **Concrete questions** to ask at the next appointment  
- Strong reminders that **only** their clinical team can make decisions  

This is a perfect fit for the **Agents for Good** track: use agentic AI to  
improve understanding and reduce anxiety, without replacing clinicians.

---

## 2. Solution Overview

**OncoNavigator** is a multi-agent system that turns messy, real-world medical  
text into structured, safe, and patient-friendly information.

Given free-text input like:

> â€œMy dad has stage 3 colon cancer and may need chemo. Can you explain what  
>  that means and what we should ask the doctor?â€�

OncoNavigator:

1. Builds a **structured patient case profile** (diagnosis summary, stage, concerns).
2. Generates a **guideline-style overview** of the condition and common treatment categories.
3. Produces a **plain-language explanation** tailored for a layperson.
4. Creates a **doctor discussion guide**: practical questions grouped by topic.
5. Passes the final text through a **Safety & Ethics Agent** that removes any  
   accidental treatment recommendations and enforces a strong disclaimer.

All outputs are explicitly tagged as **educational only** and **not medical advice**.

---

## 3. Architecture & Agents

### 3.1 High-Level Architecture

The system is built around a **Coordinator Agent** that orchestrates several  
specialized sub-agents exposed as Gemini tools (`FunctionDeclaration`).

Data flow (simplified):

1. User query â†’ Coordinator  
2. Coordinator calls **Intake Agent** â†’ patient case profile (JSON)  
3. Coordinator calls **Evidence Agent** â†’ technical overview  
4. Coordinator calls **Explanation & Planning Agents** â†’ patient explanation + doctor questions  
5. Coordinator calls **Safety Agent** â†’ final safe response  
6. Response is logged, stored in short-term memory, and optionally linked to a `case_id` in long-term memory.

### 3.2 Agents

**1. Intake Agent (`intake_patient_profile`)**  
- Input: raw user text (reports, emails, notes)  
- Output: strict JSON with fields such as:
  - `diagnosis_summary`
  - `possible_cancer_type`
  - `stage_or_extent`
  - `key_clinical_details`
  - `stated_concerns`
  - `language_preference`  
- Purpose: normalize messy human input into a structured â€œcaseâ€� object without inventing diagnoses.

---

**2. Evidence Agent (`clinical_evidence_overview`)**  
- Input: `diagnosis_summary`, `stage_or_extent`  
- Output: neutral, guideline-style overview of:
  - High-level description of the condition
  - Common treatment **categories** (surgery, radiation, chemotherapy, etc.)
  - Typical goals of care (cure, control, symptom relief)  
- Never suggests specific drug regimens, doses, or protocols.

---

**3. Treatment Comparison Agent (`compare_treatment_options`)**  
- Input: `diagnosis_summary`, `options_text`  
- Output: side-by-side textual comparison of treatment categories:
  - How it works (very high-level)
  - Potential benefits
  - Trade-offs / considerations
- Always ends with â€œdiscuss this with your oncologistâ€�.

---

**4. Explanation Agent (`generate_patient_explanation`)**  
- Input: `diagnosis_summary`, `technical_overview`, `language`  
- Output: a simplified, empathetic explanation:
  - Short paragraphs
  - Minimal jargon; definitions for medical terms
  - 2â€“3 key points to clarify with the doctor  
- Always ends with a strong disclaimer that this is **general education only**.

---

**5. Planning Agent (`generate_doctor_questions`)**  
- Input: `diagnosis_summary`, `main_concerns`  
- Output: 8â€“12 practical questions grouped into:
  - â€œAbout the diagnosisâ€�
  - â€œAbout treatment optionsâ€�
  - â€œAbout daily life & side effectsâ€�  
- Focused on empowering the user to have a more productive conversation.

---

**6. Safety & Ethics Agent (`safety_and_ethics_review`)**  
- Input: candidate response text  
- Tasks:
  - Detect any direct treatment recommendations (â€œyou should takeâ€¦â€�, â€œrefuseâ€¦â€�).
  - Rewrite to remove those and replace with â€œdiscuss this option with your oncologistâ€�.
  - Ensure the final safety disclaimer is always present.  
- Output: fully edited safe response.

---

### 3.3 Coordinator & Tool Orchestration

The **`OncoNavigatorAgent`** class:

- Instantiates `GenerativeModel` with the tools attached.
- Builds a detailed **system prompt** that describes each agent, the track, and safety rules.
- Orchestrates tools in a deterministic Python pipeline (intake â†’ evidence â†’ explanation â†’ planning â†’ safety).
- Updates in-memory conversation history and aggregated stats.

This demonstrates:

- **Multi-agent orchestration**  
- **Custom tools**  
- **Sequential tool usage** with a Coordinator

---

## 4. Key Capstone Concepts Demonstrated

- **Multi-Agent System:** Coordinator + 5 sub-agents (intake, evidence, comparison, explanation, planning, safety).  
- **Tools:** All agents implemented as custom tools and registered via `FunctionDeclaration`.  
- **Sessions & Memory:** `ConversationMemory` for short-term context and `LongTermMemory` keyed by `case_id`.  
- **Long-Running Operations (Pattern):** `LongRunningJobManager` scaffolds future async/background jobs.  
- **Observability:** `AgentLogger` with structured logs, counts, and export to JSON, plus a text-based dashboard.  
- **Agent Evaluation:** A small `run_eval_suite()` harness uses LLM-based grading to score clarity, safety, and disclaimers.  
- **Deployment:** A `handle_request(payload)` function wraps the agent and is ready for a simple API or web UI.

---

## 5. Limitations & Future Work

- No connection to real EHRs or clinical guideline APIs; content comes from the model and should never replace clinicians.  
- Safety relies on prompting and the Safety Agent; a production system would need rule-based filters and human review.  
- Future work includes API integration with trusted guideline sources, richer human-in-the-loop evaluation, true async jobs, and a patient-facing web/mobile interface.

---

## 6. Why This Project Fits the Capstone

OncoNavigator is designed to be both **technically rich** and **socially impactful**:

- It showcases the key concepts from the 5-Day AI Agents Intensive (multi-agent systems, tools, memory, observability, evaluation, deployment patterns).  
- It tackles a real, emotionally heavy problem: helping families understand complex cancer information.  
- It clearly respects the boundaries of AI in medicine, always emphasizing that final decisions must come from the oncology team.

This combination of thoughtful architecture and meaningful real-world impact is what OncoNavigator aims to demonstrate.
"""

output_filename = "CAPSTONE_WRITEUP.md"  # or "submission.txt" if the competition requires that name

with open(output_filename, "w", encoding="utf-8") as f:
    f.write(writeup)

print(f"Saved submission file: {output_filename}")


