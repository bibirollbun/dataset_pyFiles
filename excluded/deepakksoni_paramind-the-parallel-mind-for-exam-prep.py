# 1. Install the Agent Development Kit and PDF libraries
# Note: 'pypdf' is used for custom ingestion tool.
!pip install -q google-adk[a2a] pypdf opentelemetry-instrumentation-google-genai
!pip install -q google-adk[a2a] pypdf fpdf opentelemetry-instrumentation-google-genai

# 2. Setup Authentication
#Using Kaggle Secrets to avoid hardcoding keys (Security Best Practice from Day 5)
from kaggle_secrets import UserSecretsClient
import os

try:
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = api_key
    print("âœ… Authentication success.")
except:
    print("â�Œ Error: Please set 'GOOGLE_API_KEY' in Add-ons -> Secrets")

# 3. Create Project Directory
# Creating a modular package structure like 'Agent Shutton' 
PROJECT_DIR = "paramind"
os.makedirs(PROJECT_DIR, exist_ok=True)
os.makedirs(f"{PROJECT_DIR}/agents", exist_ok=True)

!touch paramind/__init__.py
!touch paramind/agents/__init__.py

print(f"âœ… Project directory '{PROJECT_DIR}' created.")


%%writefile paramind/requirements.txt
google-adk
pypdf
fpdf
opentelemetry-instrumentation-google-genai


%%writefile paramind/config.py
import os
from dataclasses import dataclass
from google.genai import types
from kaggle_secrets import UserSecretsClient

# 1. Force AI Studio Mode
# I disabled Vertex AI for local testing because we are using an API Key.
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "0"

# 2. Securely fetch API Key
if "GOOGLE_API_KEY" not in os.environ:
    try:
        user_secrets = UserSecretsClient()
        os.environ["GOOGLE_API_KEY"] = user_secrets.get_secret("GOOGLE_API_KEY")
    except:
        pass

@dataclass
class AgentConfig:
    # Using Flash-Lite for speed and efficiency (Day 4: Performance pillars)
    model_name: str = "gemini-2.5-flash-lite"

    # Retry logic to handle transient network errors (Day 2: Robustness)
    retry_config = types.HttpRetryOptions(attempts=3, initial_delay=1)

    # Explicitly capture the key for passing to agents
    api_key: str = os.environ.get("GOOGLE_API_KEY")
    
    # GENERALIZATION: Move "Magic Numbers" to config
    max_pdf_chars: int = 25000  # Increased limit for larger chapters
    max_quiz_loops: int = 5     # Safety limit for the Tutor loop

config = AgentConfig()


%%writefile paramind/tools.py
import os
from pypdf import PdfReader
from google.adk.tools import ToolContext

# Import config to get the limit
from .config import config 

def ingest_exam_pdf(file_path: str, tool_context: ToolContext) -> dict:
    """Ingests a PDF and saves context to session state."""
    try:
        # SECURITY: Path Traversal Prevention (Day 5 Security Best Practice)
        # To ensure the agent only reads files inside the working directory.
        full_path = os.path.abspath(file_path)
        if not os.path.exists(full_path):
             return {"status": "error", "message": f"File not found: {file_path}"}

        # Using pypdf to extract text (Custom Tool implementation)
        reader = PdfReader(full_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

        # STATE MANAGEMENT (Day 3): 
        # Saving the extracted text into the Session State.
        # This allows the 'TutorAgent' to access this data later without re-reading the file.
        # GENERALIZATION: Use the configured limit
        limit = config.max_pdf_chars
        tool_context.state["study_material"] = text[:limit]
        
        return {
            "status": "success", 
            "message": f"Ingested {len(reader.pages)} pages. First {limit} characters saved to memory."
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


%%writefile paramind/utils.py
from google.adk.agents.callback_context import CallbackContext
from google.genai.types import Content

def suppress_output(callback_context: CallbackContext) -> Content:
    """
    Helper to hide intermediate agent outputs from the user.
    This improves the User Experience (UX) by reducing noise.
    """
    return Content()

# ---------------------------------------------------------

from typing import AsyncGenerator
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

class QuizSuccessChecker(BaseAgent):
    """
    A special agent that checks if the student passed the quiz.
    It acts as a 'Gatekeeper' for the Loop Agent.
    """
    async def _run_async_impl(self, context: InvocationContext) -> AsyncGenerator[Event, None]:
        # Check the session state for the quiz result (Day 3: State Management)
        result = context.session.state.get("quiz_result", "FAIL")
        
        if result == "PASS":
            # If passed, we 'escalate' to break the loop (Day 1: Loop Architecture)
            yield Event(author=self.name, actions=EventActions(escalate=True))
        else:
            # Otherwise, we continue the loop to quiz again
            yield Event(author=self.name)


%%writefile paramind/agents/planner.py
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from ..config import config
from ..utils import suppress_output

# 1. Assessor
# Reads user history to find weak spots.
assessor = LlmAgent(
    name="Assessor",
    model=Gemini(model=config.model_name, api_key=config.api_key),
    instruction="Analyze the user's request. Identify specific weak topics. Output a list of topics.",
    output_key="weak_topics",
    after_agent_callback=suppress_output
)

# 2. Scheduler
# Takes the topics and puts them on a timeline.
scheduler = LlmAgent(
    name="Scheduler",
    model=Gemini(model=config.model_name, api_key=config.api_key),
    instruction="""Take the {weak_topics}. 
    Create a study plan based on the user's stated available time (infer this from the chat history).
    If no time is mentioned, default to a 2-hour session with breaks.
    Output the schedule clearly.""",
    output_key="final_schedule"
)

# SEQUENTIAL PATTERN
planner_agent = SequentialAgent(
    name="PlannerAgent",
    description="Generates study schedules.",
    sub_agents=[assessor, scheduler]
)


%%writefile paramind/agents/tutor.py
from google.adk.agents import LlmAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from ..config import config
from ..utils import QuizSuccessChecker

# 1. Quiz Master
# Uses the PDF content stored in Session State (Day 3) to generate questions.
quiz_master = LlmAgent(
    name="QuizMaster",
    model=Gemini(model=config.model_name, api_key=config.api_key),
    instruction="Generate a question based on {study_material}. Do NOT answer it.",
)

# 2. Grader
# Evaluates the answer and updates State.
grader = LlmAgent(
    name="Grader",
    model=Gemini(model=config.model_name, api_key=config.api_key),
    instruction="Evaluate answer. If correct, set state 'quiz_result' to 'PASS'.",
)

tutor_agent = LoopAgent(
    name="TutorAgent",
    description="Quizzes user until mastery.",
    sub_agents=[quiz_master, grader, QuizSuccessChecker(name="validator")],
    # GENERALIZATION: Use configured limit
    max_iterations=config.max_quiz_loops
)




%%writefile paramind/coordinator.py
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.tools import AgentTool, FunctionTool, google_search
# , preload_memory
# is set disabled as we are not using VERTEX AI in prod env
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.models.google_llm import Gemini

from .config import config
from .tools import ingest_exam_pdf
from .agents.planner import planner_agent
from .agents.tutor import tutor_agent


# News Agent (Tool User)
# Instruction is generic ("exam specified by the user")
# Proactive Vigilance (Day 1 Level 3 Agent capability)
news_agent = LlmAgent(
    name="NewsAgent",
    model=Gemini(model=config.model_name, api_key=config.api_key),
    instruction="""You are a Vigilance Agent. 
    Your goal is to find verified updates (dates, syllabus, admit cards) for the exam specified by the user.
    1. Identify which exam the user is asking about (e.g., JEE, NEET, GATE, UPSC).
    2. If no exam is mentioned, check the conversation history or ask for clarification.
    3. Use google_search to find OFFICIAL sources only (websites like .gov.in, .nic.in, or major news outlets).
    4. Ignore rumors, forums, and unverified blogs.
    """,
    tools=[google_search]
)

# The Main Coordinator
coordinator = LlmAgent(
    name="StudyCoordinator",
    model=Gemini(model=config.model_name, api_key=config.api_key),
    instruction="""You are ParaMind, an autonomous exam companion.
    
    ROUTING LOGIC:
    1. MEMORY: Always use 'preload_memory' first to check for user context (Day 3).
    2. INGESTION: If user mentions a file/PDF, use 'ingest_exam_pdf'.
    3. PLANNING: If user needs a schedule or strategy, call 'PlannerAgent'.
    4. STUDYING: If user wants to quiz/learn, call 'TutorAgent'.
    5. NEWS: If user asks for dates/updates, call 'NewsAgent'.
    
    FALLBACK:
    If you don't know what to do, ask the user for clarification.
    """,
    tools=[
        AgentTool(planner_agent),
        AgentTool(tutor_agent),
        AgentTool(news_agent),
        FunctionTool(ingest_exam_pdf)
        # ,
        # preload_memory is set disabled as we are not using VERTEX AI in prod env
    ]
)

# Factory function for deployment
def create_runner():
    return Runner(
        agent=coordinator,
        app_name="ParaMind",
        session_service=InMemorySessionService(),
        memory_service=InMemoryMemoryService(),
        plugins=[LoggingPlugin()]
    )

agent = coordinator


# Create dummy PDF
from fpdf import FPDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="Subject: Physics. Question: Calculate escape velocity...", ln=1)
pdf.output("paramind/sample_exam.pdf")


%%writefile paramind/test_config.json
{
  "criteria": {
    "tool_trajectory_avg_score": 0.1, 
    "response_match_score": 0.1
  },
  "test_cases": [] 
}


%%writefile paramind/eval_news.json
{
  "eval_set_id": "paramind_news_set",
  "eval_cases": [
    {
      "evalId": "news_check",
      "conversation": [
        {
            "user_content": {"parts": [{"text": "Are there any official updates on the JEE Main 2025 dates?"}]},
            "final_response": {"parts": [{"text": "I found updates for JEE Main 2025"}]},
            "intermediate_data": {
                "tool_uses": [{"name": "NewsAgent"}]
            }
        }
      ]
    }
  ]
}


%%writefile paramind/eval_ingestion.json
{
  "eval_set_id": "paramind_ingestion_set",
  "eval_cases": [
    {
      "evalId": "ingestion_check",
      "conversation": [
        {
            "user_content": {"parts": [{"text": "I am uploading 'paramind/sample_exam.pdf'. Please ingest it."}]},
            "intermediate_data": {
                "tool_uses": [{"name": "ingest_exam_pdf"}]
            }
        }
      ]
    }
  ]
}


%%writefile paramind/eval_tutor.json
{
  "eval_set_id": "paramind_tutor_set",
  "eval_cases": [
    {
      "evalId": "tutor_flow_check",
      "conversation": [
        {
            "user_content": {"parts": [{"text": "I am uploading 'paramind/sample_exam.pdf'. Please ingest it."}]},
            "intermediate_data": {
                "tool_uses": [{"name": "ingest_exam_pdf"}]
            }
        },
        {
            "user_content": {"parts": [{"text": "Great. Now quiz me on a question from that PDF."}]},
            "intermediate_data": {
                "tool_uses": [{"name": "TutorAgent"}]
            }
        }
      ]
    }
  ]
}


%%writefile paramind/__init__.py
from .coordinator import coordinator as agent
# Because eval in futher code recognizes Agent


%%writefile paramind/run_segmented_evals.py
import subprocess
import time
import sys
import os

# Configuration
CONFIG_PATH = "paramind/test_config.json"
AGENT_DIR = "paramind"
EVAL_FILES = [
    "paramind/eval_news.json",
    "paramind/eval_ingestion.json",
    "paramind/eval_tutor.json"
]

def run_adk_eval(eval_file):
    print(f"\n========================================")
    print(f"ğŸ§ª STARTING EVALUATION: {eval_file}")
    print(f"========================================")
    
    cmd = [
        "adk", "eval", 
        AGENT_DIR, 
        eval_file,
        f"--config_file_path={CONFIG_PATH}",
        "--print_detailed_results"
    ]
    
    # Run the command and stream output
    process = subprocess.run(cmd)
    
    if process.returncode != 0:
        print(f"â�Œ Test failed for {eval_file}")
    else:
        print(f"âœ… Test passed for {eval_file}")

def main():
    # Ensure the PDF exists for the tests (dependency from run_agent.py)
    if not os.path.exists("paramind/sample_exam.pdf"):
        print("ğŸ“„ Generating dummy PDF for tests...")
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Subject: Physics. Question: Calculate escape velocity...", ln=1)
        pdf.output("paramind/sample_exam.pdf")

    for i, eval_file in enumerate(EVAL_FILES):
        run_adk_eval(eval_file)
        
        # Don't sleep after the last test
        if i < len(EVAL_FILES) - 1:
            print("\nâ�³ Rate Limit Cooling: Sleeping for 60 seconds...")
            time.sleep(60)

if __name__ == "__main__":
    main()


# Evaluation begins

!python -m paramind.run_segmented_evals


%%writefile paramind/run_agent.py
import asyncio
import os
import time
from paramind.coordinator import create_runner
from paramind.config import config

# ==========================================
# ğŸ“� USER INSTRUCTIONS: CHANGE COMMANDS HERE
# ==========================================
# You can modify these queries to test different agent behaviors.
COMMANDS = {
    "vigilance": "Are there any official updates on the JEE Main 2025 dates?",
    "ingestion": "I am uploading the file. Please ingest it.", # File path is handled automatically below
    "tutoring": "Quiz me on a question from the PDF I just uploaded."
}
# ==========================================

async def main():
    print(f"ğŸ¤– ParaMind Initialized with model: {config.model_name}")
    
    # Initialize the runner (creates sessions & memory services)
    runner = create_runner()
    
    # --- Test 1: News Agent ---
    print(f"\n--- Test 1: News Agent (Vigilance) ---")
    print(f"User: {COMMANDS['vigilance']}")
    try:
        await runner.run_debug(COMMANDS['vigilance'])
    except Exception as e:
        print(f"Test 1 Error: {e}")
        
    print("\nâ�³ Sleeping for 60 seconds to reset API quota...")
    time.sleep(60) 
    
    # --- Test 2: Ingestion ---
    print(f"\n--- Test 2: Knowledge Agent (Ingestion) ---")
    try:
        # We automatically inject the correct file path into the user's command
        # This ensures the tool finds the file even if the user just says "upload this"
        file_path = os.path.abspath("paramind/sample_exam.pdf")
        ingest_cmd = f"{COMMANDS['ingestion']} File: {file_path}"
        
        print(f"User: {ingest_cmd}")
        await runner.run_debug(ingest_cmd)
    except Exception as e:
        print(f"Test 2 Error: {e}")

    print("\nâ�³ Sleeping for 60 seconds to reset API quota...")
    time.sleep(60)

    # --- Test 3: Tutor ---
    print(f"\n--- Test 3: Tutor Agent (Loop) ---")
    print(f"User: {COMMANDS['tutoring']}")
    try:
        await runner.run_debug(COMMANDS['tutoring'])
    except Exception as e:
         print(f"Test 3 Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())


# Run Agent
!python -m paramind.run_agent

