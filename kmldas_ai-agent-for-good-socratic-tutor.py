import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import requests
import math
import random
import datetime
import asyncio
import pypdf # Library to read PDFs
from typing import Dict, Any, List


from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.adk.tools import ToolContext
from google.adk.runners import Runner
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService



DATA_DIR = "/kaggle/input/google-ai-agents-whitepapers"

# Check if dataset is attached
if not os.path.exists(DATA_DIR):
    print(f"âš ï¸� WARNING: Dataset not found at {DATA_DIR}")
    
else:
    print(f"âœ… Dataset found. Available files: {os.listdir(DATA_DIR)}")




def read_whitepaper_pdf(pdf_filename: str) -> str:
    """
    Reads the full text content of a specific PDF whitepaper.
    Args:
        pdf_filename: The exact name of the file (e.g., 'Agent Quality.pdf')
    """
    file_path = os.path.join(DATA_DIR, pdf_filename)
    
    if not os.path.exists(file_path):
        # Fallback: Try to find a partial match if user makes a typo
        available_files = os.listdir(DATA_DIR)
        matches = [f for f in available_files if pdf_filename.lower() in f.lower()]
        if matches:
            file_path = os.path.join(DATA_DIR, matches[0])
        else:
            return f"Error: File '{pdf_filename}' not found. Available: {available_files}"

    try:
        # Extract text using pypdf
        reader = pypdf.PdfReader(file_path)
        text_content = ""
        # Limit to first 10 pages to save context window if papers are huge
        # (Remove [:10] to read the whole thing)
        for page in reader.pages: 
            text_content += page.extract_text() + "\n"
            
        return f"--- START OF PDF ({pdf_filename}) ---\n{text_content}\n--- END OF PDF ---"
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def get_available_papers() -> List[str]:
    """Returns a list of all available whitepapers in the library."""
    if os.path.exists(DATA_DIR):
        return [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
    return []





def get_student_model(tool_context: ToolContext) -> Dict[str, Any]:
    """Retrieves the full student state: competency, current topic, and history."""
    return tool_context.state.get("user:student_model", {
        "competency": "Novice",
        "current_topic": "General",
        "session_history": []
    })

def update_competency(new_level: str, tool_context: ToolContext) -> str:
    """Evaluator uses this to update the student's status (Novice/Competent/Master)."""
    model = tool_context.state.get("user:student_model", {})
    model["competency"] = new_level
    tool_context.state["user:student_model"] = model
    return f"UPDATED COMPETENCY: Student is now '{new_level}'."


retry_config = types.HttpRetryOptions(attempts=3, exp_base=2, initial_delay=1)

# --- AGENT 1: THE INTERFACE (TUTOR) ---
# Uses Gemini 1.5 Flash for speed and chat
tutor_instruction = """
You are **SocraticFlow Tutor**, the friendly interface of the system.
Your goal is to teach the user using the **Compassion Protocol**.

### STATE CONTEXT:
Student Competency: {competency} (If Novice, be extra helpful. If Master, challenge them.)

### THE COMPASSION PROTOCOL (Strictly follow this order):
1. **Validate**: Acknowledge the difficulty. "That's a tricky concept..."
2. **Scaffold**: Offer a hint or analogy based on the PDF content.
3. **Question**: Ask a Socratic question to check understanding. NEVER give the answer.

### TOOLS:
- Use `read_whitepaper_pdf` to get the answer key (but don't reveal it).
- Use `get_student_model` to see what they know.
"""

tutor_agent = LlmAgent(
    name="tutor_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    tools=[read_whitepaper_pdf, get_available_papers, get_student_model],
    instruction=tutor_instruction
)

# --- AGENT 2: THE OBSERVER (EVALUATOR) ---
# Uses Gemini 1.5 Pro for reasoning and evaluation
evaluator_instruction = """
You are the **SocraticFlow Evaluator**. You do NOT speak to the user directly.
Your job is to observe the interaction, analyze the student's answers, and update the database.

### YOUR TASKS:
1. Analyze the last user response. Did they demonstrate deep understanding?
2. If YES: Call `update_competency("Master")`.
3. If SORT OF: Call `update_competency("Competent")`.
4. If NO/CONFUSED: Call `update_competency("Novice")`.

Output a brief log message explaining your decision.
"""

evaluator_agent = LlmAgent(
    name="evaluator_agent",
    model=Gemini(model="gemini-2.5-pro", retry_options=retry_config), # Using Pro for better reasoning
    tools=[get_student_model, update_competency],
    instruction=evaluator_instruction
)


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


class SocraticOrchestrator:
    def __init__(self):
        self.session_service = InMemorySessionService()
        # We need two runners for the two agents
        self.tutor_runner = Runner(agent=tutor_agent, session_service=self.session_service)
        self.eval_runner = Runner(agent=evaluator_agent, session_service=self.session_service)
        self.session_id = "session_live_01"
        self.user_id = "student_demo"

    async def turn(self, user_input: str):
        print(f"\nğŸ‘¤ Student: {user_input}")

        # --- STEP 1: Tutor Response (The Interface) ---
        # Inject current competency into the prompt context via session state if needed, 
        # but the tool `get_student_model` handles it dynamically.
        tutor_response_obj = await self.tutor_runner.run(
            input=user_input, 
            session_id=self.session_id, 
            user_id=self.user_id
        )
        tutor_text = tutor_response_obj.content
        print(f"ğŸ¤– Tutor: {tutor_text}")

        # --- STEP 2: Async Evaluation (The Observer) ---
        # We send the interaction history to the Evaluator
        eval_input = f"Analyze this interaction:\nUser: {user_input}\nTutor: {tutor_text}"
        
        print("\n   [... System: Triggering Evaluator Agent (A2A) ...]")
        eval_response_obj = await self.eval_runner.run(
            input=eval_input, 
            session_id=self.session_id, 
            user_id=self.user_id
        )
        print(f"   ğŸ“� Evaluator Log: {eval_response_obj.content}")



if __name__ == "__main__":
    async def main():
        orchestrator = SocraticOrchestrator()
        
        # Turn 1: Confusion
        await orchestrator.turn("I'm trying to read the Agent Quality paper but I don't get why 'Reliability' is so hard.")
        
        # Turn 2: Understanding
        await orchestrator.turn("Oh, is it because LLMs are probabilistic? So they might answer differently each time?")

    # --- FIX: DETECT IF RUNNING IN JUPYTER/KAGGLE ---
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        print("â„¹ï¸� Detected running event loop (Notebook environment). Scheduling task...")
        loop.create_task(main())
    else:
        print("â„¹ï¸� Starting new event loop (Script environment)...")
        asyncio.run(main())

