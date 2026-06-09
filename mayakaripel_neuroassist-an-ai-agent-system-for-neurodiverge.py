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


%%writefile requirements.txt
streamlit
tinydb
pandas
pyngrok
# To run with real LLMs locally, also add:
# openai
# google-generativeai


!pip install -r requirements.txt -q


%%writefile agents_impl.py

# agents_impl.py
"""
Integrated multi-agent implementation for the NeuroAssist project.

This file contains the core logic for all specialized agents, including:
- LLM Abstractions: A system to auto-detect the environment (Kaggle vs. Local)
  and switch between a SafeDummyLLM and real LLMs (OpenAI/Gemini).
- SafetyAgent: Monitors for user distress and notifies caregivers.
- EmotionAgent: Infers the user's emotional state from their pace.
- AIReasoningAgent: Uses an LLM to simplify instructions and provide support.
- RoutineAgent: Executes the step-by-step routines.
- OrchestratorAgent: Coordinates all agents to run the system.
"""

import os
import json
import time
import random
from datetime import datetime

# tinydb is a lightweight database, used here for logging routine data.
try:
    from tinydb import TinyDB
except ImportError:
    TinyDB = None

# Create data folder and initialize TinyDB if available
os.makedirs("data", exist_ok=True)
DB = TinyDB("data/db.json") if TinyDB else None

# -----------------------
# LLM Abstractions
# -----------------------
class LLMBase:
    """Abstract base class for Large Language Model interactions."""
    def ask(self, prompt: str) -> str:
        raise NotImplementedError

class SafeDummyLLM(LLMBase):
    """
    A Kaggle-safe, offline LLM simulator.
    Provides plausible, non-API-based responses for public notebook execution.
    """
    def ask(self, prompt: str) -> str:
        # Use simple heuristics to generate safe, plausible responses.
        prompt_lower = prompt.lower()
        if "simplify" in prompt_lower:
            # Return a shortened version of the last line plus a calming message.
            out = prompt.splitlines()[-1].strip() if "\n" in prompt else prompt
            short = out[:77].rstrip() + "..." if len(out) > 80 else out
            calm = ["Take your time.", "You can do this.", "One step at a time."]
            return f"{short} â€” {random.choice(calm)}"
        if "calm" in prompt_lower:
            return "Try 3 calm breaths: in... hold... out. Slow and steady."
        if "caregiver summary" in prompt_lower:
            return "Routine executed. Minor hesitations observed. Recommend brief encouragement."
        return "ğŸ¤– (Simulated LLM response) â€” This is a placeholder."

class OpenAILLM(LLMBase):
    """Wrapper for the OpenAI API (for local use)."""
    # NOTE: This will only work if you have the 'openai' package installed
    # and the OPENAI_API_KEY environment variable set.
    def __init__(self, model="gpt-4o-mini"):
        try:
            import openai
        except ImportError as e:
            raise RuntimeError("openai package not found. `pip install openai` to use OpenAILLM.") from e
        
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        
        self.openai = openai
        self.model = os.environ.get("OPENAI_MODEL", model)

    def ask(self, prompt: str) -> str:
        response = self.openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()

class GeminiLLM(LLMBase):
    """Wrapper for the Google Gemini API (for local use)."""
    # NOTE: This will only work if you have the 'google-generativeai' package installed
    # and the GENIE_API_KEY environment variable set.
    def __init__(self, model="gemini-1.5-flash"):
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise RuntimeError("google.generativeai package not found. `pip install google-generativeai` to use GeminiLLM.") from e

        api_key = os.environ.get("GENIE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GENIE_API_KEY or GEMINI_API_KEY environment variable not set.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(os.environ.get("GEMINI_MODEL", model))

    def ask(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text.strip()

def load_llm():
    """
    Auto-detects the execution environment and loads the appropriate LLM.
    - On Kaggle, it defaults to the SafeDummyLLM to avoid API key errors.
    - Locally, it checks for OpenAI or Gemini API keys and loads the real LLM.
    """
    running_on_kaggle = os.path.exists("/kaggle") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
    preferred = os.environ.get("PREFERRED_LLM", "").lower()

    if running_on_kaggle:
        print("âœ… Kaggle environment detected â€” using SafeDummyLLM (no external API calls).")
        return SafeDummyLLM()

    # Local environment: try to load real LLMs based on preference or availability.
    if preferred == "openai" and "OPENAI_API_KEY" in os.environ:
        try: return OpenAILLM()
        except Exception as e: print(f"Could not load preferred OpenAI LLM: {e}")
    
    if preferred == "gemini" and ("GENIE_API_KEY" in os.environ or "GEMINI_API_KEY" in os.environ):
        try: return GeminiLLM()
        except Exception as e: print(f"Could not load preferred Gemini LLM: {e}")

    # If no preference, try any available LLM
    if "OPENAI_API_KEY" in os.environ:
        try: return OpenAILLM()
        except Exception as e: print(f"OpenAI LLM init failed: {e}")
    
    if "GENIE_API_KEY" in os.environ or "GEMINI_API_KEY" in os.environ:
        try: return GeminiLLM()
        except Exception as e: print(f"Gemini LLM init failed: {e}")
    
    print("âš ï¸� No LLM credentials found â€” falling back to SafeDummyLLM.")
    return SafeDummyLLM()

# -----------------------
# Agents
# -----------------------
class SafetyAgent:
    """Monitors user safety and well-being during routines."""
    def __init__(self, caregiver_notify_fn=None):
        self.caregiver_notify_fn = caregiver_notify_fn

    def detect_meltdown(self, event: str) -> bool:
        """Detects keywords indicating user distress."""
        distress_keywords = ["crying", "screaming", "refusing", "panic", "overwhelmed"]
        return any(k in event.lower() for k in distress_keywords)

    def notify_step_stuck(self, user_profile, step, actual_duration):
        """Notifies caregiver if user is stuck on a step for too long."""
        contact = user_profile.get("caregiver_contact")
        msg = f"User {user_profile.get('name')} may be stuck on step '{step.get('title')}' (took {int(actual_duration)}s)."
        if self.caregiver_notify_fn:
            self.caregiver_notify_fn(contact, msg)
        else:
            print(f"[SafetyAgent Notification] {msg}")

class EmotionAgent:
    """Infers the user's emotional state based on routine performance."""
    def interpret(self, routine_log: dict) -> dict:
        """Analyzes step timings to detect hesitation or rushing."""
        steps = routine_log.get('steps', [])
        slow_steps = 0
        fast_steps = 0
        for s in steps:
            planned = s.get('planned_duration', 1)
            actual = s.get('actual_duration', planned)
            if actual > planned * 1.8:  # Took 80% longer than planned
                slow_steps += 1
            if actual < planned * 0.6:  # Finished 40% faster than planned
                fast_steps += 1
        
        if slow_steps >= 1 and fast_steps == 0:
            return {"state": "hesitant", "confidence": 0.8, "advice": "Offer a calm break (e.g., 2 deep breaths)."}
        if fast_steps >= 1 and slow_steps == 0:
            return {"state": "rushed", "confidence": 0.7, "advice": "Encourage a slower, steady pace."}
        return {"state": "ok", "confidence": 0.9, "advice": "Good job! Routine followed at a steady pace."}

class AIReasoningAgent:
    """Handles LLM-based tasks like simplification and summarization."""
    def __init__(self, llm: LLMBase = None):
        self.llm = llm or load_llm()

    def simplify_instruction(self, instruction: str) -> str:
        """Uses the LLM to simplify a routine instruction."""
        prompt = f"Simplify this instruction for an individual with autism:\n'{instruction}'\n\nKeep it short, direct, and calming."
        try:
            return self.llm.ask(prompt)
        except Exception as e:
            print(f"LLM simplify_instruction error: {e}")
            return instruction + " â€” Let's do this step calmly."

    def generate_calming_message(self, emotion_info: dict) -> str:
        """Generates a supportive message based on emotional state."""
        prompt = f"A user's emotional state is detected as '{emotion_info.get('state')}'. Write a very short, reassuring, and calming message for them (1-2 sentences)."
        try:
            return self.llm.ask(prompt)
        except Exception as e:
            print(f"LLM calming message error: {e}")
            return "It's okay. Take a few slow, deep breaths."

    def caregiver_summary(self, log: dict) -> str:
        """Generates a concise summary of a routine for a caregiver."""
        prompt = f"Summarize this routine log for a caregiver. Highlight any steps where the user was hesitant or rushed and provide one simple, actionable piece of advice.\n\nLog:\n{json.dumps(log, default=str)}"
        try:
            return self.llm.ask(prompt)
        except Exception as e:
            print(f"LLM summary error: {e}")
            return "Routine finished. Please review the log for details."

class RoutineAgent:
    """Executes the steps of a given routine."""
    def __init__(self, safety_agent: SafetyAgent, ai_agent: AIReasoningAgent):
        self.safety_agent = safety_agent
        self.ai_agent = ai_agent

    def run_routine(self, routine_json: dict, user_profile: dict, on_step_update=None, simulate_behavior=True):
        """
        Runs a routine, logs progress, and interacts with other agents.
        
        Args:
            routine_json (dict): The routine to execute.
            user_profile (dict): The user's profile and preferences.
            on_step_update (function, optional): A callback for UI updates.
            simulate_behavior (bool): If True, simulates random user events.
        """
        log = {"routine_id": routine_json['id'], "start_time": datetime.utcnow().isoformat(), "steps": []}
        
        for step in routine_json['steps']:
            step_record = {"step_id": step['id'], "title": step['title'], "start": datetime.utcnow().isoformat()}
            instruction = step.get('instruction', step['title'])
            
            # Use AI agent to simplify the instruction.
            simplified_instruction = self.ai_agent.simplify_instruction(instruction)
            print(f"STEP {step['id']}: {simplified_instruction}")

            # Simulate the user performing the step.
            planned_duration = step.get('duration_sec', 5)
            start_time = time.time()
            
            # This loop simulates the passage of time for the step.
            for sec in range(int(planned_duration)):
                time.sleep(1)  # In a real app, this would be non-blocking.
                if on_step_update:
                    on_step_update(step, simplified_instruction, sec + 1, planned_duration)

            end_time = time.time()
            actual_duration = end_time - start_time
            
            step_record['end'] = datetime.utcnow().isoformat()
            step_record['planned_duration'] = planned_duration
            step_record['actual_duration'] = actual_duration
            log['steps'].append(step_record)

            # Check if the user is stuck.
            if actual_duration > (planned_duration * 2): # If step took twice as long
                self.safety_agent.notify_step_stuck(user_profile, step, actual_duration)

        log['end_time'] = datetime.utcnow().isoformat()
        if DB:
            DB.insert({'type': 'routine_log', 'data': log})
        return log

class OrchestratorAgent:
    """The main agent that coordinates all other agents to run a routine."""
    def __init__(self, user_profile: dict, routines_dir: str = "routines", use_ai: bool = True):
        self.user_profile = user_profile
        self.routines_dir = routines_dir
        
        self.safety_agent = SafetyAgent(caregiver_notify_fn=self.caregiver_notify)
        self.ai_agent = AIReasoningAgent() if use_ai else AIReasoningAgent(llm=SafeDummyLLM())
        self.routine_agent = RoutineAgent(self.safety_agent, self.ai_agent)
        self.emotion_agent = EmotionAgent()

    def caregiver_notify(self, contact, message):
        """Placeholder for sending notifications (e.g., SMS, email)."""
        print(f"[Caregiver Notification] To: {contact['name']} | Message: {message}")

    def load_routine(self, routine_id: str) -> dict:
        """Loads a routine JSON file from the specified directory."""
        path = os.path.join(self.routines_dir, f"{routine_id}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Routine file not found: {path}")
        with open(path, 'r') as f:
            return json.load(f)

    def start_routine(self, routine_id: str, on_step_update=None):
        """
        Starts and manages a full routine, from execution to analysis.
        
        Returns:
            A dictionary containing the final log, emotion analysis, and any calming message.
        """
        print(f"Starting routine: {routine_id}")
        routine = self.load_routine(routine_id)
        
        # Run the routine and get the log
        log = self.routine_agent.run_routine(routine, self.user_profile, on_step_update)
        
        # Analyze the log to interpret emotion
        emotion_summary = self.emotion_agent.interpret(log)
        
        if DB:
            DB.insert({'type': 'emotion_summary', 'data': {'routine_id': routine_id, 'emotion': emotion_summary, 'time': datetime.utcnow().isoformat()}})
            
        result = {"log": log, "emotion": emotion_summary}

        # If stress is detected, generate a calming message
        if emotion_summary.get('state') in ['hesitant', 'rushed']:
            calm_message = self.ai_agent.generate_calming_message(emotion_summary)
            result["calm_message"] = calm_message
            
        # For demonstration, also get a caregiver summary
        summary = self.ai_agent.caregiver_summary(log)
        result["caregiver_summary"] = summary
        
        print(f"Routine {routine_id} finished.")
        return result


%%writefile simulator.py

# simulator.py
import os
import json

def create_demo_files():
    """Creates JSON files for routines and a user profile if they don't exist."""
    os.makedirs('routines', exist_ok=True)
    
    # --- Define Routine Data ---
    routines = {}
    routines['brushing_teeth_v1'] = {
      "id":"brushing_teeth_v1", "title":"Brush Teeth - Morning", "steps":[
        {"id":1,"title":"Pick up toothbrush","duration_sec":6,"instruction":"Gently pick up your toothbrush."},
        {"id":2,"title":"Apply toothpaste","duration_sec":4,"instruction":"Put a small bit of toothpaste on the brush."},
        {"id":3,"title":"Brush top teeth","duration_sec":8,"instruction":"Brush the teeth on the top of your mouth."},
        {"id":4,"title":"Brush bottom teeth","duration_sec":8,"instruction":"Brush the teeth on the bottom of your mouth."},
        {"id":5,"title":"Rinse mouth","duration_sec":4,"instruction":"Rinse your mouth with water."},
        {"id":6,"title":"Put away toothbrush","duration_sec":4,"instruction":"Put your toothbrush back in its place."}
      ]}
    routines['get_ready_for_school_v1'] = {
      "id":"get_ready_for_school_v1", "title":"Get Ready For School", "steps":[
        {"id":1,"title":"Wake up","duration_sec":10,"instruction":"Wake up and sit on the bed."},
        {"id":2,"title":"Use bathroom","duration_sec":15,"instruction":"Use the bathroom if you need to."},
        {"id":3,"title":"Get dressed","duration_sec":20,"instruction":"Put on your clothes for school."},
        {"id":4,"title":"Eat breakfast","duration_sec":30,"instruction":"Eat your breakfast slowly."},
        {"id":5,"title":"Pack bag","duration_sec":25,"instruction":"Pack your books and lunch in your bag."},
        {"id":6,"title":"Put on jacket and shoes","duration_sec":15,"instruction":"Put on your jacket and shoes."},
        {"id":7,"title":"Leave for school","duration_sec":10,"instruction":"It's time to go to the school bus or car."}
      ]}

    # --- Write Routine Files ---
    for name, data in routines.items():
        path = os.path.join('routines', f"{name}.json")
        if not os.path.exists(path):
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)

    # --- Create a default user profile ---
    upath = 'routines/user_profile.json'
    if not os.path.exists(upath):
        user_profile = {
            "user_id": "maya_g", "name": "Maya", "age": 12,
            "caregiver_contact": {"name": "Pauly", "phone": "123-456-7890"},
            "sensory_preferences": {"max_step_duration": 600},
            "safety_settings": {"notify_if_unfinished": True}
        }
        with open(upath, 'w') as f:
            json.dump(user_profile, f, indent=2)

if __name__ == "__main__":
    create_demo_files()
    print("âœ… Demo routine files and user profile created in ./routines/")


%%writefile app_streamlit.py

# app_streamlit.py
import streamlit as st
import json
import os
import pandas as pd
from tinydb import TinyDB, Query
from agents_impl import OrchestratorAgent
from simulator import create_demo_files

# --- Page Config ---
st.set_page_config(page_title="NeuroAssist Demo", layout="wide")

# --- Initial Setup ---
@st.cache_resource
def setup():
    """Create demo files and initialize database."""
    create_demo_files()
    return TinyDB('data/db.json')

db = setup()

# --- Load Data ---
@st.cache_data
def load_user_profile():
    """Load the user profile from JSON."""
    with open('routines/user_profile.json', 'r') as f:
        return json.load(f)

user_profile = load_user_profile()
ROUTINES_DIR = 'routines'

# --- UI Sidebar ---
st.sidebar.title("NeuroAssist")
st.sidebar.markdown("---")
mode = st.sidebar.selectbox('Select Mode', ['User Demo', 'Caregiver Dashboard'])
use_ai = st.sidebar.toggle("Enable AI Reasoning", value=True, help="When enabled, uses an LLM for instructions and summaries. On Kaggle, this uses a safe, simulated LLM.")
st.sidebar.markdown("---")
st.sidebar.subheader("Current User")
st.sidebar.write(f"**Name:** {user_profile['name']}")
st.sidebar.write(f"**Age:** {user_profile['age']}")


# --- Main App Logic ---

# ============================
# 1. CAREGIVER DASHBOARD MODE
# ============================
if mode == 'Caregiver Dashboard':
    st.header('Caregiver Dashboard')
    st.markdown("Here you can review routine history and create/edit routines for the user.")

    # --- Analytics Section ---
    st.subheader("ğŸ“Š Analytics")
    logs = db.search(Query().type == 'routine_log')
    st.metric("Total Routines Run", len(logs))
    
    if logs:
        last_log_data = logs[-1]['data']
        st.write("**Last Routine Analysis:**")
        
        df = pd.DataFrame([
            {"step": s["title"], "Planned (s)": s["planned_duration"], "Actual (s)": round(s["actual_duration"], 1)} 
            for s in last_log_data["steps"]
        ])
        st.dataframe(df, use_container_width=True)
        
        # Display summary from the AI agent
        emotion_logs = db.search(Query().type == 'emotion_summary')
        if emotion_logs:
            st.write(f"**Emotion Detected:** {emotion_logs[-1]['data']['emotion']['state']}")
            st.info(f"**AI Advice:** {emotion_logs[-1]['data']['emotion']['advice']}")

    # --- Log Viewer Section ---
    with st.expander("View Raw Routine Logs"):
        st.json(logs)

# ============================
# 2. USER DEMO MODE
# ============================
elif mode == 'User Demo':
    st.header(f"Hi {user_profile['name']}, let's start a routine!")
    
    orchestrator = OrchestratorAgent(user_profile, use_ai=use_ai)
    
    routine_files = [f for f in os.listdir(ROUTINES_DIR) if f.endswith('.json') and 'user_profile' not in f]
    chosen_file = st.selectbox('Choose a routine to run:', routine_files, format_func=lambda x: x.replace(".json", "").replace("_", " ").title())

    if st.button('âœ¨ Start Routine', use_container_width=True):
        routine_id = chosen_file.replace('.json', '')
        
        # --- UI Placeholders for the routine runner ---
        st.markdown("---")
        title_placeholder = st.empty()
        instruction_placeholder = st.empty()
        progress_placeholder = st.empty()
        status_placeholder = st.empty()

        def on_step_update(step, simplified_instruction, sec, duration):
            """Callback function to update the Streamlit UI during a routine."""
            title_placeholder.subheader(f"Step {step['id']}: {step['title']}")
            instruction_placeholder.info(f"ğŸ¤–: {simplified_instruction}")
            progress_placeholder.progress(int((sec / duration) * 100))
            status_placeholder.write(f"Time: {sec} / {duration}s")

        # --- Run Routine ---
        with st.spinner("Running routine..."):
            result = orchestrator.start_routine(routine_id, on_step_update=on_step_update)
        
        st.success("ğŸ�‰ Routine Completed!")
        st.balloons()
        
        # --- Display Results ---
        st.markdown("---")
        st.subheader("Routine Summary")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Final Emotion State", result['emotion']['state'].title())
        
        if "calm_message" in result:
            st.info(f"**A helpful message for you:** {result['calm_message']}")
            
        with st.expander("View Final Log"):
            st.json(result['log'])
        with st.expander("View AI Caregiver Summary"):
            st.write(result.get('caregiver_summary', 'No summary generated.'))


import os
import simulator
import subprocess
from pyngrok import ngrok
from kaggle_secrets import UserSecretsClient

# --- Create demo files ---
simulator.create_demo_files()

# --- Configure ngrok with your secret authtoken ---
try:
    user_secrets = UserSecretsClient()
    NGROK_AUTH_TOKEN = user_secrets.get_secret("NGROK_AUTH_TOKEN")
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    print("âœ… ngrok authtoken configured successfully.")
except Exception as e:
    print("âš ï¸� Could not set ngrok authtoken. The tunnel will be temporary. Error:", e)
    print("-> For a stable URL, add your ngrok token to Kaggle Secrets as 'NGROK_AUTH_TOKEN'")

# --- Launch the Streamlit app and create the tunnel ---
port = 8501

# Terminate any existing ngrok tunnels
ngrok.kill()

# Start a new tunnel to the Streamlit port
public_url = ngrok.connect(port)
print("======================================================================================")
print(f"âœ… Your Streamlit App is live! Click this link to open:\n{public_url}")
print("======================================================================================")

# --- Define the command to run Streamlit ---
# We pass the command as a list of strings to subprocess
command = [
    "streamlit", "run", "app_streamlit.py",
    "--server.port", str(port),
    "--server.headless", "true"  # Important for running in a notebook
]

# --- Run the Streamlit app in the background ---
# This starts the app without blocking the notebook cell
subprocess.Popen(command)

