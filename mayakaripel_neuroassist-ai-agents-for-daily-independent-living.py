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
(Refactored to use a singleton DB instance for state consistency)
"""
import os
import json
import time
import random
from datetime import datetime

try:
    from tinydb import TinyDB
except ImportError:
    TinyDB = None

# --- DATABASE SINGLETON ---
# This ensures every part of the application uses the exact same DB instance.
os.makedirs("data", exist_ok=True)
_DB_INSTANCE = None
def get_db():
    global _DB_INSTANCE
    if _DB_INSTANCE is None:
        _DB_INSTANCE = TinyDB("data/db.json")
    return _DB_INSTANCE

# ... (LLM Abstraction classes: LLMBase, SafeDummyLLM, OpenAILLM, GeminiLLM are unchanged) ...
class LLMBase:
    def ask(self, prompt: str) -> str:
        raise NotImplementedError

class SafeDummyLLM(LLMBase):
    def ask(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        if "simplify" in prompt_lower:
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
    def __init__(self, model="gpt-4o-mini"):
        import openai
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.openai = openai
        self.model = os.environ.get("OPENAI_MODEL", model)
    def ask(self, prompt: str) -> str:
        resp = self.openai.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content.strip()

class GeminiLLM(LLMBase):
    def __init__(self, model="gemini-1.5-flash"):
        import google.generativeai as genai
        api_key = os.environ.get("GENIE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GENIE_API_KEY or GEMINI_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(os.environ.get("GEMINI_MODEL", model))
    def ask(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text.strip()

def load_llm():
    running_on_kaggle = os.path.exists("/kaggle") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
    if running_on_kaggle:
        print("âœ… Kaggle environment detected â€” using SafeDummyLLM.")
        return SafeDummyLLM()
    if "OPENAI_API_KEY" in os.environ:
        try: return OpenAILLM()
        except Exception as e: print(f"OpenAI LLM init failed: {e}")
    if "GENIE_API_KEY" in os.environ or "GEMINI_API_KEY" in os.environ:
        try: return GeminiLLM()
        except Exception as e: print(f"Gemini LLM init failed: {e}")
    print("âš ï¸� No LLM credentials found â€” falling back to SafeDummyLLM.")
    return SafeDummyLLM()


# ... (SafetyAgent, EmotionAgent, AIReasoningAgent are unchanged) ...
class SafetyAgent:
    def __init__(self, caregiver_notify_fn=None): self.caregiver_notify_fn = caregiver_notify_fn
    def detect_meltdown(self, event: str) -> bool: return any(k in event.lower() for k in ["crying", "screaming", "refusing", "panic", "overwhelmed"])
    def notify_step_stuck(self, user_profile, step, actual_duration):
        contact = user_profile.get("caregiver_contact")
        msg = f"User {user_profile.get('name')} may be stuck on step '{step.get('title')}' (took {int(actual_duration)}s)."
        if self.caregiver_notify_fn: self.caregiver_notify_fn(contact, msg)
        else: print(f"[SafetyAgent Notification] {msg}")

class EmotionAgent:
    def interpret(self, routine_log: dict) -> dict:
        steps, slow, fast = routine_log.get('steps', []), 0, 0
        for s in steps:
            planned, actual = s.get('planned_duration', 1), s.get('actual_duration', s.get('planned_duration', 1))
            if actual > planned * 1.8: slow += 1
            if actual < planned * 0.6: fast += 1
        if slow >= 1 and fast == 0: return {"state": "hesitant", "advice": "Offer a calm break."}
        if fast >= 1 and slow == 0: return {"state": "rushed", "advice": "Encourage a slower pace."}
        return {"state": "ok", "advice": "Good job! Routine followed at a steady pace."}

class AIReasoningAgent:
    def __init__(self, llm: LLMBase = None): self.llm = llm or load_llm()
    def simplify_instruction(self, instruction: str) -> str:
        prompt = f"Simplify for an individual with autism:\n'{instruction}'\n\nKeep it short and calm."
        try: return self.llm.ask(prompt)
        except Exception as e:
            print(f"LLM simplify error: {e}")
            return instruction + " â€” Let's do this step calmly."
    def generate_calming_message(self, emotion_info: dict) -> str:
        prompt = f"A user is '{emotion_info.get('state')}'. Write a short, calming message (1-2 sentences)."
        try: return self.llm.ask(prompt)
        except Exception as e:
            print(f"LLM calming message error: {e}")
            return "It's okay. Take a few slow, deep breaths."
    def caregiver_summary(self, log: dict) -> str:
        prompt = f"Summarize this routine log for a caregiver. Note hesitations/rushes and give one piece of advice.\n\nLog:\n{json.dumps(log, default=str)}"
        try: return self.llm.ask(prompt)
        except Exception as e:
            print(f"LLM summary error: {e}")
            return "Routine finished. Please review the log."

class RoutineAgent:
    def __init__(self, safety_agent: SafetyAgent, ai_agent: AIReasoningAgent):
        self.safety_agent = safety_agent
        self.ai_agent = ai_agent

    def run_routine(self, routine_json: dict, user_profile: dict, on_step_update=None):
        log = {"routine_id": routine_json['id'], "start_time": datetime.utcnow().isoformat(), "steps": []}
        for step in routine_json['steps']:
            simplified_instruction = self.ai_agent.simplify_instruction(step.get('instruction', step['title']))
            print(f"STEP {step['id']}: {simplified_instruction}")
            
            start_time = time.time()
            planned_duration = step.get('duration_sec', 5)
            for sec in range(int(planned_duration)):
                time.sleep(1)
                if on_step_update:
                    on_step_update(step, simplified_instruction, sec + 1, planned_duration)
            
            actual_duration = time.time() - start_time
            step_record = {
                "step_id": step['id'], "title": step['title'], "start": datetime.utcfromtimestamp(start_time).isoformat(),
                "end": datetime.utcnow().isoformat(), "planned_duration": planned_duration, "actual_duration": actual_duration
            }
            log['steps'].append(step_record)
            if actual_duration > (planned_duration * 2):
                self.safety_agent.notify_step_stuck(user_profile, step, actual_duration)
        
        log['end_time'] = datetime.utcnow().isoformat()
        # Use the singleton to write to the database
        get_db().insert({'type': 'routine_log', 'data': log})
        return log

class OrchestratorAgent:
    def __init__(self, user_profile: dict, routines_dir: str = "routines", use_ai: bool = True):
        self.user_profile = user_profile
        self.routines_dir = routines_dir
        
        self.safety_agent = SafetyAgent(caregiver_notify_fn=self.caregiver_notify)
        self.ai_agent = AIReasoningAgent() if use_ai else AIReasoningAgent(llm=SafeDummyLLM())
        self.routine_agent = RoutineAgent(self.safety_agent, self.ai_agent)
        self.emotion_agent = EmotionAgent()

    def caregiver_notify(self, contact, message):
        print(f"[Caregiver Notification] To: {contact['name']} | Message: {message}")

    def load_routine(self, routine_id: str) -> dict:
        path = os.path.join(self.routines_dir, f"{routine_id}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Routine file not found: {path}")
        with open(path, 'r') as f:
            return json.load(f)

    def start_routine(self, routine_id: str, on_step_update=None):
        print(f"Starting routine: {routine_id}")
        routine = self.load_routine(routine_id)
        
        log = self.routine_agent.run_routine(routine, self.user_profile, on_step_update)
        emotion_summary = self.emotion_agent.interpret(log)
        
        # Use the singleton to write to the database
        get_db().insert({'type': 'emotion_summary', 'data': {'routine_id': routine_id, 'emotion': emotion_summary, 'time': datetime.utcnow().isoformat()}})
            
        result = {"log": log, "emotion": emotion_summary}
        if emotion_summary.get('state') in ['hesitant', 'rushed']:
            result["calm_message"] = self.ai_agent.generate_calming_message(emotion_summary)
            
        result["caregiver_summary"] = self.ai_agent.caregiver_summary(log)
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
from tinydb import Query
# Import the get_db singleton function and the OrchestratorAgent
from agents_impl import get_db, OrchestratorAgent
from simulator import create_demo_files

# --- Page Config and Initial Setup ---
st.set_page_config(page_title="NeuroAssist Demo", layout="wide")
ROUTINES_DIR = 'routines'

# Create the necessary files on disk. This is safe to run multiple times.
create_demo_files()
# Get the one-and-only database instance using our singleton function.
db = get_db()

# --- Load Data ---
@st.cache_data
def load_user_profile():
    with open('routines/user_profile.json', 'r') as f:
        return json.load(f)
user_profile = load_user_profile()

# --- UI Sidebar ---
st.sidebar.title("NeuroAssist")
st.sidebar.markdown("---")
mode = st.sidebar.selectbox('Select Mode', ['User Demo', 'Caregiver Dashboard'])
use_ai = st.sidebar.toggle("Enable AI Reasoning", value=True, help="When enabled, uses an LLM. On Kaggle, this uses a safe, simulated LLM.")
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

    # Always read logs from the singleton DB object
    logs = db.search(Query().type == 'routine_log')
    st.metric("Total Routines Run", len(logs))
    
    if logs:
        logs.sort(key=lambda x: x['data']['start_time'], reverse=True)
        last_log_data = logs[0]['data']
        st.write("**Last Routine Analysis:**")
        
        df = pd.DataFrame([
            {"step": s["title"], "Planned (s)": s["planned_duration"], "Actual (s)": round(s["actual_duration"], 1)} 
            for s in last_log_data["steps"]
        ])
        st.dataframe(df, use_container_width=True)
        
        emotion_logs = db.search(Query().type == 'emotion_summary')
        if emotion_logs:
            emotion_logs.sort(key=lambda x: x['data']['time'], reverse=True)
            st.write(f"**Emotion Detected:** {emotion_logs[0]['data']['emotion']['state']}")
            st.info(f"**AI Advice:** {emotion_logs[0]['data']['emotion']['advice']}")

    with st.expander("View All Raw Routine Logs"):
        st.json(logs)

# ============================
# 2. USER DEMO MODE
# ============================
elif mode == 'User Demo':
    st.header(f"Hi {user_profile['name']}, let's start a routine!")
    
    # The agent will automatically use the same singleton DB instance
    orchestrator = OrchestratorAgent(user_profile=user_profile, use_ai=use_ai)
    
    routine_files = [f for f in os.listdir(ROUTINES_DIR) if f.endswith('.json') and 'user_profile' not in f]
    chosen_file = st.selectbox('Choose a routine to run:', routine_files, format_func=lambda x: x.replace(".json", "").replace("_", " ").title())

    if st.button('âœ¨ Start Routine', use_container_width=True):
        routine_id = chosen_file.replace('.json', '')
        
        title_placeholder = st.empty()
        instruction_placeholder = st.empty()
        progress_placeholder = st.empty()
        status_placeholder = st.empty()

        def on_step_update(step, simplified_instruction, sec, duration):
            title_placeholder.subheader(f"Step {step['id']}: {step['title']}")
            instruction_placeholder.info(f"ğŸ¤–: {simplified_instruction}")
            progress_placeholder.progress(int((sec / duration) * 100))
            status_placeholder.write(f"Time: {sec} / {duration}s")

        with st.spinner("Running routine..."):
            result = orchestrator.start_routine(routine_id, on_step_update=on_step_update)
        
        st.success("ğŸ�‰ Routine Completed!")
        st.balloons()
        
        st.subheader("Routine Summary")
        st.metric("Final Emotion State", result['emotion']['state'].title())
        if "calm_message" in result:
            st.info(f"**A helpful message for you:** {result['calm_message']}")
            
        with st.expander("View AI Caregiver Summary"):
            st.write(result.get('caregiver_summary', 'No summary generated.'))


import os
import time
import subprocess

# Ensure simulator is available and create demo files
try:
    import simulator
    simulator.create_demo_files()
except Exception as e:
    print("âš ï¸� Could not create demo files:", e)

# Configure ngrok auth token (Kaggle Secrets or env var)
try:
    from kaggle_secrets import UserSecretsClient
    from pyngrok import ngrok
    try:
        user_secrets = UserSecretsClient()
        token = user_secrets.get_secret("NGROK_AUTH_TOKEN")
        if token:
            ngrok.set_auth_token(token)
            print("âœ… ngrok authtoken configured from Kaggle Secrets.")
    except Exception:
        # fallback to env var
        token = os.environ.get("NGROK_AUTH_TOKEN") or os.environ.get("NGROK_TOKEN")
        if token:
            ngrok.set_auth_token(token)
            print("âœ… ngrok authtoken configured from environment variable.")
except Exception as e:
    ngrok = None
    print("âš ï¸� pyngrok or kaggle_secrets not available:", e)

# Launch Streamlit and ngrok tunnel
port = int(os.environ.get("STREAMLIT_PORT", 8501))

if ngrok:
    try:
        # Ensure any previous tunnels are closed
        ngrok.kill()
    except Exception:
        pass

    try:
        tunnel = ngrok.connect(port, bind_tls=True)
        public_url = getattr(tunnel, "public_url", str(tunnel))
        print("=" * 90)
        print(f"âœ… Your Streamlit App should be available at:\n{public_url}")
        print("=" * 90)
    except Exception as e:
        print("âš ï¸� Failed to open ngrok tunnel:", e)
        public_url = None
else:
    print("â„¹ï¸� ngrok not configured; run locally and open http://localhost:8501")

# Command to run Streamlit
command = [
    "streamlit", "run", "app_streamlit.py",
    "--server.port", str(port),
    "--server.headless", "true"
]

# Start Streamlit in background (stdout/stderr suppressed)
try:
    # Use startup logs files so the process isn't lost
    out_log = open("streamlit.out.log", "a")
    err_log = open("streamlit.err.log", "a")
    subprocess.Popen(command, stdout=out_log, stderr=err_log)
    # give Streamlit a moment to start
    time.sleep(2)
    print("âœ… Streamlit process started (logs: streamlit.out.log / streamlit.err.log).")
    if public_url:
        print(f"Open the public URL above (ngrok) or visit http://localhost:{port}")
    else:
        print(f"Visit http://localhost:{port}")
except FileNotFoundError:
    print("â�Œ 'streamlit' executable not found. Install Streamlit (`pip install streamlit`) and try again.")
except Exception as e:
    print("â�Œ Failed to start Streamlit:", e)


from pyngrok import ngrok
print(ngrok.get_tunnels())

