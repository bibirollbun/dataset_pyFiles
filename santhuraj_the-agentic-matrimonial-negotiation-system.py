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


# --- âš™ï¸� SECTION 1: SETUP ---
import os
import logging
import sqlite3
import json
import random
import uuid
from datetime import datetime
from typing import List, Optional

# Kaggle Secrets (Uncomment when running on Kaggle)
from kaggle_secrets import UserSecretsClient
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(f"âš ï¸� Auth Warning: {e}")

# ADK Imports
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import FunctionTool, AgentTool
from google.genai import types
from pydantic import BaseModel, Field

# Suppress verbose logs
logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("google.adk").setLevel(logging.ERROR)

# Configuration
DB_NAME = "matrimony_council.sqlite"
MODEL_FAST = "gemini-2.5-flash-lite"
MODEL_SMART = "gemini-2.5-flash"
retry_config = types.HttpRetryOptions(attempts=3, exp_base=2, initial_delay=1)

print("âœ… Dependencies imported & Config set.")


# --- ğŸ’½ SECTION 2: DATA LAYER (SQLite) ---

def setup_database():
    """Initializes the in-memory database with synthetic profiles."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Cleanup old runs
    cursor.execute("DROP TABLE IF EXISTS profiles;")
    cursor.execute("DROP TABLE IF EXISTS deals;")
    cursor.execute("DROP TABLE IF EXISTS agent_logs;")
    
    # 1. Profiles Table
    cursor.execute("""
    CREATE TABLE profiles (
        id TEXT PRIMARY KEY,
        name TEXT,
        gender TEXT,
        age INTEGER,
        location TEXT,
        job TEXT,
        salary TEXT,
        family_type TEXT,
        horoscope_sign TEXT,
        risk_factor TEXT -- Hidden field for Detective Agent
    );
    """)
    
    # 2. Deals Table (To store successful negotiations)
    cursor.execute("""
    CREATE TABLE deals (
        id TEXT PRIMARY KEY,
        groom_id TEXT,
        bride_id TEXT,
        final_terms TEXT,
        utility_score INTEGER,
        timestamp TEXT
    );
    """)
    
    # 3. Observability Log Table
    cursor.execute("""
    CREATE TABLE agent_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        agent_name TEXT,
        action TEXT,
        details TEXT
    );
    """)
    
    # Generate Synthetic Data
    locations = ["Bangalore", "Mumbai", "Delhi", "Chennai", "Hyderabad"]
    jobs = ["Software Engineer", "Doctor", "Banker", "Artist", "Entrepreneur"]
    risks = ["Clean"] * 8 + ["High Debt", "Fake Job Profile"] # 20% risk chance
    signs = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra", "Scorpio"]
    
    profiles = []
    
    # Grooms
    for i in range(1, 11):
        pid = f"G-{i:03d}"
        profiles.append((
            pid, f"Groom_Arjun_{i}", "Male", random.randint(26, 35),
            random.choice(locations), random.choice(jobs), f"{random.randint(15,50)} LPA",
            "Joint" if i % 2 == 0 else "Nuclear", random.choice(signs), random.choice(risks)
        ))
        
    # Brides
    for i in range(1, 11):
        pid = f"B-{i:03d}"
        profiles.append((
            pid, f"Bride_Priya_{i}", "Female", random.randint(24, 32),
            random.choice(locations), random.choice(jobs), f"{random.randint(10,40)} LPA",
            "Nuclear", random.choice(signs), random.choice(risks)
        ))

    cursor.executemany("INSERT INTO profiles VALUES (?,?,?,?,?,?,?,?,?,?)", profiles)
    conn.commit()
    conn.close()
    print("âœ… SQLite Database initialized with 20 synthetic profiles.")

setup_database()


# --- ğŸ› ï¸� SECTION 3: TOOLS & UTILITIES  ---

# A. Search Tools
def get_random_profile_id(gender: str) -> str:
    """Fetches a random profile ID from DB. Handles synonyms like Groom/Male."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    #  Map agent terms to DB terms
    clean_gender = gender.lower().strip()
    if clean_gender in ["groom", "man", "male", "boy"]:
        db_gender = "Male"
    elif clean_gender in ["bride", "woman", "female", "girl"]:
        db_gender = "Female"
    else:
        db_gender = gender # Fallback
        
    cursor.execute("SELECT id FROM profiles WHERE gender = ?", (db_gender,))
    rows = cursor.fetchall()
    conn.close()
    
    # Return a random ID or "None" if list is empty
    return random.choice(rows)[0] if rows else "None"

def get_profile_details(profile_id: str) -> str:
    """Fetches public details for a profile."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM profiles WHERE id = ?", (profile_id,))
    row = cursor.fetchone()
    conn.close()
    if not row: return json.dumps({"error": "Not Found"})
    
    data = {
        "id": row[0], "name": row[1], "age": row[3], "location": row[4],
        "job": row[5], "salary": row[6], "family_type": row[7], "horoscope": row[8]
    }
    return json.dumps(data)

# B. Specialist Tools
def perform_background_check(profile_id: str) -> str:
    """(Detective Only) Reveals hidden risk factors."""
    conn = sqlite3.connect(DB_NAME)
    row = conn.execute("SELECT name, risk_factor FROM profiles WHERE id = ?", (profile_id,)).fetchone()
    conn.close()
    if not row: return json.dumps({"status": "Error"})
    
    _log_system_event("Detective", "Background Check", f"Checked {row[0]}: {row[1]}")
    return json.dumps({"profile_id": profile_id, "name": row[0], "background_status": row[1]})

def check_horoscope_compatibility(sign1: str, sign2: str) -> str:
    """(Astrologer Only) Calculates match score."""
    score = random.randint(10, 36)
    verdict = "Excellent" if score > 25 else "Average" if score > 18 else "Poor"
    return json.dumps({"guna_score": score, "verdict": verdict})

# C. Negotiation Math
def calculate_utility_score(groom_loc: str, bride_loc: str, proposal_loc: str, bride_career: str) -> str:
    """Calculates mathematical fairness of a deal."""
    score = 100
    if proposal_loc != groom_loc: score -= 30
    if proposal_loc != bride_loc: score -= 20
    if bride_career == "Quit Job": score -= 50 
    return json.dumps({"total_score": score, "viability": "High" if score > 60 else "Low"})

# D. System Tools
def _log_system_event(agent: str, event: str, details: str):
    """Internal logger."""
    conn = sqlite3.connect(DB_NAME)
    conn.execute("INSERT INTO agent_logs (timestamp, agent_name, action, details) VALUES (?,?,?,?)",
                 (datetime.now().isoformat(), agent, event, details))
    conn.commit()
    conn.close()

print("âœ…  Added logic to handle 'Groom' vs 'Male' inputs.")


# --- ğŸ¤– SECTION 4: AGENT DEFINITIONS ---

# 1. The Council: Specialists
detective_agent = LlmAgent(
    name="detective_agent",
    model=Gemini(model=MODEL_FAST, retry_options=retry_config),
    tools=[perform_background_check],
    instruction="""You are a Private Investigator. 
    1. Receive Profile IDs. 
    2. Call `perform_background_check` on BOTH.
    3. If Status is 'Clean', report PASS. 
    4. If Status is 'High Debt' or 'Fake Job', report CRITICAL FAIL immediately."""
)

astrologer_agent = LlmAgent(
    name="astrologer_agent",
    model=Gemini(model=MODEL_FAST, retry_options=retry_config),
    tools=[check_horoscope_compatibility],
    instruction="""You are the Family Astrologer.
    1. Extract horoscopes from profile details.
    2. Call `check_horoscope_compatibility`.
    3. If score < 18, warn the broker. Otherwise, give blessings."""
)

vetting_council = ParallelAgent(
    name="vetting_council",
    sub_agents=[detective_agent, astrologer_agent],
    description="Runs background and astrology checks simultaneously."
)

# 2. The Negotiators: Personas
groom_rep = LlmAgent(
    name="groom_rep",
    model=Gemini(model=MODEL_FAST),
    tools=[get_profile_details],
    instruction="""You represent the Groom. 
    - PRIORITY: Groom must stay in his current location.
    - FLEXIBILITY: You are okay if the bride works, but prefer she transfers to his city.
    - TONE: Stubborn but polite."""
)

bride_rep = LlmAgent(
    name="bride_rep",
    model=Gemini(model=MODEL_FAST),
    tools=[get_profile_details],
    instruction="""You represent the Bride.
    - PRIORITY: She MUST keep her job (Career is non-negotiable).
    - FLEXIBILITY: She can move cities ONLY if she gets a transfer.
    - TONE: Firm and protective of her rights."""
)

# 3. The Judge (Quality Assurance)
judge_agent = LlmAgent(
    name="judge_agent",
    model=Gemini(model=MODEL_SMART),
    instruction="""You are an Ethics & Quality Judge.
    Evaluate the final deal string provided by the user.
    Rate it 1-5 based on: Fairness, clarity, and logical consistency.
    Output: "Score: X/5. Reason: ..." """
)

# 4. The Root Broker (Orchestrator)
broker_agent = Agent(
    name="marriage_broker_agent",
    model=Gemini(model=MODEL_SMART, retry_options=retry_config),
    tools=[
        AgentTool(vetting_council, "Phase 1: Run parallel background and star checks."),
        AgentTool(groom_rep, "Phase 2: Get Groom's demands."),
        AgentTool(bride_rep, "Phase 2: Get Bride's demands."),
        FunctionTool(calculate_utility_score),
        FunctionTool(get_random_profile_id),
        FunctionTool(get_profile_details)
    ],
    instruction="""You are the Chief Marriage Broker.
    
    **PHASE 1: VETTING**
    1. Identify the Couple IDs (Groom/Bride) from context or find random ones.
    2. Call `vetting_council` to vet them.
    3. **CRITICAL GATEKEEPER:** If Detective finds "Fake Job" or "Debt" -> STOP negotiation immediately. Report "Match Rejected".
    
    **PHASE 2: NEGOTIATION (Only if Phase 1 Passed)**
    1. Consult `groom_rep` and `bride_rep` for their constraints (Location, Job).
    2. Propose a compromise (e.g., "Bride moves but keeps job").
    3. Call `calculate_utility_score` to verify fairness.
    4. If Score > 60, declare "MATCH SUCCESSFUL" and summarize terms.
    """
)

print("âœ… Multi-Agent Architecture Defined.")


# --- ğŸ§ª SECTION 5: EXECUTION & OBSERVABILITY ---

runner = Runner(
    agent=broker_agent,
    app_name="MarriageCouncilApp",
    session_service=InMemorySessionService(),
    memory_service=InMemoryMemoryService()
)

async def run_simulation(scenario_name: str, prompt: str):
    print(f"\n{'='*60}")
    print(f"ğŸ�¬ SCENARIO: {scenario_name}")
    print(f"{'='*60}")
    
    session_id = f"sim_{uuid.uuid4().hex[:8]}"
    
    #  Use keyword arguments here
    await runner.session_service.create_session(
        app_name="MarriageCouncilApp",
        user_id="user",
        session_id=session_id
    )
    
    user_msg = types.Content(role="user", parts=[types.Part(text=prompt)])
    
    final_output = ""
    
    async for event in runner.run_async(user_id="user", session_id=session_id, new_message=user_msg):
        # 1. Interceptor Log
        if event.get_function_calls():
            fn = event.get_function_calls()[0]
            print(f"   âš™ï¸� [TOOL USE] {fn.name}")
            _log_system_event("System", "ToolCall", fn.name)
            
        # 2. Print Output
        if event.content and event.content.parts:
            text = event.content.parts[0].text
            if text:
                print(f"   ğŸ¤– {text[:100]}...") # Print preview to keep log clean
                final_output = text
                
    # 3. Post-Run Evaluation (LLM-as-a-Judge)
    if "MATCH SUCCESSFUL" in final_output:
        print(f"\nâš–ï¸� JUDGE EVALUATION:")
        judge_runner = Runner(
            agent=judge_agent, 
            app_name="Eval", 
            session_service=InMemorySessionService()
        )
        
        # Create a session for the judge as well
        judge_session_id = f"judge_{session_id}"
        await judge_runner.session_service.create_session(
            app_name="Eval",
            user_id="judge",
            session_id=judge_session_id
        )
        
        async for e in judge_runner.run_async(
            user_id="judge", 
            session_id=judge_session_id, 
            new_message=types.Content(role="user", parts=[types.Part(text=f"Evaluate this deal: {final_output}")])
        ):
            if e.is_final_response() and e.content:
                print(f"   {e.content.parts[0].text}")

print("âœ… Simulation Engine Ready.")


# --- â–¶ï¸� RUN SCENARIOS ---

# Scenario 1: Random Negotiation (Likely to succeed or fail based on random data)
await run_simulation(
    "Random Match Attempt", 
    "Find a random Groom and Bride. Run full vetting and negotiation."
)

# Scenario 2: Forced Failure (We inject a specific ID if we knew it, 
# but here we rely on the agent finding one. We can simulate a specific prompt constraint)
await run_simulation(
    "Strict Requirements",
    "Find a random couple. Note: The Bride refuses to move locations. Try to negotiate."
)

# Final: Show Observability Logs
print("\n\nğŸ“Š SYSTEM LOGS (From SQLite):")
conn = sqlite3.connect(DB_NAME)
for row in conn.execute("SELECT timestamp, agent_name, action, details FROM agent_logs LIMIT 5"):
    print(row)
conn.close()




