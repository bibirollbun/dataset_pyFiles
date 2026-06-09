import os
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
os.environ["GOOGLE_API_KEY"] = secrets.get_secret("GOOGLE_API_KEY")


import os
import json
import sqlite3
import asyncio

# ADK / GenAI imports
from google.adk.agents import Agent, SequentialAgent
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

APP_NAME = "NightingaleNexus"
USER_ID = "demo_user"   # Any stable string for the "user"

# Use the model required by the competition / environment
MODEL_NAME = "gemini-2.5-flash"
# MODEL_NAME = "gemini-2.0-flash"  # uncomment if allowed and available

if "GOOGLE_API_KEY" not in os.environ:
    print("âš ï¸� WARNING: GOOGLE_API_KEY not set. Make sure you set it before running.")

# -----------------------------------------------------------------------------
# 1. STORAGE LAYER (SQLite Medical DB)
# -----------------------------------------------------------------------------

class MedicalRecordsDB:
    def __init__(self, db_name="nightingale_adk.db"):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._init_tables()

    def _init_tables(self):
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY,
                name TEXT,
                email TEXT
            )
            """
        )
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS visits (
                id INTEGER PRIMARY KEY,
                patient_id INTEGER,
                date TEXT,
                symptoms TEXT,
                diagnosis TEXT,
                plan TEXT
            )
            """
        )
        self.conn.commit()

    def get_patient_id(self, name, email):
        res = self.cursor.execute(
            "SELECT id FROM patients WHERE email = ?", (email,)
        ).fetchone()
        if res:
            return res[0]
        self.cursor.execute(
            "INSERT INTO patients (name, email) VALUES (?, ?)", (name, email)
        )
        self.conn.commit()
        return self.cursor.lastrowid

    def log_visit(self, patient_id, symptoms, date):
        self.cursor.execute(
            "INSERT INTO visits (patient_id, date, symptoms) VALUES (?, ?, ?)",
            (patient_id, date, symptoms),
        )
        self.conn.commit()
        return self.cursor.lastrowid

    def get_history(self, patient_id):
        rows = self.cursor.execute(
            "SELECT id, date, symptoms, diagnosis, plan FROM visits WHERE patient_id = ?",
            (patient_id,),
        ).fetchall()
        return [
            {
                "visit_id": r[0],
                "date": r[1],
                "symptoms": r[2],
                "diagnosis": r[3],
                "plan": r[4],
            }
            for r in rows
        ]

    def update_visit(self, visit_id, diagnosis, plan):
        self.cursor.execute(
            "UPDATE visits SET diagnosis=?, plan=? WHERE id=?",
            (diagnosis, plan, visit_id),
        )
        self.conn.commit()


db_instance = MedicalRecordsDB()

# -----------------------------------------------------------------------------
# 2. TOOL FUNCTIONS (used by agents via ADK)
# -----------------------------------------------------------------------------

def check_calendar(date_str: str) -> str:
    """Checks doctor availability for a specific date."""
    # For demo purposes we hardcode some slots
    return f"Slots available on {date_str} at 10:00 AM and 2:00 PM."

def register_visit(name: str, email: str, symptoms: str, date: str) -> str:
    """Registers a new patient visit request in the database. Returns Visit ID."""
    pid = db_instance.get_patient_id(name, email)
    vid = db_instance.log_visit(pid, symptoms, date)
    return f"Visit confirmed. Patient ID: {pid}, Visit ID: {vid}"

def fetch_history(name: str, email: str) -> str:
    """Retrieves medical history for context engineering."""
    pid = db_instance.get_patient_id(name, email)
    history = db_instance.get_history(pid)
    if not history:
        return "No previous history found."
    return json.dumps(history, indent=2)

def update_record(visit_id: int, diagnosis: str, treatment_plan: str) -> str:
    """Updates a visit record with doctor's diagnosis."""
    db_instance.update_visit(visit_id, diagnosis, treatment_plan)
    return "Database updated successfully."

# -----------------------------------------------------------------------------
# 3. AGENTS (Sheila, Brian_Retriever, Brian_Synthesizer, Clara)
# -----------------------------------------------------------------------------

# Agent 1: Sheila â€“ Intake & Scheduling
sheila = Agent(
    name="Sheila",
    model=MODEL_NAME,
    instruction="""
    You are Sheila, a medical scheduler for a clinic.

    Your job:
    1. From the user's message, extract:
       - Patient name
       - Email
       - Symptoms
       - Requested appointment date
    2. ALWAYS call the `check_calendar` tool with the extracted date.
    3. If the calendar shows availability, ALWAYS call `register_visit`
       with (name, email, symptoms, date). Use the earliest available slot.
    4. After all tool calls, send a final friendly message
       to the patient that:
       - Confirms the booking
       - Clearly states the date and time
       - Clearly states the Visit ID from `register_visit`.

    Important rules:
    - You MUST use the tools. Do not just answer in plain text.
    - Never end your turn with only a function/tool call.
      After using tools, you MUST send a natural-language confirmation.
    """,
    tools=[check_calendar, register_visit],
)

# Agent 2a: Brian_Retriever â€“ Fetches history from DB
brian_retriever = Agent(
    name="Brian_Retriever",
    model=MODEL_NAME,
    instruction="""
    You are Brian_Retriever, a patient history retriever.

    Task:
    - From the user's message, extract the patient's name and email.
    - ALWAYS call `fetch_history(name, email)` exactly once.
    - Do not summarize. Just ensure the correct history is retrieved
      for downstream use.
    """,
    tools=[fetch_history],
)

# Agent 2b: Brian_Synthesizer â€“ Clinical context writer
brian_synthesizer = Agent(
    name="Brian_Synthesizer",
    model=MODEL_NAME,
    instruction="""
    You are Brian_Synthesizer, a Clinical Context Synthesizer.

    You will receive as input either:
    - A JSON string containing a list of visit objects, OR
    - The text "No previous history found."

    If JSON is provided:
    - Mentally parse it.
    - Generate a concise 'Doctor's Handoff Brief' in Markdown:
      - Patient overview
      - Recurring symptoms
      - Notable past diagnoses or treatments
      - Potential complications / things to watch out for.

    If there is no previous history:
    - Generate a very short brief stating that this is the first presentation.

    Keep it short, clinical, and directly useful for a doctor.
    """,
)

# Sequential Agent: Brian (Retriever â†’ Synthesizer)
brian = SequentialAgent(
    name="Brian_Context_Engine",
    sub_agents=[brian_retriever, brian_synthesizer],
)

# Agent 3: Clara â€“ Data Clerk (writes back to DB)
clara = Agent(
    name="Clara",
    model=MODEL_NAME,
    instruction="""
    You are Clara, a Data Clerk for the clinic.

    Input:
    - A doctor's note written in natural language.

    Your job:
    1. Extract:
       - Visit ID (an integer)
       - Diagnosis (short phrase)
       - Treatment plan (short paragraph)
    2. Call `update_record(visit_id, diagnosis, treatment_plan)` exactly once.
    3. After the tool call, send a confirmation message to the doctor that:
       - Mentions the Visit ID
       - Restates the diagnosis
       - Confirms that the record was updated.

    Very important:
    - You MUST call `update_record`.
    - Never end with only a function/tool call. Always send a final
      natural-language confirmation after using the tool.
    """,
    tools=[update_record],
)

# -----------------------------------------------------------------------------
# 4. RUNNERS
# -----------------------------------------------------------------------------

runner_sheila = InMemoryRunner(agent=sheila, app_name=APP_NAME)
runner_brian = InMemoryRunner(agent=brian, app_name=APP_NAME)
runner_clara = InMemoryRunner(agent=clara, app_name=APP_NAME)

# -----------------------------------------------------------------------------
# 5. SESSION INITIALIZATION (async)
# -----------------------------------------------------------------------------

async def init_sessions():
    """
    Create sessions for each runner before first use.
    Call this once in the notebook:  await init_sessions()
    """
    session_map = [
        (runner_sheila, "session_alice_1"),
        (runner_clara, "session_doc_1"),
        (runner_brian, "session_alice_context"),
    ]

    for runner, sid in session_map:
        session_service = runner.session_service
        try:
            await session_service.create_session(
                app_name=runner.app_name,
                user_id=USER_ID,
                session_id=sid,
            )
        except Exception:
            # If session already exists or any benign error, ignore
            pass

# -----------------------------------------------------------------------------
# 6. HELPERS TO EXTRACT TEXT FROM EVENTS
# -----------------------------------------------------------------------------

def extract_text_from_content(content: Content):
    if not content or not getattr(content, "parts", None):
        return None
    texts = []
    for p in content.parts:
        if getattr(p, "text", None):
            texts.append(p.text)
    joined = "\n".join(texts).strip()
    return joined or None

# -----------------------------------------------------------------------------
# 7. MAIN DEMO (async) â€“ CALL THIS WITH:  await run_hackathon_demo()
# -----------------------------------------------------------------------------

async def run_hackathon_demo():
    print("--- ğŸ�¥ NIGHTINGALE NEXUS: ADK EXECUTION ---")

    # Ensure sessions exist
    await init_sessions()

    session_sheila = "session_alice_1"
    session_clara = "session_doc_1"
    session_brian = "session_alice_context"

    # --- SCENARIO 1: Sheila (Intake & Scheduling) ---

    user_msg_1 = "I'm Alice (alice@test.com). Severe migraine. Need slot on Oct 10th."
    print(f"\nğŸ‘¤ User: {user_msg_1}")

    events_1 = runner_sheila.run_async(
        user_id=USER_ID,
        session_id=session_sheila,
        new_message=Content(role="user", parts=[Part(text=user_msg_1)]),
    )

    sheila_reply = "[No final response from Sheila]"
    async for event in events_1:
        txt = extract_text_from_content(getattr(event, "content", None))
        if txt:
            sheila_reply = txt
    print(f"ğŸ‘©â€�ğŸ’¼ Sheila: {sheila_reply}")

    # --- SCENARIO 2: Clara (Doctor note â†’ DB update) ---

    doc_msg = (
        "For Visit ID 1 (Alice): Confirmed Migraine. "
        "Prescribed Sumatriptan. Advised rest."
    )
    print(f"\nğŸ‘¨â€�âš•ï¸� Doctor Note: {doc_msg}")

    events_2 = runner_clara.run_async(
        user_id=USER_ID,
        session_id=session_clara,
        new_message=Content(role="user", parts=[Part(text=doc_msg)]),
    )

    clara_reply = "[No final response from Clara]"
    async for event in events_2:
        txt = extract_text_from_content(getattr(event, "content", None))
        if txt:
            clara_reply = txt
    print(f"ğŸ‘©â€�ğŸ’» Clara: {clara_reply}")

    # --- SCENARIO 3: Brian (Context Engine for return visit) ---

    return_msg = "Alice (alice@test.com) is back. She feels dizzy now."
    print(f"\nğŸ�¥ Return Visit Trigger: {return_msg}")
    print("ğŸ§  Brian (Sequential Agent) is processing...")

    events_3 = runner_brian.run_async(
        user_id=USER_ID,
        session_id=session_brian,
        new_message=Content(role="user", parts=[Part(text=return_msg)]),
    )

    handoff_brief = "[No handoff brief generated]"
    async for event in events_3:
        txt = extract_text_from_content(getattr(event, "content", None))
        if txt:
            handoff_brief = txt

    print(f"\nğŸ“‹ Doctor's Handoff Brief:\n{'-'*40}\n{handoff_brief}\n{'-'*40}")

    # âœ… Write outputs to CSV
    import pandas as pd
    submission_df = pd.DataFrame({
        'Sheila Reply': [sheila_reply],
        'Clara Reply': [clara_reply],
        'Brian Handoff Brief': [handoff_brief]
    })
    submission_df.to_csv('/kaggle/working/submission.csv', index=False)
    print("âœ… Submission file created: /kaggle/working/submission.csv")



await init_sessions()
await run_hackathon_demo()

