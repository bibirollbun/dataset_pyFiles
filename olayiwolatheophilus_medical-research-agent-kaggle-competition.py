#import libraries
import os
import sqlite3
import logging
import asyncio
from typing import List, Union

# Import ADK Components
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.sessions import DatabaseSessionService
from google.adk.runners import Runner
from google.genai import types


# 1. Setup API Key (Replace with your actual key processing if using Kaggle Secrets)
from kaggle_secrets import UserSecretsClient
os.environ["GOOGLE_API_KEY"] = UserSecretsClient().get_secret("GOOGLE_API_KEY")
print("Api key set")


#Set app and config
APP_NAME = "medical_research_assistant"
USER_ID = "med_student_001" # Constant ID ensures DB finds you after restart
DB_FILE = "my_agent_data.db"
MODEL_NAME = "gemini-2.5-flash-lite"

# 3. Configure Logging (Observability)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
print("App and Config set up complete")


#database setup
def init_research_db():
    """ Initialises the persistent storage for research notes."""
    # Connect to the same SQLite file used by the session service
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Create a table to store research notes if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS research_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    print(f"âœ… Research table initialized in {DB_FILE}")

# Initialise the DB immediately
init_research_db()


# Tool 1: The Search Engine (Simulated for Reliability)

def search_medical_papers(query: str) -> str:
    """
    Searches for medical and AI research papers based on a keyword.
    Returns a list of titles and abstracts.
    """
    print(f"   ðŸ”Ž [TOOL] Searching database for: '{query}'")
    
    # Simulated database (Ensures reproducibility for judges)
    mock_db = {
        "stroke": [
            "Title: AI in Acute Stroke Imaging.\n   Abstract: unexpected benefits of ML in CT scans...",
            "Title: Rural Stroke Management in Nigeria.\n   Abstract: Analysis of telemedicine interventions in Oyo State...",
        ],
        "malaria": [
            "Title: Nanobots for Malaria Eradication.\n   Abstract: Feasibility study of blood-stream nanobots...",
        ],
        "default": [
            "Title: General AI in Healthcare.\n   Abstract: Overview of LLMs in patient triage..."
        ]
    }
    
    results = []
    for key, papers in mock_db.items():
        if key in query.lower():
            results.extend(papers)
            
    if not results:
        results = mock_db["default"]
        
    return f"Found {len(results)} papers: \n" + "\n".join(results)


# Tool 2: The Writer (Saves to Disk) 
def save_research_note(topic: str, summary: str) -> str:
    """
    Saves a summary or finding to the permanent research notebook.
    Use this when the user says 'save this' or 'remember this'.
    """
    print(f"   ðŸ’¾ [TOOL] Saving note on: '{topic}'")
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO research_notes (topic, content) VALUES (?, ?)", 
        (topic, summary)
    )
    conn.commit()
    conn.close()
    
    return "âœ… Note saved successfully to database."


# Tool 3: The Reader (Recalls from Disk) 

def read_my_notebook() -> str:
    """
    Reads all previously saved research notes from the database.
    Use this to remind the user what we have worked on.
    """
    print("   ðŸ“– [TOOL] Reading notebook from disk...")
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT topic, content, timestamp FROM research_notes")
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return "The notebook is empty."
    
    notes = "Here are your saved notes:\n"
    for r in rows:
        notes += f"- [{r[2]}] TOPIC: {r[0]} | NOTE: {r[1]}\n"
        
    return notes

print("âœ… Tools defined.")




#  Session Runner
async def run_session(
    runner_instance: Runner,
    user_queries: List[str] = [],
    session_name: str = "default",
):
    print(f"\n ### Session: {session_name}")
    app_name = runner_instance.app_name

    # --- PERSISTENCE LOGIC ---
    try:
        # Try creating new (throws error if exists)
        session = await session_service.create_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
        print("   ðŸ†• Created NEW session.")
    except Exception:
        # Load existing (Persistence working!)
        session = await session_service.get_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
        print("   ðŸ”„ Loaded EXISTING session from DB.")

    # Process queries
    for query_text in user_queries:
        print(f"\nUser > {query_text}")
        
        # Convert text to Content object
        msg = types.Content(role="user", parts=[types.Part(text=query_text)])

        # Run Agent
        async for event in runner_instance.run_async(
            user_id=USER_ID, session_id=session.id, new_message=msg
        ):
            if event.content and event.content.parts:
                text = event.content.parts[0].text
                if text and text != "None":
                    print(f"{MODEL_NAME} > {text}")

print("âœ… Helper functions ready.")


# 1. Define Tools List
research_tools = [search_medical_papers, save_research_note, read_my_notebook]

# 2. Define Agent
research_agent = LlmAgent(
    model=Gemini(model=MODEL_NAME),
    name="research_assistant",
    description="An AI that helps manage medical research.",
    tools=research_tools,
    instruction=(
        "You are a helpful Research Assistant for a final year medical student."
        "You have access to a tool to search for papers and a tool to save notes to a database."
        "1. Always search for information first if asked about a topic."
        "2. If the user finds a paper interesting, offer to save a summary using 'save_research_note'."
        "3. If the user asks 'what have we done?' or refers to past work, ALWAYS use 'read_my_notebook'."
    )
)

# 3. Define Session Service (The Chat Memory)
db_url = f"sqlite:///{DB_FILE}"
session_service = DatabaseSessionService(db_url=db_url)

# 4. Define Runner (The Orchestrator)
runner = Runner(
    agent=research_agent, 
    app_name=APP_NAME, 
    session_service=session_service
)

print("âœ… Agent Architecture Assembled (Fixed).")


# --- STEP 1: Research Phase ---
await run_session(
    runner,
    [
        "Can you find any papers on stroke management in rural areas?",
        "That second paper looks relevant. Save a note saying: 'Key reference for thesis chapter 2'."
    ],
    session_name="thesis_research_session"
)

print("\n" + "="*40)
print(" ðŸ•’ SIMULATING TIME PASSING (Restart/Break)")
print("="*40)

# --- STEP 2: Recall Phase (Proves Persistence) ---
await run_session(
    runner,
    ["What notes did I save earlier?"],
    session_name="thesis_research_session"
)

