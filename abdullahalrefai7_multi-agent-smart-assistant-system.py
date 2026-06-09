Project Architecture (ASCII Diagram):
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
      +---------------+
      | User Interface|
      +-------+-------+
              |
              v
      +---------------+
      |  Main Agent   |
      +-------+-------+
              |
      +-------+-------+
      |               |
      v               v
   +------+        +------+
   |Search|        |Image |
   |Agent |        |Agent |
   +------+        +------+
              |
              v
      +---------------+
      | Firestore     |
      | Session Store |
      +---------------+

Flow Summary:
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
User -> Main Agent -> Sub-Agents -> Firestore -> User


import asyncio
import os
import re
import hashlib
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import json
from io import BytesIO

# ADK Agents
from google.adk.agents import LlmAgent
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.models.google_llm import Gemini
from google.adk.sessions import BaseSessionService, Session
from google.adk.runners import Runner
from google.adk.events import Event, EventActions
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools import google_search
from google.genai import types

# Image Processing
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import requests

# Optional DuckDuckGo for image search
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    print("DuckDuckGo search library not available. Use: pip install duckduckgo-search")


# ==================== API Keys Setup ====================
os.environ["GEMINI_API_KEY"] = "<YOUR_GEMINI_API_KEY>"

# Email Settings
EMAIL_SENDER = "example@gmail.com"
EMAIL_PASSWORD = "<YOUR_EMAIL_PASSWORD>"

# Firebase Setup
db = None
FIREBASE_INITIALIZED = False

try:
    cred = credentials.Certificate("firebase.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    FIREBASE_INITIALIZED = True
    print("Firebase connected successfully")
except Exception as e:
    print(f"Failed to connect to Firebase: {e}")
    FIREBASE_INITIALIZED = False

# VerificationCodeManager, UserManager classes
# ... (Same as in your last code, English comments, no emoji)


# Image search function
def search_images(query: str):
    if not DDGS_AVAILABLE:
        return ["DuckDuckGo search not available"]
    try:
        all_links = []
        ddgs = DDGS()
        results = ddgs.images(query, max_results=5)
        for img in results:
            link = img.get("image") or img.get("thumbnail")
            if link:
                all_links.append(link)
        return all_links if all_links else ["No images found"]
    except Exception as e:
        return [f"Search error: {str(e)}"]

# Image processing function
def execute_image_processing(task_type: str, image_source: str, save_location: str = "./output.jpg",
                             text: Optional[str] = None, color: tuple = (255, 0, 0), **kwargs) -> Dict[str, Any]:
    # ... (Same as in last code, English comments)
    pass

# AppointmentManager class
# ... (Same as in last code, English comments)


# FirestoreSessionService with limited memory (last 10 sessions)
class FirestoreSessionService(BaseSessionService):
    def __init__(self, collection_name="agent_sessions"):
        super().__init__()
        if db is None:
            raise RuntimeError("Firestore not connected")
        self.collection = db.collection(collection_name)

    async def create_session(self, app_name: str, user_id: str, state: dict = None, session_id: str = None):
        doc_id = session_id or self.collection.document().id
        session = Session(id=doc_id, app_name=app_name, user_id=user_id, state=state or {})
        self.collection.document(doc_id).set({
            "appName": app_name,
            "userId": user_id,
            "state": state or {},
            "events": [],
            "created_at": datetime.now(timezone.utc)
        })
        return session

    async def append_event(self, session, event: Event):
        # Keep only last 10 events in memory
        session.events.append(event)
        if len(session.events) > 10:
            session.events = session.events[-10:]
        simple = {
            "id": event.id,
            "author": event.author,
            "text": event.content.parts[0].text if (event.content and event.content.parts) else "",
            "timestamp": event.timestamp
        }
        self.collection.document(session.id).update({"events": firestore.ArrayUnion([simple])})
        return event


# Gemini model
gemini_model = Gemini(model="gemini-2.5-flash")

# Image Agent
image_agent = LlmAgent(
    name="Image_Agent",
    model=gemini_model,
    tools=[search_images, execute_image_processing],
    instruction="You are an image expert. Search and process images precisely."
)

# Search Agent
search_agent = LlmAgent(
    name="Search_Agent",
    model=gemini_model,
    tools=[google_search, AgentTool(image_agent)],
    instruction="You are a smart search agent. Search accurately and collaborate with Image Agent when needed."
)

# Main Root Agent
root_agent = LlmAgent(
    name="Main_Agent",
    model=gemini_model,
    tools=[AgentTool(search_agent), AgentTool(image_agent)],
    instruction="""You are the main intelligent assistant. Your tasks:
    1. Understand user requests precisely
    2. Select the appropriate agent
    3. Provide clear and organized answers
    4. Collaborate between agents to complete complex tasks"""
)

# App initialization
app = App(
    name="smart_chat_system",
    root_agent=root_agent,
    events_compaction_config=EventsCompactionConfig(compaction_interval=5, overlap_size=1)
)


if FIREBASE_INITIALIZED:
    try:
        user_manager = UserManager(db)
        session_service = FirestoreSessionService()
        runner = Runner(app=app, session_service=session_service)
    except Exception as e:
        print(f"Initialization failed: {e}")
        user_manager = None
        session_service = None
        runner = None
else:
    user_manager = None
    session_service = None
    runner = None


# async functions: signup_flow, signin_flow, show_user_dashboard, manage_appointments, continuous_chat
# ... (Same as last code with English comments, no emoji)


if __name__ == "__main__":
    if FIREBASE_INITIALIZED and user_manager and runner:
        try:
            print("Starting Smart Chat System...")
            asyncio.run(main())
        except KeyboardInterrupt:
            print("Program stopped by user.")
        except Exception as e:
            print(f"Error occurred: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("System failed to start")
        if not FIREBASE_INITIALIZED:
            print("â€¢ Firebase not connected")
        if not user_manager:
            print("â€¢ UserManager not available")
        if not runner:
            print("â€¢ Runner not available")
        print("Ensure:")
        print("1. firebase.json is in the correct path")
        print("2. All required libraries are installed")
        print("3. GEMINI_API_KEY is correct")


# ==================== Evaluation & Metrics ====================
class EvaluationManager:
    def __init__(self, db):
        self.db = db
        self.collection = db.collection("evaluation_logs")

    async def log_event(self, user_id: str, session_id: str, event_type: str, content: str):
        """Save each event for evaluation and analytics"""
        log_doc = {
            "user_id": user_id,
            "session_id": session_id,
            "event_type": event_type,  # e.g., 'user_message', 'agent_response'
            "content": content,
            "timestamp": datetime.now(timezone.utc)
        }
        self.collection.add(log_doc)

    async def get_user_metrics(self, user_id: str):
        """Generate summary metrics for a user"""
        docs = self.collection.where("user_id", "==", user_id).stream()
        events = [doc.to_dict() for doc in docs]

        total_user_messages = sum(1 for e in events if e['event_type'] == 'user_message')
        total_agent_responses = sum(1 for e in events if e['event_type'] == 'agent_response')
        sessions = list(set(e['session_id'] for e in events))

        return {
            "total_user_messages": total_user_messages,
            "total_agent_responses": total_agent_responses,
            "total_sessions": len(sessions)
        }

    async def list_sessions(self, user_id: str):
        """Return all session IDs for a user"""
        docs = self.collection.where("user_id", "==", user_id).stream()
        return list(set(doc.to_dict().get("session_id") for doc in docs))


# ==================== Logging Wrapper ====================
evaluation_manager = EvaluationManager(db)

async def log_user_event(user_id: str, session_id: str, user_input: str, agent_response: str):
    """Log both user input and agent response for evaluation"""
    await evaluation_manager.log_event(user_id, session_id, "user_message", user_input)
    await evaluation_manager.log_event(user_id, session_id, "agent_response", agent_response)


# In continuous_chat(), after sending user message and receiving agent response:

async for ev in runner.run_async(user_id=user_id, session_id=session.id, new_message=user_content):
    response = "".join(p.text or "" for p in ev.content.parts) if (ev.content and ev.content.parts) else ""
    if response:
        print(response, end="", flush=True)
        # Log evaluation
        await log_user_event(user_id, session.id, user_input, response)
    await session_service.append_event(session, ev)


# ==================== Display User Metrics ====================
async def show_user_metrics(user_id: str):
    metrics = await evaluation_manager.get_user_metrics(user_id)
    print("\n" + "="*60)
    print(f"ðŸ“Š Evaluation Metrics for User ID: {user_id}")
    print("="*60)
    print(f"Total user messages: {metrics['total_user_messages']}")
    print(f"Total agent responses: {metrics['total_agent_responses']}")
    print(f"Total sessions: {metrics['total_sessions']}")
    print("="*60 + "\n")

