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


# =================================================================
# SETUP: Install and Import necessary libraries
# =================================================================
# Install nest_asyncio to prevent the "RuntimeError: asyncio.run() cannot be called from a running event loop"
try:
    import nest_asyncio
except ImportError:
    print("Installing nest_asyncio...")
    !pip install nest_asyncio
    import nest_asyncio

import os
import sqlite3
import threading
import random
import logging
import asyncio
import re
import json
import warnings
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# --- LLM Imports ---
from google import genai
from google.genai import types
from kaggle_secrets import UserSecretsClient


# ---------- Logging and Authentication ----------
'''
logging.basicConfig(level=logging.NOTSET) 
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('google').setLevel(logging.WARNING)
logging.getLogger('google_genai').setLevel(logging.WARNING)'''
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UnifiedSupport")
warnings.filterwarnings("ignore") # Hide warnings

# Check for and set the Gemini API key
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )

DB_PATH = "/kaggle/working/customer_data.db"
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
    
# Global model initialization
try:
    gemini_client = genai.Client()
    CLASSIFICATION_MODEL = "gemini-2.5-pro"
    GENERATION_MODEL = "gemini-2.5-pro"
    print(f"âœ… Gemini Client initialized using {CLASSIFICATION_MODEL}.")
except Exception as e:
    print(f"â�Œ Failed to initialize Gemini Client: {e}")



class DatabaseManager:
    """Manages SQLite connections and operations for customer data, tickets, and known issues."""
    def __init__(self, path=DB_PATH):
        self.path = path
        self._lock = threading.RLock()
        # Set a longer timeout for notebook environments
        self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30.0) 
        with self._lock:
            self.conn.execute("PRAGMA journal_mode=WAL;")
        self._init_schema()

    def _init_schema(self):
        """Initializes the database tables and seeds sample data."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cur = self.conn.cursor()
            
            # Simplified schema creation (re-used your original tables)
            # tables: users, orders, tickets, known_issues, feedback, sessions
            cur.execute("""CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, email TEXT UNIQUE NOT NULL, subscription_tier TEXT, renewal_date TEXT, active INTEGER DEFAULT 1, balance REAL DEFAULT 0.0, created_at TEXT, updated_at TEXT);""")
            cur.execute("""CREATE TABLE IF NOT EXISTS orders (order_id TEXT PRIMARY KEY, user_id INTEGER, product TEXT, quantity INTEGER, status TEXT, created_at TEXT, updated_at TEXT, FOREIGN KEY(user_id) REFERENCES users(id));""")
            cur.execute("""CREATE TABLE IF NOT EXISTS tickets (ticket_id TEXT PRIMARY KEY, session_id TEXT, user_id INTEGER, intent TEXT, message TEXT, status TEXT, resolution TEXT, created_at TEXT, updated_at TEXT, FOREIGN KEY(user_id) REFERENCES users(id));""")
            cur.execute("""CREATE TABLE IF NOT EXISTS known_issues (issue_key TEXT PRIMARY KEY, title TEXT, category TEXT, fix TEXT, confidence_boost REAL, customer_id INTEGER, created_at TEXT, updated_at TEXT, FOREIGN KEY(customer_id) REFERENCES users(id));""")
            cur.execute("""CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, ticket_id TEXT, user_id INTEGER, intent TEXT, confidence REAL, reasoning TEXT, status TEXT, created_at TEXT, FOREIGN KEY(user_id) REFERENCES users(id));""")
            cur.execute("""CREATE TABLE IF NOT EXISTS sessions (session_id TEXT PRIMARY KEY, user_id INTEGER, created_at TEXT, last_active TEXT, FOREIGN KEY(user_id) REFERENCES users(id));""")
            self.conn.commit()

            # Seed data
            cur.execute("SELECT COUNT(*) FROM users")
            if cur.fetchone()[0] == 0:
                users = [
                    ("Alice", "alice@example.com", "PLATINUM", "2025-12-31", 1, 100.0),
                    ("Bob", "bob@example.com", "GOLD", "2024-06-30", 1, 50.0),
                    ("Dave", "dave@example.com", "GOLD", "2025-09-15", 1, 75.0),
                    ("Alan Smith", "alan@example.com", "PLATINUM", "2025-12-31", 1, 200.0)
                ]
                for u in users:
                    cur.execute("""INSERT INTO users (name,email,subscription_tier,renewal_date,active,balance,created_at,updated_at) 
                                 VALUES (?,?,?,?,?,?,?,?)""", (*u, now, now))
                
                # Sample orders (Alice has ID 1, Alan Smith has ID 4)
                cur.execute("""INSERT OR IGNORE INTO orders (order_id,user_id,product,quantity,status,created_at,updated_at) 
                                 VALUES (?,?,?,?,?,?,?)""", ("2024-03-A-1234", 1, "Wireless Mouse", 1, "Processing", now, now))
                cur.execute("""INSERT OR IGNORE INTO orders (order_id,user_id,product,quantity,status,created_at,updated_at) 
                                 VALUES (?,?,?,?,?,?,?)""", ("2024-10-K-9876", 4, "Mechanical Keyboard", 1, "Shipped", now, now))

                # Known issues
                issues = [
                    ("api-auth-401","API 401 Unauthorized Error","API Failure", "Check API key validity and scope.", 0.8, 4),
                    ("latency-eu","Latency in EU Region","Performance Issue", "Check regional server status and load.", 0.6, 2),
                    ("api-rate-limit","API Rate Limit Exceeded","API Failure", "Increase rate limit or reduce request frequency.", 0.9, None)
                ]
                for i in issues:
                    cur.execute("""INSERT OR REPLACE INTO known_issues (issue_key,title,category,fix,confidence_boost,customer_id,created_at,updated_at) 
                                 VALUES (?,?,?,?,?,?,?,?)""", (*i, now, now))
                self.conn.commit()

    # --- Users ---
    def find_user(self, identifier: Any) -> Optional[Dict[str,Any]]:
        with self._lock:
            cur = self.conn.cursor()
            if isinstance(identifier, int) or (isinstance(identifier, str) and identifier.isdigit()):
                cur.execute("SELECT * FROM users WHERE id=?", (str(identifier),))
            else:
                if not isinstance(identifier, str): return None
                cur.execute("SELECT * FROM users WHERE LOWER(email)=? OR LOWER(name)=?", (identifier.lower(), identifier.lower()))
            row = cur.fetchone()
            if not row: return None
            cols = [c[0] for c in cur.description]
            return dict(zip(cols, row))

    def get_last_order_by_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT * FROM orders WHERE user_id=? ORDER BY created_at DESC LIMIT 1", (user_id,))
            row = cur.fetchone()
            return dict(zip([c[0] for c in cur.description], row)) if row else None
        
    # --- Orders ---
    def get_order(self, order_id: str) -> Optional[Dict[str,Any]]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT * FROM orders WHERE order_id=?", (order_id,))
            row = cur.fetchone()
            return dict(zip([c[0] for c in cur.description], row)) if row else None

    def update_order_status(self, order_id: str, **updates) -> bool:
        with self._lock:
            cur = self.conn.cursor()
            sets = [f"{k}=?" for k in updates.keys()]
            vals = list(updates.values())
            if not sets: return False
            vals.append(datetime.now(timezone.utc).isoformat())
            vals.append(order_id)
            sql = f"UPDATE orders SET {', '.join(sets)}, updated_at=? WHERE order_id=?"
            cur.execute(sql, tuple(vals))
            self.conn.commit()
            return cur.rowcount > 0

    # --- Sessions ---
    def create_or_touch_session(self, session_id: str, user_id: int):
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("""
                INSERT INTO sessions (session_id, user_id, created_at, last_active) 
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET 
                    last_active=excluded.last_active, 
                    user_id=CASE 
                        WHEN excluded.user_id > 0 THEN excluded.user_id 
                        ELSE sessions.user_id 
                    END 
            """, (session_id, user_id, now, now))
            self.conn.commit()

    # --- Tickets ---
    def create_ticket(self, ticket_id: str, session_id: str, user_id: int, intent: str, message: str, status: str="New"):
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("""INSERT INTO tickets (ticket_id,session_id,user_id,intent,message,status,created_at,updated_at) 
                             VALUES (?,?,?,?,?,?,?,?)""", (ticket_id,session_id,user_id,intent,message,status,now,now))
            self.conn.commit()
        
    def update_ticket(self, ticket_id: str, **updates):
        with self._lock:
            cur = self.conn.cursor()
            sets = [f"{k}=?" for k in updates.keys()]
            vals = list(updates.values())
            if not sets: return False
            vals.append(datetime.now(timezone.utc).isoformat())
            vals.append(ticket_id)
            sql = f"UPDATE tickets SET {', '.join(sets)}, updated_at=? WHERE ticket_id=?"
            cur.execute(sql, tuple(vals))
            self.conn.commit()
            return cur.rowcount > 0

    # --- Feedback ---
    def save_feedback(self, ticket_id: str, user_id: int, intent: str, confidence: float, reasoning: str, status: str):
        with self._lock:
            cur = self.conn.cursor()
            ts = datetime.now(timezone.utc).isoformat()
            cur.execute("""INSERT INTO feedback (ticket_id,user_id,intent,confidence,reasoning,status,created_at) 
                             VALUES (?,?,?,?,?,?,?)""", (ticket_id, user_id, intent, confidence, reasoning, status, ts))
            self.conn.commit()

    def list_recent_feedback(self, limit=10) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT * FROM feedback ORDER BY id DESC LIMIT ?", (limit,))
            rows = cur.fetchall()
            return [dict(zip([c[0] for c in cur.description], r)) for r in rows]

    # --- Known issues ---
    def list_known_issues_dicts(self) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT issue_key, title, category, fix, confidence_boost, customer_id FROM known_issues")
            rows = cur.fetchall()
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, r)) for r in rows]


class Ticket:
    """A data structure to hold information about a customer support ticket."""
    def __init__(self, session_id: str, user_ref: str, message: str, intent: str = "Unknown"):
        self.ticket_id = f"TKT-{random.randint(10000,99999)}"
        self.session_id = session_id
        self.user_ref = user_ref
        self.user = None
        self.intent = intent
        self.message = message
        self.status = "New"
        self.resolution = None
        self.context = {} 
        self.created_at = datetime.now(timezone.utc).isoformat()




class IntentClassifierOutput(BaseModel):
    """Schema for classifying intent and extracting user reference."""
    intent: str = Field(description="The primary intent. One of TECH, ORDER, FAQ, SUPPORT, or GENERAL.")
    user_ref: Optional[str] = Field(description="The user's name or email address extracted from the message, if provided. If the user is correcting their name, this should be the corrected name.")

def llm_classify_intent_and_user(message: str) -> Dict[str, Any]:
    """Uses an LLM to determine intent and extract a user reference."""
    
    tool = types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="classify_input",
            description="Classifies the user's message and extracts the user reference.",
            parameters=IntentClassifierOutput.schema(),
        )
    ])

    system_instruction = (
        "You are an expert customer service classification agent. Your task is to analyze the "
        "user's message and categorize their intent into one of the following, without generating conversational text: "
        "'TECH' (for errors, bugs, system issues), "
        "'ORDER' (for tracking, canceling, delivery, returns), "
        "'FAQ' (for general policy questions), "
        "'SUPPORT' (for formal complaints or escalation), or "
        "'GENERAL' (for greetings, unclear queries, or name corrections). "
        "Extract any identifiable user information (name or email) into user_ref. "
        "Always respond by calling the 'classify_input' tool with structured JSON output."
    )
    
    try:
        response = gemini_client.models.generate_content(
            model=CLASSIFICATION_MODEL,
            contents=[message],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[tool],
                temperature=0.0
            ),
        )
        
        if response.function_calls:
            args = response.function_calls[0].args
            intent = args.get("intent", "GENERAL").upper() 
            user_ref = args.get("user_ref")

            return {"intent": intent, "user_ref": user_ref or ""}

    except Exception as e:
        logger.error(f"LLM Classification Failed: {e}")
        return {"intent": "GENERAL", "user_ref": ""}


class UnifiedAgents:
    """Contains specialized agents for handling different ticket intents."""
    def __init__(self, db_manager: DatabaseManager, llm_client: Any, llm_config: Any, logger: logging.Logger):
        self.db = db_manager
        self.llm = llm_client
        self.config = llm_config
        self.logger = logger
    
    def faq_agent(self, ticket: Ticket):
        """Handles frequently asked questions."""
        if "return" in ticket.message.lower():
            reply = "Our return policy: items can be returned within 30 days with receipt. Refunds processed within 5-7 business days."
        else:
            reply = "FAQ: Please check our help center for details."
        return {"reply": reply, "status": "Answered"}

    def order_agent(self, ticket: Ticket) -> Dict[str, str]:
        """Handles order-related inquiries."""
        user = ticket.user
        if not user:
            return {"reply": "Authentication Required: Your order query requires account identification. Please state your name or email.", "status": "Failed: No User Context"}
        
        user_id = user["id"]
        ORDER_ID_PATTERN = r"\d{4}-\d{2}-[A-Z]-\d{4}" 
        order = None
        order_id_ref = None

        # a. Try to extract an explicit Order ID
        order_id_match = re.search(ORDER_ID_PATTERN, ticket.message)
        if order_id_match:
            order_id_ref = order_id_match.group(0)
            order = self.db.get_order(order_id_ref)
            if order and order.get("user_id") != user_id:
                return {"reply": f"The order ID **{order_id_ref}** was found, but it is not associated with the authenticated account of **{user['name']}**.", "status": "Failed: Security Mismatch"}
        
        # b. Default to the last order for the user
        if not order:
            order = self.db.get_last_order_by_user(user_id) 
        
        # --- Failure: No Order Found ---
        if not order:
            name = user['name'] if user and 'name' in user else f"User ID: {user_id}"
            if order_id_ref:
                 return {"reply": f"Order ID **{order_id_ref}** was not found in our system for user **{name}**. Please verify the number.", "status": "Failed: Order Not Found"}
            return {"reply": f"No recent orders were found for user **{name}**. Please provide the specific order ID or place an order first.", "status": "Failed: No Order"}

        # --- Order Action Processing ---
        order_id = order["order_id"]
        order_status = order["status"]
        m = ticket.message.lower()
        
        if "cancel" in m or "stop" in m:
            if order_status in ["Delivered", "Cancelled"]:
                reply = f"Order **{order_id}** (Status: {order_status}) cannot be cancelled. It is already {order_status.lower()}."
                return {"reply": reply, "status": "Failed: Invalid Status"}

            self.db.update_order_status(order_id, status="Cancelled") 
            reply = f"ORDER **{order_id}** for {order.get('product', 'product')} was **cancelled**. A refund will be processed shortly."
            return {"reply": reply, "status": "OK"}

        if "delivery date" in m or "change the date" in m or "change delivery" in m:
            if order_status in ["Delivered", "Cancelled"]:
                reply = f"Order **{order_id}** (Status: {order_status}) cannot have its delivery date changed. It is already {order_status.lower()}."
                return {"reply": reply, "status": "Failed: Invalid Status"}
            
            reply = f"I see you want to change the delivery date for order **{order_id}**. Please provide the **new preferred date**."
            return {"reply": reply, "status": "Pending: Date Needed"}

        if "status" in m or "where is" in m or "track" in m:
            reply = f"The current status for order **{order_id}** ({order.get('product', 'product')}) is **{order_status}**. It is scheduled for delivery on {order.get('delivery_date', 'an unconfirmed date')}."
            return {"reply": reply, "status": "Answered"}
            
        return {"reply": f"Order **{order_id}** found. Could you specify your request (e.g., cancel, check status, or change delivery)?", "status": "Failed: No Action Recognized"}

    def tech_agent(self, ticket: Ticket):
        """Handles technical issues, leveraging the Known Issues knowledge base (KB)."""
        kb_match = ticket.context.get("kb_match")
        confidence = ticket.context.get("confidence_score", 0.5)

        if kb_match:
            reply = f"**Known Issue:** {kb_match['title']}. **Resolution:** {kb_match['fix']}"
        else:
            reply = "General troubleshooting: try restarting the affected service and checking any available system logs. If the issue persists, a human agent will review your ticket."
            confidence = 0.5
        
        ticket.context["diagnostic_reasoning"] = kb_match["fix"] if kb_match else reply
        ticket.context["confidence_score"] = confidence

        return {"reply": reply, "confidence": confidence, "status": "Answered"}

    def support_agent(self, ticket: Ticket):
        """Handles formal complaints and escalation requests."""
        return {"reply": f"Support: We've noted your formal complaint/escalation request (Ticket ID: **{ticket.ticket_id}**) and will follow up with you within 4 business hours.", "status": "Escalated"}
    
    def general_agent_llm(self, ticket: Ticket):
        """Generates a conversational response for general/unrouted inquiries using the LLM."""
        
        is_auth_needed = ticket.user is None and ticket.intent != "FAQ"
        
        system_instruction = (
            "You are a helpful, polite, and professional customer service agent. "
            f"The user's input is: '{ticket.message}'. "
        )

        if is_auth_needed:
             system_instruction += (
                "The user is requesting a specific action (e.g., Order, Tech, Support, or FAQ) but has not been authenticated. "
                "Your **primary goal** is to provide a brief, polite response that **immediately requests their name or email address**. "
                "Explain that this is necessary to identify their account and proceed with their request. "
                "Do NOT attempt to answer their original question, but acknowledge the type of request (e.g., 'I see you're trying to cancel an order')."
            )
        else:
             system_instruction += (
                "You should provide a brief, helpful, and conversational response. Acknowledge greetings or simple statements. "
                "If the user is correcting or confirming information (e.g., 'Correct user name is Alan Smith'), confirm the name has been noted and ask how you can help next. "
                "If the query is a simple, non-account-specific question, you can answer it briefly. "
                "For complex or highly technical questions, provide a brief, standard acknowledgment (e.g., 'I understand you're facing timeout errors') and then suggest a next step or ask for more detail."
            )

        try:
            local_config = types.GenerateContentConfig(
                temperature=self.config.temperature,
                system_instruction=system_instruction
            )

            response = self.llm.models.generate_content(
                model=GENERATION_MODEL,
                contents=[{"role": "user", "parts": [{"text": ticket.message}]}],
                config=local_config,
            )
            
            reply_text = response.text.strip()
            status = "LLM Answered"
            
            # If we were requesting auth, change status for logging
            if is_auth_needed:
                 status = "Authentication Requested"
            
            return {"reply": reply_text, "status": status}
        
        except Exception as e:
            self.logger.error(f"LLM Generation Failed for General Agent: {e}")
            return {"reply": "Sorry, I am currently experiencing system difficulties. Please try again in a moment.", "status": "Failed: LLM Error"}



class OrchestratorRunner:
    """The central component that classifies intent, routes the request, and logs the outcome."""
    def __init__(self, db_manager: DatabaseManager, agents: UnifiedAgents):
        self.db = db_manager
        self.agents = agents

    async def run_session(self, session_id: str, message: str):
        """Processes a single customer message."""
        # 1. Initial Classification (NOW USING LLM)
        cls = llm_classify_intent_and_user(message)
        intent = cls["intent"]
        user_ref = cls["user_ref"]
        user = self.db.find_user(user_ref) if user_ref else None
        
        # --- Session History Check: Load user from session if not found by user_ref ---
        if not user:
            with self.db._lock:
                cur = self.db.conn.cursor()
                # Need to handle case where session_id doesn't exist yet
                cur.execute("SELECT user_id FROM sessions WHERE session_id=?", (session_id,))
                session_user_id_tuple = cur.fetchone()
            
            if session_user_id_tuple:
                session_user_id = session_user_id_tuple[0]
                if session_user_id != 0:
                    user_from_session = self.db.find_user(session_user_id) 
                    if user_from_session:
                        user = user_from_session
                        logger.info(f"Context: Reusing user {user['name']} from session {session_id}")

        # Create a new ticket object
        ticket = Ticket(session_id, user_ref, message, intent)
        ticket.user = user
        user_id = user["id"] if user else 0

        # --- MANDATORY USER CHECK/Authentication Required ---
        if user_id == 0 and intent != "FAQ" and not (intent == "GENERAL" and not user_ref):
            # If user is not found and intent requires ID (i.e., not a general policy FAQ or a simple GENERAL greeting), ask for ID.
            
            # Temporarily set intent to GENERAL for LLM agent to handle auth request
            original_intent = intent
            ticket.intent = "GENERAL" 
            
            result = self.agents.general_agent_llm(ticket)
            
            ticket.status = result.get("status", "Failed")
            ticket.resolution = result.get("reply")
            
            # Log session and ticket creation/failure
            self.db.create_or_touch_session(session_id, user_id) 
            self.db.create_ticket(ticket.ticket_id, session_id, user_id, original_intent, message)
            
            self.db.save_feedback(
                ticket.ticket_id, 
                user_id, 
                original_intent, 
                0.0, 
                "Failed due to missing user authentication. Routed to LLM for prompt.", 
                ticket.status
            )
            self.db.update_ticket(ticket.ticket_id, status=ticket.status, resolution=ticket.resolution)

            return {"ticket_id": ticket.ticket_id, "intent": original_intent, "reply": ticket.resolution, "status": ticket.status}

        # 2. Knowledge Base (KB) Override for TECH/High-Confidence Issues
        kb_match = None
        
        if intent == "TECH": # Only search KB for TECH intents
            for kb in self.db.list_known_issues_dicts():
                keywords = []
                for field in ["title","category","issue_key"]:
                    val = kb.get(field,"")
                    if val:
                        keywords.extend(val.lower().split())
                
                # Check for keyword match
                if any(word in message.lower() for word in keywords):
                    kb_match = kb
                    ticket.context["kb_match"] = kb_match
                    ticket.context["diagnostic_reasoning"] = kb_match["fix"]
                    
                    confidence = kb.get("confidence_boost", 0.5)
                    ticket.context["confidence_score"] = confidence
                    
                    if confidence >= 0.75: 
                        ticket.intent = "TECH" # Re-affirm TECH for high-confidence match
                        intent = "TECH"
                    break 

        # 3. Database Operations (Session and Ticket Creation)
        self.db.create_or_touch_session(session_id, user_id)
        self.db.create_ticket(ticket.ticket_id, session_id, user_id, intent, message)

        # 4. Route to Specialized Agent
        if intent == "FAQ":
            agent_name = "FAQ_Agent"
            result = self.agents.faq_agent(ticket)
        elif intent == "ORDER":
            agent_name = "Order_Agent"
            result = self.agents.order_agent(ticket)
        elif intent == "TECH":
            agent_name = "Tech_Agent"
            result = self.agents.tech_agent(ticket)
        elif intent == "SUPPORT":
            agent_name = "Support_Agent"
            result = self.agents.support_agent(ticket)
        else:
            # General Intent now uses the LLM agent for conversational reply
            agent_name = "General_Agent"
            result = self.agents.general_agent_llm(ticket)


        # 5. Log Feedback and Update Ticket Status
        ticket.status = result.get("status", "Answered")
        ticket.resolution = result.get("reply")
        
        self.db.save_feedback(
            ticket.ticket_id, 
            user_id, 
            ticket.intent, 
            ticket.context.get("confidence_score", 0.5), 
            ticket.context.get("diagnostic_reasoning", "N/A"), 
            ticket.status 
        )
        
        self.db.update_ticket(ticket.ticket_id, status=ticket.status, resolution=ticket.resolution)

        return {"ticket_id": ticket.ticket_id, "intent": ticket.intent, "reply": result.get("reply"), "status": ticket.status, "agent": agent_name}


db = DatabaseManager()
llm_config = types.GenerateContentConfig(temperature=0.0)
agents = UnifiedAgents(db, llm_client=gemini_client, llm_config=llm_config, logger=logger)
orchestrator = OrchestratorRunner(db, agents)

async def run_session_case(orchestrator, msg, sid):
    """Wrapper function to run the session for the demo."""
    return await orchestrator.run_session(sid, msg)

async def demo_runs():
    """Runs a series of demo cases to demonstrate the system's routing and response logic."""
    print("ğŸ¤– === Demo: LLM-Integrated combined system ===\n")
    
    tests = [
        ("test-session-01", "Hi, I'm Alan Solly. I need to change the delivery date for my mouse order."), # Needs user ref (Alan Solly/Smith), then Order Agent
        ("test-session-01", "Correct user name is Alan Smith"), # Should update session context to Alan Smith, use General Agent
        ("test-session-01", "I want to cancel my mouse order."), # Uses Alan Smith's context, runs Order Agent
        ("test-session-02", "What is your platform's return policy for products purchased in the last 30 days?"), # Runs FAQ Agent (No auth needed)
        ("test-session-03", "Alan here. My keyboard that I ordered last month is completely dead. I want to raise a formal complaint."), # Needs auth (Alan), runs Support Agent
        ("test-session-04", "I received an email about a legal document update. Can you explain the implications of this change?"), # Runs General Agent
        ("test-session-05", "My API key stopped working with 401 error. Please help."), # Runs Tech Agent (High-confidence KB match)
        ("test-session-06", "I'm dave.We're seeing latency in the EU region causing timeouts."), # Needs auth (Dave), runs Tech Agent (KB match)
        ("test-session-07", "I'm Alice.I want to cancel my recent order 2024-03-A-1234."), # Needs auth (Alice), runs Order Agent (Specific Order)
        ("test-session-08", "Latency in EU region is high."), # Needs auth, but falls to General Agent (due to no user ref in message)
        ("test-session-09", "Exceeded API rate limits."), # Runs Tech Agent (High-confidence KB match)
        ("test-session-10", "Timeout errors on API endpoints."), # Runs Tech Agent
        ("test-session-07", "I want to cancel my recent order 2024-03-A-1234."), # Uses Alice's context, runs Order Agent
        ("test-session-12", "I want to check what is the general delivery time in your company?"), # Runs General Agent
        ("test-session-13", "what is the mode of delivery you are having"), # Runs General Agent
    ]

    async def run_case(session_id, text):
        # FIX: Introduce a 1.5 second pause to avoid hitting the 429 API rate limit
        await asyncio.sleep(1.5) 
        
        resp = await orchestrator.run_session(session_id, text) # Simplified call
        
        # Check if the user was authenticated for display purposes
        # Note: session_user_id is extracted from the DB session table using the session_id
        user_id_tuple = orchestrator.db.conn.cursor().execute("SELECT user_id FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        user_id = user_id_tuple[0] if user_id_tuple else 0
        user_info = orchestrator.db.find_user(user_id)
        user_name = user_info['name'] if user_info and user_info['id'] != 0 else "Unauthenticated"
        
        print(f"\n[{session_id}] **User: {user_name}** | Input: **{text}**")
        print(f"[{session_id}] -> **{resp['intent']}** Agent: {resp.get('agent', 'N/A')} | Reply: {resp['reply']} (Status: **{resp['status']}**)")
        print(f"---")
        return resp

    # Run all cases concurrently using asyncio.gather
    print("Starting concurrent execution of all 15 demo cases...")
    await asyncio.gather(*(run_case(s, t) for s, t in tests))

    # Unified feedback + learning output
    print("\n" + "="*50)
    print("ğŸ“‹ Recent Feedback Entries (for analysis/learning):")
    print("="*50)
    for fb in db.list_recent_feedback(limit=10):
        # Retrieve user name for clearer logging
        user_info = db.find_user(fb['user_id'])
        user_name = user_info['name'] if user_info and user_info['id'] != 0 else "Unauthenticated"
        
        print(f"  TKT: {fb['ticket_id'][-10:]} | Intent: {fb['intent']} | Status: {fb['status']} | User: {user_name}")
        print(f"  Confidence: {fb['confidence']:.2f} | Reasoning: {fb['reasoning'][:80]}...")
        print("-" * 40)
            
    print("\n=== Demo complete ===")


if __name__ == '__main__':
    # Patch the asyncio library to allow nested event loops (required for notebooks)
    nest_asyncio.apply()
    
    # Run the asynchronous demo using asyncio.run()
    asyncio.run(demo_runs())




