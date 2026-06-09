import sys
import os
import time
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings('ignore')

import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool
from kaggle_secrets import UserSecretsClient
from IPython.display import display, HTML, clear_output

print("âœ“ Libraries Loaded")


# ============================================================
# HealthGuard Config - API & Agent settings
# ============================================================

import google.generativeai as genai
from kaggle_secrets import UserSecretsClient

# Load API Key from Kaggle Secrets
try:
    user_secrets = UserSecretsClient()
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
    print("âœ“ API Key Configured")
except Exception as e:
    print(f"âš  API Key Error: {str(e)}")
    GOOGLE_API_KEY = None

# Agent Configuration
CONFIG = {
    "team": "HealthGuard",
    "model": "models/gemini-2.5-flash",
    "max_tokens": 2000,
    "temperature": 0.3,
    "version": "2.0.0"
}

print(f"\n{'='*60}")
print(f"{'HEALTHGUARD CONFIGURATION':^60}")
print(f"{'='*60}")
for k, v in CONFIG.items():
    print(f"{k:.<25} {v}")
print(f"{'='*60}")



# Create healthguard_config.py in the current working directory
config_code = """
import google.generativeai as genai
from kaggle_secrets import UserSecretsClient

# Load API Key from Kaggle Secrets
try:
    user_secrets = UserSecretsClient()
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
    print("âœ“ API Key Configured")
except Exception as e:
    print(f"âš  API Key Error: {str(e)}")
    GOOGLE_API_KEY = None

CONFIG = {
    "team": "HealthGuard",
    "model": "models/gemini-2.5-flash",
    "max_tokens": 2000,
    "temperature": 0.3,
    "version": "2.0.0"
}

print(f"âœ“ CONFIG Loaded: {CONFIG}")
"""

# Write to file
with open("healthguard_config.py", "w") as f:
    f.write(config_code)

print("âœ“ healthguard_config.py created!")



from healthguard_config import CONFIG
print(CONFIG)


from google.generativeai.types import FunctionDeclaration, Tool

# -----------------------------
# HealthGuard Agent Function Declarations
# -----------------------------
function_declarations = [
    FunctionDeclaration(
        name="check_vitals",
        description="Checks real-time vitals such as BP, heart rate, SpO2 from user data",
        parameters={
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "Unique ID of the user"},
                "data": {"type": "object", "description": "Dictionary of sensor readings: bp, hr, spo2"}
            },
            "required": ["user_id", "data"]
        }
    ),
    FunctionDeclaration(
        name="analyze_fitness_activity",
        description="Analyzes user's activity data and provides fitness insights",
        parameters={
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "Unique ID of the user"},
                "activity_log": {"type": "array", "description": "List of activity entries with timestamps"}
            },
            "required": ["user_id", "activity_log"]
        }
    ),
    FunctionDeclaration(
        name="predict_health_risk",
        description="Predicts potential health risks based on vitals and activity data",
        parameters={
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "Unique ID of the user"},
                "vitals": {"type": "object", "description": "User's vitals readings"},
                "activity_log": {"type": "array", "description": "Recent activity data"}
            },
            "required": ["user_id", "vitals", "activity_log"]
        }
    ),
    FunctionDeclaration(
        name="suggest_health_improvements",
        description="Suggests personalized health improvements for the user",
        parameters={
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "Unique ID of the user"},
                "vitals": {"type": "object", "description": "User's vitals readings"},
                "activity_log": {"type": "array", "description": "Recent activity data"},
                "preferences": {"type": "string", "description": "User preferences for diet/exercise"}
            },
            "required": ["user_id", "vitals", "activity_log"]
        }
    ),
    FunctionDeclaration(
        name="generate_health_report",
        description="Generates a summary report of user's health and activity data",
        parameters={
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "Unique ID of the user"},
                "timeframe_days": {"type": "integer", "description": "Number of days to include in the report"}
            },
            "required": ["user_id", "timeframe_days"]
        }
    )
]

tools = Tool(function_declarations=function_declarations)
print(f"âœ“ HealthGuard Function Declarations Created ({len(function_declarations)} tools)")



import json
import os
from typing import Dict, Any, List

MEMORY_FILE = "healthguard_memory.json"

# -----------------------------
# Memory System for HealthGuard
# -----------------------------
class HealthGuardMemory:
    def __init__(self, memory_file: str = MEMORY_FILE):
        self.memory_file = memory_file
        self.memory: Dict[str, Any] = {}
        self.load_memory()
        print(f"âœ“ HealthGuard Memory Initialized ({len(self.memory)} users loaded)")

    def load_memory(self):
        """Load memory from JSON file"""
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r") as f:
                self.memory = json.load(f)
        else:
            self.memory = {}

    def save_memory(self):
        """Save memory to JSON file"""
        with open(self.memory_file, "w") as f:
            json.dump(self.memory, f, indent=2)

    def store_user_vitals(self, user_id: str, vitals: Dict[str, Any]):
        """Store or update user's vitals"""
        user_data = self.memory.get(user_id, {})
        user_data["vitals"] = vitals
        self.memory[user_id] = user_data
        self.save_memory()
        print(f"âœ“ Stored vitals for user {user_id}")

    def store_user_activity(self, user_id: str, activity_log: List[Dict[str, Any]]):
        """Store or update user's activity log"""
        user_data = self.memory.get(user_id, {})
        user_data["activity_log"] = activity_log
        self.memory[user_id] = user_data
        self.save_memory()
        print(f"âœ“ Stored activity log for user {user_id}")

    def store_user_predictions(self, user_id: str, predictions: Dict[str, Any]):
        """Store health risk predictions"""
        user_data = self.memory.get(user_id, {})
        user_data["predictions"] = predictions
        self.memory[user_id] = user_data
        self.save_memory()
        print(f"âœ“ Stored predictions for user {user_id}")

    def store_health_report(self, user_id: str, report: str):
        """Store generated health report"""
        user_data = self.memory.get(user_id, {})
        user_data["report"] = report
        self.memory[user_id] = user_data
        self.save_memory()
        print(f"âœ“ Stored health report for user {user_id}")

    def get_user_data(self, user_id: str) -> Dict[str, Any]:
        """Retrieve all stored data for a user"""
        return self.memory.get(user_id, {})


# -----------------------------
# Example usage (for testing)
# -----------------------------
if __name__ == "__main__":
    memory = HealthGuardMemory()

    # Test storing vitals
    memory.store_user_vitals("user123", {"bp": "120/80", "hr": 72, "spo2": 98})

    # Test storing activity
    memory.store_user_activity("user123", [{"time": "08:00", "activity": "walking"}])

    # Test storing predictions
    memory.store_user_predictions("user123", {"risk": "low", "advice": "keep walking daily"})

    # Test storing report
    memory.store_health_report("user123", "User is healthy. Maintain current lifestyle.")

    # Retrieve data
    user_data = memory.get_user_data("user123")
    print("Retrieved User Data:", user_data)



import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any

# -----------------------------
# HealthGuard Agent Logging System
# -----------------------------
@dataclass
class AgentLogger:
    """Comprehensive logging for agent operations"""
    logs: List[Dict[str, Any]] = field(default_factory=list)
    
    def log(self, level: str, event: str, details: Dict[str, Any] = None):
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "event": event,
            "details": details or {}
        })
        print(f"[{level}] {event}")  # Optional: real-time log output
    
    def info(self, event: str, **kwargs):
        self.log("INFO", event, kwargs)
    
    def error(self, event: str, **kwargs):
        self.log("ERROR", event, kwargs)
    
    def warning(self, event: str, **kwargs):
        self.log("WARNING", event, kwargs)
    
    def get_recent_logs(self, count: int = 10) -> List[Dict]:
        return self.logs[-count:]
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_logs": len(self.logs),
            "info_count": sum(1 for log in self.logs if log['level'] == 'INFO'),
            "error_count": sum(1 for log in self.logs if log['level'] == 'ERROR'),
            "warning_count": sum(1 for log in self.logs if log['level'] == 'WARNING')
        }
    
    def export_logs(self, filename: str = "agent_logs.json"):
        with open(filename, 'w') as f:
            json.dump(self.logs, f, indent=2)
        print(f"âœ“ Logs exported to {filename}")


# -----------------------------
# Example Usage (for testing)
# -----------------------------
if __name__ == "__main__":
    logger = AgentLogger()
    logger.info("Logger initialized")
    
    logger.info("Vitals checked", user_id="user123", bp="120/80", hr=72, spo2=98)
    logger.warning("Low activity detected", user_id="user123")
    logger.error("Sensor failure", user_id="user123", sensor="SpO2")
    
    print("Recent Logs:", logger.get_recent_logs())
    print("Stats:", logger.get_stats())
    
    logger.export_logs("healthguard_logs.json")
    print("âœ“ Logging System Ready")



# ============================================================
# HealthGuard System - Secure Deployment Version
# ============================================================

import time
import json
import os
from typing import Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

import google.generativeai as genai

# ============================================================
# 0. CONFIG - Secure API Key Handling
# ============================================================
# Option 1: Load from environment variable
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")  # <-- Set this in your deployment environment

if not GOOGLE_API_KEY:
    # Option 2: Load from Kaggle Secrets (if running in Kaggle)
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    except Exception:
        GOOGLE_API_KEY = None

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("âœ“ API Key Configured")
else:
    print("âš  API Key not found. Agent will not initialize!")

CONFIG = {
    "team": "HealthGuard",
    "model": "models/gemini-2.5-flash",
    "max_tokens": 2000,
    "temperature": 0.3,
    "version": "2.0.0"
}

# ============================================================
# 1. Logging System
# ============================================================
@dataclass
class AgentLogger:
    logs: List[Dict[str, Any]] = field(default_factory=list)
    
    def log(self, level: str, event: str, details: Dict[str, Any] = None):
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "event": event,
            "details": details or {}
        })
        print(f"[{level}] {event}")
    
    def info(self, event: str, **kwargs): self.log("INFO", event, kwargs)
    def error(self, event: str, **kwargs): self.log("ERROR", event, kwargs)
    def warning(self, event: str, **kwargs): self.log("WARNING", event, kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_logs": len(self.logs),
            "info_count": sum(1 for log in self.logs if log['level'] == 'INFO'),
            "error_count": sum(1 for log in self.logs if log['level'] == 'ERROR'),
            "warning_count": sum(1 for log in self.logs if log['level'] == 'WARNING')
        }

# ============================================================
# 2. Memory System
# ============================================================
MEMORY_FILE = "healthguard_memory.json"

class HealthGuardMemory:
    def __init__(self, memory_file: str = MEMORY_FILE):
        self.memory_file = memory_file
        self.memory: Dict[str, Any] = {}
        self.load_memory()
        print(f"âœ“ HealthGuard Memory Initialized ({len(self.memory)} users loaded)")

    def load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r") as f:
                self.memory = json.load(f)

    def save_memory(self):
        with open(self.memory_file, "w") as f:
            json.dump(self.memory, f, indent=2)

    def store_user_vitals(self, user_id: str, vitals: Dict[str, Any]):
        self.memory.setdefault(user_id, {})["vitals"] = vitals
        self.save_memory()

    def get_user_data(self, user_id: str) -> Dict[str, Any]:
        return self.memory.get(user_id, {})

# ============================================================
# 3. HealthGuard Tool Functions
# ============================================================
def estimate_health_risk(vitals: Dict[str, float]) -> str:
    prompt = f"Medical AI. Given vitals: {vitals}, output risk score 0â€“1."
    model = genai.GenerativeModel(CONFIG["model"])
    return model.generate_content(prompt).text

def generate_coach_advice(risk: float, vitals: Dict[str, Any]) -> str:
    prompt = f"Health coach AI. Risk: {risk}, Vitals: {vitals}. Give short advice."
    model = genai.GenerativeModel(CONFIG["model"])
    return model.generate_content(prompt).text

# ============================================================
# 4. HealthGuard Main Agent
# ============================================================
class HealthGuardAgent:
    def __init__(self, config: Dict, memory: HealthGuardMemory, logger: AgentLogger):
        self.config = config
        self.memory = memory
        self.logger = logger
        self.stats = {"queries_processed":0, "tools_called":0, "total_response_time":0.0, "errors":0}
        if GOOGLE_API_KEY:
            self.model = genai.GenerativeModel(model_name=config['model'])
            self.logger.info("Agent initialized", model=config['model'])
        else:
            self.model = None
            self.logger.warning("Agent not initialized due to missing API key")

    def run(self, user_query: str) -> str:
        if not self.model:
            return "âš  Agent not initialized due to missing API key."
        start_time = time.time()
        self.logger.info("Query received", query=user_query[:100])
        system_prompt = f"HealthGuard AI Assistant. Context: {self.memory.memory}. User Query: {user_query}"
        chat = self.model.start_chat()
        response = chat.send_message(system_prompt)
        elapsed = time.time() - start_time
        self.stats["queries_processed"] += 1
        self.stats["total_response_time"] += elapsed
        self.logger.info("Query completed", response_time=f"{elapsed:.2f}s")
        return getattr(response, "text", "No response received")

    def get_stats(self):
        avg_response_time = (
            self.stats["total_response_time"] / self.stats["queries_processed"]
            if self.stats["queries_processed"] > 0 else 0
        )
        return {
            **self.stats,
            "avg_response_time": round(avg_response_time, 2),
            "memory_stats": self.memory.memory,
            "logger_stats": self.logger.get_stats()
        }

# ============================================================
# 5. Initialize Everything
# ============================================================
memory = HealthGuardMemory()
logger = AgentLogger()
agent = HealthGuardAgent(config=CONFIG, memory=memory, logger=logger)
print("âœ“ HealthGuard Agent Initialized âœ…")



# ============================================================
# Test function for HealthGuard Agent
# ============================================================

def test_agent(query: str):
    """Test the HealthGuard agent with a user query"""
    if not agent:
        print("âš  Agent not initialized")
        return
    
    print(f"\n{'='*60}")
    print(f"USER QUERY: {query}")
    print(f"{'='*60}\n")
    
    # Run query through the agent
    response = agent.run(query)
    
    print("AGENT RESPONSE:")
    print(f"{'-'*60}")
    print(response)
    print(f"{'='*60}\n")

print("âœ“ Test function ready")
print("ğŸ“Œ Usage: test_agent('your question here')")



# Test the agent with a sample health query
test_agent("My blood pressure is 140/90 and heart rate is 85. What should I do?")


# ================================
# Extend HealthGuardMemory for Dashboard
# ================================
class HealthGuardMemoryWithStats(HealthGuardMemory):
    """Extended memory to track messages for dashboard"""
    
    def __init__(self, memory_file: str = MEMORY_FILE):
        super().__init__(memory_file)
        self.messages: List[Dict[str, Any]] = []  # Track chat history
    
    def add_message(self, sender: str, content: str):
        """Add user/agent message"""
        self.messages.append({"sender": sender, "content": content})
    
    def get_stats(self):
        total = len(self.messages)
        user_msgs = sum(1 for m in self.messages if m['sender'] == 'user')
        agent_msgs = sum(1 for m in self.messages if m['sender'] == 'agent')
        return {
            "total_messages": total,
            "user_messages": user_msgs,
            "agent_messages": agent_msgs
        }

# ================================
# Dashboard Display Function
# ================================
def display_statistics():
    """Display agent performance metrics"""
    if not agent:
        print("âš  Agent not initialized")
        return
    
    stats = agent.get_stats()
    
    print(f"\n{'='*60}")
    print(f"{'AGENT PERFORMANCE DASHBOARD':^60}")
    print(f"{'='*60}")
    
    print(f"\nğŸ“Š Query Statistics:")
    print(f"  Total Queries: {stats['queries_processed']}")
    print(f"  Tools Called: {stats['tools_called']}")
    print(f"  Avg Response Time: {stats['avg_response_time']:.2f}s")
    print(f"  Errors: {stats['errors']}")
    
    print(f"\nğŸ’­ Memory Statistics:")
    mem = stats['memory_stats']
    print(f"  Total Messages: {mem['total_messages']}")
    print(f"  User Messages: {mem['user_messages']}")
    print(f"  Agent Messages: {mem['agent_messages']}")
    
    print(f"\nğŸ“� Logger Statistics:")
    log = stats['logger_stats']
    print(f"  Total Logs: {log['total_logs']}")
    print(f"  Info: {log['info_count']} | Warning: {log['warning_count']} | Error: {log['error_count']}")
    
    print(f"{'='*60}\n")

# ================================
# Initialize Dashboard-ready Agent
# ================================
memory = HealthGuardMemoryWithStats()
logger = AgentLogger()

# No tools parameter needed
agent = HealthGuardAgent(config=CONFIG, memory=memory, logger=logger)

print("âœ“ HealthGuard Agent & Dashboard Ready âœ…")




# ============================================================
# HealthGuard Agent Demo
# ============================================================

print("="*60)
print("DEMO 1: Blood Pressure & Heart Rate Advice")
print("="*60)
test_agent("My blood pressure is 140/90 and heart rate is 85. What should I do?")

print("\n" + "="*60)
print("DEMO 2: Fitness Recommendation")
print("="*60)
test_agent("I want to improve my stamina and lose 3 kg in a month. Suggest a daily routine.")

print("\n" + "="*60)
print("DEMO 3: Diet Suggestion")
print("="*60)
test_agent("What should I eat for breakfast to maintain healthy blood pressure?")

print("\n" + "="*60)
print("UPDATED PERFORMANCE METRICS")
print("="*60)

# Show updated statistics
if agent:
    stats = agent.get_stats()
    
    print(f"\nğŸ“Š Query Statistics:")
    print(f"  Total Queries: {stats['queries_processed']}")
    print(f"  Avg Response Time: {stats['avg_response_time']:.2f}s")
    print(f"  Errors: {stats['errors']}")
    
 
    
    print(f"\nğŸ“� Logger Statistics:")
    log = stats['logger_stats']
    print(f"  Total Logs: {log['total_logs']}")
    print(f"  Info: {log['info_count']} | Warning: {log['warning_count']} | Error: {log['error_count']}")





import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List
from dataclasses import dataclass, field

import warnings
warnings.filterwarnings("ignore")

# UI & plotting
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import numpy as np

# Optional: Google Generative AI
try:
    import google.generativeai as genai
except Exception:
    genai = None

# -------------------------------
# 0. CONFIG & Secure API Loading
# -------------------------------
CONFIG = {
    "team": "HealthGuard",
    "model": "models/gemini-2.5-flash",  # update if needed
    "max_tokens": 512,
    "temperature": 0.2
}

# Load API key from environment or Kaggle secrets (no hard-coded keys)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    try:
        from kaggle_secrets import UserSecretsClient
        GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    except Exception:
        GOOGLE_API_KEY = None

if GOOGLE_API_KEY and genai:
    genai.configure(api_key=GOOGLE_API_KEY)
    LLM_AVAILABLE = True
else:
    LLM_AVAILABLE = False

# -------------------------------
# 1. Logger
# -------------------------------
@dataclass
class AgentLogger:
    logs: List[Dict[str, Any]] = field(default_factory=list)

    def _append(self, level: str, event: str, details: Dict[str, Any] = None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "event": event,
            "details": details or {}
        }
        self.logs.append(entry)

    def info(self, event: str, **kwargs):
        self._append("INFO", event, kwargs)

    def warning(self, event: str, **kwargs):
        self._append("WARNING", event, kwargs)

    def error(self, event: str, **kwargs):
        self._append("ERROR", event, kwargs)

    def recent(self, n=10):
        return self.logs[-n:]

logger = AgentLogger()
logger.info("Logger initialized", llm=LLM_AVAILABLE)

# -------------------------------
# 2. Memory (simple json file)
# -------------------------------
MEMORY_FILE = "healthguard_memory.json"

class HealthGuardMemory:
    def __init__(self, memory_file=MEMORY_FILE):
        self.file = memory_file
        self.memory: Dict[str, Any] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.file):
            try:
                with open(self.file, "r") as f:
                    self.memory = json.load(f)
            except Exception:
                self.memory = {}
        else:
            self.memory = {}

    def _save(self):
        with open(self.file, "w") as f:
            json.dump(self.memory, f, indent=2)

    def save_entry(self, user_id: str, entry: Dict[str, Any]):
        user = self.memory.get(user_id, {"history": []})
        user["history"].append(entry)
        self.memory[user_id] = user
        self._save()
        logger.info("Saved memory entry", user_id=user_id)

    def get_history(self, user_id: str, limit: int = 50):
        return self.memory.get(user_id, {}).get("history", [])[-limit:]

memory = HealthGuardMemory()
logger.info("Memory initialized", users_loaded=len(memory.memory))

# -------------------------------
# 3. LLM-backed helpers (or dummy)
# -------------------------------
def llm_estimate_risk(vitals: Dict[str, Any]) -> float:
    """Use LLM for reasoning if available; else basic heuristic."""
    if LLM_AVAILABLE and genai:
        try:
            prompt = (
                "You are a medical reasoning assistant. Given these vitals, return a risk score "
                "between 0 and 1 (only the number). Vitals:\n" + json.dumps(vitals)
            )
            model = genai.GenerativeModel(CONFIG["model"])
            resp = model.generate_content(prompt)
            text = getattr(resp, "text", "")
            # try parse float
            val = None
            try:
                val = float(text.strip().split()[0])
            except Exception:
                # fallback later
                val = None
            if val is None:
                raise ValueError("LLM returned non-numeric")
            logger.info("LLM risk estimate", value=val)
            return max(0.0, min(1.0, float(val)))
        except Exception as e:
            logger.warning("LLM risk failed", error=str(e))
    # Dummy heuristic
    try:
        bp = vitals.get("bp", "")
        hr = float(vitals.get("hr", 0) or 0)
        if isinstance(bp, str) and "/" in bp:
            s, d = bp.split("/", 1)
            s = float(s); d = float(d)
        else:
            s = float(vitals.get("systolic", 120) or 120)
            d = float(vitals.get("diastolic", 80) or 80)
        # simple normalized score
        score = 0.0
        score += max(0, (s - 110) / 100) * 0.6
        score += max(0, (hr - 70) / 100) * 0.4
        score = float(np.clip(score, 0, 1))
        logger.info("Heuristic risk estimate", value=score)
        return score
    except Exception:
        logger.error("Risk heuristic error")
        return 0.5

def llm_generate_advice(risk: float, vitals: Dict[str, Any]) -> str:
    if LLM_AVAILABLE and genai:
        try:
            prompt = (
                "You are a friendly health coach. Given a risk score (0-1) and vitals, "
                "provide 3 short actionable recommendations. Return plain text.\n"
                f"Risk: {risk}\nVitals: {json.dumps(vitals)}"
            )
            model = genai.GenerativeModel(CONFIG["model"])
            resp = model.generate_content(prompt)
            text = getattr(resp, "text", "")
            logger.info("LLM advice returned")
            return text.strip()
        except Exception as e:
            logger.warning("LLM advice failed", error=str(e))
    # Dummy advice
    advice = []
    if risk > 0.6:
        advice.append("High risk detected: recheck vitals, rest, and contact healthcare.")
    elif risk > 0.35:
        advice.append("Moderate risk: relax, hydrate, re-measure in 30 minutes.")
    else:
        advice.append("Low risk: maintain healthy habits and regular monitoring.")
    advice.append("Reduce salt and stay active.")
    return " ".join(advice)

# -------------------------------
# 4. Agent function (simple wrapper)
# -------------------------------
def analyze_vitals_and_return(user_id: str, vitals: Dict[str, Any]) -> Dict[str, Any]:
    """Main analysis pipeline used by the UI."""
    t0 = time.time()
    risk = llm_estimate_risk(vitals)
    advice = llm_generate_advice(risk, vitals)
    result = {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "vitals": vitals,
        "risk_score": round(float(risk), 3),
        "advice": advice
    }
    memory.save_entry(user_id, result)
    logger.info("Analysis completed", user_id=user_id, runtime=f"{time.time()-t0:.2f}s")
    return result

# -------------------------------
# 5. UI (ipywidgets) - Tabs: Vitals, History, Prediction
# -------------------------------
# Widgets for Vitals Tab
user_id_w = widgets.Text(value="user_001", description="User ID:", layout=widgets.Layout(width="50%"))
bp_w = widgets.Text(value="120/80", description="BP:", placeholder="systolic/diastolic")
hr_w = widgets.IntText(value=72, description="HR (bpm):")
spo2_w = widgets.IntText(value=98, description="SpOâ‚‚ (%):")
activity_w = widgets.Dropdown(options=["Resting","Walking","Exercise","Upstairs","Downstairs"], description="Activity:")
notes_w = widgets.Text(value="", description="Notes:")

analyze_btn = widgets.Button(description="Analyze", button_style="primary", icon="heartbeat")
save_btn = widgets.Button(description="Save to History", icon="save")
clear_btn = widgets.Button(description="Clear Output", icon="trash")

output_area = widgets.Output(layout={'border': '1px solid #ddd'})

# Widgets for History Tab
history_user_w = widgets.Text(value="user_001", description="User ID:", layout=widgets.Layout(width="50%"))
history_limit_w = widgets.IntSlider(value=10, min=1, max=100, description="Limit:")
history_out = widgets.Output(layout={'border': '1px solid #ddd'})

# Widgets for Prediction Tab (simple trend chart)
chart_user_w = widgets.Text(value="user_001", description="User ID:", layout=widgets.Layout(width="50%"))
chart_out = widgets.Output()

# Simple helper to parse BP string
def parse_bp(bp_str):
    try:
        if isinstance(bp_str, str) and "/" in bp_str:
            s,d = bp_str.split("/",1)
            return float(s), float(d)
    except:
        pass
    return 120.0, 80.0

# Analyze button callback
def on_analyze_clicked(b):
    with output_area:
        clear_output()
        user_id = user_id_w.value.strip() or "user_001"
        bp = bp_w.value.strip()
        hr = int(hr_w.value or 0)
        spo2 = int(spo2_w.value or 0)
        activity = activity_w.value
        notes = notes_w.value.strip()
        systolic, diastolic = parse_bp(bp)
        vitals = {
            "bp": bp,
            "systolic": systolic,
            "diastolic": diastolic,
            "hr": hr,
            "spo2": spo2,
            "activity": activity,
            "notes": notes
        }
        print("Analyzing... (LLM available: {})".format(LLM_AVAILABLE))
        res = analyze_vitals_and_return(user_id, vitals)
        print("\n=== VitalGuard AI Report ===")
        print(f"Time: {res['timestamp']}")
        print(f"User: {res['user_id']}")
        print(f"Risk score: {res['risk_score']}")
        print("Advice:")
        print(res['advice'])
        print("\nSaved to memory (history).")

analyze_btn.on_click(on_analyze_clicked)

# Save to history callback (manual save an arbitrary entry)
def on_save_clicked(b):
    with output_area:
        user_id = user_id_w.value.strip() or "user_001"
        bp = bp_w.value.strip()
        systolic, diastolic = parse_bp(bp)
        vitals = {"bp": bp, "systolic": systolic, "diastolic": diastolic, "hr": hr_w.value, "spo2": spo2_w.value}
        entry = {
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "vitals": vitals,
            "note": notes_w.value.strip()
        }
        memory.save_entry(user_id, entry)
        with output_area:
            print("âœ“ Manual entry saved to history.")

save_btn.on_click(on_save_clicked)

def on_clear_clicked(b):
    with output_area:
        clear_output()
clear_btn.on_click(on_clear_clicked)

# History callbacks
def show_history(_=None):
    with history_out:
        clear_output()
        uid = history_user_w.value.strip() or "user_001"
        hist = memory.get_history(uid, limit=history_limit_w.value)
        if not hist:
            print("No history for user:", uid)
            return
        print(f"History for {uid} (latest {len(hist)}):\n")
        for i, item in enumerate(reversed(hist[-history_limit_w.value:])):
            ts = item.get("timestamp", "N/A")
            rs = item.get("risk_score", None)
            vit = item.get("vitals", item.get("vitals", {}))
            print(f"[{i+1}] {ts} | risk={rs} | vitals={vit}")
            if item.get("advice"):
                print("    Advice:", item["advice"])
            if item.get("note"):
                print("    Note:", item["note"])
            print("-"*40)

history_refresh_btn = widgets.Button(description="Show History", icon="list")
history_refresh_btn.on_click(lambda b: show_history())

# Chart callback (plot risk trend or HR trend)
def show_chart(_=None):
    with chart_out:
        clear_output()
        uid = chart_user_w.value.strip() or "user_001"
        hist = memory.get_history(uid, limit=50)
        if not hist:
            print("No history to plot for user:", uid)
            return
        # extract timestamps and hr/risk
        timestamps = [h.get("timestamp") for h in hist]
        risk_scores = [h.get("risk_score", np.nan) for h in hist]
        hrs = [h.get("vitals", {}).get("hr", np.nan) for h in hist]

        # convert to simple index for plotting
        idx = list(range(len(hist)))
        fig, ax = plt.subplots(1,2, figsize=(10,3))
        ax[0].plot(idx, risk_scores, marker='o')
        ax[0].set_title("Risk score trend")
        ax[0].set_xlabel("Entry (older -> newer)")
        ax[0].set_ylim(0,1)

        ax[1].plot(idx, hrs, marker='o')
        ax[1].set_title("Heart Rate trend")
        ax[1].set_xlabel("Entry (older -> newer)")
        plt.tight_layout()
        plt.show()

chart_refresh_btn = widgets.Button(description="Show Chart", icon="line-chart")
chart_refresh_btn.on_click(lambda b: show_chart())

# Build layouts
vitals_controls = widgets.VBox([
    widgets.HBox([user_id_w, activity_w]),
    widgets.HBox([bp_w, hr_w, spo2_w]),
    widgets.HBox([notes_w]),
    widgets.HBox([analyze_btn, save_btn, clear_btn]),
    output_area
])

history_controls = widgets.VBox([
    widgets.HBox([history_user_w, history_limit_w]),
    history_refresh_btn,
    history_out
])

chart_controls = widgets.VBox([
    widgets.HBox([chart_user_w]),
    chart_refresh_btn,
    chart_out
])

tab = widgets.Tab(children=[vitals_controls, history_controls, chart_controls])
tab.set_title(0, "Vitals")
tab.set_title(1, "History")
tab.set_title(2, "Prediction / Chart")

# Display header and help
header = widgets.HTML(
    "<h2 style='color:#2b7cff'>HealthGuard â€” AI agent</h2>"
    "<p>Enter vitals and click <b>Analyze</b>. Uses Google Generative AI if a valid API key is configured.</p>"
)
display(header, tab)

# Show initial small status box
status_text = widgets.HTML(value=f"<b>LLM available:</b> {LLM_AVAILABLE} &nbsp; | &nbsp; <b>Memory users:</b> {len(memory.memory)}")
display(status_text)

# Keep a small instruction
display(widgets.HTML("<small>Tip: Add your Google API key to Kaggle Secrets named <code>GOOGLE_API_KEY</code> to enable LLM-powered reasoning.</small>"))

logger.info("UI displayed")


