!pip install -q google-generativeai
import google.generativeai as genai

import re
import logging
from datetime import datetime

print("Environment ready.")





# MEMORY SERVICE

class MemoryService:
    def __init__(self):
        self.session_data = {}
        self.events = []
        self.preferences = {}

    # Generic key–value memory
    def save(self, key, value):
        self.session_data[key] = value

    def get(self, key, default=None):
        return self.session_data.get(key, default)

    # Store events (for reminder agent)
    def add_event(self, event_dict):
        """
        event_dict = {
            "title": "...",
            "date": "...",
            "time": "..."
        }
        """
        self.events.append(event_dict)

    def get_events(self):
        return self.events

    # Preferences (optional but useful)
    def set_preference(self, key, value):
        self.preferences[key] = value

    def get_preference(self, key, default=None):
        return self.preferences.get(key, default)


# Create global memory object
memory = MemoryService()

print("Memory Service initialized!")



memory.save("username", "Ayushman")
memory.add_event({"title": "AI Agent", "date": "2025-12-05", "time": "3 PM"})

print("Username:", memory.get("username"))
print("Events:", memory.get_events())




# BASE AGENT + RESEARCH AGENT


import os
import textwrap
from datetime import datetime

# BaseAgent (extendable)
class BaseAgent:
    def __init__(self, name: str, memory=None, tools: dict=None):
        """
        name: agent name (string)
        memory: MemoryService instance (optional)
        tools: dict of callable tools the agent can use (optional)
        """
        self.name = name
        self.memory = memory
        self.tools = tools or {}
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)

    def use_tool(self, tool_name, *args, **kwargs):
        tool = self.tools.get(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not available for agent {self.name}")
        return tool(*args, **kwargs)

    def respond(self, prompt: str, **kwargs):
        """Default respond method — override in child agents."""
        raise NotImplementedError("Child agents must implement respond()")

#  Helper: safe genai check
def _has_genai_key():
    # Common env var names (user may set either)
    return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GENAI_API_KEY") or os.environ.get("GOOGLE_API_KEY_JSON"))

# We try to import and configure genai only if user set API keys.
try:
    import google.generativeai as genai
    genai_available = True
except Exception:
    genai_available = False

# Simple local summarizer (fallback)
def simple_local_summary(topic_text: str, max_sentences: int = 3):
    """
    Lightweight fallback summarizer:
    - Splits into sentences and returns the first few
    - Adds a short 'key points' bullet template
    This ensures the notebook is runnable offline and still useful for demos.
    """
    # crude sentence splitting
    sents = [s.strip() for s in topic_text.replace("\n", " ").split(".") if s.strip()]
    summary_sents = sents[:max_sentences] or ["No detail available."]
    # Build a small structured summary
    summary = " ".join(summary_sents)
    key_points = []
    # Generate a few heuristic key points from words (very simple)
    words = topic_text.split()
    unique = []
    for w in words:
        w_clean = re.sub(r"[^A-Za-z0-9\-]", "", w).lower()
        if len(w_clean) > 5 and w_clean not in unique:
            unique.append(w_clean)
        if len(unique) >= 3:
            break
    for i, k in enumerate(unique):
        key_points.append(f"{i+1}. {k}")
    if not key_points:
        key_points = ["1. Core idea", "2. Use-cases", "3. Where to learn more"]
    return {
        "summary": summary,
        "key_points": key_points,
        "note": "This is a local fallback summary. For web search and richer summaries, set GOOGLE_API_KEY or GENAI_API_KEY in environment."
    }

# ResearchAgent
class ResearchAgent(BaseAgent):
    def __init__(self, name="ResearchAgent", memory=None, tools: dict=None, model="gpt-4o-mini"):
        super().__init__(name=name, memory=memory, tools=tools)
        self.model = model

    def _call_genai(self, prompt: str, max_output_tokens: int = 512):
        """
        Minimal wrapper to call google.generativeai if configured.
        We check environment keys and the presence of the genai package.
        """
        if not genai_available or not _has_genai_key():
            raise RuntimeError("GenAI not configured in environment or package missing.")
        # If user has set up correctly, they can edit this block to use their preferred genai call.
        # We'll try a common pattern, but keep it guarded to avoid crashing for different SDK versions.
        try:
            # Try a generic call (the exact function can vary by installed genai version)
            # Keep it minimal — users with API access may replace this with advanced calls.
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY") or os.environ.get("GENAI_API_KEY"))
            response = genai.generate_text(model=self.model, input=prompt, max_output_tokens=max_output_tokens)
            # response.text or response.output[0].content depends on the genai package version
            text = getattr(response, "text", None) or getattr(response, "output", [None])[0]
            if isinstance(text, dict):  # some versions return a dict-like
                text = text.get("content") or str(text)
            return text or str(response)
        except Exception as e:
            self.logger.warning("GenAI call failed: %s", e)
            raise

    def research(self, query: str, context: str = None, use_genai_if_available: bool = True):
        self.logger.info(f"ResearchAgent received query: {query}")

        """
        Main research method.
        - If genai is configured and use_genai_if_available True -> call API
        - Otherwise -> run local heuristic summarizer using provided context or using the query itself
        Returns a dict: {summary, key_points, sources (maybe empty), meta}
        """
        prompt_text = (context + "\n\n" + query) if context else query
        # If genai is available and the user wants it, try to call it (guarded)
        if use_genai_if_available and genai_available and _has_genai_key():
            try:
                out = self._call_genai(prompt_text)
                # Minimal post-processing into structured dict
                return {
                    "summary": out if isinstance(out, str) else str(out),
                    "key_points": [],
                    "sources": [],
                    "meta": {"via": "genai", "timestamp": datetime.utcnow().isoformat()}
                }
            except Exception:
                # fallback to local summarizer on failure
                pass

        # Local heuristic fallback
        fallback = simple_local_summary(prompt_text, max_sentences=3)
        return {
            "summary": fallback["summary"],
            "key_points": fallback["key_points"],
            "sources": [],
            "meta": {"via": "local_fallback", "timestamp": datetime.utcnow().isoformat()}
        }

# Quick test/demo 
if __name__ == "__main__" or True:
    ra = ResearchAgent(memory=memory)
    demo_query = "Explain Bloom filters: what they are, how they work, and common use-cases."
    result = ra.research(demo_query)
    print("---- Research Agent Demo ----")
    print("Summary:\n", textwrap.fill(result["summary"], width=90))
    print("\nKey points:")
    for kp in result["key_points"]:
        print("-", kp)
    print("\nMeta:", result["meta"])




# REMINDER & EVENT TRACKING AGENT


import re
from datetime import datetime, timedelta

class ReminderAgent(BaseAgent):
    def __init__(self, name="ReminderAgent", memory=None):
        super().__init__(name=name, memory=memory)

    
    # Utility: Basic date extraction
   
    def extract_date(self, text):
        # Matches: 2025-12-05 OR 05-12-2025 OR 5 Dec 2025 OR Dec 5, 2025
        patterns = [
            r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b",   # YYYY-MM-DD
            r"\b(\d{1,2})-(\d{1,2})-(\d{4})\b",   # DD-MM-YYYY
            r"\b(\d{1,2})\s+([A-Za-z]{3,})\s+(\d{4})\b",  # 5 Dec 2025
            r"\b([A-Za-z]{3,})\s+(\d{1,2}),\s+(\d{4})\b"  # Dec 5, 2025
        ]

        for p in patterns:
            match = re.search(p, text)
            if match:
                try:
                    parts = match.groups()
                    if len(parts) == 3:
                        # Try different interpretations
                        try:
                            # Case: YYYY-MM-DD
                            return datetime.strptime("-".join(parts), "%Y-%m-%d").date()
                        except:
                            pass
                        try:
                            # Case: DD-MM-YYYY
                            d, m, y = parts
                            return datetime.strptime(f"{d}-{m}-{y}", "%d-%m-%Y").date()
                        except:
                            pass
                        try:
                            # Case: 5 Dec 2025
                            return datetime.strptime(" ".join(parts), "%d %b %Y").date()
                        except:
                            pass
                        try:
                            # Case: Dec 5, 2025
                            return datetime.strptime(f"{parts[0]} {parts[1]} {parts[2]}", "%b %d %Y").date()
                        except:
                            pass
                except:
                    pass
        return None

    
    # Utility: Basic time extraction
    
    def extract_time(self, text):
        # Matches: 3 PM, 3:30 PM, 14:00
        match = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(AM|PM|am|pm)", text)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2)) if match.group(2) else 0
            ampm = match.group(3).lower()
            if ampm == "pm" and hour != 12:
                hour += 12
            return datetime.strptime(f"{hour}:{minute}", "%H:%M").time()

        # 24-hour time
        match = re.search(r"\b(\d{1,2}):(\d{2})\b", text)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
            return datetime.strptime(f"{hour}:{minute}", "%H:%M").time()

        return None

    
    # Extract event title
    
    def extract_title(self, text):
        # Simplify: remove dates/times; return meaningful words
        clean = re.sub(r"\d+.*", "", text).strip()
        clean = clean.replace("on", "").replace("at", "")
        return clean[:60].strip() or "Untitled Event"

    
    # Create event object
    
    def parse_event(self, text):
        date = self.extract_date(text)
        time = self.extract_time(text)
        title = self.extract_title(text)

        if not date:
            return None  # date is essential

        return {
            "title": title,
            "date": str(date),
            "time": str(time) if time else "Not specified"
        }

    
    # Generate reminders automatically
   
    def generate_reminders(self, event):
        """
        Creates reminder times:
        - 24 hours before
        - 1 hour before
        """
        date_str = event["date"]
        time_str = event["time"]

        try:
            event_datetime = datetime.strptime(date_str + " " + time_str, "%Y-%m-%d %H:%M:%S")
        except:
            # If time not provided, default to 09:00 AM
            event_datetime = datetime.strptime(date_str + " 09:00", "%Y-%m-%d %H:%M")

        reminders = [
            event_datetime - timedelta(hours=24),
            event_datetime - timedelta(hours=1)
        ]

        return [str(r) for r in reminders]

    
    # Main function (called by orchestrator)
    
    def process_event(self, text):
        self.logger.info(f"ReminderAgent received text: {text}")

        event = self.parse_event(text)
        if not event:
            return {"error": "Could not extract event date. Please give a clearer date."}
        self.logger.info(f"Extracted event: {event}")
    

        # Save to memory
        self.memory.add_event(event)

        # Generate reminder times
        reminders = self.generate_reminders(event)

        return {
            "event": event,
            "reminders": reminders,
            "saved_to_memory": True
        }



# Quick Demo

if True:
    ra = ReminderAgent(memory=memory)
    test = ra.process_event("I have an AWS Webinar on 5 Dec 2025 at 3 PM")
    print("Reminder Agent Output:\n", test)




#  ORCHESTRATOR AGENT (FINAL WORKING VERSION)

class OrchestratorAgent(BaseAgent):
    def __init__(self, name="OrchestratorAgent", memory=None, research_agent=None, reminder_agent=None):
        super().__init__(name=name, memory=memory)
        self.research_agent = research_agent
        self.reminder_agent = reminder_agent

   
    # INTENT DETECTION
    
    def detect_intent(self, text):
        text_lower = text.lower()
        words = text_lower.replace(",", " ").replace(".", " ").split()

        reminder_keywords = {"webinar", "meeting", "event", "remind", "schedule", "at", "on"}
        research_keywords = {"explain", "research", "what", "summarize", "how", "topic", "learn"}

        date_present = bool(self.reminder_agent.extract_date(text))
        time_present = bool(self.reminder_agent.extract_time(text))

        if date_present or time_present:
            intent = "reminder"
        elif any(k in words for k in research_keywords):
            intent = "research"
        elif any(k in words for k in reminder_keywords):
            intent = "reminder"
        else:
            intent = "unknown"

        self.logger.info(f"[Orchestrator] Intent detected: {intent} | Query: {text}")

        return intent

   
    # AGENT ROUTING
    
    def handle(self, text):
        intent = self.detect_intent(text)

        if intent == "research":
            result = self.research_agent.research(query=text)
            return {"intent": "research", "result": result, "meta": {"handled_by": "ResearchAgent"}}

        elif intent == "reminder":
            result = self.reminder_agent.process_event(text)
            return {"intent": "reminder", "result": result, "meta": {"handled_by": "ReminderAgent"}}

        else:
            self.logger.info("[Orchestrator] Unknown request.")
            return {
                "intent": "unknown",
                "message": "I could not understand your request. Try asking a topic or mentioning an event.",
                "meta": {"handled_by": "OrchestratorAgent"}
            }

    
    # RESPOND() — REQUIRED BY BaseAgent
   
    def respond(self, prompt: str, **kwargs):
        return self.handle(prompt)



orchestrator = OrchestratorAgent(
    memory=memory,
    research_agent=research_agent,
    reminder_agent=reminder_agent
)

run_agent("Explain binary search")
run_agent("Set a reminder for exam on 5 Feb 2026 at 2 PM")




#  WORKFLOW EXECUTION FUNCTION

# Create global agent instances (for user usage)
research_agent = ResearchAgent(memory=memory)
reminder_agent = ReminderAgent(memory=memory)
orchestrator = OrchestratorAgent(
    memory=memory,
    research_agent=research_agent,
    reminder_agent=reminder_agent
)

def run_agent(query: str):
    print("\n==============================")
    print(f" USER QUERY: {query}")
    print("==============================")

    output = orchestrator.respond(query)

    intent = output.get("intent")

    if intent == "research":
        result = output["result"]
        print("\n[Research Summary]")
        print(result["summary"])
        print("\n[Key Points]")
        for kp in result["key_points"]:
            print("-", kp)
        print("\n[Meta]", result["meta"])

    elif intent == "reminder":
        result = output["result"]
        print("\n[Event Detected]")
        print(result["event"])
        print("\n[Reminder Times]")
        for r in result["reminders"]:
            print("-", r)
        print("\nSaved to memory:", result["saved_to_memory"])

    else:
        print("\n[Unknown Intent]")
        print(output["message"])

    print("\n==============================\n")

    return output



run_agent("Explain hash tables in simple words")



run_agent("Set a reminder for SDE Mock Interview on 10 Jan 2026 at 11 AM")



run_agent("Yo bro what's up?")



def show_system_state():
    print("\n===== SYSTEM STATE =====")

    print("\nStored Events:")
    for e in memory.get_events():
        print(" -", e)

    print("\nStored Session Data:")
    for k, v in memory.session_data.items():
        print(f" - {k}: {v}")

    print("\nStored Preferences:")
    for k, v in memory.preferences.items():
        print(f" - {k}: {v}")

    print("\n===== END STATE =====\n")




# LOGGING SYSTEM SETUP

import logging

logger = logging.getLogger("MultiAgentSystem")
logger.setLevel(logging.INFO)

# Stream handler (prints logs in notebook output)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)

# Prevent duplicate handlers
if not logger.handlers:
    logger.addHandler(handler)

logger.info("Logging initialized successfully.")



import pandas as pd
import os

# Creating a valid test submission file for Kaggle competition
dummy_submission = pd.DataFrame({
    "id": [1],
    "output": ["Smart Personal Life Concierge Agent Completed"]
})

file_path = "/kaggle/working/submission.csv"
dummy_submission.to_csv(file_path, index=False)

print("Submission file generated at:", file_path)
print(os.listdir("/kaggle/working/"))

