import os
import sys

# Standard setup code from original notebook
folders = [
    "project/agents",
    "project/tools",
    "project/memory",
    "project/core"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("Project structure created successfully.")


%%writefile project/requirements.txt
google-generativeai
python-dotenv
termcolor


# Install dependencies immediately to ensure the next imports work
!pip install -r project/requirements.txt

# --- Gemini Client Setup ---
from google import genai
from google.genai.errors import APIError

# ⚠️ IMPORTANT: Replace "YOUR_GEMINI_API_KEY" with your actual Gemini API key.
# Alternatively, set it in Kaggle's "Add-ons" -> "Secrets" as GOOGLE_API_KEY
API_KEY = "AIzaSyBWmg6tZLhcNfIgObyWG92_S9KA0RdBXSk" # <--- REPLACE THIS LINE
os.environ["GEMINI_API_KEY"] = API_KEY

if not API_KEY or API_KEY == "YOUR_GEMINI_API_KEY":
    raise ValueError("GEMINI_API_KEY not set. Please set your API key in the cell above.")

try:
    GLOBAL_GEMINI_CLIENT = genai.Client(api_key=API_KEY)
    print("Gemini Client Initialized successfully.")
except Exception as e:
    print(f"Error initializing Gemini Client: {e}")
    GLOBAL_GEMINI_CLIENT = None

# Default model name for the agent system
MODEL_NAME = "gemini-2.5-flash"


%%writefile project/core/a2a_protocol.py
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class AgentMessage:
    sender: str
    content: str
    msg_type: str = "text"  # text, command, error, search_result
    metadata: Dict[str, Any] = field(default_factory=dict)

class AgentProtocol:
    def format_message(self, sender: str, content: str, msg_type: str = "text", **kwargs) -> AgentMessage:
        return AgentMessage(sender=sender, content=content, msg_type=msg_type, metadata=kwargs)


%%writefile project/core/observability.py
import logging
from termcolor import colored
import sys # <--- The critical fix is here, ensuring sys is imported

class Observer:
    def __init__(self, name="System"):
        # Prevent duplicate handlers in notebook environments
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        # Check if handlers exist before adding one to avoid duplicate logs in notebooks
        if not self.logger.handlers:
            # Use sys.stderr for observability logs
            handler = logging.StreamHandler(sys.stderr) 
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log_event(self, agent_name: str, action: str, details: str):
        msg = f"[{agent_name}] {action}: {details}"
        # Print to stderr using termcolor for visual distinction in the notebook
        print(colored(msg, "cyan"), file=sys.stderr) 
        self.logger.info(msg)

    def log_error(self, agent_name: str, error: str):
        msg = f"[{agent_name}] ERROR: {error}"
        print(colored(msg, "red"), file=sys.stderr)
        self.logger.error(msg)


%%writefile project/core/context_engineering.py
import os
from google import genai
from google.genai.errors import APIError

class LLMService:
    """
    Wrapper for LLM calls, connecting to Google Gemini API.
    """
    def __init__(self, client: genai.Client, model_name: str = "gemini-2.5-flash"):
        self.client = client
        self.model_name = model_name
        self.observer = None # Will be set by MainAgent for better logging

    def _call_gemini(self, system_instruction: str, prompt: str) -> str:
        if not self.client:
            return "System: LLM Service not available (API Key Error)."
            
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "system_instruction": system_instruction,
                    "temperature": 0.0, # Use low temp for deterministic tasks
                }
            )
            return response.text.strip()
        except APIError as e:
            return f"API_ERROR: {e}"
        except Exception as e:
            return f"LLM_ERROR: {e}"

    def classify_intent(self, text: str) -> str:
        system_instruction = (
            "You are an Intent Classifier Agent. Your task is to analyze the user's message "
            "and respond ONLY with one single, all-caps category tag: "
            "'OFFENSIVE_CONTENT', 'SEARCH_QUERY', 'SUPPORT_REQUEST', or 'BROADCAST'. "
            "Criteria: 'OFFENSIVE_CONTENT' for hate speech/profanity. 'SEARCH_QUERY' "
            "for factual questions, or explicit requests. 'SUPPORT_REQUEST' "
            "for pleas for help (e.g., 'I need help'). 'BROADCAST' for general conversation."
        )
        prompt = f"MESSAGE: {text}"
        
        # The result from Gemini will be the single tag required by the prompt
        return self._call_gemini(system_instruction, prompt)

    def generate_response(self, context: str, query: str) -> str:
        system_instruction = (
            "You are an Intelligent Chat Bot. Respond to the user's query professionally "
            "and concisely based ONLY on the CONTEXT provided below. Do not use outside "
            "knowledge. If the context is empty or irrelevant, politely state that you "
            "cannot answer based on the available information."
        )
        prompt = f"CONTEXT:\n{context}\n\nUSER QUERY: {query}"
        return self._call_gemini(system_instruction, prompt)


%%writefile project/memory/session_memory.py
from typing import Dict, Any

class SessionMemory:
    """
    Long Term Memory simulation for User Preferences.
    """
    def __init__(self):
        # Simulating a database
        self._user_store: Dict[str, Dict[str, Any]] = {
            "default_user": {"preferred_room": "general", "language": "en"}
        }

    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        return self._user_store.get(user_id, {"preferred_room": "general", "language": "en"})

    def update_preference(self, user_id: str, key: str, value: Any):
        if user_id not in self._user_store:
            self._user_store[user_id] = {}
        self._user_store[user_id][key] = value


%%writefile project/tools/tools.py
import datetime

class GoogleSearchTool:
    """
    Built-in Tool: Google Search simulation.
    """
    def search(self, query: str) -> str:
        # Real implementation would use Google Search JSON API (SerpAPI or Google Custom Search)
        return f"Top search result for '{query}': [Simulated Web Data: The I-MUCS system is an intelligent, multi-agent chat platform for proactive moderation and knowledge retrieval. It was developed by ADK Capstone.]"

class LogSaverTool:
    """
    Custom Tool: Log Saver Utility.
    """
    def log_chat_event(self, user: str, message: str, status: str):
        timestamp = datetime.datetime.now().isoformat()
        log_entry = f"{timestamp} | {user} | {status} | {message}"
        # In prod: write to database or file
        # print(f"DEBUG: Logged -> {log_entry}")
        return True


%%writefile project/agents/worker.py
from project.core.context_engineering import LLMService
from project.core.observability import Observer
from project.tools.tools import GoogleSearchTool

class ContentModerator:
    """
    Classifies intent and performs safety checks using the LLM.
    """
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service
        self.observer = Observer("ContentModerator")

    def process(self, message: str) -> dict:
        self.observer.log_event("Moderator", "Analyzing", message)
        
        # 1. Call the real Gemini model for intent classification
        intent_raw = self.llm.classify_intent(message)
        
        # 2. Standardize/Clean the intent
        intent_clean = intent_raw.strip().upper()
        
        # 3. Determine Safety status (The fix for blocking is here)
        is_safe = True
        final_intent = "BROADCAST" # Default to BROADCAST

        # Safety/Error check: if the LLM call failed or returned offensive content
        if "OFFENSIVE_CONTENT" in intent_clean:
            is_safe = False
            final_intent = "OFFENSIVE_CONTENT"
            self.observer.log_error("Moderator", f"Safety Violation Detected. Raw Intent: {intent_raw}")
        elif "API_ERROR" in intent_clean or "LLM_ERROR" in intent_clean:
            is_safe = False
            final_intent = "API_ERROR" # Use a distinct intent tag for routing diagnosis
            self.observer.log_error("Moderator", f"LLM Service Error Encountered. Raw Intent: {intent_raw}")
        
        # 4. Map to the final intent tag for the router (Only if safe)
        if is_safe:
            if "SEARCH_QUERY" in intent_clean:
                final_intent = "SEARCH_QUERY"
            elif "SUPPORT_REQUEST" in intent_clean:
                final_intent = "SUPPORT_REQUEST"
            # Otherwise, it remains the default BROADCAST

        self.observer.log_event("Moderator", "Intent Classified (Final)", final_intent)
            
        return {"intent": final_intent, "is_safe": is_safe}

class KnowledgeAgent:
    """
    Handles external knowledge retrieval and uses LLM to synthesize a natural answer.
    """
    def __init__(self, llm_service: LLMService):
        self.tool = GoogleSearchTool()
        self.llm = llm_service
        self.observer = Observer("KnowledgeAgent")

    def fetch_info(self, query: str) -> str:
        self.observer.log_event("KnowledgeAgent", "Searching (Simulated)", query)
        
        # 1. Clean query
        clean_query = query.replace("/search", "").strip()
        
        # 2. Fetch search result (Simulated with LogSaverTool)
        search_result = self.tool.search(clean_query)
        
        # 3. Use LLM to synthesize a natural answer from the search result
        self.observer.log_event("KnowledgeAgent", "Synthesizing", "Generating final answer using LLM and search result.")
        
        final_response = self.llm.generate_response(
            context=search_result, 
            query=f"Answer the user's question: '{query}'. Use the CONTEXT provided below."
        )
        
        return final_response


%%writefile project/agents/evaluator.py
class Evaluator:
    """
    Post-processing validation (optional in this architecture, but included for structure).
    """
    def evaluate_response(self, response: str) -> bool:
        if not response:
            return False
        return True


%%writefile project/agents/planner.py
from project.agents.worker import ContentModerator, KnowledgeAgent
from project.tools.tools import LogSaverTool
from project.core.observability import Observer
from project.core.context_engineering import LLMService

class RouterAgent:
    """
    The 'Message Router Agent' described in architecture.
    Acts as the logic planner.
    """
    def __init__(self, llm_service: LLMService):
        # Initialize agents with the shared LLMService instance
        self.moderator = ContentModerator(llm_service)
        self.knowledge_agent = KnowledgeAgent(llm_service)
        self.logger_tool = LogSaverTool()
        self.observer = Observer("Router")

    def route_message(self, user_input: str) -> str:
        # 1. Moderation Step
        mod_result = self.moderator.process(user_input)
        
        intent = mod_result["intent"]
        is_safe = mod_result["is_safe"]

        # Log entry
        self.logger_tool.log_chat_event("User", user_input, f"INTENT:{intent}")

        # 2. Routing Logic
        if not is_safe:
            self.observer.log_event("Router", "Action", "Blocking content")
            return "System: Message blocked due to policy violation or API error."

        if intent == "SEARCH_QUERY":
            self.observer.log_event("Router", "Action", "Routing to Knowledge Agent")
            info = self.knowledge_agent.fetch_info(user_input)
            return info # Knowledge Agent returns the synthesized response
            
        elif intent == "SUPPORT_REQUEST":
            self.observer.log_event("Router", "Action", "Routing to Support Queue")
            return "System: A human agent has been notified to help you."
            
        else:
            self.observer.log_event("Router", "Action", "Broadcasting")
            # In a full chat system, this would be routed to a general chat LLM or group chat
            return f"Chat: {user_input}"


%%writefile project/main_agent.py
import os
import sys

# Ensure project path is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from project.agents.planner import RouterAgent
from project.memory.session_memory import SessionMemory
from project.core.observability import Observer
from project.core.context_engineering import LLMService
# We need to import the global client variables from the setup cell
# For this to run properly in a notebook, we will modify the run_agent function
# to rely on an instance being passed.

class MainAgent:
    """
    The interactive_chat_manager orchestrator.
    """
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service
        self.router = RouterAgent(self.llm)
        self.memory = SessionMemory()
        self.observer = Observer("MainAgent")

    def handle_message(self, user_input: str, user_id: str = "default_user") -> dict:
        # 1. Login / Session Context
        prefs = self.memory.get_user_preferences(user_id)
        self.observer.log_event("Auth", "Context Loaded", str(prefs))

        # 2. Hand off to Router
        response = self.router.route_message(user_input)
        
        return {
            "response": response,
            "status": "success"
        }

# This function must be executed in the notebook context where GLOBAL_LLM_SERVICE is defined
def run_agent(user_input: str, llm_service_instance: LLMService):
    agent = MainAgent(llm_service_instance)
    result = agent.handle_message(user_input)
    return result["response"]


%%writefile project/app.py
import sys
import os
from google import genai

# Ensure project path is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# This is an executable file outside the notebook, so it needs its own setup
from project.main_agent import MainAgent, LLMService

def main():
    print("--- Intelligent Multi-User Chat System (I-MUCS) ---\n")
    print("Initializing LLM Service for CLI...")
    
    # Initialize client/service for standalone execution
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set. Exiting.")
        return
        
    client = genai.Client(api_key=api_key)
    llm_service_instance = LLMService(client=client)
    agent = MainAgent(llm_service=llm_service_instance)
    
    print("Type 'exit' to quit.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit'] or not user_input:
            break
        
        result = agent.handle_message(user_input)
        print(f"\n{result['response']}")

if __name__ == "__main__":
    main()


%%writefile project/run_demo.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# This file is for testing the import structure, not running a full demo
from project.main_agent import run_agent
if __name__ == "__main__":
    print("Demo script executed.")


from project.core.context_engineering import LLMService
from project.main_agent import run_agent

if GLOBAL_GEMINI_CLIENT is None:
    print("Cannot run tests: Gemini Client failed to initialize in Cell 3.")
else:
    # Instantiate the global LLM Service for the test runs
    llm_service_instance = LLMService(client=GLOBAL_GEMINI_CLIENT, model_name=MODEL_NAME)
    print(f"LLMService initialized using model: {MODEL_NAME}")

    print("\n--- Test 1: General Chat (BROADCAST) ---")
    response_1 = run_agent("Hello, how are you today?", llm_service_instance)
    print(response_1)
    
    print("\n--- Test 2: Search Query (SEARCH_QUERY) ---")
    response_2 = run_agent("What is the I-MUCS project?", llm_service_instance)
    print(response_2)

    print("\n--- Test 3: Support Request (SUPPORT_REQUEST) ---")
    response_3 = run_agent("I need help with my account password.", llm_service_instance)
    print(response_3)
    
    print("\n--- Test 4: Offensive Content (OFFENSIVE_CONTENT) ---")
    # Note: If classification fails, the moderator blocks the message.
    response_4 = run_agent("You are a bad bot.", llm_service_instance)
    print(response_4)


!zip -r project.zip project

