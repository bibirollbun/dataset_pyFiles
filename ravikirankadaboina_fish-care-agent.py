!pip install -q -U google-adk google-generativeai chromadb sentence-transformers


import os
import logging
from kaggle_secrets import UserSecretsClient

# Api Key Setup
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("API Key loaded from Secrets")
except Exception:
    # Fallback for local testing
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if GOOGLE_API_KEY:
        print("API Key loaded from Environment")
    else:
        print("API Key not found. Please set GOOGLE_API_KEY.")

# Observability Setup (logging)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("agent_debug.log"), logging.StreamHandler()]
)


import chromadb
from chromadb.utils import embedding_functions

class FishKnowledgeBase:
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(name="fish_facts")
        self._populate_db()

    def _populate_db(self):
        documents = [
            "Betta fish require a minimum of 5 gallons and a heater (78-80Â°F). They are aggressive towards other Bettas.",
            "Neon Tetras are schooling fish and need at least 6 in a group. They prefer planted tanks.",
            "The Nitrogen Cycle consists of Ammonia -> Nitrite -> Nitrate. Ammonia and Nitrite are toxic.",
            "Ich is a parasitic disease appearing as white spots. Treat with heat (86Â°F) and aquarium salt.",
            "Goldfish are cold water fish and produce high waste. They need 20+ gallons for the first fish."
        ]
        ids = [f"id_{i}" for i in range(len(documents))]
        self.collection.add(documents=documents, ids=ids)

    def search(self, query: str, n_results: int = 2) -> str:
        results = self.collection.query(query_texts=[query], n_results=n_results)
        return "\n".join(results['documents'][0])

# Initialize Knowledge Base
kb = FishKnowledgeBase()

def retrieve_knowledge(query: str) -> str:
    """Retrieves relevant fish care info from the knowledge base."""
    return kb.search(query)


import os

# Create directory for agents
os.makedirs("fish_agent", exist_ok=True)
os.makedirs("fish_agent/tools", exist_ok=True)

# Create __init__.py to make it a package
with open("fish_agent/__init__.py", "w") as f:
    f.write("")


%%writefile fish_agent/tools/tools.py
from typing import List

# Calculator tool
def calculate(expression: str) -> str:
    """
    Evaluates a mathematical expression.
    Useful for calculating tank volume (L*W*H/231) or medication dosages.
    """
    try:
        allowed_names = {"abs": abs, "round": round}
        code = compile(expression, "<string>", "eval")
        for name in code.co_names:
            if name not in allowed_names:
                raise NameError(f"Use of '{name}' is not allowed")
        result = eval(code, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error calculating expression: {e}"

# Rag tool 
# In production, we'd import the KB. For this notebook, we'll inject it.
def search_fish_db(query: str) -> str:
    """Searches the internal database for fish care facts."""
    return "[Database Search Result]"


%%writefile fish_agent/agent.py
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools.agent_tool import AgentTool
from google.genai import types
from .tools.tools import calculate

# Retry Config
retry_config = types.HttpRetryOptions(
    attempts=3,
    exp_base=2,
    initial_delay=1,
    http_status_codes=[429, 500, 503],
)

# Model Config
model = Gemini(model="gemini-2.0-flash", retry_options=retry_config)

# Specialist Agents Setup

setup_agent = LlmAgent(
    name="setup_specialist",
    model=model,
    description="Helps with aquarium setup, cycling, and equipment.",
    instruction="""
    You are the Setup Specialist.
    Explain the Nitrogen Cycle simply.
    Use the 'calculate' tool for tank volume (L*W*H/231 for gallons).
    """,
    tools=[calculate]
)

livestock_agent = LlmAgent(
    name="livestock_specialist",
    model=model,
    description="Advises on fish compatibility, diet, and stocking.",
    instruction="""
    You are the Livestock Specialist.
    Prevent 'tank busters' (fish that grow too big).
    Check compatibility.
    """
)

health_agent = LlmAgent(
    name="health_specialist",
    model=model,
    description="Diagnoses diseases and water quality issues.",
    instruction="""
    You are the Health Specialist.
    Diagnose issues based on symptoms.
    Recommend treatments.
    """
)

# Router Agent Setup

root_agent = LlmAgent(
    name="fish_care_router",
    model=model,
    instruction="""
    You are the Fish Care Agent Router.
    Analyze the user's request and delegate to the appropriate specialist:
    - setup_specialist: New tanks, equipment, cycling.
    - livestock_specialist: Fish choices, compatibility.
    - health_specialist: Sickness, algae, water problems.
    
    If the request is general, answer it yourself.
    """,
    tools=[AgentTool(setup_agent), AgentTool(livestock_agent), AgentTool(health_agent)]
)


# from IPython.core.display import display, HTML
# from jupyter_server.serverapp import list_running_servers

# def get_adk_proxy_url():
#     PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
#     ADK_PORT = "8000"
#     servers = list(list_running_servers())
#     if not servers: raise Exception("No running Jupyter servers found.")
#     baseURL = servers[0]["base_url"]
#     path_parts = baseURL.split("/")
#     kernel = path_parts[2]
#     token = path_parts[3]
#     url_prefix = f"/k/{kernel}/{token}/proxy/proxy/{ADK_PORT}"
#     return PROXY_HOST + url_prefix, url_prefix

# full_url, url_prefix = get_adk_proxy_url()
# print(f"ADK Web UI: {full_url}")
# !adk web --log_level DEBUG --url_prefix {url_prefix}


import json

# Define Test Cases
test_cases = [
    {
        "input": "I want to set up a 20 gallon tank.",
        "expected_specialist": "setup_specialist",
        "expected_keywords": ["cycle", "ammonia", "nitrate"]
    },
    {
        "input": "Can I put a Betta with a Goldfish?",
        "expected_specialist": "livestock_specialist",
        "expected_keywords": ["temperature", "space"]
    },
    {
        "input": "My fish has white spots.",
        "expected_specialist": "health_specialist",
        "expected_keywords": ["ich", "white spot"]
    }
]

print("Running Evaluation...")

# We import the agent dynamically
from fish_agent.agent import root_agent
from google.adk.runners import InMemoryRunner

# Create a runner
runner = InMemoryRunner(agent=root_agent)

for test in test_cases:
    print(f"\nInput: {test['input']}")
    
    # Run Agent
    response = await runner.run_debug(test['input'])
    
    # Evaluation Logic
    text_response = ""
    try:
        # response is a list of Event objects
        for event in response:
            if hasattr(event, 'content') and hasattr(event.content, 'parts'):
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        text_response += part.text + " "
    except Exception:
        text_response = str(response)
    
    print(f"Response: {text_response[:100]}...")
    
    # Check Keywords
    passed = all(k.lower() in text_response.lower() for k in test['expected_keywords'])
    status = "PASS" if passed else "FAIL"
    print(f"Status: {status}")

