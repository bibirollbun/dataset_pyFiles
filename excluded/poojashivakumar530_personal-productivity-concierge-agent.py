import os
from kaggle_secrets import UserSecretsClient

# CRITICAL FIX: Remove hardcoded fallback key
# Only use Kaggle Secrets API key
try:
    user_secrets = UserSecretsClient()
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("Gemini API key loaded from Kaggle Secrets")
except Exception as e:
    print(f"ERROR: Cannot retrieve GOOGLE_API_KEY from Kaggle Secrets")
    print(f"Please add your Gemini API key to Kaggle User Secrets")
    print(f"Error: {str(e)}")
    raise



# =====================================================================
# CELL 2: Import ADK Components & Setup
# =====================================================================

from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types
import json
import logging
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any
from collections import defaultdict

# Configure retry options
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504]
)

# Setup logging for observability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("âœ… ADK components imported successfully!")


# =====================================================================
# CELL 3: Custom Tools, Memory & Multi-Agent System
# =====================================================================

# ğŸ“¦ MEMORY BANK - Long-term memory storage
@dataclass
class MemoryBank:
    """Long-term memory storage for agent"""
    memories: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    
    def store(self, category: str, memory: str):
        """Store a memory in a specific category"""
        self.memories[category].append({
            'content': memory,
            'timestamp': datetime.now().isoformat()
        })
        logger.info(f"Stored memory in category '{category}'")
    
    def retrieve(self, category: str) -> List[str]:
        """Retrieve memories from a category"""
        return self.memories.get(category, [])
    
    def get_all(self) -> Dict:
        """Get all memories"""
        return dict(self.memories)

# Initialize memory bank
memory_bank = MemoryBank()

print("âœ… Memory Bank initialized!")
print("âœ… System ready for multi-agent deployment!")


# =====================================================================
# CELL 4: Multi-Agent System with Custom Tools & Gradio UI
# =====================================================================

import gradio as gr
from collections import defaultdict

# ğŸ› ï¸� CUSTOM TOOLS - Demonstrating custom tool creation
def meal_planner_tool(preferences: str = "healthy") -> dict:
    """Custom tool: Generate meal plans based on preferences"""
    logger.info(f"Meal planner called with preferences: {preferences}")
    
    meals = {
        "healthy": {
            "breakfast": "Oatmeal with berries and nuts",
            "lunch": "Grilled chicken salad with quinoa",
            "dinner": "Baked salmon with vegetables"
        },
        "vegetarian": {
            "breakfast": "Smoothie bowl with fruits",
            "lunch": "Veggie wrap with hummus",
            "dinner": "Lentil curry with rice"
        },
        "quick": {
            "breakfast": "Greek yogurt with granola",
            "lunch": "Sandwich and soup",
            "dinner": "Pasta with marinara"
        }
    }
    
    plan = meals.get(preferences.lower(), meals["healthy"])
    memory_bank.store("meal_plans", json.dumps(plan))
    return plan

def task_manager_tool(action: str, task: str = "") -> dict:
    """Custom tool: Manage tasks (add/list/complete)"""
    logger.info(f"Task manager called: {action} - {task}")
    
    if action == "add" and task:
        memory_bank.store("tasks", task)
        return {"status": "success", "message": f"Task added: {task}"}
    elif action == "list":
        tasks = memory_bank.retrieve("tasks")
        return {"status": "success", "tasks": tasks}
    else:
        return {"status": "error", "message": "Invalid action"}

print("âœ… Custom tools created!")


# ===================================================================================
# CELL 5: Create ADK Agent with Proper Directory Structure
# ===================================================================================
import os
from kaggle_secrets import UserSecretsClient

# Create directory for the agent
os.makedirs('Concierge_Agent', exist_ok=True)

# Create agent.py with root_agent
agent_code = '''import os
from google.adk.agents import Agent
from google.adk.tools import google_search
from google.genai import types
from kaggle_secrets import UserSecretsClient

# Get API key from Kaggle secrets
user_secrets = UserSecretsClient()
GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")

# Define the root agent
root_agent = Agent(
    model="gemini-2.0-flash-exp",
    name="Concierge_Agent",
    instruction="""You are a helpful Personal Productivity Assistant.
    
    You help users with:
    - Planning meals based on dietary preferences
    - Managing daily tasks (add, list, complete)
    - Searching for information on the web
    - Organizing schedules and priorities
    
    Be friendly, efficient, and proactive. Always provide actionable suggestions.
    """,
    tools=[google_search],
)
'''

# Write the agent.py file
with open('Concierge_Agent/agent.py', 'w') as f:
    f.write(agent_code)

print("Agent directory created: Concierge_Agent/")
print("Agent file created: Concierge_Agent/agent.py")
print(f"Current directory: {os.getcwd()}")
print(f"Contents: {os.listdir('Concierge_Agent')}")


# CELL 6: Setup ADK Web Interface (SAFE VERSION - NO BUTTON)
print("\n" + "="*60)
print("ADK Web Interface Setup")
print("="*60)
print("\nConfiguration:")
print("- Concierge Agent ready for deployment")
print("- ADK framework configured")
print("- Google Gemini integrated")
print("\nNOTE: ADK web server NOT started (prevents quota exhaustion)")
print("\nTo use this agent:")
print("1. Cells 1-5: Agent setup and configuration")
print("2. Test the agent with small test queries")
print("3. Monitor your API usage carefully")
print("4. Do NOT run indefinite web servers in production")
print("\n" + "="*60)
print("Setup Complete!")
print("="*60)


# Fix the agent.py file directly
import os

# Create directory first
os.makedirs('Concierge_Agent', exist_ok=True)

agent_code_fixed = '''import os
from google.adk.agents import Agent
from google.adk.tools import google_search
from kaggle_secrets import UserSecretsClient

# Get API key from Kaggle secrets
user_secrets = UserSecretsClient()
GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")

# Define the root agent
root_agent = Agent(
    model="gemini-2.0-flash-exp",
    name="Concierge_Agent",
    instruction="""You are a helpful Personal Productivity Assistant.
    
    You help users with:
    - Planning meals based on dietary preferences
    - Managing daily tasks (add, list, complete)
    - Searching for information on the web
    - Organizing schedules and priorities
    
    Be friendly, efficient, and proactive. Always provide actionable suggestions.
    """,
    tools=[google_search],
)
'''
# Write the corrected file
with open('Concierge_Agent/agent.py', 'w') as f:
    f.write(agent_code_fixed)
    
print("âœ… Agent file has been corrected!")
print("   - Fixed: name='Concierge_Agent' (underscore instead of space)")
print("   - Removed: generation_config parameter")
print("   - Removed: api_key parameter")


# CRITICAL FIX: DO NOT RUN !adk web indefinitely
# This prevents quota exhaustion
print("\n" + "="*60)
print("FINAL SETUP - ADK Web Interface")
print("="*60)
print("\nWARNING: Prevent API Quota Exhaustion")
print("-" * 60)
print("The original code had: !adk web --url_prefix ...")
print("This runs INDEFINITELY and exhausts your API quota!")
print("\nFIX: This cell now just displays configuration.")
print("\nTo use the ADK interface:")
print("1. Run only cells 1-6 for setup")
print("2. Test agent with small queries")
print("3. Monitor your API usage")
print("4. Never run !adk web in production notebook")
print("\n" + "="*60)
print("Setup Complete - Agent Ready for Use")
print("="*60)


# QUOTA EXHAUSTED - Display Status Message
from IPython.core.display import display, HTML
from jupyter_server.serverapp import list_running_servers

# Display warning message
print("\n" + "="*80)
print("âš ï¸�  GEMINI API QUOTA EXHAUSTED")
print("="*80)
print("\nYour free tier quota has been exceeded for today.")
print("\nFree Tier Limits:")
print("  â€¢ 15 requests per minute")
print("  â€¢ 1M tokens per day")
print("\nğŸ“… Quota resets daily at midnight UTC")
print("\n" + "="*80)

# Try to show the UI link
try:
    PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
    ADK_PORT = "8000"
    servers = list(list_running_servers())
    if servers:
        baseURL = servers[0]["base_url"]
        path_parts = baseURL.split("/")
        kernel = path_parts[2]
        token = path_parts[3]
        url_prefix = f"/k/{kernel}/{token}/proxy/proxy/{ADK_PORT}"
        url = f"{PROXY_HOST}{url_prefix}"
        
        # Show warning with link
        link_html = f'<div style="margin: 20px; padding: 20px; background-color: #ffebee; border: 3px solid #d32f2f; border-radius: 8px;"><h2 style="color: #d32f2f;">âš ï¸�  Quota Exhausted</h2><p><strong>Your Gemini API quota has been exceeded.</strong></p><p>Agent responses will fail until quota resets (midnight UTC).</p><a href="{url}" target="_blank" style="display: inline-block; padding: 15px 30px; background-color: #d32f2f; color: white; text-decoration: none; border-radius: 4px; font-weight: bold;">View ADK UI (Read-Only)</a></div>'
        display(HTML(link_html))
        print(f"\nUI Link: {url}")
except Exception as e:
    print(f"Error: {str(e)}")
print("\n" + "="*80)
print("Next steps: Wait for quota reset or upgrade to paid plan")
print("="*80)

