import os
import json
from typing import Any, Dict, List, Optional
import logging

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.tools import AgentTool, load_memory, preload_memory
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.genai import types

from kaggle_secrets import UserSecretsClient
import asyncio
import time


# Configure your Gemini API Key
try:
    user_secrets = UserSecretsClient()
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_Key")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


# Configure Retry Options
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# Constants
TASK_DB_FILE = "task_database.json"
USER_ID = "user123"
APP_NAME = "ai_personal_assistant"
model = "gemini-2.5-flash"


# ---  Define the Callback Function ---
async def auto_save_to_memory(callback_context):
    """Automatically save session to memory after each agent turn."""
    # This ensures that as soon as the agent finishes, the data is stored in long-term memory
    await callback_context._invocation_context.memory_service.add_session_to_memory(
        callback_context._invocation_context.session
    )
    print("System: ğŸ’¾ [Auto-Saved Session to Memory]")


# Mock Database for Financial Transactions
transaction_db = []

def log_expense(item: str, amount: float, category: str) -> dict:
    """
    Logs a financial transaction into the database.
    
    Args:
        item: Description of the purchase (e.g., "Coffee", "Laptop").
        amount: Cost of the item.
        category: Budget category (e.g., "Food", "Tech", "Bills").
    """
    transaction = {"item": item, "amount": amount, "category": category}
    transaction_db.append(transaction)
    return {"status": "success", "message": f"Logged ${amount} for {item} in {category}.", "current_db_size": len(transaction_db)}

def get_recent_transactions() -> List[Dict]:
    """Returns the list of all logged transactions for analysis."""
    return transaction_db


# Helper Agent to do calculation
calculation_agent = LlmAgent(
    name="CalculationAgent",
    model=model,
    instruction="""You are a specialized calculator that ONLY responds with Python code.
    
    **RULES:**
    1. Your output MUST be ONLY a Python code block.
    2. Do NOT write any text before or after the code block.
    3. The Python code MUST calculate the result.
    4. The Python code MUST print the final result to stdout.
    5. You are PROHIBITED from performing the calculation yourself.
    
    Failure to follow these rules will result in an error.
    """,
    code_executor=BuiltInCodeExecutor()
)


# Agent A: Financial Organizer (Uses Custom Tools)
finance_agent = LlmAgent(
    name="FinancialOrganizer",
    model=model,
    instruction="""You are a financial assistant. 
    Use the `log_expense` tool to track user spending.
    After successfully logging an expense, **always generate a simple, direct confirmation** of the transaction,
    for example: 'OK. I've logged [AMOUNT] for [ITEM].'
    
    **IMPORTANT**: Always categorize expenses immediately using logical categories if the user doesn't specify one.
    Common categories: Food, Clothing, Tech, Bills, Entertainment, Transportation, Other.
    Do NOT ask for category - just pick the most appropriate one and log it.""",
    tools=[log_expense, get_recent_transactions]
)

# Agent B: Accountability Coach (Uses Memory)
# We enable `load_memory` so it can recall past goals/conversations.
coach_agent = LlmAgent(
    name="AccountabilityCoach",
    model=model,
    instruction="""You are a tough-love accountability coach.
    
    **IMPORTANT BEHAVIOR:**
    1. When a user states a goal (like "I want to save $5000"), acknowledge it forcefully
    2. This conversation is AUTOMATICALLY saved to your long-term memory
    3. When asked about past goals, use `load_memory` with a relevant search query
    
    **SEARCH TIPS:**
    - Search for keywords like "save", "goal"
    - Try multiple searches if the first one doesn't work
    - The memory system has stored previous conversations
    """,
    tools=[load_memory] 
)

# Agent C: Reporting Agent (Uses Code Execution)
# This agent writes Python code to calculate stats, ensuring math accuracy.
reporting_agent = LlmAgent(
    name="ReportingAgent",
    model=model,
    instruction="""You are a data analyst with access to transaction data.
    
    **MANDATORY WORKFLOW:**
    1. **YOUR FIRST ACTION MUST BE TO CALL `get_recent_transactions()` to retrieve the transaction data.** Do NOT skip this step and do NOT ask for permission.
    2. Analyze the returned list of dictionaries.
    3. Use CalculationAgent to compute totals/stats ONLY AFTER successfully getting the data.
    4. Present clear results.
    
    The `get_recent_transactions` tool returns all logged expenses automatically.""",
    tools=[
        get_recent_transactions,
        AgentTool(agent=calculation_agent)  # Use calculation agent as a tool!
    ]
)


# This agent routes traffic to the specialists.
root_agent = LlmAgent(
    name="AutopilotOrchestrator",
    model=model,
    instruction="""You are the Chief of Staff for the user.
    Analyze the user's request and route it to the correct specialist.
    
    **CRITICAL MEMORY RULE:** You have access to the user's long-term memory. 
    If the memory contains information relevant to the user's request (like past goals), 
    you MUST include that information in the arguments when calling the specialist agent.
    
    **Delegation Rules:**
    - For spending/buying/logging expenses -> FinancialOrganizer
    - For goals, saving, motivation -> AccountabilityCoach
    - For summaries/stats/reports -> ReportingAgent
    
    If the user just says hello or makes small talk, answer directly.
    """,
    tools=[
        AgentTool(finance_agent),
        AgentTool(coach_agent),
        AgentTool(reporting_agent),
        preload_memory
    ],
    # This creates the automatic link between Session and Memory
    after_agent_callback=auto_save_to_memory
)


# Sessions & Runner

session_service = InMemorySessionService() # Short-term context
memory_service = InMemoryMemoryService()   # Long-term recall

# Initialize Runner with Observability Plugin
runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service,
    plugins=[LoggingPlugin()] # Logs inputs/outputs for debugging
)


async def run_demo():
    user_id = USER_ID
    session_id = "session_01"

    print("--- ğŸ�� STARTING AUTOPILOT DEMO ---")

    # Scenario 1: Set a Goal (Populate Memory)
    user_prompt = "My goal is to learn the fundamentals of AI and python by the end of this January 2026"
    print("\nUser: " + user_prompt)
    await runner.run_debug(
        user_prompt,
        user_id=user_id, 
        session_id=session_id
    )

    # After first conversation, check what's stored
    search_result = await memory_service.search_memory(
        app_name=APP_NAME,
        user_id=user_id, 
        query="goal"  # Simple keyword
    )
    
    print(f"ğŸ”� Found {len(search_result.memories)} memories")
    for mem in search_result.memories:
        if mem.content and mem.content.parts:
            print(f"  Memory: {mem.content.parts[0].text[:100]}")
    
    # Add delay between requests to prevent Gemini API rate limits
    time.sleep(20)
    
    # Scenario 2: Log an Expense
    user_prompt = "I just bought a vintage jacket for $150."
    print("\nUser: " + user_prompt)
    await runner.run_debug(
        user_prompt,
        user_id=user_id, 
        session_id=session_id
    )

    # Add delay between requests to prevent Gemini API rate limits
    time.sleep(20)
    
    # Scenario 3: Log an Expense - New Session
    new_session_id = "session_02"
    user_prompt = "I purchased a $200 smartwatch."
    print("\nUser: " + user_prompt)
    await runner.run_debug(
        user_prompt,
        user_id=user_id, 
        session_id=new_session_id
    )

    # Add delay between requests to prevent Gemini API rate limits
    time.sleep(20)
    
    # Scenario 4: Reporting (Code Execution via CalculationAgent)
    user_prompt = "Calculate my total spending so far based on what I told you."
    print("\nUser: " + user_prompt)
    await runner.run_debug(
        user_prompt,
        user_id=user_id, 
        session_id=new_session_id
    )

    # Add delay between requests to prevent Gemini API rate limits
    time.sleep(20)

    # Scenario 5: Ask reminder of the goal (Memory Recall via Accountability Agent)
    user_prompt = "What goal do I want to achieve?"
    print("\nUser: " + user_prompt)
    await runner.run_debug(
        user_prompt,
        user_id=user_id, 
        session_id=new_session_id
    )
    
    print("\n--- âœ… DEMO COMPLETE ---")


# --- Execute ---
# (If running in Jupyter/Colab, use await run_demo())
await run_demo()

