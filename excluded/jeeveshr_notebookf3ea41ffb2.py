import os
import pandas as pd
from kaggle_secrets import UserSecretsClient
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import FunctionTool
from google.adk.runners import InMemoryRunner # Used for running tests in this notebook

# --- 1. Environment Setup and Configuration ---
try:
    # Set API Key from Kaggle secrets
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print(" âœ… Gemini API key setup complete.")
except Exception as e:
    print(f" â�Œ Authentication Error: {e}")

# Configure Retry Options
retry_config = types.HttpRetryOptions(
    attempts=5, exp_base=7, initial_delay=1, http_status_codes=[429, 500, 503, 504]
)

# --- 2. Tool Definitions ---

# KNOWLEDGE_BASE is simplified to avoid complex external dependencies for stability
KNOWLEDGE_BASE = {
    "leave_policy": {
        "title": "Employee Vacation and Sick Leave Policy",
        "content": "All full-time employees are granted 20 days of paid annual leave. Sick leave is 10 days per year. Unused sick leave does not roll over. Unused vacation rolls over up to a maximum of 5 days.",
    },
    "inventory_steps": {
        "title": "Monthly Inventory Count SOP",
        "content": "Inventory must be counted on the last business day of every month. Procedure: 1. Print 'Current Stock Report'. 2. Count stock. 3. Enter discrepancies.",
    },
}

def search_docs(query: str) -> dict:
    """Searches internal docs (SOPs, Policies)."""
    query_lower = query.lower()
    results = {}
    for key, doc in KNOWLEDGE_BASE.items():
        if query_lower in key or any(word in doc['content'].lower() for word in query_lower.split()):
            results[doc['title']] = doc['content']
    return {"status": "success", "search_results": results} if results else {"status": "info", "message": "No documents found."}

def create_ticket_placeholder(summary: str, requester: str, priority: str = "Medium") -> dict:
    """Acknowledges a ticket request with a placeholder ID."""
    return {
        "status": "success", 
        "ticket_id": f"HLP-{abs(hash(summary)) % 1000:04d}", 
        "message": f"Placeholder: Ticket for '{summary}' acknowledged for {requester} with {priority} priority."
    }
    

# --- 3. Agent Definition (In-Memory for Notebook Stability) ---
helpdesk_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="LocalHelpdeskAgent",
    description="Helpdesk agent for small business questions.",
    # --- Corrected Instruction Block for Synthesis ---
instruction="""You are a powerful AI assistant dedicated to the S.O.I.A. helpdesk. Your primary function is to resolve user requests and **ALWAYS** provide a final, synthesized text response.

RULES:
1.  **Tool Use is Mandatory:** For Policy/Issue queries, you MUST use the correct tool.
2.  **Synthesis is CRITICAL:** After every tool call, you MUST take the information returned (e.g., policy content or ticket ID) and synthesize it into a polite, complete, user-facing sentence.
3.  **NO RAW OUTPUT:** You are strictly prohibited from showing the raw output of the tool. Your final message must be pure, conversational text.

Workflow:
- Policy Questions (Knowledge Retrieval): Use `search_docs`. Output the policy summary.
- Issue/Action Requests (Service Delegation): Use `create_ticket_placeholder`. Output the confirmed Ticket ID and message.

You MUST NOT answer with "No explicit final response from agent."
""",
# ---------------------------------------------------
    tools=[FunctionTool(func=search_docs), FunctionTool(func=create_ticket_placeholder)],
)

# --- 4. Define In-Memory Runner for Testing ---
runner = InMemoryRunner(agent=helpdesk_agent, app_name="local_helpdesk_app_in_memory")
print(" âœ… Agent loaded and In-Memory Runner configured.")


async def run_test(query: str):
    """
    Runs a query, gets the full list of events synchronously, 
    and safely processes the final response, protecting against NoneType errors.
    """
    print(f"\nğŸ‘¤ User: {query}")
    print("-" * 40)
    
    # 1. Await the result to get the full list of events 
    response_events = await runner.run_debug(query)
    
    final_response_text = "No explicit final response from agent."
    
    # 2. Iterate safely over the synchronous list of events
    if response_events is not None:
        for event in response_events:
            # Check for the final response event AND check if event.content exists
            if event.is_final_response() and event.content:
                
                # --- CRITICAL FIX: Check if the 'parts' list is not None before iterating ---
                if event.content.parts is not None:
                    for part in event.content.parts:
                        if part.text:
                            final_response_text = part.text
                            break
                
                # Since this is the final response event, we break the outer loop
                break
            
    print(f"ğŸ¤– Agent: {final_response_text}")
    print("-" * 40)


# Assuming the corrected run_test function has been executed.

# Test 1: Policy Retrieval
await run_test("What is the maximum vacation rollover?")

# Test 2: Action/Ticket Creation
await run_test("My name is John, and I need a new mouse and high priority.")

