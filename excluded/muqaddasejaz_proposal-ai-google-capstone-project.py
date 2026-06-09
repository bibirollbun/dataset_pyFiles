# --- INSTALLATION AND IMPORTS ---
!pip install google-adk --upgrade
import os
import json
from typing import Any, Dict, List
from datetime import datetime
import uuid
import time
import asyncio # For running run_debug

# Core ADK imports
from google.genai import types
from google.adk.agents import Agent, LlmAgent, SequentialAgent
from google.adk.models.google_llm import Gemini 
from google.adk.runners import InMemoryRunner 
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool 

# --- Authentication and Configuration ---
try:
    from kaggle_secrets import UserSecretsClient
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… Gemini API key setup complete.")
except Exception:
    print("ğŸ”‘ Assuming API key is set or using external secrets.")

# Configuration
retry_config = types.HttpRetryOptions(attempts=5, exp_base=7, initial_delay=1, http_status_codes=[429, 500, 503, 504])
session_service = InMemorySessionService()
GEMINI_MODEL = Gemini(model_name="gemini-2.5-flash", retry_config=retry_config)

def pretty_print_json(data: Any):
    print(json.dumps(data, indent=2, ensure_ascii=False))
    
print("âœ… Configuration ready.")


# --- MEMORY BANK (For Robust State Transfer) ---
GRANT_PROPOSAL_MEMORY: Dict[str, Dict[str, Any]] = {}
GRANT_BUDGET_MEMORY: Dict[str, Dict[str, Any]] = {}
GRANT_STATUS_MEMORY: Dict[str, Dict[str, Any]] = 


# === Proposal & Idea Tools ===
def save_proposal_data_tool(proposal_id: str, data: Dict[str, Any]) -> dict:
    """Tool: Save or update the core structured research idea and methodology."""
    GRANT_PROPOSAL_MEMORY[proposal_id] = data
    return {"status": "success", "proposal_id": proposal_id}

def get_proposal_data_tool(proposal_id: str) -> dict:
    """Tool: Retrieve the core structured research idea and methodology."""
    data = GRANT_PROPOSAL_MEMORY.get(proposal_id)
    if data is None:
        return {"status": "error", "error_message": f"No proposal idea found for {proposal_id}"}
    return {"status": "success", "data": data}


# === Budget & Finance Tools ===
def save_budget_tool(proposal_id: str, budget: Dict[str, Any]) -> dict:
    """Tool: Save or update the project's financial plan/budget."""
    GRANT_BUDGET_MEMORY[proposal_id] = budget
    return {"status": "success", "proposal_id": proposal_id}

def get_budget_tool(proposal_id: str) -> dict:
    """Tool: Retrieve the project's financial plan/budget."""
    budget = GRANT_BUDGET_MEMORY.get(proposal_id)
    if budget is None:
        return {"status": "error", "error_message": f"No budget found for {proposal_id}"}
    return {"status": "success", "budget": budget}



# === Status & Review Tools ===
def save_status_tool(proposal_id: str, status_update: str) -> dict:
    """Tool: Save the latest review or status update for the proposal (e.g., critique)."""
    GRANT_STATUS_MEMORY[proposal_id] = {"status": status_update, "timestamp": datetime.now().isoformat()}
    return {"status": "success", "proposal_id": proposal_id}

def get_status_tool(proposal_id: str) -> dict:
    """Tool: Retrieve the latest saved status or review of the proposal."""
    status = GRANT_STATUS_MEMORY.get(proposal_id)
    if status is None:
        return {"status": "error", "error_message": f"No status found for {proposal_id}"}
    return {"status": "success", "status": status}


# --- Custom Tool (Budget Calculation) ---
def calculate_project_budget(duration_months: int, personnel_count: int, equipment_cost: int) -> dict:
    """
    Custom Tool: Calculates a mock grant budget based on key parameters.
    """
    PERSONNEL_RATE = 8000 * personnel_count
    PERSONNEL_TOTAL = PERSONNEL_RATE * duration_months
    
    INDIRECT_COST_RATE = 0.15 
    
    TOTAL_DIRECT_COSTS = PERSONNEL_TOTAL + equipment_cost
    INDIRECT_COSTS = TOTAL_DIRECT_COSTS * INDIRECT_COST_RATE
    TOTAL_GRANT_REQUEST = TOTAL_DIRECT_COSTS + INDIRECT_COSTS

    return {
        "status": "success",
        "budget_breakdown": {
            "Personnel Total": PERSONNEL_TOTAL,
            "Equipment Cost": equipment_cost,
            "Indirect Costs": INDIRECT_COSTS,
        },
        "total_grant_request": TOTAL_GRANT_REQUEST,
        "duration_months": duration_months,
    }

budget_calculation_tool = FunctionTool(func=calculate_project_budget)
print("âœ… Memory and Function Tools defined.")


# === 1) Idea Agent ===
idea_agent = LlmAgent(
    name="IdeaAgent",
    model=GEMINI_MODEL,
    instruction="""You will receive a research topic and parameters. STEP 1: From the topic, generate a structured JSON summary of the research idea: {"title": "...", "abstract": "...", "methodology": "..."}. STEP 2: CRITICAL: Call save_proposal_data_tool(proposal_id, idea_json) to save the structured idea. STEP 3: Return ONLY the structured idea JSON.""",
    tools=[save_proposal_data_tool],
)

# === 2) Budget Agent ===
budget_agent = LlmAgent(
    name="BudgetAgent",
    model=GEMINI_MODEL,
    instruction="""You will receive the proposal ID and the parameters. STEP 1: Call calculate_project_budget(duration_months, personnel_count, equipment_cost) using the provided parameters. STEP 2: CRITICAL: Call save_budget_tool(proposal_id, calculated_budget_json) to save the calculated budget. STEP 3: Return ONLY the budget breakdown JSON.""",
    tools=[budget_calculation_tool, save_budget_tool],
)

# === 3) Review Agent (Final Reporter) ===
review_agent = LlmAgent(
    name="ReviewAgent",
    model=GEMINI_MODEL,
    instruction="""You are the **FINAL REPORTER**. STEP 1: Retrieve the 'research_idea' using get_proposal_data_tool(proposal_id). STEP 2: Retrieve the 'budget' using get_budget_tool(proposal_id). STEP 3: Generate a single, comprehensive, professional **Markdown Grant Proposal** using ALL retrieved information. STEP 4: CRITICAL: Summarize the proposal's strength (e.g., "Highly fundable") and call save_status_tool(proposal_id, summary_status). STEP 5: **FINAL REPORTER**: Output the full Markdown Proposal.""",
    tools=[get_proposal_data_tool, get_budget_tool, save_status_tool],
)

# === Root Sequential Multi-Agent System ===
root_agent = SequentialAgent(
    name="ProposalPipeline",
    sub_agents=[idea_agent, budget_agent, review_agent],
)

# Initialize Runner
runner = InMemoryRunner(root_agent)
print("âœ… All Agents and Sequential Pipeline created.")


# --- 1. EXTRACT FINAL AGENT REPORT (ReviewAgent) ---
final_report_text = "[No Final Report Found]"
final_turn = None

# Iterate backward to find the last agent in the pipeline (ReviewAgent)
for turn in reversed(response_turns):
    
    # 1. Attempt to get the source attribute (original, now broken method)
    source = getattr(turn, 'source', None)
    
    # 2. Check the raw 'message' for the agent's name (Robust Fallback)
    message = getattr(turn, 'message', '')
    
    # Check if the turn is from the final agent
    is_review_agent = (source == "ReviewAgent") or ("ReviewAgent" in str(message))
    
    if is_review_agent:
        final_turn = turn
        break # Found the final agent's turn

# If the loop finished and we found the final turn, extract the text
if final_turn is not None:
    # Safely extract the final Markdown text from content or parts
    if hasattr(final_turn, "content") and final_turn.content:
        if hasattr(final_turn.content, "text") and final_turn.content.text:
            final_report_text = final_turn.content.text
        elif hasattr(final_turn.content, "parts") and final_turn.content.parts:
            texts = [p.text for p in final_turn.content.parts if hasattr(p, "text") and p.text]
            if texts:
                final_report_text = "\n".join(texts)
else:
    # ABSOLUTE FALLBACK: If we couldn't identify the source, assume the very last turn contains the output.
    if response_turns and hasattr(response_turns[-1], "content"):
        last_turn = response_turns[-1]
        if hasattr(last_turn.content, "text") and last_turn.content.text:
            final_report_text = last_turn.content.text
        elif hasattr(last_turn.content, "parts") and last_turn.content.parts:
             texts = [p.text for p in last_turn.content.parts if hasattr(p, "text") and p.text]
             if texts:
                final_report_text = "\n".join(texts)


print("\n==================== FINAL GRANT PROPOSAL REPORT ====================\n")
print(final_report_text.strip())
print("\n" + "="*80)



import uuid
import time
import asyncio

# ============================================================
# DEMO: FULL MULTI-AGENT PIPELINE RUN 
# ============================================================
# --- Define the necessary variables (MISSING in the input) ---
DEMO_SESSION_ID = str(uuid.uuid4()) # Unique ID
INPUT_TOPIC = "Developing a quantum-resistant encryption protocol using topological materials."

# Parameters provided in the initial prompt for the Budget Agent
PARAMETERS = {
    "duration_months": 36,
    "personnel_count": 4,
    "equipment_cost": 50000,
}

# Format the prompt to include the unique ID and all necessary parameters
prompt = f"""PROPOSAL_ID: {DEMO_SESSION_ID}
RESEARCH_TOPIC: {INPUT_TOPIC}
PARAMETERS: {json.dumps(PARAMETERS)}"""

# --- Reset memory for a clean run ---
if DEMO_SESSION_ID in GRANT_PROPOSAL_MEMORY: del GRANT_PROPOSAL_MEMORY[DEMO_SESSION_ID]
if DEMO_SESSION_ID in GRANT_BUDGET_MEMORY: del GRANT_BUDGET_MEMORY[DEMO_SESSION_ID]
if DEMO_SESSION_ID in GRANT_STATUS_MEMORY: del GRANT_STATUS_MEMORY[DEMO_SESSION_ID]
print(f"âœ… Memory reset for proposal {DEMO_SESSION_ID[:8]}...")

# Run through full multi-agent pipeline with debug info
print(f"ğŸš€ Running pipeline for Topic: {INPUT_TOPIC}")
start_time = time.time()

# CRITICAL: Execute the pipeline and capture response_turns
response_turns = await runner.run_debug(prompt) 

end_time = time.time()
print(f"â�±ï¸� Pipeline finished in {end_time - start_time:.2f} seconds.")


# --- 1. EXTRACT FINAL AGENT REPORT (ReviewAgent) ---
final_report_text = "[No Final Report Found]"
final_turn = None

# Iterate backward to find the last agent in the pipeline (ReviewAgent)
for turn in reversed(response_turns):
    
    # 1. Attempt to get the source attribute (original, now broken method)
    source = getattr(turn, 'source', None)
    
    # 2. Check the raw 'message' for the agent's name (Robust Fallback)
    message = getattr(turn, 'message', '')
    
    # Check if the turn is from the final agent
    is_review_agent = (source == "ReviewAgent") or ("ReviewAgent" in str(message))
    
    if is_review_agent:
        final_turn = turn
        break # Found the final agent's turn

# If the loop finished and we found the final turn, extract the text
if final_turn is not None:
    # Safely extract the final Markdown text from content or parts
    if hasattr(final_turn, "content") and final_turn.content:
        if hasattr(final_turn.content, "text") and final_turn.content.text:
            final_report_text = final_turn.content.text
        elif hasattr(final_turn.content, "parts") and final_turn.content.parts:
            texts = [p.text for p in final_turn.content.parts if hasattr(p, "text") and p.text]
            if texts:
                final_report_text = "\n".join(texts)
else:
    # ABSOLUTE FALLBACK: If we couldn't identify the source, assume the very last turn contains the output.
    if response_turns and hasattr(response_turns[-1], "content"):
        last_turn = response_turns[-1]
        if hasattr(last_turn.content, "text") and last_turn.content.text:
            final_report_text = last_turn.content.text
        elif hasattr(last_turn.content, "parts") and last_turn.content.parts:
            texts = [p.text for p in last_turn.content.parts if hasattr(p, "text") and p.text]
            if texts:
                final_report_text = "\n".join(texts)


print("\n==================== FINAL GRANT PROPOSAL REPORT ====================\n")
print(final_report_text.strip())
print("\n" + "="*80)

# ============================================================
# OBSERVABILITY CHECK: TRACING THE MEMORY BANK
# ============================================================

print("\n===== ğŸ”� OBSERVABILITY CHECK: GRANT MEMORY STATE =====")
print(f"--- Data stored for Proposal ID: {DEMO_SESSION_ID[:8]} ---")

# 1. Check Proposal Idea (Saved by IdeaAgent)
print("\n[PROPOSAL IDEA MEMORY]:")
if GRANT_PROPOSAL_MEMORY.get(DEMO_SESSION_ID):
    pretty_print_json(GRANT_PROPOSAL_MEMORY[DEMO_SESSION_ID])
    print("âœ… Idea Agent successfully saved its output.")
else:
    print("âš ï¸� Idea Agent output NOT found.")

# 2. Check Budget (Saved by BudgetAgent)
print("\n[BUDGET MEMORY]:")
if GRANT_BUDGET_MEMORY.get(DEMO_SESSION_ID):
    pretty_print_json(GRANT_BUDGET_MEMORY[DEMO_SESSION_ID])
    print("âœ… Budget Agent successfully saved its output.")
else:
    print("âš ï¸� Budget Agent output NOT found.")
    
# 3. Check Status/Review (Saved by ReviewAgent)
print("\n[STATUS MEMORY]:")
if GRANT_STATUS_MEMORY.get(DEMO_SESSION_ID):
    pretty_print_json(GRANT_STATUS_MEMORY[DEMO_SESSION_ID])
    print("âœ… Review Agent successfully saved the final status.")
else:
    print("âš ï¸� Review Agent status NOT found.")

print("\n======================================================\n")

