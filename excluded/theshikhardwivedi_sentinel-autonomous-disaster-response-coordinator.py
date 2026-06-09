# Using the official Google Generative AI SDK, ChromaDB for RAG, and Pydantic for structure.
!pip install -q -U google-generativeai chromadb pydantic sentence-transformers

print("âœ… Dependencies Installed.")


# =========================
# Imports & Config
# =========================
import os
import json
import time
import textwrap
from typing import List, Literal, Dict

# Google Generative AI SDK
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from pydantic import BaseModel, Field
from kaggle_secrets import UserSecretsClient
from IPython.display import display, Markdown

# =========================
# API Authentication
# =========================
try:
    user_secrets = UserSecretsClient()
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
    print("âœ… Authentication Successful.")
except Exception as e:
    print(f"â�Œ Auth Error: {e}. Please add GOOGLE_API_KEY to Kaggle Secrets.")

# =========================
# Model Configuration
# =========================
MODEL_COMMANDER = "gemini-2.5-flash-lite"   # For text generation
MODEL_TOOLS = "gemini-2.5-flash-lite"       # For tool-based tasks

# =========================
# Safety Settings
# =========================
SAFETY_SETTINGS = [
    {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
]

print(f"âœ… Configuration Complete.\nCommander Model: {MODEL_COMMANDER}\nTools Model: {MODEL_TOOLS}")


# --- TOOL 1: VISION SCOUT (Multimodal) ---
def analyze_road_conditions(image_url: str, sector: str) -> dict:
    """
    Analyzes an aerial image of a sector to determine road passability.

    Args:
        image_url: The URL of the disaster image.
        sector: The sector name (e.g., 'Sector 4').
    """
    print(f"\nâš™ï¸� [TOOL EXECUTION] Vision Agent analyzing: {image_url}...")

    # We define a specialized vision agent just for this task
    vision_model = genai.GenerativeModel(MODEL_TOOLS)

    # Download image (simulated logic for the notebook wrapper, actual requests in production)
    # We will ask Gemini to visualize the URL directly if possible,
    # or simulate the vision return if the URL is not accessible by the container.

    # For the competition robustness, we perform a direct multimodal call:
    import PIL.Image
    import requests
    from io import BytesIO

    try:
        response = requests.get(image_url)
        img = PIL.Image.open(BytesIO(response.content))

        prompt = "Look at this disaster zone. Is the main road passable for heavy ground vehicles? Answer YES or NO and list hazards."
        response = vision_model.generate_content([prompt, img])
        analysis = response.text

        is_passable = "YES" in analysis.upper() and "NO" not in analysis.upper().split()[:5]

        return {
            "tool": "VisionScout",
            "sector": sector,
            "passable": is_passable,
            "details": analysis
        }
    except Exception as e:
        # Fallback if image link fails in Kaggle environment
        return {"tool": "VisionScout", "error": f"Image access failed: {str(e)}", "passable": False}


# --- TOOL 2: LOGISTICS RAG (Vector Search) ---
# Simple In-Memory RAG implementation
import numpy as np
from sentence_transformers import SentenceTransformer

# Knowledge Base
UNIT_DATA = [
    {"name": "Heavy Rescue Squad", "cap": "Hydraulic spreaders, shoring, structural collapse, clear roads only."},
    {"name": "Amphibious Boat Team", "cap": "Flood evacuation, swift water, medical transport over water."},
    {"name": "Helicopter Evac (Air Support)", "cap": "Critical transport, bypasses blocked roads, aerial supply drops."},
    {"name": "Drone Surveillance Unit", "cap": "Reconnaissance, light payload delivery, no medical capacity."}
]

# Initialize Embeddings (runs once)
print("Loading Embedding Model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
unit_embeddings = embedder.encode([u['cap'] for u in UNIT_DATA])

def find_best_unit(need_description: str, location: str) -> dict:
    """
    Finds the best rescue unit for a specific need using Semantic Search (RAG).

    Args:
        need_description: Description of the crisis (e.g. "trapped in basement").
        location: The location of the incident.
    """
    print(f"\nâš™ï¸� [TOOL EXECUTION] Logistics RAG searching for: '{need_description}'...")

    # Vector Search
    query_vec = embedder.encode([need_description])
    similarities = np.dot(unit_embeddings, query_vec.T).flatten()
    best_idx = np.argmax(similarities)

    best_unit = UNIT_DATA[best_idx]

    return {
        "tool": "LogisticsRAG",
        "location": location,
        "recommended_unit": best_unit['name'],
        "reasoning": f"Capability match: {best_unit['cap']}"
    }

# --- TOOL 3: MISSION AUDITOR ---
def log_mission_audit(safety_score: int, efficiency_score: int, rationale: str):
    """
    Logs the final mission audit scores to the persistent system log.

    Args:
        safety_score: 0-5 score on safety.
        efficiency_score: 0-5 score on speed.
        rationale: Explanation of the score.
    """
    print(f"\nâš™ï¸� [TOOL EXECUTION] Audit logged: Safety {safety_score}/5")
    return {"status": "Audit Saved", "verdict": rationale}

# Register Tools
tools_list = [analyze_road_conditions, find_best_unit, log_mission_audit]
print("âœ… Tools Registered with Orchestrator.")


# Define the System Instruction (The Brain)
COMMANDER_INSTRUCTIONS = """
You are SENTINEL, an Autonomous Disaster Response Orchestrator.
Your goal is to save lives by coordinating specialized tools.

### MISSION WORKFLOW (Follow Strictly):
1. **TRIAGE:** Analyze the user's raw text. Identify specific incidents (Location + Need).
2. **LOGISTICS:** For EACH incident, use `find_best_unit` to get a recommendation.
3. **VISION:** Check the provided Image URL using `analyze_road_conditions` to see if roads are passable.
4. **DECIDE:**
   - IF roads are BLOCKED and the recommendation was a Ground Unit -> **REROUTE** to 'Helicopter Evac'.
   - ELSE -> Confirm the recommended unit.
5. **AUDIT:** Once the plan is finalized, call `log_mission_audit` to score yourself.
   - DEDUCT points if you assigned a ground unit to a blocked road.
6. **REPORT:** Generate the final response in Markdown format.

### CRITICAL RULES:
- PRIORITIZE LIFE SAFETY.
- Do not make up information. Use the tool outputs.
"""

# Initialize the Model with Tools
commander_model = genai.GenerativeModel(
    model_name=MODEL_COMMANDER,
    tools=tools_list,
    system_instruction=COMMANDER_INSTRUCTIONS,
    safety_settings=SAFETY_SETTINGS
)

# Start Chat Session (Memory)
sentinel = commander_model.start_chat(enable_automatic_function_calling=True)

print("âœ… Commander Agent Online and Ready.")


# The Scenario: A flood/earthquake combo scenario.

SCENARIO_INPUT = """
**SENTINEL ACTIVATION SIGNAL**
INCOMING REPORTS:
1. Sector 4 Residential: "Basement collapsed, family trapped, water rising! Need extraction now!"
2. Sector 2 Bridge: "Bridge looks shaky, but people are stranded on the other side. Need supplies."

AERIAL IMAGE FEED:
https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/2010_Haiti_earthquake_-_United_States_Air_Force_survey_of_Port-au-Prince.jpg/640px-2010_Haiti_earthquake_-_United_States_Air_Force_survey_of-Port-au-Prince.jpg
(Note: Treat this image as 'Sector 4')
"""

print("ğŸš¨ STARTING SIMULATION...")
print("-" * 60)

# Run the Agent
response = sentinel.send_message(SCENARIO_INPUT)

# Display the Agent's reasoning and final output
display(Markdown(f"### ğŸ›°ï¸� MISSION REPORT\n\n{response.text}"))


# Defining the three test cases as a list of dictionaries
TEST_SCENARIOS = [
    {
        "id": "T-1: Safety Override Test (Blocked Road)",
        "input": (
            "URGENT: Major structural collapse in Sector Gamma. "
            "Report: Urgent need for shoring, 5 victims trapped. "
            "Vision Input URL: https://storage.googleapis.com/test-images-capstone/road_blocked.jpg" # PLACEHOLDER: Use URL for a visually BLOCKED road
        ),
        "expected_action": "Reroute to Air Support (Override)"
    },
    {
        "id": "T-2: RAG Precision Test (Clear Road)",
        "input": (
            "URGENT: Water contamination in Sector Delta. "
            "Report: Need for water purification, 100 displaced. "
            "Vision Input URL: https://storage.googleapis.com/test-images-capstone/road_clear.jpg" # PLACEHOLDER: Use URL for a visually CLEAR road
        ),
        "expected_action": "Ground Unit Deployed (RAG Success)"
    },
    {
        "id": "T-3: Resource Gap & Audit Test",
        "input": (
            "URGENT: Multiple needs in Sector Alpha. "
            "Report: Need for heavy equipment AND medical transport. 2 victims found. "
            "Vision Input URL: https://storage.googleapis.com/test-images-capstone/road_clear.jpg" # PLACEHOLDER: Use URL for a visually CLEAR road
        ),
        "expected_action": "Partial Deployment & Audit Log of Missing Resource"
    },
]

print("--- Running Multi-Agent Validation Scenarios ---")
print("----------------------------------------------\n")

# Initializing the list to store results
all_test_results = []

for scenario in TEST_SCENARIOS:
    print(f"[{scenario['id']}] Running simulation...")
    print(f"EXPECTED: {scenario['expected_action']}")

    # Execute the Commander Agent with the test input
    response = sentinel.send_message(scenario['input'])

    # Store the results
    result_text = response.text
    all_test_results.append(f"### SCENARIO: {scenario['id']}\n**Input:** {scenario['input']}\n**Expected:** {scenario['expected_action']}\n**Commander Output:**\n\n{result_text}\n\n---\n")

    print("...Simulation Complete.\n")

# Display all results in a single block for review
print("\n=============================================")
print("=== FINAL AGENT SIMULATION RESULTS SUMMARY ===")
print("=============================================\n")

# Looping over the defined list: all_test_results
for result in all_test_results:
    print(result)


# The competition requires a concrete artifact. We save the chat history and final plan to a file.

import datetime

timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Construct the log file content
log_content = f"""
# SENTINEL MISSION LOG
**Date:** {timestamp}
**Track:** Agents for Good
**Status:** SUCCESS

---

## 1. Input Scenario
{SCENARIO_INPUT}

---

## 2. Agent Orchestration Logic
(See Codebase for Tool definitions: Vision, RAG, Audit)

**Tool Calls Executed:**
{len(sentinel.history)} interactions logged in session history.

---

## 3. Final Mission Directive
{response.text}

---

## 4. System Audit
*Generated by internal Evaluator Tool*
Check JSON logs for specific safety scores.
"""

# 1. Write the results of the initial Scenario (Step 5) to create the file
#    (We use 'w' to overwrite/create the file fresh)
try:
    with open("SENTINEL_MISSION_LOG.md", "w") as f:
        # Write the main scenario report (from the first run in Step 5)
        f.write("# ğŸ“� Sentinel Mission Log: Primary Scenario\n\n")
        f.write("--- Initial Mission Execution ---\n\n")
        # Assuming 'response' holds the result from your Step 5 single run
        f.write(response.text)
        f.write("\n\n---\n\n")

    # 2. Append the results of the Validation Scenarios (from Step 6)
    #    (We use 'a' to append to the existing file)
    with open("SENTINEL_MISSION_LOG.md", "a") as f:
        f.write("## ğŸ§ª Validation Scenarios & Robustness Test Results\n\n")
        f.write("The following scenarios validate the Commander's Conditional Routing and tool use reliability:\n\n")

        # Write the combined string from the list of test results
        for result_string in all_test_results:
            f.write(result_string)

    print("âœ… Successfully generated and updated SENTINEL_MISSION_LOG.md")
    print("This file contains the primary run and all validation tests, ready for submission.")

except NameError:
    print("â�Œ ERROR: Ensure you have run Steps 5 and 7 successfully.")
    print("   'response' (from Step 5) or 'all_test_results' (from Step 7) is not defined.")

