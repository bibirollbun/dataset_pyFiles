import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print("ğŸ”‘ Authentication Error: Please add 'GOOGLE_API_KEY' to your Kaggle secrets.")
    raise e


from typing import Any, Dict, List
import json

from google.genai import types

from google.adk.agents import Agent, LlmAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import AgentTool
from google.adk.tools.tool_context import ToolContext

print("âœ… ADK components imported successfully.")

# Retry configuration
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# Session service (conceptual)
session_service = InMemorySessionService()
print("âœ… Session service (conceptual) created.")


def pretty_print_json(data: Any):
    print(json.dumps(data, indent=2, ensure_ascii=False))

print("âœ… Helper pretty printer ready.")


# === UniversitySearchTool: custom tool used by the University Agent ===

UNIVERSITY_DB: List[Dict[str, Any]] = [
    {
        "name": "Carnegie Mellon University",
        "country": "USA",
        "tuition_band": "high",
        "has_cs": True,
        "has_ds": True,
        "has_business": False,
        "has_psychology": False,
        "notes": "Strong CS/AI program",
    },
    {
        "name": "Arizona State University",
        "country": "USA",
        "tuition_band": "medium",
        "has_cs": True,
        "has_ds": True,
        "has_business": True,
        "has_psychology": False,
        "notes": "Good CS and Data Science options",
    },
    {
        "name": "University of Toronto",
        "country": "Canada",
        "tuition_band": "high",
        "has_cs": True,
        "has_ds": True,
        "has_business": True,
        "has_psychology": True,
        "notes": "Top global university",
    },
    {
        "name": "University of Waterloo",
        "country": "Canada",
        "tuition_band": "high",
        "has_cs": True,
        "has_ds": True,
        "has_business": False,
        "has_psychology": False,
        "notes": "Excellent for CS/Engineering",
    },
    {
        "name": "National University of Singapore",
        "country": "Singapore",
        "tuition_band": "high",
        "has_cs": True,
        "has_ds": True,
        "has_business": True,
        "has_psychology": False,
        "notes": "Top Asian tech school",
    },
    {
        "name": "BITS Pilani",
        "country": "India",
        "tuition_band": "medium",
        "has_cs": True,
        "has_ds": True,
        "has_business": False,
        "has_psychology": False,
        "notes": "Strong for CS and engineering in India",
    },
    {
        "name": "IIT Bombay",
        "country": "India",
        "tuition_band": "medium",
        "has_cs": True,
        "has_ds": True,
        "has_business": False,
        "has_psychology": False,
        "notes": "Premier Indian tech institute",
    },
    {
        "name": "Michigan State University",
        "country": "USA",
        "tuition_band": "low",
        "has_cs": True,
        "has_ds": False,
        "has_business": True,
        "has_psychology": True,
        "notes": "Affordable local option",
    },
    {
        "name": "Azimji Premji University",
        "country": "India",
        "tuition_band": "low",
        "has_cs": False,
        "has_ds": False,
        "has_business": True,
        "has_psychology": True,
        "notes": "Business and psychology focused",
    },
    {
        "name": "UT Dallas",
        "country": "USA",
        "tuition_band": "medium",
        "has_cs": True,
        "has_ds": True,
        "has_business": True,
        "has_psychology": False,
        "notes": "Popular for CS/data with moderate tuition",
    },
]


def UniversitySearchTool(profile: Dict[str, Any], main_track: str) -> dict:
    """
    Custom Tool: Filters universities based on profile constraints and chosen track.

    This function is used by the University Agent to separate universities into:
      - reach
      - target
      - safe

    Args:
        profile: structured student profile (including constraints)
        main_track: the main recommended track/major string

    Returns:
        dict: {
          "status": "success",
          "main_track": "...",
          "reach": [...],
          "target": [...],
          "safe": [...]
        }
    """
    constraints = profile.get("constraints", {})
    countries = constraints.get("countries")
    budget = constraints.get("budget_band")

    lt = main_track.lower()
    want_cs = any(x in lt for x in ["computer", "cs", "ai", "software"])
    want_ds = "data" in lt or "analytics" in lt
    want_business = any(x in lt for x in ["business", "management", "entrepreneur"])
    want_psych = any(x in lt for x in ["psychology", "behavior", "mind"])

    filtered = []
    for uni in UNIVERSITY_DB:
        if countries and uni["country"] not in countries:
            continue
        if budget and uni["tuition_band"] != budget:
            continue
        if want_cs and not uni["has_cs"]:
            continue
        if want_ds and not uni["has_ds"]:
            continue
        if want_business and not uni["has_business"]:
            continue
        if want_psych and not uni["has_psychology"]:
            continue
        filtered.append(uni)

    # Basic logic to tier universities based on their order in the list (proxy for selectivity)
    reach = filtered[:2]
    target = filtered[2:5]
    safe = filtered[5:]

    return {
        "status": "success",
        "main_track": main_track,
        "reach": reach,
        "target": target,
        "safe": safe,
    }


print("âœ… UniversitySearchTool defined.")


# === Memory Bank: Long-term student profile + plan storage ===

STUDENT_MEMORY: Dict[str, Dict[str, Any]] = {}
PLAN_MEMORY: Dict[str, Dict[str, Any]] = {}


def save_profile_tool(student_id: str, profile: Dict[str, Any]) -> dict:
    """
    Tool: Save or update a student's profile in the memory bank.
    """
    STUDENT_MEMORY[student_id] = profile
    return {"status": "success", "student_id": student_id}


def get_profile_tool(student_id: str) -> dict:
    """
    Tool: Retrieve a stored student profile from the memory bank.
    """
    profile = STUDENT_MEMORY.get(student_id)
    if profile is None:
        return {"status": "error", "error_message": f"No profile found for {student_id}"}
    return {"status": "success", "profile": profile}


def save_plan_tool(student_id: str, plan: Dict[str, Any]) -> dict:
    """
    Tool: Save or update a student's action plan in the memory bank.
    """
    PLAN_MEMORY[student_id] = plan
    return {"status": "success", "student_id": student_id}


def get_plan_tool(student_id: str) -> dict:
    """
    Tool: Retrieve a stored student action plan from the memory bank.
    """
    plan = PLAN_MEMORY.get(student_id)
    if plan is None:
        return {"status": "error", "error_message": f"No plan found for {student_id}"}
    return {"status": "success", "plan": plan}


print("âœ… Memory Bank tools defined (profiles + plans).")


# === 1) Profile Agent ===

profile_agent = LlmAgent(
    name="ProfileAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
You are the Profile Agent for Clarity AI.

You will receive a message like:

STUDENT_ID: some_id

STUDENT_DESCRIPTION:
<free-form description here>

STEP 1: Extract a structured profile JSON object with fields:
- student_id: string
- name: string (if not given, default to "Student")
- grade: integer (e.g. 9, 10, 11, 12)
- board: string (e.g. "CBSE", "ICSE", "IB", "State")
- interests: list of short strings
- strengths: list of short strings
- constraints:
    - countries: list of country names
    - budget_band: "low" | "medium" | "high"
- scores: mapping from subject to integer if mentioned, else {}.

STEP 2: Call save_profile_tool(student_id, profile) using a tool function call.
STEP 3: Return ONLY this JSON:

{
  "profile": {
    "student_id": "...",
    "name": "...",
    "grade": ...,
    "board": "...",
    "interests": [...],
    "strengths": [...],
    "constraints": {
      "countries": [...],
      "budget_band": "low|medium|high"
    },
    "scores": { ... }
  }
}

IMPORTANT:
- The final answer MUST be ONLY valid JSON.
- Do NOT include markdown, headings, or extra text.
""",
    tools=[save_profile_tool],
)

print("âœ… ProfileAgent created.")


# === 2) Mapper Agent ===

mapper_agent = LlmAgent(
    name="MapperAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
You are the Mapper Agent for Clarity AI.

You must perform two phases strictly:

## PHASE 1: Track Mapping
1. Propose 2â€“3 academic/career tracks or majors based on the profile.
2. Choose ONE main_track that is most aligned.

## PHASE 2: Action Plan Synthesis & Saving (CRITICAL)
3. Using the chosen main_track and the student's grade/constraints, generate the Plan JSON (summary, short_term_tasks, medium_term_tasks, long_term_tasks).
4. CRITICAL: You MUST call save_plan_tool(student_id, plan_json) with the structured plan JSON you generated.

Return ONLY this JSON: { "tracks": [ ... ], "main_track": "string" }
""",
    # Mapper Agent now needs the Plan saving tool
    tools=[save_plan_tool], 
)
print("âœ… MapperAgent created.")


# === 3) University Agent ===
university_agent = LlmAgent(
    name="UniversityAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
You are the University Agent and **FINAL REPORTER** for Clarity AI.

You must follow these steps:

1. Retrieve the saved plan data using get_plan_tool(student_id).
2. Call UniversitySearchTool(profile, main_track) as a tool to get the shortlist.
3. Generate a single, comprehensive, human-readable **Markdown report** that includes:
   - All data captured from the Profile Agent.
   - The Main Track chosen by the Mapper Agent.
   - The full Reach/Target/Safe University Shortlist returned by the tool.
   - The Action Plan Timeline retrieved in Step 1.
   
**IMPORTANT OUTPUT RULE:**
- **DO NOT** return any JSON or code fences.
- The final response MUST be a single, **human-readable Markdown report** following the structure you defined.
""",
    # Tools: UniversitySearchTool and the memory retrieval tool
    tools=[UniversitySearchTool, get_plan_tool], 
)

print("âœ… UniversityAgent created.")


# === 5) Mentor Agent (Loop Agent) ===

mentor_agent = LlmAgent(
    name="MentorAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
You are the Mentor Agent for Clarity AI. You support long-running, continuous guidance for the student.

You will receive:
- A student_id string
- A progress_update string (student describing what they have done since the last plan)

Your steps:
1. Use get_profile_tool(student_id) and get_plan_tool(student_id) to retrieve the original profile and plan.
2. Compare the student's progress_update with the existing timeline.
3. Generate a single, encouraging, and actionable **Markdown message** for the student.

**IMPORTANT OUTPUT RULE:**
- **DO NOT** return any JSON or code fences.
- The final response MUST be a single, **human-readable Markdown message**.

**Message Structure:**
# â­� Progress Check-in for [Student Name]
---
## Status & Feedback
* **Great Job:** Summarize the student's accomplishments (e.g., Python course, coding club) and relate them to their original track.
* **Next Focus:** Gently remind them of an overdue or crucial next task (e.g., SAT Prep) from the Medium-Term tasks.

## ğŸ’¡ Updated Suggestions (Next 30-90 Days)
Based on your progress, focus on:
* [Refined Step 1 for the next 30 days]
* [Refined Step 2 for the next 30 days]
* [Refined Step 3 for the next 30 days]
""",
    tools=[get_profile_tool, get_plan_tool],
)

print("âœ… MentorAgent revised for clear Markdown guidance.")


# === Root Sequential Multi-Agent System ===

root_agent = SequentialAgent(
    name="ClarityPipeline",
    sub_agents=[
        profile_agent,
        mapper_agent,
        university_agent,
        #action_plan_agent,
    ],
)

print("âœ… Root SequentialAgent (ClarityPipeline) created.")

# In this ADK version, InMemoryRunner only takes the agent
runner = InMemoryRunner(root_agent)
print("âœ… InMemoryRunner created for ClarityPipeline.")

# Separate runner for the MentorAgent loop
mentor_runner = InMemoryRunner(mentor_agent)
print("âœ… InMemoryRunner created for MentorAgent.")


# ============= UNIVERSAL DEBUG PRINTER (Unchanged) ===============

def debug_print_events(events):
    print("\n==================== AGENT TURNS ====================\n")

    for turn in events:
        who = getattr(turn, "source", "Unknown")
        print(f"{who} >\n")

        text_found = False

        # 1) content.text
        if (
            hasattr(turn, "content")
            and turn.content is not None
            and hasattr(turn.content, "text")
            and turn.content.text
        ):
            print(turn.content.text)
            text_found = True

        # 2) content.parts[].text
        if (
            hasattr(turn, "content")
            and turn.content is not None
            and hasattr(turn.content, "parts")
            and turn.content.parts is not None
        ):
            for p in turn.content.parts:
                if hasattr(p, "text") and p.text:
                    print(p.text)
                    text_found = True

        # 3) delta.text
        if hasattr(turn, "delta") and turn.delta is not None:
            if hasattr(turn.delta, "text") and turn.delta.text:
                print(turn.delta.text)
                text_found = True

        # 4) message str
        if hasattr(turn, "message") and isinstance(turn.message, str):
            print(turn.message)
            text_found = True

        if not text_found:
            print("[No text output]")

        print("\n------------------------------\n")


# ============================================================
# DEMO: FULL MULTI-AGENT PIPELINE RUN (WITH PARSED FINAL JSON)
# ============================================================

import asyncio # Ensure asyncio is imported here

demo_description = """
My name is Ayesha. I'm in Grade 12 IBDP in India.
I love computer science, coding, math, and business studies and I enjoy watching tech videos, tv series.
I'm also curious about startups and entrepreneurship.
My family can afford high tuition, and Iâ€™d like to study in the USA or Canada if possible.
My scores: Math 92, Physics 88, English 80, business studies 95.
"""

prompt = f"""
STUDENT_ID: student_001

STUDENT_DESCRIPTION:
{demo_description}
"""

# Reset memory for a clean run
if "student_001" in STUDENT_MEMORY:
    del STUDENT_MEMORY["student_001"]
if "student_001" in PLAN_MEMORY:
    del PLAN_MEMORY["student_001"]
print("âœ… Memory reset for student_001.")


# Run through full multi-agent pipeline with debug info
# The runner handles all agent calls, tool calls, and sequential execution.
response = await runner.run_debug(prompt)


# ---- 1. FILTERED AGENT TURNS (Show the final clear Markdown output) ----
print("\n==================== AGENT TURNS (FILTERED OUTPUT) ====================\n")
for i, turn in enumerate(response):
    who = getattr(turn, "source", f"Turn {i}")
    
    text = None
    if hasattr(turn, "content") and turn.content is not None:
        if hasattr(turn.content, "text") and turn.content.text:
            text = turn.content.text
        elif hasattr(turn.content, "parts") and turn.content.parts is not None:
            # Join all text parts from the turn
            texts = [p.text for p in turn.content.parts if hasattr(p, "text") and p.text]
            if texts:
                text = ("\n".join(texts))
    
    # CRITICAL FILTER: Only print the final agent's Markdown output
    if text and who == "ActionPlanAgent":
        print(f"--- OUTPUT FROM: {who} ---")
        print(text.strip())
        print("\n------------------------------\n")


# ---- 2. CONFIRMATION OF INTERNAL JSON SAVE (Addressing the error) ----
print("\n===== CONFIRMATION OF INTERNAL JSON SAVE (Plan Memory Check) =====\n")

if PLAN_MEMORY.get("student_001"):
    print("âœ… Plan successfully found in PLAN_MEMORY.")
    print("This confirms the ActionPlanAgent successfully called the save_plan_tool.")
    pretty_print_json(PLAN_MEMORY["student_001"])
else:
    print("âš ï¸� Plan NOT found in PLAN_MEMORY.")
    print("This indicates the ActionPlanAgent either did not call save_plan_tool or the tool call failed.")
    # Show the last response content to help debug the failure reason
    last_turn = response[-1]
    print("\n--- Last Agent Response Content (Check for missing tool call) ---")
    print(last_turn)


# ============================================================
# PROGRESS UPDATE: MENTOR AGENT LOOP RUN (WITH PARSED OUTPUT)
# ============================================================

import asyncio

progress_update = """
Hi, this is Ayesha again. Since the last plan:
- I finished one Python course and built a small project.
- I joined the school coding club.
- I started SAT prep yet, but I'm planning to start next month.
What should I focus on next?
"""

mentor_prompt = json.dumps(
    {
        "student_id": "student_001",
        "progress_update": progress_update,
    },
    indent=2,
)

# Use the separate runner for the Mentor Agent
mentor_runner = InMemoryRunner(mentor_agent)

# Run the Mentor Agent
mentor_response = await mentor_runner.run_debug(mentor_prompt)

# === CRITICAL FIX: AGGRESSIVE READ LOOP ===
# We will explicitly pause AND aggressively re-read the response object 
# until it is populated with the final JSON content.

MAX_WAIT_TIME = 10 # Total time to wait for output (in seconds)
sleep_interval = 1 
elapsed_time = 0

while elapsed_time < MAX_WAIT_TIME:
    # Check if the last response part contains content (text or a function response)
    if hasattr(mentor_response[-1], 'content') and mentor_response[-1].content:
        # Check if any text was successfully streamed. If so, break the loop.
        if hasattr(mentor_response[-1].content, 'text') and mentor_response[-1].content.text:
            break
        elif hasattr(mentor_response[-1].content, 'parts') and mentor_response[-1].content.parts:
             if any(p.text for p in mentor_response[-1].content.parts if hasattr(p, 'text')):
                break
    
    # If no final output yet, pause and wait
    # We re-run the runner here to ensure we get a fresh object if the first call was partial/interrupted
    mentor_response = await mentor_runner.run_debug(mentor_prompt) 
    
    await asyncio.sleep(sleep_interval)
    elapsed_time += sleep_interval
    print(f"[Wait: {elapsed_time}s] Waiting for MentorAgent output...")

# Ensure mentor_response is updated after the loop finishes waiting
mentor_response = await mentor_runner.run_debug(mentor_prompt) 


# ---- Extract the FINAL MentorAgent Markdown Output ----
mentor_last = mentor_response[-1]
mentor_text = None

if hasattr(mentor_last, "content") and mentor_last.content is not None:
    if hasattr(mentor_last.content, "text") and mentor_last.content.text:
        mentor_text = mentor_last.content.text
    elif hasattr(mentor_last.content, "parts") and mentor_last.content.parts is not None:
        texts = [p.text for p in mentor_last.content.parts if hasattr(p, "text") and p.text]
        if texts:
            mentor_text = "\n".join(texts)

print("\n===== FINAL MENTOR AGENT OUTPUT =====\n")
print(mentor_text or "[No final text]")

print("\nâœ… MentorAgent successfully demonstrated retrieval and updated suggestions.")


# ============================================================
# FINAL OUTPUT SUMMARY AND CREATION OF SUBMISSION FILE FOR KAGGLE(IF NEEDED)
# ============================================================

import json

print("ğŸ”„ Running Clarity AI pipeline to regenerate final output...")

demo_description = """
My name is Ayesha. I'm in Grade 12 IBDP in India.
I love computer science, coding, math, and business studies and I enjoy watching tech videos, tv series.
I'm also curious about startups and entrepreneurship.
My family can afford high tuition, and Iâ€™d like to study in the USA or Canada if possible.
My scores: Math 92, Physics 88, English 80, business studies 95.
"""

prompt = f"""
STUDENT_ID: student_001
STUDENT_DESCRIPTION:
{demo_description}
"""

# Run the pipeline fully again (Profile -> Mapper -> University -> Plan)
demo_response = await runner.run_debug(prompt)

# Extract final turn (ActionPlanAgent)
last_turn = demo_response[-1]
final_text = None

# Safe text extraction (covers text, parts, delta)
if hasattr(last_turn, "content") and last_turn.content:
    if hasattr(last_turn.content, "text") and last_turn.content.text:
        final_text = last_turn.content.text
    elif hasattr(last_turn.content, "parts") and last_turn.content.parts:
        pieces = [p.text for p in last_turn.content.parts if hasattr(p, "text") and p.text]
        if pieces:
            final_text = "\n".join(pieces)

# Fallback
if final_text is None:
    final_text = "[No final text produced by ActionPlanAgent]"

# Try parsing JSON
json_payload = {}
try:
    json_candidate = final_text[final_text.find("{"):]
    json_payload = json.loads(json_candidate)
    print("âœ… Parsed final output JSON.")
except Exception as e:
    print("âš ï¸� JSON parsing failed:", e)
    json_payload = {"raw_output": final_text}

# Save output file
output_filename = "clarity_output.json"
with open(output_filename, "w") as f:
    json.dump(json_payload, f, indent=2, ensure_ascii=False)

print(f"ğŸ“� Saved submission file: {output_filename}")
print("ğŸ�‰ You can now SUBMIT this file to the competition.")

