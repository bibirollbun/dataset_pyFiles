# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Installing Google's Agent Development Kit (ADK)
!pip install -q google-adk


import os

# ğŸ”‘ Google AI Studio key
GOOGLE_API_KEY = "AIzaSyAJb04ZrZBX_zmJ0ZFVMQ8ChGj99xDnGvw"

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE" 

# ğŸ”� Smoke-test ADK imports
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

print("ADK imported OK âœ…")



'''Generating small, realistic JSON files in /kaggle/working that my agent tools will read.'''


# ==== RoboShift: Synthetic factory data generation ====

import json
import random

random.seed(42)

# 1ï¸�âƒ£ Zones: where tasks happen
zones = [
    {
        "id": "Z1",
        "name": "Assembly Line A",
        "risk_level": "medium",
        "is_narrow_corridor": False,
        "max_people": 6,
        "robot_allowed": True,
        "human_allowed": True,
    },
    {
        "id": "Z2",
        "name": "Assembly Line B",
        "risk_level": "high",
        "is_narrow_corridor": True,   # narrow corridor â†’ congestion risk
        "max_people": 3,
        "robot_allowed": True,
        "human_allowed": True,
    },
    {
        "id": "Z3",
        "name": "Packaging",
        "risk_level": "low",
        "is_narrow_corridor": False,
        "max_people": 8,
        "robot_allowed": True,
        "human_allowed": True,
    },
    {
        "id": "Z4",
        "name": "Heavy Materials Storage",
        "risk_level": "high",
        "is_narrow_corridor": False,
        "max_people": 4,
        "robot_allowed": True,
        "human_allowed": False,  # prefer robots here
    },
    {
        "id": "Z5",
        "name": "Inspection Area",
        "risk_level": "medium",
        "is_narrow_corridor": False,
        "max_people": 5,
        "robot_allowed": False,
        "human_allowed": True,
    },
]

# 2ï¸�âƒ£ Robots: simple mobile robots with different capabilities
robots = [
    {
        "id": "R1",
        "name": "RoboCarrier-1",
        "max_speed_mps": 1.2,
        "battery_level": 0.9,
        "can_carry_kg": 80,
        "zone_access": ["Z1", "Z2", "Z3", "Z4"],
        "safety_rating": "high",
    },
    {
        "id": "R2",
        "name": "RoboCarrier-2",
        "max_speed_mps": 1.0,
        "battery_level": 0.6,
        "can_carry_kg": 60,
        "zone_access": ["Z1", "Z2", "Z3"],
        "safety_rating": "medium",
    },
    {
        "id": "R3",
        "name": "InspectorBot",
        "max_speed_mps": 0.8,
        "battery_level": 0.8,
        "can_carry_kg": 20,
        "zone_access": ["Z1", "Z3", "Z5"],
        "safety_rating": "high",
    },
]

# 3ï¸�âƒ£ Workers: humans with skills & PPE compliance
workers = [
    {
        "id": "W1",
        "name": "Alex",
        "role": "Operator",
        "skill_level": "high",
        "ppe_compliance_score": 0.95,
        "max_shift_hours": 8,
        "preferred_zones": ["Z1", "Z2"],
    },
    {
        "id": "W2",
        "name": "Jordan",
        "role": "Operator",
        "skill_level": "medium",
        "ppe_compliance_score": 0.85,
        "max_shift_hours": 8,
        "preferred_zones": ["Z1", "Z3"],
    },
    {
        "id": "W3",
        "name": "Taylor",
        "role": "Inspector",
        "skill_level": "high",
        "ppe_compliance_score": 0.9,
        "max_shift_hours": 8,
        "preferred_zones": ["Z3", "Z5"],
    },
    {
        "id": "W4",
        "name": "Morgan",
        "role": "Loader",
        "skill_level": "medium",
        "ppe_compliance_score": 0.7,  # slightly lower compliance
        "max_shift_hours": 8,
        "preferred_zones": ["Z3", "Z4"],
    },
    {
        "id": "W5",
        "name": "Riley",
        "role": "Operator",
        "skill_level": "low",
        "ppe_compliance_score": 0.65,
        "max_shift_hours": 6,
        "preferred_zones": ["Z1", "Z3"],
    },
    {
        "id": "W6",
        "name": "Sam",
        "role": "Supervisor",
        "skill_level": "high",
        "ppe_compliance_score": 0.98,
        "max_shift_hours": 8,
        "preferred_zones": ["Z1", "Z2", "Z3", "Z5"],
    },
]

# 4ï¸�âƒ£ Tasks: shift tasks needing robots, humans, or both
time_windows = ["day", "evening", "night"]
risk_levels = ["low", "medium", "high"]

tasks = []
task_id = 1

def add_task(description, zone_id, duration, requires_robot, requires_human,
             risk_level, time_window, line_id):
    global task_id
    tasks.append(
        {
            "id": f"T{task_id}",
            "description": description,
            "location_zone": zone_id,
            "duration_minutes": duration,
            "requires_robot": requires_robot,
            "requires_human": requires_human,
            "risk_level": risk_level,
            "time_window": time_window,
            "line_id": line_id,
        }
    )
    task_id += 1

# Hereby adding a few hand-crafted tasks to ensure interesting patterns 
add_task(
    "Move heavy material pallets to storage",
    "Z4",
    duration=45,
    requires_robot=True,
    requires_human=False,
    risk_level="high",
    time_window="night",
    line_id="Line 3",
)
add_task(
    "Manual inspection of assemblies",
    "Z5",
    duration=30,
    requires_robot=False,
    requires_human=True,
    risk_level="medium",
    time_window="night",
    line_id="Line 3",
)
add_task(
    "Load parts onto Assembly Line B",
    "Z2",
    duration=40,
    requires_robot=True,
    requires_human=True,
    risk_level="high",
    time_window="night",
    line_id="Line 3",
)

# Hereby adding some semi-random tasks for variety
for _ in range(12):
    zone = random.choice(zones)
    tw = random.choice(time_windows)
    rl = random.choice(risk_levels)
    # Ensure at least one of robot/human is required
    requires_robot = random.choice([True, False])
    requires_human = random.choice([True, False])
    if not (requires_robot or requires_human):
        requires_human = True

    add_task(
        description=f"Generic task in {zone['name']} ({rl} risk)",
        zone_id=zone["id"],
        duration=random.choice([20, 30, 45, 60]),
        requires_robot=requires_robot,
        requires_human=requires_human,
        risk_level=rl,
        time_window=tw,
        line_id=random.choice(["Line 1", "Line 2", "Line 3"]),
    )

# the below Helper function is to save JSON file
def save_json(obj, filename):
    with open(filename, "w") as f:
        json.dump(obj, f, indent=2)

save_json(zones, "zones.json")
save_json(robots, "robots.json")
save_json(workers, "workers.json")
save_json(tasks, "tasks.json")

print("Zones:", len(zones))
print("Robots:", len(robots))
print("Workers:", len(workers))
print("Tasks:", len(tasks))

print("\nSample zone:", zones[0])
print("Sample robot:", robots[0])
print("Sample worker:", workers[0])
print("Sample task:", tasks[0])



# =========
# Factory data helpers + tools
# =========
import os
import json

FACTORY_DATA_DIR = "/kaggle/working/factory_data"

os.makedirs(FACTORY_DATA_DIR, exist_ok=True)

def _load_json(filename: str):
    """
    Internal helper: load a JSON file from FACTORY_DATA_DIR.
    If the file doesn't exist yet, return an empty list.
    """
    path = os.path.join(FACTORY_DATA_DIR, filename)
    if not os.path.exists(path):
        # I am starting with empty data
        return []
    with open(path, "r") as f:
        return json.load(f)


def load_factory_data() -> dict:
    """
    Loads all factory configuration data (zones, robots, workers, tasks).

    Returns:
        dict with keys: zones, robots, workers, tasks
    """
    zones = _load_json("zones.json")
    robots = _load_json("robots.json")
    workers = _load_json("workers.json")
    tasks = _load_json("tasks.json")

    return {
        "zones": zones,
        "robots": robots,
        "workers": workers,
        "tasks": tasks,
    }


def get_shift_data(line_id: str, time_window: str = "next_8_hours") -> dict:
    """
    Tool: Return all task + resource context for a specific production line & time window.

    Args:
        line_id: ID/name of the production line (e.g. 'Line_A')
        time_window: Human label for planning window (e.g. 'next_2_hours')

    Returns:
        dict with:
            - line_id
            - time_window
            - tasks: tasks on that line
            - robots: all robots (agent will choose which ones to use)
            - workers: all workers
            - zones: all factory zones
    """
    data = load_factory_data()

    filtered_tasks = [
        t for t in data["tasks"]
        if str(t.get("line_id", "")).lower() == str(line_id).lower()
    ]

    return {
        "line_id": line_id,
        "time_window": time_window,
        "tasks": filtered_tasks,
        "robots": data["robots"],
        "workers": data["workers"],
        "zones": data["zones"],
    }

print("Factory helpers + get_shift_data tool ready âœ…")



# ==== Sync generated JSON files into FACTORY_DATA_DIR and sanity-check ====
import shutil

print("FACTORY_DATA_DIR:", FACTORY_DATA_DIR)

os.makedirs(FACTORY_DATA_DIR, exist_ok=True)

for name in ["zones", "robots", "workers", "tasks"]:
    src = f"/kaggle/working/{name}.json"
    dst = os.path.join(FACTORY_DATA_DIR, f"{name}.json")
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"Copied {src} -> {dst}")
    else:
        print(f"WARNING: {src} not found")

# smoke test for get_shift_data
sample_shift = get_shift_data(line_id="Line 3", time_window="night")
print("\nTasks for Line 3:", len(sample_shift["tasks"]))
if sample_shift["tasks"]:
    print("Example task:", sample_shift["tasks"][0])
else:
    print("No tasks found for Line 3")



# Helper: safely parse JSON from model output
def _parse_json_from_model(raw_text: str):
    """
    Parse JSON from a model response that may be wrapped
    in ```json ... ``` or ``` ... ``` fences.
    """
    cleaned = raw_text.strip()

    # If it starts with ``` (markdown code block), strip the fences
    if cleaned.startswith("```"):
        # Drop the first line (``` or ```json)
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1 :]

        # Drop trailing ```
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
    return json.loads(cleaned)


# ==== RoboShift: multi-agent setup ====

import json
import uuid

from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

APP_NAME = "roboshift_app"
USER_ID = "roboshift_user"
GEMINI_MODEL = "gemini-2.0-flash"

# ---------- 1) Define the three agents ----------

task_intake_agent = Agent(
    name="task_intake_agent",
    model=GEMINI_MODEL,
    description=(
        "Understands a supervisor's natural language request and turns it into "
        "a structured planning request for the RoboShift system."
    ),
    instruction=(
        "You are the Task Intake Agent for RoboShift, a humanâ€“robot shift planner.\n\n"
        "Your job is to read the supervisor's request and output a SINGLE JSON object "
        "with the following fields:\n"
        "{\n"
        '  "line_id": string,        // e.g. "Line 3"\n'
        '  "time_window": string,    // e.g. \"night\", \"day\", \"evening\", \"next_8_hours\"\n'
        '  "objectives": [string],   // list of high-level goals (e.g. [\"minimize risk\", \"balance workload\"])\n'
        '  "constraints": [string]   // list of important constraints the planner should respect\n'
        "}\n\n"
        "Rules:\n"
        "- If the user does NOT mention a line, default to 'Line 3'.\n"
        "- If the user mentions 'tonight', map to time_window='night'.\n"
        "- If the user says things like 'minimize risk', 'avoid congestion', "
        "  'don't overload workers', include those in objectives/constraints.\n"
        "- RESPOND WITH JSON ONLY. No extra text, no markdown, no explanation."
    ),
)

safety_planner_agent = Agent(
    name="safety_planner_agent",
    model=GEMINI_MODEL,
    description=(
        "Plans risk-aware factory shifts by assigning tasks between robots and "
        "human workers under safety and workload constraints."
    ),
    instruction=(
        "You are the Safety-Constrained Planner Agent in RoboShift.\n\n"
        "You receive:\n"
        "- A structured PLANNING_REQUEST JSON with line_id, time_window, objectives, constraints.\n"
        "- A SHIFT_DATA JSON object with:\n"
        "  - tasks: list of tasks for that line and time window\n"
        "  - robots: list of available robots and their capabilities\n"
        "  - workers: list of workers, skills, PPE compliance, max_shift_hours, preferred_zones\n"
        "  - zones: list of zones with risk_level, is_narrow_corridor, max_people, robot/human permissions\n\n"
        "Your job is to produce a draft SHIFT_PLAN that respects safety and workload constraints.\n\n"
        "When planning, follow these principles:\n"
        "- High-risk zones (risk_level='high') and zones where human_allowed=false: "
        "  prefer robots for physical tasks if possible.\n"
        "- Workers with low ppe_compliance_score (< 0.8) should avoid high-risk tasks where possible.\n"
        "- Do not assign more than zone.max_people people (workers + robots) to a zone at the same time window.\n"
        "- Avoid overloading a single worker or robot with too many long tasks; try to balance workload.\n"
        "- Be explicit about which robot/worker is assigned to which task.\n\n"
        "OUTPUT FORMAT:\n"
        "Respond with a SINGLE JSON object of the form:\n"
        "{\n"
        '  "assignments": [\n'
        "    {\n"
        '      "task_id": string,\n'
        '      "task_description": string,\n'
        '      "zone_id": string,\n'
        '      "assigned_robots": [string],\n'
        '      "assigned_workers": [string],\n'
        '      "notes": string   // brief justification, including safety reasoning\n'
        "    },\n"
        "    ...\n"
        "  ]\n"
        "}\n\n"
        "Respond with JSON ONLY. No markdown, no extra prose."
    ),
)

risk_auditor_agent = Agent(
    name="risk_auditor_agent",
    model=GEMINI_MODEL,
    description=(
        "Reviews a shift plan, highlights risk hotspots, and generates a "
        "human-readable summary report for the supervisor."
    ),
    instruction=(
        "You are the Risk Auditor & Reporter Agent for RoboShift.\n\n"
        "You receive:\n"
        "- The original natural language USER_REQUEST.\n"
        "- The PLANNING_REQUEST JSON.\n"
        "- The SHIFT_DATA summary (JSON).\n"
        "- The SHIFT_PLAN JSON produced by the planner.\n\n"
        "Your job is to:\n"
        "1) Identify potential risk hotspots (e.g., too many people in a narrow corridor, "
        "   low-compliance workers on high-risk tasks, overused robots, etc.).\n"
        "2) Suggest concrete mitigations (reassignments, staggering tasks, adding supervision).\n"
        "3) Produce a clear, human-readable report the supervisor can use directly.\n\n"
        "FORMAT YOUR FINAL ANSWER IN NATURAL LANGUAGE (NOT JSON), with sections like:\n"
        "- Overview of the plan\n"
        "- Key safety checks\n"
        "- Risk hotspots & mitigations\n"
        "- Final recommended shift summary\n"
    ),
)


# ---------- 2) Helper: run a single agent once (fresh session each time) ----------

async def _run_agent_once(agent: Agent, user_message: str) -> str:
    """
    Create a fresh in-memory session + runner for this agent call,
    run it once, and return the final text response.
    """
    session_service = InMemorySessionService()
    session_id = f"session_{agent.name}_{uuid.uuid4().hex[:8]}"

    # Create a session
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
    )

    # Runner bound to this session_service
    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    # Build user content
    content = types.Content(
        role="user",
        parts=[types.Part(text=user_message)],
    )

    final_text = ""
    events = runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=content,
    )

    async for event in events:
        if event.is_final_response():
            parts = event.content.parts if event.content and event.content.parts else []
            final_text = "".join(
                getattr(p, "text", "") for p in parts if getattr(p, "text", None)
            )

    return final_text


# ---------- 3) Full RoboShift orchestration (async) ----------

async def chat_with_roboshift(user_message: str) -> str:
    """
    Full multi-agent pipeline:
    1) Task Intake Agent -> PLANNING_REQUEST JSON
    2) get_shift_data() using line_id/time_window
    3) Safety Planner Agent -> SHIFT_PLAN JSON
    4) Risk Auditor & Reporter Agent -> final human-readable report
    """

    # 1) Task Intake
    intake_raw = await _run_agent_once(task_intake_agent, user_message)
    try:
        planning_request = _parse_json_from_model(intake_raw)
    except Exception:
        planning_request = {
            "line_id": "Line 3",
            "time_window": "night",
            "objectives": ["minimize risk", "balance workload"],
            "constraints": ["avoid congestion in narrow corridors"],
        }

    line_id = planning_request.get("line_id", "Line 3")
    time_window = planning_request.get("time_window", "night")
    
    # 2) fetching SHIFT_DATA  
    shift_data = get_shift_data(line_id=line_id, time_window=time_window)

    # 3) Safety Planner Agent
    planner_prompt = (
        "You are the Safety-Constrained Planner Agent.\n\n"
        "PLANNING_REQUEST JSON:\n"
        f"{json.dumps(planning_request, indent=2)}\n\n"
        "SHIFT_DATA JSON:\n"
        f"{json.dumps(shift_data, indent=2)}\n\n"
        "Produce the SHIFT_PLAN JSON as specified in your instructions."
    )
    planner_raw = await _run_agent_once(
        safety_planner_agent,
        planner_prompt,
    )

    try:
        shift_plan = _parse_json_from_model(planner_raw)
    except Exception:
        shift_plan = {"assignments": [], "raw": planner_raw}

    # 4) Risk Auditor & Reporter Agent
    auditor_prompt = (
        "You are the Risk Auditor & Reporter Agent.\n\n"
        "USER_REQUEST:\n"
        f"{user_message}\n\n"
        "PLANNING_REQUEST JSON:\n"
        f"{json.dumps(planning_request, indent=2)}\n\n"
        "SHIFT_DATA (JSON):\n"
        f"{json.dumps(shift_data, indent=2)}\n\n"
        "SHIFT_PLAN JSON:\n"
        f"{json.dumps(shift_plan, indent=2)}\n\n"
        "Now audit this plan for safety and produce a clear report for the supervisor."
    )
    final_report = await _run_agent_once(
        risk_auditor_agent,
        auditor_prompt,
    )

    print("RoboShift (final report):\n")
    print(final_report)
    return final_report

print("âœ… RoboShift multi-agent (per-call sessions, no asyncio.run) is ready.")



'''Testing:'''

await chat_with_roboshift("Plan tonight's shift for Line 3 focusing on minimizing safety risks.")



# ==== Debug helper: inspect internal RoboShift artifacts ====

async def debug_roboshift(user_message: str):
    """
    Runs the full multi-agent pipeline but also prints:
    - PLANNING_REQUEST JSON
    - SHIFT_DATA summary
    - SHIFT_PLAN JSON (first few assignments)
    - FINAL REPORT (natural language)

    Returns a dict with all of them.
    """
    # 1) Task Intake
    intake_raw = await _run_agent_once(task_intake_agent, user_message)
    try:
        planning_request = _parse_json_from_model(intake_raw)
    except Exception:
        planning_request = {
            "line_id": "Line 3",
            "time_window": "night",
            "objectives": ["minimize risk", "balance workload"],
            "constraints": ["avoid congestion in narrow corridors"],
        }

    line_id = planning_request.get("line_id", "Line 3")
    time_window = planning_request.get("time_window", "night")

    # 2) fetching SHIFT_DATA 
    shift_data = get_shift_data(line_id=line_id, time_window=time_window)

    # 3) Safety Planner Agent
    planner_prompt = (
        "You are the Safety-Constrained Planner Agent.\n\n"
        "PLANNING_REQUEST JSON:\n"
        f"{json.dumps(planning_request, indent=2)}\n\n"
        "SHIFT_DATA JSON:\n"
        f"{json.dumps(shift_data, indent=2)}\n\n"
        "Produce the SHIFT_PLAN JSON as specified in your instructions."
    )
    planner_raw = await _run_agent_once(
        safety_planner_agent,
        planner_prompt,
    )

    try:
        shift_plan = _parse_json_from_model(planner_raw)
    except Exception:
        shift_plan = {"assignments": [], "raw": planner_raw}

    # 4) Risk Auditor & Reporter Agent
    auditor_prompt = (
        "You are the Risk Auditor & Reporter Agent.\n\n"
        "USER_REQUEST:\n"
        f"{user_message}\n\n"
        "PLANNING_REQUEST JSON:\n"
        f"{json.dumps(planning_request, indent=2)}\n\n"
        "SHIFT_DATA (JSON):\n"
        f"{json.dumps(shift_data, indent=2)}\n\n"
        "SHIFT_PLAN JSON:\n"
        f"{json.dumps(shift_plan, indent=2)}\n\n"
        "Now audit this plan for safety and produce a clear report for the supervisor."
    )
    final_report = await _run_agent_once(
        risk_auditor_agent,
        auditor_prompt,
    )

    # ---- for displaying the debugged steps ----
    print("=== PLANNING_REQUEST ===")
    print(json.dumps(planning_request, indent=2))

    print("\n=== SHIFT_DATA (summary) ===")
    print(
        f"Tasks: {len(shift_data['tasks'])}, "
        f"Robots: {len(shift_data['robots'])}, "
        f"Workers: {len(shift_data['workers'])}, "
        f"Zones: {len(shift_data['zones'])}"
    )

    print("\n=== SHIFT_PLAN (first few assignments) ===")
    assignments = shift_plan.get("assignments", [])
    if assignments:
        for a in assignments[:5]:
            print(json.dumps(a, indent=2))
            print("-" * 40)
    else:
        print(shift_plan)

    print("\n=== FINAL REPORT ===")
    print(final_report)

    return {
        "planning_request": planning_request,
        "shift_data": shift_data,
        "shift_plan": shift_plan,
        "final_report": final_report,
    }

print("âœ… debug_roboshift updated to use _parse_json_from_model")



'''Testing:'''

_ = await debug_roboshift(
    "Plan tonight's shift for Line 3, minimizing risk and avoiding congestion in narrow corridors."
)



# === Different supervisor requests ===

print("=== Demo 1: Night shift â€“ minimize safety risk ===\n")
await chat_with_roboshift(
    "Plan tonight's shift for Line 5 focusing on minimizing safety risks."
)

print("\n\n=== Demo 2: Day shift â€“ balance workload, keep robots busy ===\n")
await chat_with_roboshift(
    "Plan today's day shift for Line 2, balance workload across workers and keep robots utilized, but avoid overcrowding narrow corridors."
)

print("\n\n=== Demo 3: Short shift â€“ prioritize human training ===\n")
await chat_with_roboshift(
    "Plan a short 4-hour shift for Line 1 that prioritizes training junior workers while keeping high-risk tasks with robots."
)


