# --- Basic setup for this notebook (no special SDK needed yet) ---

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from datetime import datetime
import textwrap

print("Setup complete âœ…")


# ---------------------------
#  KNOWLEDGE BASE (KB)
# ---------------------------

KNOWLEDGE_BASE = {
    "sin": {
        "name": "Social Insurance Number (SIN)",
        "requirements": [
            "Passport",
            "Work Permit or Study Permit or PR Confirmation",
            "Local Canadian address"
        ],
        "steps": [
            "Visit Service Canada (walk-in or appointment)",
            "Provide identity + immigration documents",
            "Receive SIN immediately"
        ],
        "processing_time": "Same day (instant)",
        "documents": ["Passport", "Permit/PR Document"]
    },
    
    "health_card": {
        "name": "Provincial Health Card",
        "requirements": [
            "Proof of address in province",
            "Work/Study Permit or PR Document",
            "Passport"
        ],
        "steps": [
            "Apply online or visit ServiceOntario/Health Office",
            "Submit ID and proof of address",
            "Receive card by mail"
        ],
        "processing_time": "2â€“4 weeks",
        "documents": ["Lease", "Bank Statement", "Passport"]
    },
    
    "bank_account": {
        "name": "Open a Bank Account",
        "requirements": [
            "Passport",
            "Permit/PR",
            "Initial deposit (optional)"
        ],
        "steps": [
            "Visit bank (TD, RBC, Scotia, CIBC, BMO)",
            "Provide ID + immigration documents",
            "Choose chequing account"
        ],
        "processing_time": "Same day",
        "documents": ["Passport", "PR/Permit"]
    },
    
    "drivers_license": {
        "name": "Get Driverâ€™s License",
        "requirements": [
            "Proof of residency",
            "Passport",
            "Driverâ€™s license from your country (optional)"
        ],
        "steps": [
            "Visit DriveTest centre",
            "Take knowledge test (G1)",
            "Take road test (G2/G)"
        ],
        "processing_time": "Varies (1â€“3 months)",
        "documents": ["Passport", "Proof of address"]
    },
    
    "school_admission": {
        "name": "Child School Admission",
        "requirements": [
            "Child passport",
            "Proof of address",
            "Parent IDs"
        ],
        "steps": [
            "Find your school board",
            "Register online",
            "Submit documents",
            "Attend school interview/orientation"
        ],
        "processing_time": "1â€“2 weeks",
        "documents": ["Passport", "Lease", "Immunization record"]
    }
}


# ---------------------------
# KB SEARCH TOOL
# ---------------------------

def search_kb(query: str):
    """
    Simple keyword search through KNOWLEDGE_BASE.
    Returns best matching settlement topic.
    """
    q = query.lower()
    for key, data in KNOWLEDGE_BASE.items():
        if key in q or data["name"].lower() in q:
            return data
    
    # secondary fuzzy match
    for key, data in KNOWLEDGE_BASE.items():
        if any(word in q for word in key.split("_")):
            return data
    
    return {"error": "No matching topic found."}

print("Knowledge Base Loaded âœ…")



# ---------------------------
#  USER PROFILE STORAGE TOOL
# ---------------------------

@dataclass
class UserProfile:
    name: str
    country_of_origin: str
    province: str
    arrival_date: str   # "2024-08-15"
    family_description: str  # "Married, 2 kids (6 and 2)"
    employment_status: str   # "Job offer", "Job searching", etc.
    priorities: str          # free text like "school first, then driving"

# simple in-memory "database"
PROFILE_DB: Dict[str, UserProfile] = {}


def save_profile(user_id: str, profile: UserProfile):
    """Save or update a user profile."""
    PROFILE_DB[user_id] = profile
    print(f"âœ… Profile saved for user: {user_id}")


def load_profile(user_id: str) -> Optional[UserProfile]:
    """Load a user profile if it exists."""
    return PROFILE_DB.get(user_id)


def pretty_print_profile(profile: UserProfile):
    """Nice text version for debugging / display."""
    return textwrap.dedent(f"""
    --- User Profile ---
    Name: {profile.name}
    From: {profile.country_of_origin}
    Province: {profile.province}
    Arrival: {profile.arrival_date}
    Family: {profile.family_description}
    Work: {profile.employment_status}
    Priorities: {profile.priorities}
    --------------------
    """)

print("Profile storage ready âœ…")


# ---------------------------
#  TASK MANAGER TOOL
# ---------------------------

@dataclass
class Task:
    id: int
    title: str
    category: str         # e.g. "ID", "Health", "Banking"
    description: str
    status: str = "pending"   # pending / done / blocked / skipped
    priority: int = 3         # 1 = high, 3 = low
    depends_on: List[int] = field(default_factory=list)


TASK_DB: Dict[int, Task] = {}
TASK_COUNTER = 0


def create_task(title: str, category: str, description: str,
                priority: int = 3, depends_on: Optional[List[int]] = None) -> Task:
    """Create a new task and store it in TASK_DB."""
    global TASK_COUNTER
    TASK_COUNTER += 1
    t = Task(
        id=TASK_COUNTER,
        title=title,
        category=category,
        description=description,
        priority=priority,
        depends_on=depends_on or []
    )
    TASK_DB[t.id] = t
    return t


def list_tasks(status_filter: Optional[str] = None) -> List[Task]:
    """Return all tasks, optionally filtered by status."""
    tasks = list(TASK_DB.values())
    if status_filter:
        tasks = [t for t in tasks if t.status == status_filter]
    # sort by priority then id
    tasks.sort(key=lambda t: (t.priority, t.id))
    return tasks


def update_task_status(task_id: int, new_status: str):
    """Update status (pending/done/blocked/skipped)."""
    if task_id in TASK_DB:
        TASK_DB[task_id].status = new_status
        print(f"ğŸ”� Task {task_id} â†’ {new_status}")
    else:
        print(f"âš ï¸� Task {task_id} not found")


def reprioritize_tasks():
    """
    Very simple reprioritization:
    - tasks that all dependencies are done â†’ priority 1
    - others stay as-is
    """
    done_ids = {t.id for t in TASK_DB.values() if t.status == "done"}
    for t in TASK_DB.values():
        if t.status == "pending" and all(dep in done_ids for dep in t.depends_on):
            t.priority = 1

    print("âœ… Reprioritized tasks based on completed dependencies")


def pretty_print_tasks(tasks: List[Task]):
    if not tasks:
        print("No tasks")
        return
    for t in tasks:
        deps = ",".join(map(str, t.depends_on)) or "-"
        print(f"[{t.id}] ({t.status}, P{t.priority}, {t.category}, deps:{deps}) {t.title}")
        print(f"    {t.description}")
    print("-" * 40)

print("Task manager ready âœ…")



# ---------------------------
#  AGENTS IMPLEMENTATION
# ---------------------------

def intake_profile_agent(
    user_id: str,
    name: str,
    country_of_origin: str,
    province: str,
    arrival_date: str,
    family_description: str,
    employment_status: str,
    priorities: str
):
    """
    'Agent 1': takes structured profile info and saves it.
    In a full LLM system this would parse free text; here we keep it simple
    and focus on tools + multi-agent pattern.
    """
    profile = UserProfile(
        name=name,
        country_of_origin=country_of_origin,
        province=province,
        arrival_date=arrival_date,
        family_description=family_description,
        employment_status=employment_status,
        priorities=priorities,
    )
    save_profile(user_id, profile)
    print(pretty_print_profile(profile))
    return profile


def roadmap_planner_agent(user_id: str):
    """
    'Agent 2': creates a basic settlement roadmap (tasks) based on profile.
    Uses the KNOWLEDGE_BASE + task tools.
    """
    profile = load_profile(user_id)
    if not profile:
        print("âš ï¸� No profile found for this user. Run intake_profile_agent first.")
        return []

    created = []

    # Everyone: SIN, health card, bank account
    sin_data = KNOWLEDGE_BASE["sin"]
    t1 = create_task(
        "Apply for Social Insurance Number (SIN)",
        category="ID",
        description="Visit Service Canada with passport + permit/PR to get your SIN.",
        priority=1
    )
    created.append(t1)

    health_data = KNOWLEDGE_BASE["health_card"]
    t2 = create_task(
        "Apply for provincial Health Card",
        category="Health",
        description="Apply for health coverage in your province using ID + proof of address.",
        priority=1,
        depends_on=[t1.id]  # ideally after SIN/ID
    )
    created.append(t2)

    bank_data = KNOWLEDGE_BASE["bank_account"]
    t3 = create_task(
        "Open a Canadian bank account",
        category="Finance",
        description="Open a chequing account to receive salary and pay bills.",
        priority=2,
        depends_on=[t1.id]
    )
    created.append(t3)

    # Driving tasks only if user mentions driving or car in priorities
    if "drive" in profile.priorities.lower() or "car" in profile.priorities.lower():
        drv_data = KNOWLEDGE_BASE["drivers_license"]
        t4 = create_task(
            "Start driverâ€™s license process",
            category="Driving",
            description="Book G1 knowledge test, then road tests as needed.",
            priority=2
        )
        created.append(t4)

    # School tasks if user has kids
    if "kid" in profile.family_description.lower() or "child" in profile.family_description.lower():
        sch_data = KNOWLEDGE_BASE["school_admission"]
        t5 = create_task(
            "Register children for school",
            category="School",
            description="Register kids with local school board and complete documentation.",
            priority=1,
            depends_on=[t2.id]  # good to have health card first
        )
        created.append(t5)

    print("âœ… Roadmap created. Current tasks:")
    pretty_print_tasks(list_tasks())
    return created


def explainer_agent(question: str, user_id: Optional[str] = None) -> str:
    """
    'Agent 3': Answers 'why' and 'how' questions using the KB.
    """
    kb_entry = search_kb(question)
    if "error" in kb_entry:
        answer = "I couldn't find an exact match in the newcomer KB, but generally you should consult your provincial or federal government website."
    else:
        answer = textwrap.dedent(f"""
        **{kb_entry['name']}**

        Why it matters:
        - It is usually required for working, accessing services, or getting benefits.

        Typical requirements:
        - {", ".join(kb_entry["requirements"])}

        Basic steps:
        - {"; ".join(kb_entry["steps"])}

        Approximate processing time: {kb_entry["processing_time"]}

        (This is general information for learning only, not legal or immigration advice.)
        """)
    print(answer)
    return answer


def progress_tracker_agent(user_id: str, completed_task_ids: List[int]):
    """
    'Agent 4': marks tasks as done and reprioritizes.
    """
    for tid in completed_task_ids:
        update_task_status(tid, "done")
    reprioritize_tasks()
    print("ğŸ“Œ Updated pending tasks:")
    pretty_print_tasks(list_tasks(status_filter="pending"))



# ---------------------------
#  ORCHESTRATOR + DEMO
# ---------------------------

def orchestrator(user_id: str, message: str):
    """
    Very simple rule-based orchestrator:
    - 'plan' in message -> roadmap planner
    - 'done' + numbers -> progress tracker
    - 'profile:' -> not used here (we call intake directly)
    - otherwise -> explainer
    """
    msg = message.lower().strip()

    if msg.startswith("plan"):
        # user asks for plan/roadmap
        return roadmap_planner_agent(user_id)

    if msg.startswith("done"):
        # e.g. "done 1 3"
        parts = msg.split()
        ids = [int(p) for p in parts[1:] if p.isdigit()]
        return progress_tracker_agent(user_id, ids)

    # default: explanation question
    return explainer_agent(message, user_id=user_id)


# ---------------------------
#  DEMO: END-TO-END
# ---------------------------

def demo_newcomer_flow():
    user_id = "demo_user"

    print("STEP 1: Intake profile\n")
    intake_profile_agent(
        user_id=user_id,
        name="Alex",
        country_of_origin="India",
        province="Ontario",
        arrival_date="2024-08-15",
        family_description="Married, 2 kids (6 and 2)",
        employment_status="Software engineer, remote job",
        priorities="health card, school for kids, then driving and car"
    )

    print("\nSTEP 2: Generate roadmap\n")
    orchestrator(user_id, "plan my first 90 days")

    print("\nSTEP 3: Ask a question (why health card?)\n")
    orchestrator(user_id, "Why do I need a health card and how to get it?")

    print("\nSTEP 4: Mark some tasks as done (e.g. 1 and 3)\n")
    orchestrator(user_id, "done 1 3")


# Run the demo once to see everything working
demo_newcomer_flow()



# ---------------------------
#  EVALUATION FRAMEWORK
# ---------------------------

def reset_tasks():
    """Helper to clear the task DB between scenarios."""
    global TASK_DB, TASK_COUNTER
    TASK_DB = {}
    TASK_COUNTER = 0


@dataclass
class Scenario:
    name: str
    user_id: str
    profile_kwargs: dict
    expected_categories: List[str]


SCENARIOS = [
    Scenario(
        name="Single adult, Ontario, no kids",
        user_id="s1",
        profile_kwargs=dict(
            name="Ravi",
            country_of_origin="India",
            province="Ontario",
            arrival_date="2024-09-01",
            family_description="Single",
            employment_status="Job searching",
            priorities="health card, job, bank account"
        ),
        expected_categories=["ID", "Health", "Finance"]
    ),
    Scenario(
        name="Family with kids, Ontario",
        user_id="s2",
        profile_kwargs=dict(
            name="Priya",
            country_of_origin="India",
            province="Ontario",
            arrival_date="2024-08-15",
            family_description="Married, 2 kids (6 and 2)",
            employment_status="Software engineer, remote job",
            priorities="health card, school for kids, then driving and car"
        ),
        expected_categories=["ID", "Health", "Finance", "School", "Driving"]
    ),
    Scenario(
        name="Couple, no kids, cares about driving",
        user_id="s3",
        profile_kwargs=dict(
            name="John",
            country_of_origin="Brazil",
            province="Ontario",
            arrival_date="2024-10-10",
            family_description="Married, no kids",
            employment_status="Hospitality worker",
            priorities="driving license first, then bank account and health card"
        ),
        expected_categories=["ID", "Health", "Finance", "Driving"]
    ),
]


def evaluate_scenario(s: Scenario) -> dict:
    """Run intake + planner, then compute a simple coverage score."""
    reset_tasks()

    intake_profile_agent(user_id=s.user_id, **s.profile_kwargs)
    roadmap_planner_agent(user_id=s.user_id)

    tasks = list_tasks()
    cats_present = {t.category for t in tasks}

    # coverage score: fraction of expected categories that appear
    covered = [c for c in s.expected_categories if c in cats_present]
    missing = [c for c in s.expected_categories if c not in cats_present]

    coverage_score = round(10 * len(covered) / max(1, len(s.expected_categories)), 1)

    return {
        "scenario": s.name,
        "tasks_created": len(tasks),
        "expected_categories": s.expected_categories,
        "present_categories": sorted(list(cats_present)),
        "missing_categories": missing,
        "coverage_score_0_to_10": coverage_score,
    }


def run_all_evaluations():
    results = []
    for s in SCENARIOS:
        print(f"\n=== Evaluating: {s.name} ===")
        res = evaluate_scenario(s)
        results.append(res)
        print("Tasks created:", res["tasks_created"])
        print("Expected:", res["expected_categories"])
        print("Present :", res["present_categories"])
        print("Missing :", res["missing_categories"])
        print("Coverage score (0-10):", res["coverage_score_0_to_10"])
    return results


# Run evaluation once
eval_results = run_all_evaluations()


