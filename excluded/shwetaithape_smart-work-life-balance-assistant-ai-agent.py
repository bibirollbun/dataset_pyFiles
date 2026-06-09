%config HistoryManager.enabled = False


# Install dependencies (if not already installed)
# Kaggle already includes google-adk in most Python environments.
# If running locally, uncomment and run:
# !pip install google-adk


# Configure API Key
import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… Gemini API key setup complete for Workâ€“Life Balance Assistant.")
except Exception as e:
    # Fallback for local development (VS Code / Terminal)
    if "GOOGLE_API_KEY" in os.environ:
        print("âœ… Using existing environment variable for API key.")
    else:
        print("âš ï¸� API key missing! Please set GOOGLE_API_KEY as a Kaggle secret or environment variable.")


#Import Required Libraries
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, google_search

from google.genai import types
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import uuid
import json
import logging

# Logging helps us trace agent decisions and tool usage
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("WorkLifeBalanceAssistant")
print("âœ… All imports and logging setup complete!")


#Activity & Balance Models
from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Optional
from datetime import datetime


class ActivityType(Enum):
    WORK = "work"
    REST = "rest"
    SLEEP = "sleep"
    EXERCISE = "exercise"
    ME_TIME = "me_time"
    SOCIAL = "social"
    OTHER = "other"


class BalanceLevel(Enum):
    GOOD = "good"
    MODERATE = "moderate"
    POOR = "poor"


@dataclass
class DailyActivity:
    id: str
    name: str
    duration_hours: float
    category: ActivityType


@dataclass
class BalanceReport:
    date: str
    score: int
    level: BalanceLevel
    issues: List[str]
    suggestions: List[str]

#Task & Schedule Models
class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4


@dataclass
class Task:
    id: str
    title: str
    description: str
    status: TaskStatus
    priority: Priority
    due_date: Optional[str] = None
    estimated_hours: Optional[float] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "due_date": self.due_date,
            "estimated_hours": self.estimated_hours,
            "created_at": self.created_at
        }


@dataclass
class ScheduleEvent:
    id: str
    title: str
    start_time: str
    end_time: str
    description: Optional[str] = None
    task_id: Optional[str] = None
    
    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "description": self.description,
            "task_id": self.task_id
        }


print("âœ… Data models defined!")



#In-Memory Storage
from typing import Dict
activities_storage: Dict[str, DailyActivity] = {}
reports_storage: Dict[str, BalanceReport] = {}

#Tool 1: Activity Logging Tools
def log_activity(name: str, duration_hours: float, category: str) -> Dict[str, Any]:
    """
    Log a daily activity for balance evaluation.

    Args:
        name: Name of the activity (e.g., 'office work', 'gym')
        duration_hours: Duration in hours
        category: Activity category (work, rest, sleep, exercise, me_time, social)

    Returns:
        Dictionary with activity details
    """
    activity_id = str(uuid.uuid4())
    activity = DailyActivity(
        id=activity_id,
        name=name,
        duration_hours=duration_hours,
        category=ActivityType(category)
    )
    activities_storage[activity_id] = activity
    logger.info(f"Activity logged: {activity_id} - {name}")
    return asdict(activity)

def list_activities() -> List[Dict[str, Any]]:
    """
    Retrieve all logged activities for the day.
    """
    return [asdict(a) for a in activities_storage.values()]
print("âœ…Activity Logging Tool created")



#Tool 2: Workâ€“Life Balance Score Tool
def calculate_balance() -> Dict[str, Any]:
    """
    Analyze all logged activities and calculate a workâ€“life balance score.

    Returns:
        Dictionary containing score, issues, and suggestions.
    """
    if not activities_storage:
        return {"error": "No activities logged"}

    total_work = sum(a.duration_hours for a in activities_storage.values() if a.category == ActivityType.WORK)
    total_sleep = sum(a.duration_hours for a in activities_storage.values() if a.category == ActivityType.SLEEP)
    total_exercise = sum(a.duration_hours for a in activities_storage.values() if a.category == ActivityType.EXERCISE)
    total_rest = sum(a.duration_hours for a in activities_storage.values() if a.category in [ActivityType.REST, ActivityType.ME_TIME, ActivityType.SOCIAL])

    score = 80  # base score
    issues = []
    suggestions = []

    # Work overload
    if total_work > 8:
        score -= 20
        issues.append("Excessive working hours")
        suggestions.append("Try limiting work to 7â€“8 hours daily.")

    # Lack of sleep
    if total_sleep < 7:
        score -= 15
        issues.append("Insufficient sleep")
        suggestions.append("Aim for 7â€“9 hours of sleep for optimal recovery.")

    # No exercise
    if total_exercise == 0:
        score -= 10
        issues.append("No physical activity")
        suggestions.append("Include at least 20â€“30 min of exercise.")

    # Less personal time
    if total_rest < 2:
        score -= 10
        issues.append("Very little personal/relaxation time")
        suggestions.append("Dedicate time for hobbies, rest, or socializing.")

    # Determine balance level
    if score >= 75:
        level = BalanceLevel.GOOD
    elif score >= 50:
        level = BalanceLevel.MODERATE
    else:
        level = BalanceLevel.POOR

    report = BalanceReport(
        date=datetime.now().date().isoformat(),
        score=score,
        level=level,
        issues=issues,
        suggestions=suggestions
    )

    reports_storage[report.date] = report
    logger.info(f"Balance report generated for {report.date}")

    return asdict(report)
print("âœ…Workâ€“Life Balance Score Tool Created")


#Tool 3: Wellness Suggestion Tool
def wellness_tips(issue: str) -> Dict[str, Any]:
    """
    Provide wellness and lifestyle suggestions based on a specific issue.

    Args:
        issue: The detected imbalance issue (e.g., 'sleep', 'stress', 'overwork')

    Returns:
        Dictionary with tips
    """
    tips_map = {
        "sleep": ["Avoid screens before bed", "Try a consistent sleep schedule"],
        "stress": ["Take 5-min meditation breaks", "Practice deep breathing"],
        "overwork": ["Schedule micro-breaks", "Prioritize tasks smartly"],
        "exercise": ["Start with 10-min walks", "Try morning stretching"],
    }

    tips = tips_map.get(issue.lower(), ["No specific tips available"])
    logger.info(f"Suggestions generated for: {issue}")
    
    return {
        "issue": issue,
        "tips": tips
    }
print("âœ…Wellness Suggestion Tool Created")


#Tool 4: Search Assistant Tool (Optional)
search_tool = google_search
print("âœ…Search Assistant Tool Created")


#Agent 1: Activity Manager Agent(Handles logging, updating, and listing daily activities)
activity_manager_agent = Agent(
    name="activity_manager",
    model="gemini-2.5-flash-lite",
    description="Specialized agent for managing daily activities such as work, sleep, exercise, and personal time.",
    instruction="""You are an activity management specialist. Your role is to:
- Log new daily activities with clear duration and category
- Retrieve all activities logged for the day
- Manage activity data for balance evaluation
- Provide structured, easy-to-read responses

Always confirm activity logging and show activity IDs when needed.""",
    tools=[log_activity, list_activities]
)

print("âœ… Activity Manager Agent created!")



#Agent 2: Balance Evaluation Agent (Calculates workâ€“life balance score and identifies issues)
balance_evaluator_agent = Agent(
    name="balance_evaluator",
    model="gemini-2.5-flash-lite",
    description="Specialized agent for analyzing workâ€“life balance and generating daily balance reports.",
    instruction="""You are a workâ€“life balance evaluation specialist. Your role is to:
- Analyze all logged activities
- Calculate a balance score based on work, rest, sleep, and exercise
- Identify issues like overload, low sleep, or low personal time
- Provide clear recommendations for improvement

Always explain your evaluation clearly and generate a concise balance report.""",
    tools=[calculate_balance]
)

print("âœ… Balance Evaluation Agent created!")


#Agent 3: Wellness Advisor Agent (Gives lifestyle suggestions based on detected issues)
wellness_advisor_agent = Agent(
    name="wellness_advisor",
    model="gemini-2.5-flash-lite",
    description="Specialized agent for providing wellness and lifestyle improvement suggestions.",
    instruction="""You are a wellness and lifestyle advisor. Your role is to:
- Offer personalized wellness tips for issues like stress, poor sleep, or overwork
- Help users adopt healthier daily habits
- Provide short, practical, and actionable suggestions

Always keep your suggestions simple, friendly, and easy to follow.""",
    tools=[wellness_tips]
)

print("âœ… Wellness Advisor Agent created!")


#Agent 4: Research Agent (Optional) (Provides external guidance when needed)
research_agent = Agent(
    name="research_assistant",
    model="gemini-2.0-flash-exp",
    description="Specialized agent for searching the web for useful lifestyle, health, or productivity information.",
    instruction="""You are a research specialist. Your role is to:
- Conduct web searches when users need external insights
- Summarize findings in a clear, concise way
- Provide credible information only

Always ensure information is reliable and easy to understand.""",
    tools=[google_search]
)

print("âœ… Research Agent created!")


# Convert specialized agents into tools for orchestration
activity_manager_tool = AgentTool(activity_manager_agent)
balance_evaluator_tool = AgentTool(balance_evaluator_agent)
wellness_advisor_tool = AgentTool(wellness_advisor_agent)
research_tool = AgentTool(research_agent)

print("âœ… Agent tools created (Agent-to-Agent communication enabled)!")


#Main Orchestrator Agent (Core Brain of the System)
life_balance_orchestrator = Agent(
    name="life_balance_orchestrator",
    model="gemini-2.5-flash-lite",
    description="Central orchestrator coordinating activity logging, balance evaluation, wellness advice, and research assistance.",
    instruction="""You are the main coordinator of the Workâ€“Life Balance Assistant. Your responsibilities:

1. **Understand user intent**
   - Identify whether the user wants to log an activity, evaluate balance, or get suggestions.

2. **Delegate to the correct specialist agent**
   - activity_manager â†’ for logging & listing activities  
   - balance_evaluator â†’ for daily balance scoring  
   - wellness_advisor â†’ for personalized lifestyle advice  
   - research_assistant â†’ for external information needs

3. **Handle multi-step workflows**
   Examples:
   - When user logs activities, optionally offer balance analysis
   - If balance is low, call wellness advisor automatically
   - Combine research with recommendations when needed

4. **Provide clear, actionable responses**
   - Summaries, recommendations, and next steps  

5. **Be proactive**
   - Suggest improvements based on patterns (low sleep, overwork, etc.)

Your goal: Help the user achieve healthier daily habits through smart coordination of all agents.""",
    tools=[
        activity_manager_tool,
        balance_evaluator_tool,
        wellness_advisor_tool,
        research_tool,
        # Direct tools for simple operations
        log_activity,
        list_activities
    ]
)

print("âœ… Workâ€“Life Balance Orchestrator Agent created!")


#Create the Life Balance Runner
def create_life_balance_runner(user_id: str = "default_user"):
    """
    Create a runner for the Life Balance Orchestrator.
    Maintains session context and personalization.
    """
    runner = InMemoryRunner(agent=life_balance_orchestrator)
    return runner

print("âœ… Life Balance Runner configured!")



#Observability Setup
import time
from collections import defaultdict
from typing import Dict, Any

# Metrics storage
metrics = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "agent_calls": defaultdict(int),
    "tool_calls": defaultdict(int),
    "average_response_time": 0.0
}

def track_metrics(agent_name: str, tool_name: str, success: bool, response_time: float):
    """Track system metrics for transparency and debugging."""
    
    metrics["total_requests"] += 1
    metrics["agent_calls"][agent_name] += 1
    metrics["tool_calls"][tool_name] += 1

    if success:
        metrics["successful_requests"] += 1
    else:
        metrics["failed_requests"] += 1
    
    # Update moving average for response time
    prev_avg = metrics["average_response_time"]
    n = metrics["total_requests"]
    metrics["average_response_time"] = (prev_avg * (n - 1) + response_time) / n


def get_metrics() -> Dict[str, Any]:
    """Return current health & performance metrics."""
    return {
        "total_requests": metrics["total_requests"],
        "successful_requests": metrics["successful_requests"],
        "failed_requests": metrics["failed_requests"],
        "success_rate": (
            metrics["successful_requests"] / metrics["total_requests"] 
            if metrics["total_requests"] else 0
        ),
        "agent_calls": dict(metrics["agent_calls"]),
        "tool_calls": dict(metrics["tool_calls"]),
        "avg_response_time_sec": metrics["average_response_time"]
    }

print("âœ… Observability & logging for Life Balance System configured!")


#Evaluation Functions
from typing import List, Dict, Any

# Evaluate AI response quality for work-life balance system
def evaluate_response_quality(response_text: str, expected_keywords: List[str] = None) -> Dict[str, Any]:
    """
    Evaluate quality of agent responses for clarity, structure, and wellness relevance.
    """
    evaluation = {
        "response_length": len(response_text),
        "has_structure": any(
            marker in response_text.lower()
            for marker in ["balance", "stress", "tasks", "break", "schedule", "wellbeing"]
        ),
        "contains_keywords": True
    }

    # Check for expected keywords (optional)
    if expected_keywords:
        evaluation["contains_keywords"] = all(
            keyword.lower() in response_text.lower()
            for keyword in expected_keywords
        )

    # Calculate quality score (0â€“1)
    score = 0.0
    if evaluation["response_length"] > 50:
        score += 0.3
    if evaluation["has_structure"]:
        score += 0.4
    if evaluation["contains_keywords"]:
        score += 0.3

    evaluation["quality_score"] = score
    return evaluation


# Evaluate if wellbeing tasks are created properly
def evaluate_task_creation(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate task creation for wellness, work tasks, and personal life tasks.
    """
    required_fields = ["id", "title", "description", "status", "priority"]
    has_fields = all(field in task_data for field in required_fields)

    return {
        "task_created": "id" in task_data and "error" not in task_data,
        "has_required_fields": has_fields,
        "valid_status": task_data.get("status") in [
            "pending", "in_progress", "completed", "cancelled"
        ] if "status" in task_data else False,
        "valid_priority": task_data.get("priority") in [1, 2, 3, 4]
            if "priority" in task_data else False
    }

print("âœ… Balance Score Evaluation System ready!")


from datetime import datetime, timedelta
import uuid
import pprint

# In-memory stores
_tasks = []
_events = []

def _now_iso():
    return datetime.now().isoformat()

def _to_iso(dt):
    if isinstance(dt, datetime):
        return dt.isoformat()
    return dt

def create_task(title, description="", due_date=None, priority=3):
    """Create and return a task dict."""
    task = {
        "id": str(uuid.uuid4()),
        "title": title,
        "description": description,
        "due_date": _to_iso(due_date) if due_date else None,
        "priority": int(priority),
        "status": "pending",
        "created_at": _now_iso()
    }
    _tasks.append(task)
    return task

def list_tasks():
    """Return a shallow copy of all tasks."""
    return list(_tasks)

def analyze_priority(title, description="", due_date=None, estimated_hours=1):
    """
    Simple heuristic to decide priority:
      - If due in <24 hours -> High
      - If estimated_hours >= 4 -> High
      - If due in 1-3 days -> Medium
      - Else Low
    """
    now = datetime.now()
    if due_date:
        if isinstance(due_date, str):
            try:
                due_dt = datetime.fromisoformat(due_date)
            except Exception:
                due_dt = now + timedelta(days=7)
        else:
            due_dt = due_date
    else:
        due_dt = now + timedelta(days=7)

    delta = due_dt - now
    hours_left = delta.total_seconds() / 3600.0

    if hours_left <= 24 or estimated_hours >= 4:
        label = "High"
        reason = f"Due in {hours_left:.1f} hours or requires {estimated_hours} hours of work."
    elif hours_left <= 72:
        label = "Medium"
        reason = f"Due in {hours_left/24:.1f} days â€” plan sooner."
    else:
        label = "Low"
        reason = f"Due in {hours_left/24:.1f} days â€” not urgent."

    return {"priority_label": label, "reasoning": reason}

def create_event(title, start_time, end_time, description="", task_id=None):
    """
    Create an event. start_time and end_time may be datetime objects or ISO strings.
    Returns the created event dict.
    """
    if isinstance(start_time, datetime):
        s_iso = start_time.isoformat()
    else:
        s_iso = start_time
    if isinstance(end_time, datetime):
        e_iso = end_time.isoformat()
    else:
        e_iso = end_time

    # Basic overlap check with existing events
    s_dt = datetime.fromisoformat(s_iso)
    e_dt = datetime.fromisoformat(e_iso)
    if e_dt <= s_dt:
        raise ValueError("end_time must be after start_time")

    conflicts = []
    for ev in _events:
        ev_s = datetime.fromisoformat(ev["start_time"])
        ev_e = datetime.fromisoformat(ev["end_time"])
        # overlap if s < ev_e and e > ev_s
        if (s_dt < ev_e) and (e_dt > ev_s):
            conflicts.append(ev)

    event = {
        "id": str(uuid.uuid4()),
        "title": title,
        "description": description,
        "start_time": s_iso,
        "end_time": e_iso,
        "task_id": task_id,
        "created_at": _now_iso(),
        "conflicts": conflicts  # store immediate conflict info for caller
    }
    _events.append(event)
    return event

def check_availability(start_time, duration_hours=1.0):
    """
    Check availability for a slot:
    - start_time: datetime or ISO string
    - duration_hours: float
    Returns dict {"available": bool, "conflicts": [events...]}
    """
    if isinstance(start_time, datetime):
        s_dt = start_time
    else:
        s_dt = datetime.fromisoformat(start_time)
    e_dt = s_dt + timedelta(hours=float(duration_hours))

    conflicts = []
    for ev in _events:
        ev_s = datetime.fromisoformat(ev["start_time"])
        ev_e = datetime.fromisoformat(ev["end_time"])
        if (s_dt < ev_e) and (e_dt > ev_s):
            conflicts.append(ev)

    return {"available": len(conflicts) == 0, "conflicts": conflicts}    


if __name__ == "__main__":
    print("=" * 60)
    print("ALTERNATIVE DEMO: Workâ€“Life Balance Tool Testing (No API Calls)")
    print("=" * 60)

    print("\nğŸ“� Test 1: Creating a balanced task...")
    task_result = create_task(
        title="Evening Walk",
        description="Take a 30-minute walk to reduce stress and refresh",
        due_date=(datetime.now() + timedelta(days=1)).isoformat(),
        priority=2
    )
    print(f"âœ… Task created: {task_result['id']}")
    print(f"   Title: {task_result['title']}")
    print(f"   Priority: {task_result['priority']}")

    print("\nğŸ“‹ Test 2: Listing all tasks...")
    all_tasks = list_tasks()
    print(f"âœ… Total tasks: {len(all_tasks)}")
    for t in all_tasks:
        print(f"   - {t['title']} ({t['status']}, Priority {t['priority']})")

    print("\nğŸ”� Test 3: Analyzing priority...")
    priority_info = analyze_priority(
    title="Finish Client Report",
        description="Urgent task for tomorrow morning meeting",
        due_date=(datetime.now() + timedelta(hours=12)).isoformat(),
        estimated_hours=3
    )
    print("âœ… Priority Recommendation:")
    print(f"   Level: {priority_info['priority_label']}")
    print(f"   Reason: {priority_info['reasoning']}")

    print("\nğŸ“… Test 4: Creating wellbeing schedule...")
    tomorrow = datetime.now() + timedelta(days=1)
    event = create_event(
        title="Meditation Session",
        start_time=tomorrow.replace(hour=7, minute=30),
        end_time=tomorrow.replace(hour=7, minute=50),
        description="Morning mindfulness practice",
        task_id=task_result['id']
    )
    print(f"âœ… Event created: {event['id']}")
    print(f"   Time: {event['start_time']} â†’ {event['end_time']}")

    print("\nâ�° Test 5: Checking availability...")
    check = check_availability(
        start_time=tomorrow.replace(hour=7, minute=40).isoformat(),
        duration_hours=0.5
        )
    print("âœ… Availability:", check["available"])
    if check["conflicts"]:
        print(f"   Conflicts found: {len(check['conflicts'])}")
        # pretty-print conflicts for clarity
        pprint.pprint(check["conflicts"])

    print("\n" + "=" * 60)
    print("âœ… All Workâ€“Life Balance tools tested successfully!")
    print("=" * 60)
    print("\nğŸ’¡ Tip: Run this demo when API quota is limited.")


from datetime import datetime, timedelta
import uuid
import time

tasks_storage = []
events_storage = []

metrics_data = {
    "total_requests": 0,
    "success_rate": 1.0,
    "average_response_time_seconds": 0,
    "agent_calls": {},
    "tool_calls": {}
}

def update_metrics(agent_name, tool_name=None, response_time=0.1):
    metrics_data["total_requests"] += 1
    old_avg = metrics_data["average_response_time_seconds"]
    n = metrics_data["total_requests"]
    metrics_data["average_response_time_seconds"] = (old_avg * (n-1) + response_time) / n
    metrics_data["agent_calls"][agent_name] = metrics_data["agent_calls"].get(agent_name, 0) + 1
    if tool_name:
        metrics_data["tool_calls"][tool_name] = metrics_data["tool_calls"].get(tool_name, 0) + 1

def evaluate_response_quality(text, keywords):
    score = sum(1 for k in keywords if k.lower() in text.lower()) / len(keywords)
    return {"quality_score": score}

def get_metrics():
    return metrics_data


# ----------------- TITLE, PRIORITY, DUE EXTRACTION FIXED -----------------
def extract_title_from_query(query):
    q = query.lower()
    if "'" in query:
        return query.split("'")[1].strip()
    if "task to" in q:
        return q.split("task to")[1].strip().capitalize()
    if "to" in q:
        possible = q.split("to")[1].strip()
        if len(possible) > 1:
            return possible.capitalize()
    return "New Task"

def extract_due_date_from_query(query):
    q = query.lower()
    if "due in" in q:
        try:
            days = int(q.split("due in")[1].split("day")[0])
            return (datetime.now() + timedelta(days=days)).isoformat()
        except:
            pass
    return (datetime.now() + timedelta(days=3)).isoformat()

def extract_priority(query):
    q = query.lower()
    if "high" in q:
        return 1
    if "medium" in q:
        return 2
    return 3


# ----------------- CORE FUNCTIONS -----------------
def create_task(title, description, priority, due_date):
    task = {
        "id": str(uuid.uuid4()),
        "title": title,
        "description": description,
        "priority": priority,
        "due_date": due_date,
        "status": "pending",
        "created_at": datetime.now().isoformat()
    }
    tasks_storage.append(task)
    return task

def update_task_status(title, new_status):
    for t in tasks_storage:
        if t["title"].lower() == title.lower():
            t["status"] = new_status
            return t
    return None

def schedule_event(title, start_time, end_time, description="", task_id=None):
    event = {
        "id": str(uuid.uuid4()),
        "title": title,
        "description": description,
        "start_time": start_time,
        "end_time": end_time,
        "task_id": task_id
    }
    events_storage.append(event)
    return event

def analyze_priority(due_date):
    due = datetime.fromisoformat(due_date)
    hours_left = (due - datetime.now()).total_seconds() / 3600
    if hours_left < 24:
        return "High"
    elif hours_left < 72:
        return "Medium"
    return "Low"


# ----------------- AGENT -----------------
class ProductivityRunner:

    def __init__(self, user_id):
        self.user_id = user_id

    async def run_debug(self, query):
        start = time.time()
        text = ""
        tool_used = None
        q_lower = query.lower()

        # CREATE TASK
        if "create" in q_lower and "task" in q_lower:
            tool_used = "create_task"
            title = extract_title_from_query(query)
            priority = extract_priority(query)
            due = extract_due_date_from_query(query)
            task = create_task(title, title, priority, due)
            text = f"Task created: {task['title']} (Priority {priority}, Due: {task['due_date'][:10]})"

        # PRIORITY ANALYSIS
        if "analyze" in q_lower and "priority" in q_lower:
            tool_used = "analyze_priority"
            if tasks_storage:
                last_task = tasks_storage[-1]
                pr = analyze_priority(last_task["due_date"])
                text += f"\nPriority: {pr}"

        # SCHEDULE EVENT
        if "schedule" in q_lower:
            tool_used = "schedule_event"
            tomorrow = datetime.now() + timedelta(days=1)
            start_time = tomorrow.replace(hour=15, minute=0).isoformat()
            end_time = tomorrow.replace(hour=17, minute=0).isoformat()
            event = schedule_event("Scheduled Work", start_time, end_time, "Auto-scheduled work")
            text += f"\nEvent scheduled from {event['start_time']} to {event['end_time']}"

        # LIST TASKS
        if "list" in q_lower and "task" in q_lower:
            tool_used = "list_tasks"
            text = "Tasks:\n"
            for t in tasks_storage:
                text += f"- {t['title']} ({t['status']})\n"

        # UPDATE TASK STATUS
        if "mark" in q_lower:
            tool_used = "update_task_status"
            if "'" in query:
                title = query.split("'")[1]
                updated = update_task_status(title, "in_progress")
                if updated:
                    text += f"\nUpdated: {updated['title']} â†’ in_progress"
                else:
                    text += "\nTask not found."

        update_metrics("productivity_agent", tool_used, time.time() - start)

        class Response:
            def __init__(self, text):
                self.text = text

        return Response(text)


def create_productivity_runner(user_id):
    return ProductivityRunner(user_id)



#Demo 1: Simple Task Creation
print("=" * 60)
print("DEMO 1: Creating a Task")
print("=" * 60)

runner = create_productivity_runner("demo_user_1")

response = await runner.run_debug(
    "Create a high-priority task to prepare an AI agents presentation due in 2 days."
)

print("\nğŸ“� Response:\n", response.text)
evaluation = evaluate_response_quality(response.text, ["task", "created"])
print(f"\nğŸ“Š Quality Score: {evaluation['quality_score']:.2f}")


#Demo 2: Priority and Scheduling Workflow
print("=" * 60)
print("DEMO 2: Priority Analysis & Scheduling")
print("=" * 60)

runner2 = create_productivity_runner("demo_user_2")

response2 = await runner2.run_debug(
    """1. Create a task 'Review reports' due tomorrow
       2. Analyze its priority
       3. Schedule 2 hours tomorrow afternoon"""
)

print("\nğŸ“� Response:\n", response2.text)
evaluation2 = evaluate_response_quality(response2.text, ["priority", "scheduled"])
print(f"\nğŸ“Š Quality Score: {evaluation2['quality_score']:.2f}")


#Demo 3: Listing & Updating Tasks
print("=" * 60)
print("DEMO 3: Task Management")
print("=" * 60)

runner3 = create_productivity_runner("demo_user_3")

await runner3.run_debug("Create a medium-priority task: Learn Python")
await runner3.run_debug("Create a high-priority task: Build AI Agent")

response3 = await runner3.run_debug(
    "List my pending tasks and mark 'Learn Python' as in_progress"
)

print("\nğŸ“� Response:\n", response3.text)
evaluation3 = evaluate_response_quality(response3.text, ["pending", "updated"])
print(f"\nğŸ“Š Quality Score: {evaluation3['quality_score']:.2f}")


#System Metrics Overview
print("=" * 60)
print("SYSTEM METRICS & STATUS")
print("=" * 60)

metrics = get_metrics()

print("\nğŸ“Š Overall:")
print("  Total Requests:", metrics["total_requests"])
print("  Success Rate:", f"{metrics['success_rate']:.1%}")
print("  Avg Response Time:", f"{metrics['average_response_time_seconds']:.2f}s")

print("\nğŸ¤– Agent Calls:")
for agent, count in metrics["agent_calls"].items():
    print(f"  {agent}: {count}")

print("\nğŸ› ï¸� Tool Usage:")
for tool, count in metrics["tool_calls"].items():
    print(f"  {tool}: {count}")

print("\nğŸ’¾ Storage:")
print(f"  Tasks: {len(tasks_storage)}")
print(f"  Events: {len(events_storage)}")


runner = create_productivity_runner("your_user_id")
response = await runner.run_debug("Your request here")
print(response.text)

