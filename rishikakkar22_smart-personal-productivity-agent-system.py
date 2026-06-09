# Install dependencies (if not already installed)
# !pip install google-adk



# Configure API Key
import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    # Fallback for local development
    if "GOOGLE_API_KEY" in os.environ:
        print("âœ… Using environment variable for API key.")
    else:
        print(f"âš ï¸� API key not found. Please set GOOGLE_API_KEY environment variable.")



# Import required libraries
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, google_search
from google.genai import types
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
import uuid
import logging

# Setup basic logging for observability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("âœ… All imports successful!")



# Data models for our productivity system
from dataclasses import dataclass, asdict
from enum import Enum

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



# In-memory storage for tasks and events (in production, use a database)
tasks_storage: Dict[str, Task] = {}
events_storage: Dict[str, ScheduleEvent] = {}

# Custom Tool 1: Task Management Tools
def create_task(title: str, description: str, due_date: Optional[str] = None, 
                priority: int = 2) -> Dict[str, Any]:
    """
    Create a new task with the given details.
    
    Args:
        title: Task title
        description: Task description
        due_date: Optional due date in ISO format
        priority: Priority level (1=Low, 2=Medium, 3=High, 4=Urgent)
    
    Returns:
        Dictionary with task details
    """
    task_id = str(uuid.uuid4())
    priority_enum = Priority(priority) if priority in [1, 2, 3, 4] else Priority.MEDIUM
    
    task = Task(
        id=task_id,
        title=title,
        description=description,
        status=TaskStatus.PENDING,
        priority=priority_enum,
        due_date=due_date
    )
    
    tasks_storage[task_id] = task
    logger.info(f"Task created: {task_id} - {title}")
    
    return task.to_dict()

def get_task(task_id: str) -> Dict[str, Any]:
    """
    Retrieve a task by its ID.
    
    Args:
        task_id: The unique identifier of the task
    
    Returns:
        Task details as dictionary
    """
    if task_id in tasks_storage:
        return tasks_storage[task_id].to_dict()
    return {"error": f"Task {task_id} not found"}

def list_tasks(status: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List all tasks, optionally filtered by status.
    
    Args:
        status: Optional status filter (pending, in_progress, completed, cancelled)
    
    Returns:
        List of task dictionaries
    """
    tasks = [task.to_dict() for task in tasks_storage.values()]
    
    if status:
        tasks = [t for t in tasks if t["status"] == status]
    
    logger.info(f"Retrieved {len(tasks)} tasks")
    return tasks

def update_task_status(task_id: str, status: str) -> Dict[str, Any]:
    """
    Update the status of a task.
    
    Args:
        task_id: The unique identifier of the task
        status: New status (pending, in_progress, completed, cancelled)
    
    Returns:
        Updated task details
    """
    if task_id not in tasks_storage:
        return {"error": f"Task {task_id} not found"}
    
    task = tasks_storage[task_id]
    task.status = TaskStatus(status)
    logger.info(f"Task {task_id} status updated to {status}")
    
    return task.to_dict()

# Note: In ADK, functions are passed directly to tools=[] list
# No need to wrap them in FunctionTool - ADK automatically converts them based on docstrings
# The functions (create_task, get_task, list_tasks, update_task_status) are ready to use as tools

print("âœ… Task management tools created!")



# Custom Tool 2: Schedule Management Tools
def create_event(title: str, start_time: str, end_time: str, 
                 description: Optional[str] = None, task_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a new calendar event.
    
    Args:
        title: Event title
        start_time: Start time in ISO format
        end_time: End time in ISO format
        description: Optional event description
        task_id: Optional associated task ID
    
    Returns:
        Dictionary with event details
    """
    event_id = str(uuid.uuid4())
    
    event = ScheduleEvent(
        id=event_id,
        title=title,
        start_time=start_time,
        end_time=end_time,
        description=description,
        task_id=task_id
    )
    
    events_storage[event_id] = event
    logger.info(f"Event created: {event_id} - {title}")
    
    return event.to_dict()

def list_events(start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List calendar events, optionally filtered by date range.
    
    Args:
        start_date: Optional start date filter (ISO format)
        end_date: Optional end date filter (ISO format)
    
    Returns:
        List of event dictionaries
    """
    events = [event.to_dict() for event in events_storage.values()]
    
    if start_date:
        events = [e for e in events if e["start_time"] >= start_date]
    if end_date:
        events = [e for e in events if e["end_time"] <= end_date]
    
    # Sort by start_time
    events.sort(key=lambda x: x["start_time"])
    
    logger.info(f"Retrieved {len(events)} events")
    return events

def check_availability(start_time: str, duration_hours: float) -> Dict[str, Any]:
    """
    Check if a time slot is available for scheduling.
    
    Args:
        start_time: Proposed start time in ISO format
        duration_hours: Duration in hours
    
    Returns:
        Dictionary indicating availability and conflicts
    """
    start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
    end = start + timedelta(hours=duration_hours)
    
    conflicts = []
    for event in events_storage.values():
        event_start = datetime.fromisoformat(event.start_time.replace('Z', '+00:00'))
        event_end = datetime.fromisoformat(event.end_time.replace('Z', '+00:00'))
        
        # Check for overlap
        if not (end <= event_start or start >= event_end):
            conflicts.append(event.to_dict())
    
    is_available = len(conflicts) == 0
    
    return {
        "available": is_available,
        "conflicts": conflicts,
        "start_time": start_time,
        "end_time": end.isoformat()
    }

# Note: Functions are passed directly to tools=[] - no wrapping needed
# ADK automatically converts these functions to tools based on docstrings and type hints

print("âœ… Schedule management tools created!")



# Custom Tool 3: Priority Analysis Tool
def analyze_priority(title: str, description: str, due_date: Optional[str] = None,
                     estimated_hours: Optional[float] = None) -> Dict[str, Any]:
    """
    Analyze and suggest priority for a task based on various factors.
    
    Args:
        title: Task title
        description: Task description
        due_date: Optional due date in ISO format
        estimated_hours: Optional estimated hours to complete
    
    Returns:
        Dictionary with priority analysis and recommendation
    """
    priority_score = 2  # Default to medium
    
    # Factor 1: Due date urgency
    if due_date:
        try:
            due = datetime.fromisoformat(due_date.replace('Z', '+00:00'))
            now = datetime.now(due.tzinfo) if due.tzinfo else datetime.now()
            days_until_due = (due - now).days
            
            if days_until_due < 1:
                priority_score += 2  # Urgent
            elif days_until_due < 3:
                priority_score += 1  # High
            elif days_until_due > 7:
                priority_score -= 1  # Lower priority
        except:
            pass
    
    # Factor 2: Task complexity (based on description length and keywords)
    urgent_keywords = ["urgent", "asap", "critical", "important", "deadline"]
    if any(keyword in description.lower() for keyword in urgent_keywords):
        priority_score += 1
    
    # Factor 3: Estimated time (longer tasks might need earlier start)
    if estimated_hours and estimated_hours > 8:
        priority_score += 1
    
    # Clamp priority between 1 and 4
    priority_score = max(1, min(4, priority_score))
    
    priority_map = {
        1: "LOW",
        2: "MEDIUM",
        3: "HIGH",
        4: "URGENT"
    }
    
    return {
        "recommended_priority": priority_score,
        "priority_label": priority_map[priority_score],
        "reasoning": f"Analyzed based on due date, keywords, and estimated duration"
    }

# Note: Function is ready to use as a tool - no wrapping needed

print("âœ… Priority analysis tool created!")



# Agent 1: Task Manager Agent
# This agent specializes in task CRUD operations
task_manager_agent = Agent(
    name="task_manager",
    model="gemini-2.5-flash-lite",
    description="Specialized agent for managing tasks - creating, retrieving, updating, and listing tasks",
    instruction="""You are a task management specialist. Your role is to:
- Create new tasks with clear titles and descriptions
- Retrieve task information when requested
- Update task statuses (pending, in_progress, completed, cancelled)
- List tasks with optional status filtering
- Provide clear, structured responses about task operations

Always confirm successful operations and provide task IDs for reference.""",
    tools=[create_task, get_task, list_tasks, update_task_status]
)

print("âœ… Task Manager Agent created!")



# Agent 2: Schedule Coordinator Agent
# This agent handles calendar and scheduling operations
schedule_coordinator_agent = Agent(
    name="schedule_coordinator",
    model="gemini-2.5-flash-lite",
    description="Specialized agent for managing calendar events and scheduling",
    instruction="""You are a scheduling specialist. Your role is to:
- Create calendar events with proper time slots
- Check availability before scheduling
- List upcoming events
- Avoid scheduling conflicts
- Suggest optimal times based on availability

Always verify availability before creating events and provide clear time information.""",
    tools=[create_event, list_events, check_availability]
)

print("âœ… Schedule Coordinator Agent created!")



# Agent 3: Priority Analyzer Agent
# This agent analyzes and assigns priorities to tasks
priority_analyzer_agent = Agent(
    name="priority_analyzer",
    model="gemini-2.5-flash-lite",
    description="Specialized agent for analyzing task priorities based on multiple factors",
    instruction="""You are a priority analysis specialist. Your role is to:
- Analyze tasks based on due dates, keywords, and complexity
- Recommend appropriate priority levels (Low, Medium, High, Urgent)
- Provide reasoning for priority recommendations
- Help users understand why certain tasks should be prioritized

Always provide clear reasoning for your priority recommendations.""",
    tools=[analyze_priority]
)

print("âœ… Priority Analyzer Agent created!")



# Agent 4: Research Agent
# This agent uses Google Search for research assistance
research_agent = Agent(
    name="research_assistant",
    model="gemini-2.0-flash-exp",
    description="Specialized agent for conducting research using web search",
    instruction="""You are a research specialist. Your role is to:
- Conduct web searches to find current information
- Summarize search results clearly and concisely
- Provide relevant sources and citations
- Help users gather information needed for their tasks

Always cite your sources and provide accurate, up-to-date information.""",
    tools=[google_search]
)

print("âœ… Research Agent created!")



# Create Agent Tools - agents that can be used as tools by other agents
# Note: AgentTool only takes the agent as a parameter
# The agent's name and description come from the agent itself

# Convert specialized agents into tools that can be used by the orchestrator
task_manager_tool = AgentTool(task_manager_agent)
schedule_coordinator_tool = AgentTool(schedule_coordinator_agent)
priority_analyzer_tool = AgentTool(priority_analyzer_agent)
research_tool = AgentTool(research_agent)

print("âœ… Agent tools created (Agent-to-Agent communication enabled)!")



# Main Orchestrator Agent
# This agent coordinates all specialized agents and handles complex requests
productivity_orchestrator = Agent(
    name="productivity_orchestrator",
    model="gemini-2.5-flash-lite",
    description="Main orchestrator agent that coordinates task management, scheduling, priority analysis, and research",
    instruction="""You are a smart personal productivity assistant. Your role is to:

1. **Understand user requests** - Parse what the user wants to accomplish
2. **Delegate to specialists** - Use the appropriate specialist agent for each task:
   - task_manager: For creating, updating, or retrieving tasks
   - schedule_coordinator: For calendar and scheduling operations
   - priority_analyzer: For analyzing and recommending task priorities
   - research_assistant: For finding information via web search

3. **Coordinate complex workflows** - For requests that need multiple agents:
   - When creating a task, first analyze its priority, then create it
   - When scheduling, check availability first, then create the event
   - Combine research with task creation when needed

4. **Provide comprehensive responses** - Always summarize what was accomplished and provide clear next steps

5. **Be proactive** - Suggest optimizations, remind about deadlines, and help prioritize work

Remember: You coordinate a team of specialists. Delegate appropriately and provide clear, helpful responses.""",
    tools=[
        task_manager_tool,
        schedule_coordinator_tool,
        priority_analyzer_tool,
        research_tool,
        # Also include direct tools for simple operations
        create_task,
        list_tasks,
        list_events
    ]
)

print("âœ… Productivity Orchestrator Agent created!")



# Create a runner
# Note: InMemoryRunner uses run_debug() method which handles sessions automatically
def create_productivity_runner(user_id: str = "default_user"):
    """
    Create a runner for the productivity orchestrator.
    
    Args:
        user_id: Unique identifier for the user (for reference)
    
    Returns:
        Configured runner
    """
    runner = InMemoryRunner(agent=productivity_orchestrator)
    
    return runner

print("âœ… Runner configured!")



# Enhanced observability setup
import time
from collections import defaultdict

# Metrics tracking
metrics = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "agent_calls": defaultdict(int),
    "tool_calls": defaultdict(int),
    "average_response_time": 0.0
}

def track_metrics(agent_name: str, tool_name: str, success: bool, response_time: float):
    """Track metrics for observability"""
    metrics["total_requests"] += 1
    if success:
        metrics["successful_requests"] += 1
    else:
        metrics["failed_requests"] += 1
    
    metrics["agent_calls"][agent_name] += 1
    metrics["tool_calls"][tool_name] += 1
    
    # Update average response time (simple moving average)
    current_avg = metrics["average_response_time"]
    n = metrics["total_requests"]
    metrics["average_response_time"] = (current_avg * (n - 1) + response_time) / n

def get_metrics() -> Dict[str, Any]:
    """Get current metrics"""
    return {
        "total_requests": metrics["total_requests"],
        "successful_requests": metrics["successful_requests"],
        "failed_requests": metrics["failed_requests"],
        "success_rate": metrics["successful_requests"] / metrics["total_requests"] if metrics["total_requests"] > 0 else 0,
        "agent_calls": dict(metrics["agent_calls"]),
        "tool_calls": dict(metrics["tool_calls"]),
        "average_response_time_seconds": metrics["average_response_time"]
    }

print("âœ… Observability and metrics tracking configured!")



# Evaluation functions
def evaluate_response_quality(response_text: str, expected_keywords: List[str] = None) -> Dict[str, Any]:
    """
    Evaluate the quality of an agent response.
    
    Args:
        response_text: The agent's response text
        expected_keywords: Optional list of keywords that should be present
    
    Returns:
        Evaluation metrics
    """
    evaluation = {
        "response_length": len(response_text),
        "has_structure": any(marker in response_text.lower() for marker in ["task", "schedule", "priority", "completed", "created"]),
        "contains_keywords": True
    }
    
    if expected_keywords:
        evaluation["contains_keywords"] = all(
            keyword.lower() in response_text.lower() 
            for keyword in expected_keywords
        )
    
    # Calculate quality score (0-1)
    score = 0.0
    if evaluation["response_length"] > 50:  # Substantial response
        score += 0.3
    if evaluation["has_structure"]:
        score += 0.4
    if evaluation["contains_keywords"]:
        score += 0.3
    
    evaluation["quality_score"] = score
    return evaluation

def evaluate_task_creation(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate if a task was created correctly.
    
    Args:
        task_data: Task data dictionary
    
    Returns:
        Evaluation metrics
    """
    required_fields = ["id", "title", "description", "status", "priority"]
    has_all_fields = all(field in task_data for field in required_fields)
    
    return {
        "task_created": "id" in task_data and "error" not in task_data,
        "has_required_fields": has_all_fields,
        "valid_status": task_data.get("status") in ["pending", "in_progress", "completed", "cancelled"] if "status" in task_data else False,
        "valid_priority": task_data.get("priority") in [1, 2, 3, 4] if "priority" in task_data else False
    }

print("âœ… Evaluation functions created!")



# Alternative Demo: Test tools directly without API calls
# This demonstrates the system functionality without hitting rate limits

print("=" * 60)
print("ALTERNATIVE DEMO: Direct Tool Testing (No API Calls)")
print("=" * 60)

# Test 1: Create a task directly
print("\nğŸ“� Test 1: Creating a task directly...")
task_result = create_task(
    title="Prepare presentation on AI agents",
    description="Create a comprehensive presentation covering AI agent architectures and use cases",
    due_date=(datetime.now() + timedelta(days=2)).isoformat(),
    priority=3  # High priority
)
print(f"âœ… Task created: {task_result['id']}")
print(f"   Title: {task_result['title']}")
print(f"   Priority: {task_result['priority']} (High)")

# Test 2: List tasks
print("\nğŸ“‹ Test 2: Listing all tasks...")
all_tasks = list_tasks()
print(f"âœ… Found {len(all_tasks)} task(s)")
for task in all_tasks:
    print(f"   - {task['title']} (Status: {task['status']}, Priority: {task['priority']})")

# Test 3: Analyze priority
print("\nğŸ”� Test 3: Analyzing task priority...")
priority_analysis = analyze_priority(
    title="Review quarterly reports",
    description="Urgent: Review and analyze Q4 financial reports for board meeting",
    due_date=(datetime.now() + timedelta(days=1)).isoformat(),
    estimated_hours=4.0
)
print(f"âœ… Priority Analysis:")
print(f"   Recommended Priority: {priority_analysis['priority_label']} ({priority_analysis['recommended_priority']})")
print(f"   Reasoning: {priority_analysis['reasoning']}")

# Test 4: Create an event
print("\nğŸ“… Test 4: Creating a calendar event...")
tomorrow = datetime.now() + timedelta(days=1)
event_result = create_event(
    title="Work on Presentation",
    start_time=tomorrow.replace(hour=14, minute=0, second=0).isoformat(),
    end_time=tomorrow.replace(hour=16, minute=0, second=0).isoformat(),
    description="Dedicated time to work on AI agents presentation",
    task_id=task_result['id']
)
print(f"âœ… Event created: {event_result['id']}")
print(f"   Title: {event_result['title']}")
print(f"   Time: {event_result['start_time']} to {event_result['end_time']}")

# Test 5: Check availability
print("\nâ�° Test 5: Checking availability...")
availability = check_availability(
    start_time=tomorrow.replace(hour=15, minute=0, second=0).isoformat(),
    duration_hours=1.0
)
print(f"âœ… Availability Check:")
print(f"   Available: {availability['available']}")
if availability['conflicts']:
    print(f"   Conflicts: {len(availability['conflicts'])} event(s)")

print("\n" + "=" * 60)
print("âœ… All tool tests completed successfully!")
print("=" * 60)
print("\nğŸ’¡ Note: These tests demonstrate the system functionality without API calls.")
print("   Once your quota resets, you can run the full agent demos above.")



# Demo 1: Simple task creation
print("=" * 60)
print("DEMO 1: Creating a Task")
print("=" * 60)

runner = create_productivity_runner("demo_user_1")

start_time = time.time()
response = await runner.run_debug(
    "Create a task to prepare a presentation on AI agents. It's due in 2 days and is high priority."
)
response_time = time.time() - start_time

# Extract text from response
response_text = response.text if hasattr(response, 'text') else str(response)

print(f"\nğŸ“� Response:\n{response_text}\n")
print(f"â�±ï¸� Response time: {response_time:.2f} seconds\n")

# Track metrics
track_metrics("productivity_orchestrator", "create_task", True, response_time)

# Evaluate response
evaluation = evaluate_response_quality(response_text, ["task", "created", "presentation"])
print(f"ğŸ“Š Response Quality Score: {evaluation['quality_score']:.2f}\n")



# Demo 2: Complex workflow - Create task with priority analysis and scheduling
print("=" * 60)
print("DEMO 2: Complex Workflow - Task with Priority & Scheduling")
print("=" * 60)

runner2 = create_productivity_runner("demo_user_2")

start_time = time.time()
response2 = await runner2.run_debug(
    """I need to:
1. Create a task for 'Review quarterly reports' due tomorrow
2. Analyze its priority
3. Schedule 2 hours tomorrow afternoon to work on it"""
)
response_time2 = time.time() - start_time

response_text2 = response2.text if hasattr(response2, 'text') else str(response2)
print(f"\nğŸ“� Response:\n{response_text2}\n")
print(f"â�±ï¸� Response time: {response_time2:.2f} seconds\n")

track_metrics("productivity_orchestrator", "multi_agent_workflow", True, response_time2)



# Demo 3: List and manage tasks
print("=" * 60)
print("DEMO 3: Task Management - List and Update")
print("=" * 60)

runner4 = create_productivity_runner("demo_user_3")

# First, create a few tasks
await runner4.run_debug("Create a task: 'Learn Python' - medium priority")
await runner4.run_debug("Create a task: 'Build AI agent' - high priority, due in 3 days")

# Then list them and update one
start_time = time.time()
# FIX 1: Use 'runner4' instead of 'runner3'
response4 = await runner4.run_debug(
    "Show me all my pending tasks and update the 'Learn Python' task to in_progress"
)
response_time4 = time.time() - start_time

# FIX 2 & 3: Use 'response4' and 'response_text4' consistently
response_text4 = response4.text if hasattr(response4, 'text') else str(response4)
print(f"\nğŸ“� Response:\n{response_text4}\n") # Changed response_text3 to response_text4
print(f"â�±ï¸� Response time: {response_time4:.2f} seconds\n")

track_metrics("productivity_orchestrator", "list_tasks", True, response_time4)


# Display metrics and system status
print("=" * 60)
print("SYSTEM METRICS & OBSERVABILITY")
print("=" * 60)

current_metrics = get_metrics()
print(f"\nğŸ“Š Overall Metrics:")
print(f"  Total Requests: {current_metrics['total_requests']}")
print(f"  Successful: {current_metrics['successful_requests']}")
print(f"  Failed: {current_metrics['failed_requests']}")
print(f"  Success Rate: {current_metrics['success_rate']:.1%}")
print(f"  Average Response Time: {current_metrics['average_response_time_seconds']:.2f}s")

print(f"\nğŸ¤– Agent Calls:")
for agent, count in current_metrics['agent_calls'].items():
    print(f"  {agent}: {count}")

print(f"\nğŸ› ï¸� Tool Usage:")
for tool, count in current_metrics['tool_calls'].items():
    print(f"  {tool}: {count}")

print(f"\nğŸ’¾ Storage Status:")
print(f"  Tasks: {len(tasks_storage)}")
print(f"  Events: {len(events_storage)}")


