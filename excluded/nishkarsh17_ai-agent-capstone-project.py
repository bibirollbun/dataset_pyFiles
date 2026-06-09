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


# =============================================================================
# SMART STUDY & PRODUCTIVITY AGENT - KAGGLE VERSION
# Capstone Project - Google AI Agents Intensive Course
# Track: Concierge Agents
# =============================================================================

"""
KAGGLE SETUP INSTRUCTIONS:
1. Go to Add-ons (right sidebar) â†’ Secrets
2. Add secret: GOOGLE_API_KEY with your API key from https://aistudio.google.com/apikey
3. Toggle it ON for this notebook
4. Run all cells in order
5. At the bottom, run: await run_demo()

KEY CONCEPTS DEMONSTRATED:
âœ… Multi-agent system (Sequential + Parallel agents)
âœ… Custom tools (4 specialized tools)
âœ… Sessions & Memory (State management + Long-term memory)
âœ… Observability (Metrics, logging, tracing)
âœ… Context engineering (Pydantic models)
"""

# =============================================================================
# CELL 1: INSTALL DEPENDENCIES
# =============================================================================

!pip install -q google-adk google-genai kaggle_secrets pydantic

print("âœ… Dependencies installed")

# =============================================================================
# CELL 2: IMPORTS & SETUP
# =============================================================================

import os
import json
import time
import logging
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Kaggle API Key Setup
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
GOOGLE_API_KEY = secrets.get_secret("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Suppress expected warnings
warnings.filterwarnings('ignore', message='.*non-text parts in the response.*')

# Google ADK imports
from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent
from google.adk.apps.app import App
from google.adk.tools.tool_context import ToolContext
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.WARNING, format='%(message)s')
logger = logging.getLogger(__name__)

print("="*70)
print("ğŸ�“ SMART STUDY & PRODUCTIVITY AGENT SYSTEM")
print("="*70)
print("âœ… All imports loaded successfully")

# =============================================================================
# CELL 3: OBSERVABILITY - METRICS COLLECTOR
# =============================================================================

class MetricsCollector:
    """Tracks agent performance and productivity metrics"""
    
    def __init__(self):
        self.study_sessions = 0
        self.total_study_minutes = 0
        self.tasks_completed = 0
        self.tasks_created = 0
        self.agent_invocations = {}
        self.tool_calls = {}
        self.start_time = time.time()
        
    def record_study_session(self, duration_minutes: int):
        self.study_sessions += 1
        self.total_study_minutes += duration_minutes
        
    def record_task(self, action: str):
        if action == "created":
            self.tasks_created += 1
        elif action == "completed":
            self.tasks_completed += 1
        
    def record_agent_call(self, agent_name: str):
        self.agent_invocations[agent_name] = self.agent_invocations.get(agent_name, 0) + 1
        
    def record_tool_call(self, tool_name: str):
        self.tool_calls[tool_name] = self.tool_calls.get(tool_name, 0) + 1
        
    def get_report(self) -> Dict[str, Any]:
        uptime = time.time() - self.start_time
        completion_rate = (self.tasks_completed / self.tasks_created * 100) if self.tasks_created > 0 else 0
        
        return {
            "uptime_seconds": round(uptime, 2),
            "study_sessions": self.study_sessions,
            "total_study_hours": round(self.total_study_minutes / 60, 2),
            "tasks_created": self.tasks_created,
            "tasks_completed": self.tasks_completed,
            "completion_rate": f"{completion_rate:.1f}%",
            "agent_invocations": self.agent_invocations,
            "tool_calls": self.tool_calls
        }
        
    def print_dashboard(self):
        print("\n" + "="*70)
        print("ğŸ“Š AGENT PERFORMANCE DASHBOARD")
        print("="*70)
        report = self.get_report()
        print(f"â�±ï¸�  System Uptime: {report['uptime_seconds']}s")
        print(f"ğŸ“š Study Sessions: {report['study_sessions']}")
        print(f"â�° Total Study Time: {report['total_study_hours']} hours")
        print(f"âœ… Tasks Created: {report['tasks_created']}")
        print(f"âœ”ï¸�  Tasks Completed: {report['tasks_completed']}")
        print(f"ğŸ“ˆ Completion Rate: {report['completion_rate']}")
        print(f"ğŸ¤– Agent Calls: {report['agent_invocations']}")
        print(f"ğŸ› ï¸�  Tool Usage: {report['tool_calls']}")
        print("="*70 + "\n")

metrics = MetricsCollector()
print("âœ… Metrics collector initialized")

# =============================================================================
# CELL 4: MEMORY BANK - LONG-TERM STORAGE
# =============================================================================

class MemoryBank:
    """Persistent storage for user data across sessions"""
    
    def __init__(self):
        self.storage = {
            "study_goals": [],
            "completed_topics": [],
            "study_schedule": [],
            "notes": [],
            "progress_tracker": {},
            "study_materials": []
        }
        
    def save(self, key: str, value: Any):
        self.storage[key] = value
        
    def get(self, key: str, default=None):
        return self.storage.get(key, default)
    
    def append(self, key: str, item: Any):
        if key not in self.storage:
            self.storage[key] = []
        self.storage[key].append(item)
        
    def update_progress(self, topic: str, progress: int):
        tracker = self.storage.get("progress_tracker", {})
        tracker[topic] = {
            "progress": progress,
            "last_updated": datetime.now().isoformat()
        }
        self.storage["progress_tracker"] = tracker
        
    def get_all(self) -> Dict[str, Any]:
        return self.storage

memory_bank = MemoryBank()
print("âœ… Memory bank initialized")

# =============================================================================
# CELL 5: CONTEXT ENGINEERING - DATA MODELS
# =============================================================================

class StudySession(BaseModel):
    """Structured study session"""
    topic: str = Field(..., description="Study topic")
    duration_minutes: int = Field(..., description="Session duration")
    start_time: str = Field(..., description="Start time (HH:MM)")
    end_time: str = Field(..., description="End time (HH:MM)")
    date: str = Field(..., description="Date (YYYY-MM-DD)")
    session_type: str = Field(..., description="Type: lecture, practice, review")
    
class StudyPlan(BaseModel):
    """Complete study plan output"""
    sessions: List[StudySession] = Field(default_factory=list)
    total_hours: float = Field(..., description="Total study hours")
    start_date: str = Field(..., description="Plan start date")
    end_date: str = Field(..., description="Plan end date")

class ProgressReport(BaseModel):
    """Learning progress analysis"""
    completed_topics: List[str] = Field(default_factory=list)
    in_progress_topics: List[str] = Field(default_factory=list)
    total_study_hours: float = Field(..., description="Hours studied")
    completion_percentage: float = Field(..., description="Overall completion %")

print("âœ… Data models initialized")

# =============================================================================
# CELL 6: CUSTOM TOOLS
# =============================================================================

def schedule_optimizer_tool(
    sessions: List[Dict[str, Any]],
    start_hour: int = 9,
    end_hour: int = 22,
    break_minutes: int = 15,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """Optimizes study schedule with breaks and time constraints"""
    metrics.record_tool_call("schedule_optimizer_tool")
    
    try:
        today = datetime.today().date()
        current_time = datetime.combine(today, datetime.min.time()) + timedelta(hours=start_hour)
        
        optimized = []
        total_minutes = 0
        
        for session in sessions:
            duration = int(session.get("duration_minutes", 60))
            topic = session.get("topic", "Study")
            session_type = session.get("session_type", "study")
            
            session_end = current_time + timedelta(minutes=duration)
            if session_end.time().hour >= end_hour:
                current_time = datetime.combine(
                    current_time.date() + timedelta(days=1),
                    datetime.min.time()
                ) + timedelta(hours=start_hour)
                session_end = current_time + timedelta(minutes=duration)
            
            optimized.append({
                "topic": topic,
                "type": session_type,
                "date": current_time.strftime("%Y-%m-%d"),
                "start": current_time.strftime("%H:%M"),
                "end": session_end.strftime("%H:%M"),
                "duration_minutes": duration
            })
            
            total_minutes += duration
            metrics.record_study_session(duration)
            current_time = session_end + timedelta(minutes=break_minutes)
        
        result = {
            "optimized_schedule": optimized,
            "total_study_hours": round(total_minutes / 60, 2),
            "sessions_count": len(optimized)
        }
        
        memory_bank.save("study_schedule", optimized)
        
        if tool_context:
            tool_context.state["optimized_schedule"] = optimized
            
        return result
        
    except Exception as e:
        return {"error": str(e)}

def progress_tracker_tool(
    topic: str,
    progress_percentage: int,
    notes: str = "",
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """Tracks learning progress for topics"""
    metrics.record_tool_call("progress_tracker_tool")
    
    try:
        memory_bank.update_progress(topic, progress_percentage)
        
        if notes:
            memory_bank.append("notes", {
                "topic": topic,
                "note": notes,
                "timestamp": datetime.now().isoformat()
            })
        
        if progress_percentage >= 100:
            memory_bank.append("completed_topics", {
                "topic": topic,
                "completed_at": datetime.now().isoformat()
            })
            metrics.record_task("completed")
        
        result = {
            "topic": topic,
            "progress": progress_percentage,
            "status": "completed" if progress_percentage >= 100 else "in_progress",
            "message": f"Progress updated for '{topic}' to {progress_percentage}%"
        }
        
        if tool_context:
            tool_context.state["last_progress_update"] = result
            
        return result
        
    except Exception as e:
        return {"error": str(e)}

def study_material_generator_tool(
    topic: str,
    material_type: str = "summary",
    difficulty: str = "intermediate",
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """Generates study materials (summaries, flashcards, quizzes)"""
    metrics.record_tool_call("study_material_generator_tool")
    
    try:
        material = {
            "topic": topic,
            "type": material_type,
            "difficulty": difficulty,
            "generated_at": datetime.now().isoformat(),
            "content": f"Generated {material_type} for {topic} at {difficulty} level"
        }
        
        memory_bank.append("study_materials", material)
        
        if tool_context:
            tool_context.state["last_material"] = material
        
        return material
        
    except Exception as e:
        return {"error": str(e)}

def motivation_coach_tool(
    current_progress: float,
    goal_description: str,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """Provides motivational messages based on progress"""
    metrics.record_tool_call("motivation_coach_tool")
    
    try:
        if current_progress >= 80:
            message = f"ğŸ�‰ Outstanding! You're at {current_progress}% - almost there!"
            tips = ["Keep momentum", "Review completed sections", "Final push"]
        elif current_progress >= 50:
            message = f"ğŸ’ª Great work! Halfway through at {current_progress}%"
            tips = ["Stay consistent", "Take breaks", "Review periodically"]
        elif current_progress >= 25:
            message = f"ğŸŒ± Good start! You've reached {current_progress}%"
            tips = ["Build routine", "Focus on fundamentals", "Track progress"]
        else:
            message = f"ğŸš€ Let's begin! Every expert was once a beginner"
            tips = ["Start small", "Set daily goals", "Stay patient"]
        
        result = {
            "message": message,
            "tips": tips,
            "progress": current_progress,
            "goal": goal_description
        }
        
        if tool_context:
            tool_context.state["motivation"] = result
        
        return result
        
    except Exception as e:
        return {"error": str(e)}

print("âœ… Custom tools initialized (4 tools)")

# =============================================================================
# CELL 7: MULTI-AGENT SYSTEM
# =============================================================================

# Agent 1: Study Planner
study_planner_agent = LlmAgent(
    name="StudyPlannerAgent",
    model="gemini-2.0-flash-lite",
    instruction="""You are a Study Planner Agent. Parse user study goals and create 
    structured study plans. Use schedule_optimizer_tool to generate optimized schedules 
    with realistic timing.""",
    description="Creates personalized study schedules",
    tools=[schedule_optimizer_tool],
    output_schema=StudyPlan,
    output_key="study_plan"
)

# Agent 2: Progress Monitor
progress_monitor_agent = LlmAgent(
    name="ProgressMonitorAgent",
    model="gemini-2.0-flash-lite",
    instruction="""You are a Progress Monitor Agent. Track learning progress using 
    progress_tracker_tool. Analyze completion rates and provide recommendations.""",
    description="Monitors and analyzes learning progress",
    tools=[progress_tracker_tool],
    output_schema=ProgressReport,
    output_key="progress_report"
)

# Agent 3: Content Generator
content_generator_agent = LlmAgent(
    name="ContentGeneratorAgent",
    model="gemini-2.0-flash-lite",
    instruction="""You are a Content Generator Agent. Generate study materials using 
    study_material_generator_tool. Create summaries, flashcards, and practice questions.""",
    description="Generates study materials",
    tools=[study_material_generator_tool]
)

# Agent 4: Motivation Coach
motivation_agent = LlmAgent(
    name="MotivationCoachAgent",
    model="gemini-2.0-flash-lite",
    instruction="""You are a Motivation Coach Agent. Provide encouraging messages using 
    motivation_coach_tool. Celebrate achievements and offer strategies.""",
    description="Provides motivation and coaching",
    tools=[motivation_coach_tool]
)

# Parallel agents (run simultaneously)
parallel_support_agents = ParallelAgent(
    name="SupportAgentsParallel",
    sub_agents=[content_generator_agent, motivation_agent],
    description="Generates content and motivation in parallel"
)

# Sequential pipeline (run in order)
main_pipeline = SequentialAgent(
    name="SmartStudyPipeline",
    sub_agents=[
        study_planner_agent,
        progress_monitor_agent,
        parallel_support_agents
    ],
    description="Complete study assistance pipeline"
)

print("âœ… Multi-agent system configured")
print("   - 4 Individual agents")
print("   - 1 Sequential pipeline")
print("   - 1 Parallel sub-pipeline")

# =============================================================================
# CELL 8: APP & SESSION MANAGEMENT
# =============================================================================

study_app = App(
    name="SmartStudyProductivityApp",
    root_agent=main_pipeline
)

session_service = InMemorySessionService()

app_runner = Runner(
    app=study_app,
    session_service=session_service
)

print("âœ… App and runner initialized")

# =============================================================================
# CELL 9: EXECUTION FUNCTIONS
# =============================================================================

import asyncio
import nest_asyncio
nest_asyncio.apply()

async def run_study_agent(
    user_input: str,
    session_id: str = "default_session",
    user_id: str = "default_user"
) -> Dict[str, Any]:
    """Main entry point for interacting with the study agent system"""
    
    metrics.record_agent_call("SmartStudyPipeline")
    
    try:
        # Create or get session
        try:
            session = await session_service.create_session(
                app_name=study_app.name,
                user_id=user_id,
                session_id=session_id
            )
        except:
            session = await session_service.get_session(
                app_name=study_app.name,
                user_id=user_id,
                session_id=session_id
            )
        
        # Create user message
        class UserMessage:
            def __init__(self, text: str):
                self.role = "user"
                self.parts = [type("Part", (), {"text": text})()]
        
        msg = UserMessage(user_input)
        
        # Run pipeline
        event_count = 0
        async for event in app_runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=msg
        ):
            event_count += 1
        
        # Extract results
        result = {
            "session_id": session.id,
            "study_plan": session.state.get("study_plan"),
            "progress_report": session.state.get("progress_report"),
            "optimized_schedule": session.state.get("optimized_schedule"),
            "motivation": session.state.get("motivation"),
            "memory_snapshot": memory_bank.get_all(),
            "metrics": metrics.get_report(),
            "event_count": event_count
        }
        
        return result
        
    except Exception as e:
        return {"error": str(e)}

def print_results(results: Dict[str, Any]):
    """Pretty print agent results"""
    print("\n" + "="*70)
    print("ğŸ“‹ SMART STUDY AGENT - RESULTS")
    print("="*70)
    
    if "error" in results:
        print(f"â�Œ Error: {results['error']}")
        return
    
    if "study_plan" in results and results["study_plan"]:
        plan = results["study_plan"]
        print(f"\nğŸ“š Study Plan Generated:")
        print(f"   Total Hours: {plan.get('total_hours', 0)}")
        print(f"   Start: {plan.get('start_date', 'N/A')}")
        print(f"   End: {plan.get('end_date', 'N/A')}")
    
    if "optimized_schedule" in results and results["optimized_schedule"]:
        print(f"\nğŸ“… Optimized Schedule ({len(results['optimized_schedule'])} sessions):")
        for i, session in enumerate(results["optimized_schedule"][:5], 1):
            print(f"   {i}. {session['date']} {session['start']}-{session['end']}: {session['topic']}")
        if len(results["optimized_schedule"]) > 5:
            print(f"   ... and {len(results['optimized_schedule']) - 5} more sessions")
    
    if "progress_report" in results and results["progress_report"]:
        report = results["progress_report"]
        print(f"\nğŸ“Š Progress Report:")
        print(f"   Completion: {report.get('completion_percentage', 0)}%")
        print(f"   Study Hours: {report.get('total_study_hours', 0)}")
    
    if "motivation" in results and results["motivation"]:
        mot = results["motivation"]
        print(f"\nğŸ’ª Motivation:")
        print(f"   {mot.get('message', '')}")
        if mot.get('tips'):
            print(f"   Tips: {', '.join(mot['tips'][:3])}")
    
    print(f"\nğŸ“Š Events Processed: {results.get('event_count', 0)}")
    print("="*70 + "\n")

print("âœ… Execution functions ready")

# =============================================================================
# CELL 10: DEMO FUNCTION
# =============================================================================

async def run_demo():
    """Run a complete demonstration"""
    print("\n" + "="*70)
    print("ğŸ�¬ RUNNING DEMONSTRATION")
    print("="*70)
    
    user_input = """
I need to prepare for a Machine Learning exam in 2 weeks. I need to study:
- Linear Regression (3 hours)
- Neural Networks (4 hours)
- Decision Trees (3 hours)
- Practice problems (5 hours)

I can study from 6 PM to 10 PM on weekdays and 9 AM to 5 PM on weekends.
Please create an optimized study schedule.
"""
    
    print("\nğŸ“� User Request:")
    print(user_input.strip())
    print("\nâ�³ Processing with AI agents...")
    print("   (This may take 30-60 seconds...)")
    
    results = await run_study_agent(user_input, session_id="demo_session")
    
    print("\nâœ… Processing complete!")
    print_results(results)
    
    # Show metrics dashboard
    metrics.print_dashboard()
    
    # Show memory bank status
    print("ğŸ“¦ Memory Bank Status:")
    memory = memory_bank.get_all()
    print(f"   Study schedules: {len(memory.get('study_schedule', []))}")
    print(f"   Completed topics: {len(memory.get('completed_topics', []))}")
    print(f"   Study materials: {len(memory.get('study_materials', []))}")
    print(f"   Progress tracked: {len(memory.get('progress_tracker', {}))}")
    
    return results

print("âœ… Demo function ready")

# =============================================================================
# CELL 11: QUICK TEST
# =============================================================================

async def quick_test():
    """Quick test to verify system works"""
    print("\nğŸ”§ Running quick system test...")
    
    try:
        result = await run_study_agent(
            "Create a simple 1-week study plan for Python basics (10 hours total)",
            session_id="test"
        )
        
        if result and not result.get('error'):
            print("âœ… SYSTEM TEST PASSED - All components working!")
            return True
        else:
            print("â�Œ SYSTEM TEST FAILED")
            return False
    except Exception as e:
        print(f"â�Œ SYSTEM TEST FAILED: {str(e)}")
        return False

print("âœ… Quick test ready")

# =============================================================================
# FINAL INSTRUCTIONS
# =============================================================================

print("\n" + "="*70)
print("ğŸ�‰ SETUP COMPLETE!")
print("="*70)
print("\nğŸ“– HOW TO USE:")
print("\n1. Quick Test (verify it works):")
print("   await quick_test()")
print("\n2. Run Full Demo:")
print("   await run_demo()")
print("\n3. Custom Query:")
print("   results = await run_study_agent('Your question here')")
print("   print_results(results)")
print("\n" + "="*70)
print("\nğŸ’¡ TIP: Start with 'await quick_test()' to verify everything works!")
print("="*70)


await quick_test()


await run_demo()

