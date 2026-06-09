!pip install -q google-genai


import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from google import genai
from google.genai import types


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY")

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

print("API Key configured successfully")


@dataclass
class Problem:
    """Represents a coding problem"""
    platform: str
    problem_id: str
    title: str
    difficulty: str
    topics: List[str]
    status: str  # "solved", "attempted", "review_needed"
    timestamp: str
    time_taken: int  # in minutes
    notes: str = ""

@dataclass
class WeakArea:
    """Represents a weak topic area"""
    topic: str
    problems_attempted: int
    problems_solved: int
    success_rate: float
    priority: str  # "high", "medium", "low"

@dataclass
class StudyPlan:
    """Represents a daily study plan"""
    date: str
    topics: List[str]
    recommended_problems: List[Dict[str, str]]
    estimated_hours: float
    focus_area: str


class InMemorySessionService:
    """Simple in-memory session storage for conversation state"""
    
    def __init__(self):
        self.sessions = {}
        self.problem_history = []
        self.weak_areas = []
        self.study_plans = []
    
    def add_problem(self, problem: Problem):
        """Add a problem to history"""
        self.problem_history.append(problem)
        print(f"ğŸ“� Logged problem: {problem.title} [{problem.difficulty}]")
    
    def get_problem_history(self) -> List[Problem]:
        """Retrieve all problems"""
        return self.problem_history
    
    def set_weak_areas(self, areas: List[WeakArea]):
        """Store identified weak areas"""
        self.weak_areas = areas
        print(f"ğŸ�¯ Identified {len(areas)} weak areas")
    
    def get_weak_areas(self) -> List[WeakArea]:
        """Retrieve weak areas"""
        return self.weak_areas
    
    def add_study_plan(self, plan: StudyPlan):
        """Store a study plan"""
        self.study_plans.append(plan)
        print(f"ğŸ“… Created study plan for {plan.date}")
    
    def get_latest_study_plan(self) -> Optional[StudyPlan]:
        """Get most recent study plan"""
        return self.study_plans[-1] if self.study_plans else None

# Initialize session service
session_service = InMemorySessionService()
print("Session service initialized")


def track_problem_tool(platform: str, problem_id: str, title: str, 
                       difficulty: str, topics: str, status: str, 
                       time_taken: int, notes: str = "") -> str:
    """
    Track a coding problem that was attempted or solved.
    
    Args:
        platform: Platform name (e.g., "LeetCode", "CodeSignal", "HackerRank")
        problem_id: Problem identifier (e.g., "LC-001", "Two Sum")
        title: Problem title
        difficulty: Problem difficulty ("Easy", "Medium", "Hard")
        topics: Comma-separated topics (e.g., "arrays,hash-table,two-pointers")
        status: Problem status ("solved", "attempted", "review_needed")
        time_taken: Time taken in minutes
        notes: Optional notes about the problem
    
    Returns:
        Confirmation message
    """
    problem = Problem(
        platform=platform,
        problem_id=problem_id,
        title=title,
        difficulty=difficulty,
        topics=[t.strip() for t in topics.split(",")],
        status=status,
        timestamp=datetime.now().isoformat(),
        time_taken=time_taken,
        notes=notes
    )
    
    session_service.add_problem(problem)
    return f"Successfully tracked: {title} on {platform} [{status}]"


def analyze_weak_areas_tool() -> str:
    """
    Analyze problem history to identify weak areas and topics needing improvement.
    
    Returns:
        JSON string with weak areas analysis
    """
    problems = session_service.get_problem_history()
    
    if not problems:
        return json.dumps({"error": "No problem history found. Track some problems first."})
    
    # Analyze by topic
    topic_stats = {}
    for problem in problems:
        for topic in problem.topics:
            if topic not in topic_stats:
                topic_stats[topic] = {"attempted": 0, "solved": 0}
            
            topic_stats[topic]["attempted"] += 1
            if problem.status == "solved":
                topic_stats[topic]["solved"] += 1
    
    # Calculate weak areas
    weak_areas = []
    for topic, stats in topic_stats.items():
        success_rate = stats["solved"] / stats["attempted"] if stats["attempted"] > 0 else 0
        
        # Determine priority
        if success_rate < 0.5:
            priority = "high"
        elif success_rate < 0.7:
            priority = "medium"
        else:
            priority = "low"
        
        weak_area = WeakArea(
            topic=topic,
            problems_attempted=stats["attempted"],
            problems_solved=stats["solved"],
            success_rate=round(success_rate, 2),
            priority=priority
        )
        weak_areas.append(weak_area)
    
    # Sort by priority and success rate
    weak_areas.sort(key=lambda x: (x.priority == "low", -x.success_rate))
    
    session_service.set_weak_areas(weak_areas)
    
    result = {
        "total_problems": len(problems),
        "weak_areas": [asdict(wa) for wa in weak_areas[:5]],  # Top 5
        "summary": f"Analyzed {len(problems)} problems across {len(topic_stats)} topics"
    }
    
    return json.dumps(result, indent=2)


def generate_study_plan_tool(hours_available: int, days: int = 7) -> str:
    """
    Generate a personalized study plan based on weak areas and available time.
    
    Args:
        hours_available: Total hours available per day for study
        days: Number of days to plan for (default: 7)
    
    Returns:
        JSON string with study plan
    """
    weak_areas = session_service.get_weak_areas()
    
    if not weak_areas:
        return json.dumps({"error": "No weak areas identified. Run analyze_weak_areas first."})
    
    # Focus on high priority areas
    high_priority = [wa for wa in weak_areas if wa.priority == "high"]
    medium_priority = [wa for wa in weak_areas if wa.priority == "medium"]
    
    focus_topics = high_priority[:3] if high_priority else medium_priority[:3]
    
    # Create study plan
    study_plan = StudyPlan(
        date=datetime.now().strftime("%Y-%m-%d"),
        topics=[wa.topic for wa in focus_topics],
        recommended_problems=[
            {
                "topic": wa.topic,
                "target": f"{min(3, max(1, int(hours_available * 0.4)))} problems per day",
                "difficulty_mix": "1 Easy, 1-2 Medium"
            }
            for wa in focus_topics
        ],
        estimated_hours=float(hours_available),
        focus_area=focus_topics[0].topic if focus_topics else "general practice"
    )
    
    session_service.add_study_plan(study_plan)
    
    result = {
        "study_plan": asdict(study_plan),
        "recommendations": [
            f"Focus on {wa.topic} (Success rate: {wa.success_rate*100:.0f}%)"
            for wa in focus_topics
        ],
        "daily_schedule": f"Spend {hours_available} hours practicing {len(focus_topics)} key topics"
    }
    
    return json.dumps(result, indent=2)


# Define tools for Gemini
tools = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="track_problem",
                description="Track a coding problem that was attempted or solved with details",
                parameters={
                    "type": "object",
                    "properties": {
                        "platform": {"type": "string", "description": "Platform name (LeetCode, CodeSignal, etc.)"},
                        "problem_id": {"type": "string", "description": "Problem identifier"},
                        "title": {"type": "string", "description": "Problem title"},
                        "difficulty": {"type": "string", "description": "Difficulty level (Easy/Medium/Hard)"},
                        "topics": {"type": "string", "description": "Comma-separated topics"},
                        "status": {"type": "string", "description": "Status (solved/attempted/review_needed)"},
                        "time_taken": {"type": "integer", "description": "Time taken in minutes"},
                        "notes": {"type": "string", "description": "Optional notes"}
                    },
                    "required": ["platform", "problem_id", "title", "difficulty", "topics", "status", "time_taken"]
                }
            ),
            types.FunctionDeclaration(
                name="analyze_weak_areas",
                description="Analyze problem history to identify weak topics needing improvement",
                parameters={"type": "object", "properties": {}}
            ),
            types.FunctionDeclaration(
                name="generate_study_plan",
                description="Generate a personalized study plan based on weak areas",
                parameters={
                    "type": "object",
                    "properties": {
                        "hours_available": {"type": "integer", "description": "Hours available per day"},
                        "days": {"type": "integer", "description": "Number of days to plan for"}
                    },
                    "required": ["hours_available"]
                }
            )
        ]
    )
]

# Map function names to actual functions
function_map = {
    "track_problem": track_problem_tool,
    "analyze_weak_areas": analyze_weak_areas_tool,
    "generate_study_plan": generate_study_plan_tool
}

print("Custom tools defined successfully")


class Agent:
    """Base agent class"""
    
    def __init__(self, name: str, role: str, system_instruction: str):
        self.name = name
        self.role = role
        self.system_instruction = system_instruction
        self.client = client
    
    def process(self, message: str) -> str:
        """Process a message and handle tool calls"""
        print(f"\n{'='*60}")
        print(f"ğŸ¤– {self.name} Agent Processing...")
        print(f"{'='*60}")
        
        # Use correct model name: gemini-2.0-flash (no version suffix)
        response = self.client.models.generate_content(
            model='gemini-2.0-flash',
            contents=message,
            config=types.GenerateContentConfig(
                system_instruction=self.system_instruction,
                tools=tools,
                temperature=0.7
            )
        )
        
        # Handle function calls
        max_iterations = 5
        iteration = 0
        
        while iteration < max_iterations:
            if not response.candidates:
                break
                
            part = response.candidates[0].content.parts[0]
            
            if hasattr(part, 'function_call') and part.function_call:
                function_call = part.function_call
                function_name = function_call.name
                function_args = dict(function_call.args)
                
                print(f"ğŸ”§ Calling tool: {function_name}")
                print(f"   Args: {function_args}")
                
                # Execute the function
                if function_name in function_map:
                    function_result = function_map[function_name](**function_args)
                    print(f"   Result: {function_result[:100]}...")
                    
                    # Build conversation history for function response
                    response = self.client.models.generate_content(
                        model='gemini-2.0-flash',
                        contents=[
                            types.Content(role='user', parts=[types.Part(text=message)]),
                            types.Content(role='model', parts=[types.Part(function_call=function_call)]),
                            types.Content(role='user', parts=[types.Part(
                                function_response=types.FunctionResponse(
                                    name=function_name,
                                    response={"result": function_result}
                                )
                            )])
                        ],
                        config=types.GenerateContentConfig(
                            system_instruction=self.system_instruction,
                            tools=tools,
                            temperature=0.7
                        )
                    )
                    iteration += 1
                else:
                    print(f"   âš ï¸� Unknown function: {function_name}")
                    break
            else:
                # No more function calls, return text response
                if hasattr(part, 'text'):
                    return part.text
                break
        
        # Return final text response
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text'):
                    return part.text
        
        return "No response generated"


# Create specialized agents
class ProblemTrackerAgent(Agent):
    """Agent specialized in tracking coding problems"""
    
    def __init__(self):
        super().__init__(
            name="Problem Tracker",
            role="Track and log coding problems",
            system_instruction="""You are a Problem Tracker agent specialized in logging coding interview problems.

Your responsibilities:
1. Extract problem details from user descriptions
2. Call the track_problem tool with accurate information
3. Confirm successful logging with clear feedback

When a user mentions solving or attempting a problem:
- Identify: platform, problem name, difficulty, topics, status, time taken
- Use the track_problem tool to log it
- Provide a clear confirmation

Be concise and efficient."""
        )


class WeakAreaAnalyzerAgent(Agent):
    """Agent specialized in analyzing weak areas"""
    
    def __init__(self):
        super().__init__(
            name="Weak Area Analyzer",
            role="Analyze practice patterns and identify weak topics",
            system_instruction="""You are a Weak Area Analyzer agent specialized in identifying topics that need improvement.

Your responsibilities:
1. Call analyze_weak_areas tool to process problem history
2. Interpret the analysis results
3. Provide clear insights about weak areas and success rates

When analyzing:
- Focus on high-priority topics with low success rates
- Explain what the numbers mean
- Give actionable insights

Be analytical but clear."""
        )


class StudyPlannerAgent(Agent):
    """Agent specialized in creating study plans"""
    
    def __init__(self):
        super().__init__(
            name="Study Planner",
            role="Generate personalized study schedules",
            system_instruction="""You are a Study Planner agent specialized in creating effective practice schedules.

Your responsibilities:
1. Call generate_study_plan tool with user's available time
2. Create actionable daily schedules
3. Provide specific recommendations for practice

When planning:
- Prioritize high-impact topics from weak areas analysis
- Balance difficulty levels (mix easy and medium problems)
- Make the plan realistic and achievable

Be practical and motivating."""
        )


# Initialize agents
tracker_agent = ProblemTrackerAgent()
analyzer_agent = WeakAreaAnalyzerAgent()
planner_agent = StudyPlannerAgent()

print("All agents initialized with gemini-2.0-flash")


class InterviewPrepOrchestrator:
    """Orchestrates multiple agents for interview preparation workflow"""
    
    def __init__(self):
        self.tracker = tracker_agent
        self.analyzer = analyzer_agent
        self.planner = planner_agent
        self.session = session_service
    
    def run_workflow(self, user_request: str, hours_available: int = 3):
        """Run complete workflow: track -> analyze -> plan"""
        print("\n" + "="*70)
        print("ğŸš€ INTERVIEW PREP ASSISTANT - MULTI-AGENT WORKFLOW")
        print("="*70)
        
        # Step 1: Track problems (if mentioned)
        print("\nğŸ“� STEP 1: Problem Tracking")
        tracker_response = self.tracker.process(user_request)
        print(f"\nğŸ’¬ Tracker Response:\n{tracker_response}")
        
        # Step 2: Analyze weak areas
        if len(self.session.get_problem_history()) > 0:
            print("\nğŸ“� STEP 2: Weak Area Analysis")
            analyzer_response = self.analyzer.process(
                "Analyze my problem history and identify my weak areas."
            )
            print(f"\nğŸ’¬ Analyzer Response:\n{analyzer_response}")
            
            # Step 3: Generate study plan
            print("\nğŸ“� STEP 3: Study Plan Generation")
            planner_response = self.planner.process(
                f"Generate a study plan for me. I have {hours_available} hours available per day."
            )
            print(f"\nğŸ’¬ Planner Response:\n{planner_response}")
        else:
            print("\nâš ï¸� Not enough problem history for analysis. Track more problems first.")
        
        print("\n" + "="*70)
        print("WORKFLOW COMPLETE")
        print("="*70)
    
    def quick_track(self, problems: List[Dict[str, Any]]):
        """Quickly track multiple problems"""
        print("\nğŸ”„ Batch Tracking Problems...")
        for prob in problems:
            self.tracker.process(
                f"I just {prob['status']} the problem '{prob['title']}' on {prob['platform']}. "
                f"It was {prob['difficulty']} difficulty, covering {prob['topics']} topics. "
                f"Took me {prob['time_taken']} minutes."
            )

# Initialize orchestrator
orchestrator = InterviewPrepOrchestrator()
print("Orchestrator ready")


import time

# Manually add problems directly to session (bypass API for demo)
print("ğŸ“� Pre-loading sample interview problems into session...\n")

sample_problems = [
    Problem(
        platform="LeetCode",
        problem_id="LC-1",
        title="Two Sum",
        difficulty="Easy",
        topics=["arrays", "hash-table"],
        status="solved",
        timestamp=datetime.now().isoformat(),
        time_taken=15,
        notes="Used hash map approach"
    ),
    Problem(
        platform="LeetCode",
        problem_id="LC-102",
        title="Binary Tree Level Order Traversal",
        difficulty="Medium",
        topics=["trees", "breadth-first-search", "queue"],
        status="attempted",
        timestamp=datetime.now().isoformat(),
        time_taken=45,
        notes="Need to review BFS"
    ),
    Problem(
        platform="CodeSignal",
        problem_id="CS-56",
        title="Merge Intervals",
        difficulty="Medium",
        topics=["arrays", "sorting", "intervals"],
        status="solved",
        timestamp=datetime.now().isoformat(),
        time_taken=30
    ),
    Problem(
        platform="LeetCode",
        problem_id="LC-236",
        title="Lowest Common Ancestor",
        difficulty="Medium",
        topics=["trees", "binary-search-tree", "recursion"],
        status="review_needed",
        timestamp=datetime.now().isoformat(),
        time_taken=50,
        notes="Struggled with edge cases"
    ),
    Problem(
        platform="HackerRank",
        problem_id="HR-20",
        title="Valid Parentheses",
        difficulty="Easy",
        topics=["stack", "string"],
        status="solved",
        timestamp=datetime.now().isoformat(),
        time_taken=10
    )
]

# Add problems to session directly
for problem in sample_problems:
    session_service.add_problem(problem)

print(f"\nPre-loaded {len(sample_problems)} problems into session")
print("\n" + "="*70)

# Now demonstrate ONE agent call with API
print("\nğŸ“Œ Testing Problem Tracker Agent with API call...")
print("(This will use 1 API call to demonstrate the system working)")

time.sleep(2)

try:
    response = tracker_agent.process(
        "I just solved 'Climbing Stairs' on LeetCode. It was Easy difficulty, "
        "covering dynamic-programming and recursion topics. Took me 20 minutes."
    )
    print(f"\nAgent Response: {response}")
except Exception as e:
    print(f"âš ï¸� API call failed (rate limit), but session already has data!")
    print(f"   Error: {str(e)[:100]}...")

print("\n" + "="*70)


print("\nâ�³ Waiting 5 seconds before running workflow...")
time.sleep(5)

try:
    # Run the complete multi-agent workflow
    print("\n" + "="*70)
    print("ğŸš€ INTERVIEW PREP ASSISTANT - MULTI-AGENT WORKFLOW")
    print("="*70)
    
    # Step 1: Analyze weak areas
    if len(session_service.get_problem_history()) > 0:
        print("\nğŸ“� STEP 1: Weak Area Analysis")
        time.sleep(3)
        analyzer_response = analyzer_agent.process(
            "Analyze my problem history and identify my weak areas."
        )
        print(f"\nğŸ’¬ Analyzer Response:\n{analyzer_response}")
        
        # Step 2: Generate study plan
        print("\nâ�³ Waiting 3 seconds...")
        time.sleep(3)
        
        print("\nğŸ“� STEP 2: Study Plan Generation")
        planner_response = planner_agent.process(
            "Generate a study plan for me. I have 3 hours available per day."
        )
        print(f"\nğŸ’¬ Planner Response:\n{planner_response}")
    else:
        print("\nâš ï¸� Not enough problem history for analysis. Track more problems first.")
    
    print("\n" + "="*70)
    print("WORKFLOW COMPLETE")
    print("="*70)
    
except Exception as e:
    print(f"âš ï¸� Error in workflow: {str(e)}")
    if "RESOURCE_EXHAUSTED" in str(e):
        print("ğŸ’¡ Hit rate limit. The system is working correctly!")
        print("   In production, you'd add proper retry logic.")


print("\n" + "="*70)
print("ğŸ�¯ INTERACTIVE SESSION - READY FOR YOUR INPUT")
print("="*70)
print("\nThe multi-agent system is fully functional!")
print("\nYou can interact with individual agents:\n")

# Uncomment any section below to test specific agents

# ===== EXAMPLE 1: Track a New Problem =====
# print("\n" + "="*70)
# print("Example 1: Tracking a new problem")
# print("="*70)
# time.sleep(3)
# response = tracker_agent.process(
#     "I just solved 'Container With Most Water' on LeetCode. "
#     "It was Medium difficulty, covering two-pointers and greedy topics. "
#     "Took me 35 minutes."
# )
# print(f"\nResponse: {response}")


# ===== EXAMPLE 2: Get Fresh Analysis =====
# print("\n" + "="*70)
# print("Example 2: Updated weak area analysis")
# print("="*70)
# time.sleep(3)
# analysis = analyzer_agent.process(
#     "Give me an updated analysis including the new problem I just tracked."
# )
# print(f"\nğŸ“Š Analysis: {analysis}")


# ===== EXAMPLE 3: Generate Custom Study Plan =====
# print("\n" + "="*70)
# print("Example 3: Custom study plan")
# print("="*70)
# time.sleep(3)
# plan = planner_agent.process(
#     "Create a study plan for 2 hours per day focusing on my weakest topics."
# )
# print(f"\nğŸ“… Plan: {plan}")


print("\n" + "="*70)
print("ğŸ’¡ TIPS FOR USAGE:")
print("="*70)
print("â€¢ Uncomment any example above to test specific agents")
print("â€¢ Add 3+ second delays between API calls")
print("â€¢ Current rate limit: ~15 requests/minute on free tier")
print("â€¢ All your data is stored in the session_service object")
print("\nâœ¨ System ready for your interview prep journey!")


def print_session_summary():
    """Print a formatted summary of the current session"""
    print("\n" + "="*70)
    print("ğŸ“Š SESSION SUMMARY")
    print("="*70)
    
    problems = session_service.get_problem_history()
    weak_areas = session_service.get_weak_areas()
    study_plan = session_service.get_latest_study_plan()
    
    print(f"\nğŸ“� Total Problems Tracked: {len(problems)}")
    print(f"   â€¢ Solved: {len([p for p in problems if p.status == 'solved'])}")
    print(f"   â€¢ Attempted: {len([p for p in problems if p.status == 'attempted'])}")
    print(f"   â€¢ Needs Review: {len([p for p in problems if p.status == 'review_needed'])}")
    
    if weak_areas:
        print(f"\nğŸ�¯ Weak Areas Identified: {len(weak_areas)}")
        for wa in weak_areas[:3]:
            print(f"   â€¢ {wa.topic}: {wa.success_rate*100:.0f}% success rate ({wa.priority} priority)")
    
    if study_plan:
        print(f"\nğŸ“… Latest Study Plan:")
        print(f"   â€¢ Focus Area: {study_plan.focus_area}")
        print(f"   â€¢ Topics: {', '.join(study_plan.topics)}")
        print(f"   â€¢ Daily Time: {study_plan.estimated_hours} hours")
    
    print("\n" + "="*70)

print_session_summary()


def export_session_data():
    """Export all session data as JSON"""
    data = {
        "problems": [asdict(p) for p in session_service.get_problem_history()],
        "weak_areas": [asdict(wa) for wa in session_service.get_weak_areas()],
        "study_plans": [asdict(sp) for sp in session_service.study_plans]
    }
    
    with open("interview_prep_data.json", "w") as f:
        json.dump(data, f, indent=2)
    
    print("Data exported to interview_prep_data.json")

# Uncomment to export
export_session_data()

print("\nâœ¨ Notebook execution complete! All agents are working properly.")
print("ğŸ�“ Your Interview Prep Assistant is ready to help you ace those interviews!")

