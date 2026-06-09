!pip install google-adk python-dotenv


import os
from google.colab import userdata

# Set your Gemini API key (you need to get this from Google AI Studio)
# Go to: https://aistudio.google.com/app/apikey
os.environ["GOOGLE_API_KEY"] = "your_actual_api_key_here"  # REPLACE WITH YOUR ACTUAL KEY

print("API key set up!")


import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List

# --- Core ADK Imports ---
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService # Corrected import for MemoryService
from google.adk.tools import AgentTool


import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Callable

# --- Core ADK Imports ---
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService 

# --- Tool Decorator/Class Discovery ---
# We use a try/except block to handle the varying import path for the tool decorator/class.
Tool = None
FunctionTool = None


try:
    # Attempt to import the decorator directly (most common path)
    from google.adk.tools import Tool
    print("Tool decorator imported from google.adk.tools")
except ImportError:
    try:
        # Fallback 1: Try importing the explicit FunctionTool class
        from google.adk.tools import FunctionTool
        print(" FunctionTool class imported from google.adk.tools")
    except ImportError:
        # Fallback 2: Try importing the FunctionTool class from a submodule
        try:
            from google.adk.tools.function_tool import FunctionTool
            print(" FunctionTool class imported from google.adk.tools.function_tool")
        except ImportError:
            print("Neither Tool decorator nor FunctionTool found. Custom tools will not work.")



AgentEngine = None
try:
    from google.adk.engine import AgentEngine
    print("AgentEngine imported from google.adk.engine")
except ImportError:
    try:
        from google.adk import AgentEngine
        print(" AgentEngine imported from google.adk")
    except ImportError:
        print(" AgentEngine not found. Falling back to simple tool demo.")
        # Define a mock class for smooth execution if the engine is missing
        class MockAgentEngine:
            def __init__(self, **kwargs): pass
            async def invoke(self, query: str): 
                return {"response": "AgentEngine is not available for full invocation."}
        AgentEngine = MockAgentEngine


logging.basicConfig(level=logging.INFO)


# Simple in-memory storage for tasks and notes (non-persistent for demo)
user_tasks: List[Dict[str, Any]] = []
user_notes: List[Dict[str, Any]] = []

# Define standard Python functions for tool logic
def add_task_func(task_description: str, priority: str = "medium") -> Dict[str, Any]:
    """Add a task to the user's task list. Requires task_description and optional priority (low, medium, high)."""
    task = {
        "id": len(user_tasks) + 1,
        "description": task_description,
        "priority": priority.lower(),
        "created_at": datetime.now().isoformat(),
        "completed": False
    }
    user_tasks.append(task)
    return {"status": "success", "message": f"Task '{task_description}' added", "task_id": task["id"]}



def list_tasks_func() -> Dict[str, Any]:
    """List all current, incomplete tasks, including their ID, description, and priority."""
    incomplete_tasks = [task for task in user_tasks if not task["completed"]]
    return {
        "status": "success",
        "tasks": incomplete_tasks,
        "count": len(incomplete_tasks)
    }

def complete_task_func(task_id: int) -> Dict[str, Any]:
    """Mark a task as completed. Requires the numerical task_id."""
    for task in user_tasks:
        if task["id"] == task_id:
            task["completed"] = True
            task["completed_at"] = datetime.now().isoformat()
            return {"status": "success", "message": f"Task '{task['description']}' completed"}
    return {"status": "error", "message": f"Task {task_id} not found"}

def add_note_func(note_content: str, category: str = "general") -> Dict[str, Any]:
    """Add a personal note, memo, or reminder. Requires note_content and optional category."""
    note = {
        "id": len(user_notes) + 1,
        "content": note_content,
        "category": category,
        "created_at": datetime.now().isoformat()
    }
    user_notes.append(note)
    return {"status": "success", "message": "Note added successfully", "note_id": note["id"]}



# --- Apply Decorator or Wrapper ---
if Tool:
    # If the decorator was imported successfully, apply it to the functions
    add_task = Tool(name="add_task")(add_task_func)
    list_tasks = Tool(name="list_tasks")(list_tasks_func)
    complete_task = Tool(name="complete_task")(complete_task_func)
    add_note = Tool(name="add_note")(add_note_func)
    TOOL_LIST = [add_task, list_tasks, complete_task, add_note]
    print(" Tools wrapped using the @Tool decorator.")

elif FunctionTool:
    # If the decorator failed, explicitly wrap the functions using FunctionTool
    add_task = FunctionTool(add_task_func)
    list_tasks = FunctionTool(list_tasks_func)
    complete_task = FunctionTool(complete_task_func)
    add_note = FunctionTool(add_note_func)
    TOOL_LIST = [add_task, list_tasks, complete_task, add_note]
    print("Tools wrapped using the FunctionTool class.")

else:
    # If all tool imports failed, the demo will only work via direct function calls
    TOOL_LIST = []
    add_task = add_task_func
    list_tasks = list_tasks_func
    complete_task = complete_task_func
    add_note = add_note_func
    print("❌ Custom tools cannot be registered with agents.")



# --- 2. AGENT DEFINITIONS (Multi-Agent System) ---

GEMINI_MODEL = "gemini-2.5-flash" 

# 2.1 Specialist Agents
research_agent = LlmAgent(
    name="research_agent",
    model=GEMINI_MODEL,
    instruction="You are a research specialist. Help users find and explain concepts. Use Google Search or knowledge retrieval tools when available. Do not use task or note tools.",
    description="Research and information specialist."
)

task_agent = LlmAgent(
    name="task_agent",
    model=GEMINI_MODEL,
    instruction="You specialize in task management. Use the add_task, list_tasks, and complete_task tools to manage the user's to-do list.",
    description="Task management specialist.",
    # Pass the wrapped tools to the agent
    tools=[t for t in TOOL_LIST if t.name in ["add_task", "list_tasks", "complete_task"]]
)

decision_agent = LlmAgent(
    name="decision_agent",
    model=GEMINI_MODEL,
    instruction="You help with structured decision making. When a user asks for advice on choices, analyze the options, request necessary context, and provide pros/cons. Do not use task or note tools.",
    description="Decision support specialist."
)

# 2.2 Main Coordinator Agent (The brain)
life_sync_assistant = LlmAgent(
    name="life_sync_assistant",
    model=GEMINI_MODEL,
    instruction=f"""You are LifeSync Assistant - a comprehensive personal AI coordinator.

Your primary role is to coordinate requests and manage sub-agents:
1. When a request is about tasks (adding, listing, completing), delegate to the 'task_agent'.
2. When a request is about general information or explanations, delegate to the 'research_agent'.
3. When a request requires analyzing options or providing structured advice, delegate to the 'decision_agent'.
4. For adding general notes or reminders, use the 'add_note' tool directly.
5. Be helpful, friendly, and proactively use the appropriate tool or sub-agent.
""",
    description="Main personal assistant coordinator for life management.",
    sub_agents=[research_agent, task_agent, decision_agent],
    # The coordinator only needs the note tool directly
    tools=[t for t in TOOL_LIST if t.name == "add_note"]
)

print("✅ Multi-agent system (Coordinator + 3 Specialists) created.")




# --- 3. INITIALIZATION AND RUNNER ---

async def initialize_assistant() -> AgentEngine:
    """Initialize the complete assistant system with ADK services."""
    
    session_service = InMemorySessionService()
    memory_service = InMemoryMemoryService() 
    
    global assistant_engine
    
    if AgentEngine and AgentEngine != MockAgentEngine:
        try:
            engine = AgentEngine(
                agent=life_sync_assistant,
                session_service=session_service,
                memory_service=memory_service 
            )
            print(" LifeSync Assistant Engine initialized successfully!")
            return engine
        except Exception as e:
            print(f" Error creating AgentEngine (Final attempt): {e}")
            return None
    
    print(" Agent Engine unavailable. Assistant not fully initialized for full ADK runtime.")
    return None

# Global variable to hold the initialized engine
assistant_engine: AgentEngine = None

# Run initialization
print("Initializing LifeSync Assistant...")

async def main_init():
    global assistant_engine
    assistant_engine = await initialize_assistant()

# Execute the main initialization (for notebook compatibility)
try:
    # Use a check to run the async setup if not already running in a loop
    if not asyncio.get_event_loop().is_running():
        asyncio.run(main_init())
    else:
        # In some notebook environments, you must run this manually
        print("Async environment detected. Run 'await main_init()' manually to initialize the engine.")
except Exception as e:
    print(f"Error during async setup: {e}")


def interactive_demo():
    """Interactive demo of LifeSync Assistant capabilities (Tool Focus)."""
    
    print("""
\n=============================================
 LifeSync Assistant - Interactive Demo
=============================================
This demo tests the functionality of your custom tools:

 COMMANDS:
1. 'tasks' - View your task list
2. 'add [task]' - Add a new task (e.g., add Buy groceries)
3. 'complete [id]' - Complete a task (e.g., complete 1)
4. 'note [text]' - Add a note (e.g., note Meeting is at 3 PM)
5. 'test' - Run the simple tool test again
6. 'quit' - Exit demo
""")
    
    while True:
        try:
            command = input("\n Enter command (or 'help'): ").strip().lower()
        except EOFError:
            print("\n Thanks for testing LifeSync Assistant!")
            break
        
        if command in ['quit', 'exit']:
            print("Thanks for testing LifeSync Assistant!")
            break
        elif command == 'tasks':
            # FIX: Always call the original function for direct demo
            tasks = list_tasks_func()
            if tasks.get('tasks'):
                print("Your Tasks:")
                for task in tasks['tasks']:
                    status = " wait"
                    print(f"  {status} {task['id']}. {task['description']} ({task['priority']})")
            else:
                print(" No tasks found!")
        elif command.startswith('add '):
            task_desc = command[4:].strip()
            if task_desc:
                # FIX: Always call the original function for direct demo
                result = add_task_func(task_desc)
                print(f" {result['message']} (ID: {result['task_id']})")
            else:
                print(" Please provide a task description")
        elif command.startswith('complete '):
            try:
                task_id = int(command[9:].strip())
                # FIX: Always call the original function for direct demo
                result = complete_task_func(task_id)
                print(f" {result['message']}")
            except ValueError:
                print(" Please provide a valid task ID (e.g., complete 1)")
        elif command.startswith('note '):
            note_text = command[5:].strip()
            if note_text:
                # FIX: Always call the original function for direct demo
                result = add_note_func(note_text)
                print(f" {result['message']} (ID: {result['note_id']})")
            else:
                print("Please provide note content")
        elif command == 'test':
            if not asyncio.get_event_loop().is_running():
                asyncio.run(simple_test())
            else:
                print("Cannot run nested asyncio. Please call 'await simple_test()' manually.")
        elif command == 'help':
            print(interactive_demo.__doc__)
        else:
            print("Unknown command. Type 'help' for available commands.")


# Simple test function (can be run directly)
async def simple_test():
    """Simple test of our agents and tools"""
    
    print("\n Testing tools and agent invocation...")
    
    # Test task tools using the direct functions to ensure they work regardless of wrapping
    print("\n-- Tool Testing --")
    add_task_func("Submit Capstone Report", "high")
    add_task_func("Review ADK documentation", "low")
    print(f"Current tasks: {list_tasks_func()}")
    
    complete_task_func(1)
    print(f"After completing Task 1: {list_tasks_func()}")
    
    add_note_func("Project feedback due next Friday.", "project")
    print("Notes created.")
    
    # Test agent invocation (if engine is available)
    if assistant_engine and assistant_engine != MockAgentEngine:
        print("\n-- Agent Invocation Testing --")
        query_task = "I need to add 'Call mentor' to my high priority list."
        print(f"User Query (Task): '{query_task}'")
        
        # We need to manually create a session to invoke the agent outside of a runner context
        # This is a common pattern for testing ADK agents locally.
        try:
            # Note: In a real environment, you'd use a Runner.invoke(agent, query)
            # or Session.invoke(query). Here we simulate the process.
            session_service = assistant_engine.session_service
            temp_session = await session_service.create_session(
                app_name="life_sync_test",
                user_id="test_user"
            )
            
            # This part is highly dependent on the ADK version. We use a generic approach.
            print("Attempting agent invocation...")
            response_obj = await assistant_engine.agent.invoke(temp_session, query_task)
            
            # Attempt to extract response text
            response_task_text = ""
            if response_obj.response and response_obj.response.parts:
                response_task_text = response_obj.response.parts[0].text
            
            print(f"Agent Response: {response_task_text if response_task_text else 'No direct response object could be parsed.'}")
            print(f"New Task List: {list_tasks_func()}")
            
        except Exception as e:
             print(f"Error during agent invocation: {e}. Check documentation for Runner/Session usage.")




# --- Start Interactive Demo ---
print("\nStarting interactive demo...")
interactive_demo()


await simple_test()




