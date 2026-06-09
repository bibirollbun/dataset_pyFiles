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


# Install necessary packages
!pip install google-generativeai -q


import os
from kaggle_secrets import UserSecretsClient

# Get API key from Kaggle Secrets
try:
    user_secrets = UserSecretsClient()
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ“ API Key loaded from Kaggle Secrets")
except Exception as e:
    print(f"âš ï¸� Warning: Could not load from Kaggle Secrets: {e}")
    print("Using environment variable or hardcoded key...")
    # Fallback - only for testing, remove before making public
    GOOGLE_API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual key
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Verify the key is set
if os.environ.get("GOOGLE_API_KEY"):
    print("âœ“ API Key is configured")
else:
    print("âœ— API Key is NOT configured")


import json
from typing import List, Dict, Any
from datetime import datetime
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

print("âœ“ Libraries imported and model initialized!")


import os
import json
from typing import List, Dict, Any
from datetime import datetime
import google.generativeai as genai

# Configure Gemini API
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Use the correct model name
model = genai.GenerativeModel('models/gemini-1.5-flash-embedding')

print("âœ“ Libraries imported and model initialized!")


import google.generativeai as genai

# List all available models
print("Available models:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"  - {m.name}")


# ============================================================================
# CAPABILITY 1: MEMORY SYSTEM
# ============================================================================

class ConversationMemory:
    """Implements conversation memory to maintain context across interactions"""
    
    def __init__(self, max_history: int = 10):
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history = max_history
        self.user_preferences: Dict[str, Any] = {}
        
    def add_interaction(self, user_message: str, agent_response: str):
        """Store a user-agent interaction"""
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_message,
            "agent": agent_response
        })
        
        # Keep only recent history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def get_context(self) -> str:
        """Retrieve recent conversation context"""
        if not self.conversation_history:
            return "No previous conversation."
        
        context = "Previous conversation:\n"
        for interaction in self.conversation_history[-5:]:
            context += f"User: {interaction['user']}\n"
            context += f"Agent: {interaction['agent']}\n\n"
        return context
    
    def update_preferences(self, key: str, value: Any):
        """Store user preferences"""
        self.user_preferences[key] = value
    
    def get_preferences(self) -> Dict[str, Any]:
        """Retrieve stored preferences"""
        return self.user_preferences

print("âœ“ Memory System created!")


# ============================================================================
# CAPABILITY 2: TOOL INTEGRATION - Task Manager
# ============================================================================

class TaskManager:
    """Tool for managing tasks and to-do lists"""
    
    def __init__(self):
        self.tasks: List[Dict[str, Any]] = []
        self.task_id_counter = 1
    
    def add_task(self, title: str, priority: str = "medium", due_date: str = None) -> str:
        """Add a new task"""
        task = {
            "id": self.task_id_counter,
            "title": title,
            "priority": priority,
            "due_date": due_date,
            "status": "pending",
            "created_at": datetime.now().isoformat()
        }
        self.tasks.append(task)
        self.task_id_counter += 1
        return f"Task '{title}' added with ID {task['id']}"
    
    def list_tasks(self, status: str = None) -> str:
        """List all tasks or filter by status"""
        filtered_tasks = self.tasks
        if status:
            filtered_tasks = [t for t in self.tasks if t["status"] == status]
        
        if not filtered_tasks:
            return "No tasks found."
        
        result = "Tasks:\n"
        for task in filtered_tasks:
            result += f"[{task['id']}] {task['title']} - Priority: {task['priority']} - Status: {task['status']}\n"
        return result
    
    def complete_task(self, task_id: int) -> str:
        """Mark a task as complete"""
        for task in self.tasks:
            if task["id"] == task_id:
                task["status"] = "completed"
                return f"Task {task_id} marked as complete!"
        return f"Task {task_id} not found."
    
    def get_statistics(self) -> Dict[str, int]:
        """Get task statistics"""
        total = len(self.tasks)
        completed = len([t for t in self.tasks if t["status"] == "completed"])
        pending = total - completed
        return {"total": total, "completed": completed, "pending": pending}

print("âœ“ Task Manager created!")


# ============================================================================
# CAPABILITY 2: TOOL INTEGRATION - Note Taker
# ============================================================================

class NoteTaker:
    """Tool for taking and managing notes"""
    
    def __init__(self):
        self.notes: List[Dict[str, Any]] = []
        self.note_id_counter = 1
    
    def create_note(self, title: str, content: str, tags: List[str] = None) -> str:
        """Create a new note"""
        note = {
            "id": self.note_id_counter,
            "title": title,
            "content": content,
            "tags": tags or [],
            "created_at": datetime.now().isoformat()
        }
        self.notes.append(note)
        self.note_id_counter += 1
        return f"Note '{title}' created with ID {note['id']}"
    
    def search_notes(self, query: str) -> str:
        """Search notes by title or content"""
        results = [
            n for n in self.notes 
            if query.lower() in n["title"].lower() or query.lower() in n["content"].lower()
        ]
        
        if not results:
            return "No notes found matching your query."
        
        output = "Found notes:\n"
        for note in results:
            output += f"[{note['id']}] {note['title']}\n"
            output += f"  {note['content'][:100]}...\n\n"
        return output
    
    def list_notes(self) -> str:
        """List all notes"""
        if not self.notes:
            return "No notes available."
        
        output = "All notes:\n"
        for note in self.notes:
            output += f"[{note['id']}] {note['title']} - Tags: {', '.join(note['tags'])}\n"
        return output

print("âœ“ Note Taker created!")


# ============================================================================
# CAPABILITY 3: ORCHESTRATION (Agent Orchestrator)
# ============================================================================

class ProductivityAgent:
    """Main agent that orchestrates tools and maintains memory"""
    
    def __init__(self):
        self.memory = ConversationMemory()
        self.task_manager = TaskManager()
        self.note_taker = NoteTaker()
        self.model = model
        
    def process_message(self, user_message: str) -> str:
        """Process user message and orchestrate appropriate actions"""
        
        # Create system prompt with context
        system_prompt = f"""You are a helpful productivity assistant. You can help users manage their tasks and notes.

Available actions:
- Add tasks with priority levels
- List and complete tasks
- Create and search notes
- Remember conversation context

{self.memory.get_context()}

User preferences: {json.dumps(self.memory.get_preferences())}

Respond naturally and helpfully. If you need to perform an action, do so and explain what you did."""
        
        full_prompt = f"{system_prompt}\n\nUser: {user_message}\n\nAssistant:"
        
        try:
            # Generate response
            response = self.model.generate_content(full_prompt)
            agent_response = response.text
            
            # Check if we need to execute any actions based on the user's request
            # This is a simple implementation - in production, use function calling API
            actions_performed = []
            
            if "add task" in user_message.lower() or "create task" in user_message.lower():
                # Extract task details (simplified)
                result = self.task_manager.add_task(user_message, "medium")
                actions_performed.append(result)
            
            if "list task" in user_message.lower() or "show task" in user_message.lower():
                result = self.task_manager.list_tasks()
                actions_performed.append(result)
            
            if "complete task" in user_message.lower():
                # Try to find task ID in message
                import re
                task_ids = re.findall(r'\d+', user_message)
                if task_ids:
                    result = self.task_manager.complete_task(int(task_ids[0]))
                    actions_performed.append(result)
            
            if "create note" in user_message.lower() or "add note" in user_message.lower():
                result = self.note_taker.create_note("Note", user_message)
                actions_performed.append(result)
            
            if "list note" in user_message.lower() or "show note" in user_message.lower():
                result = self.note_taker.list_notes()
                actions_performed.append(result)
            
            # If actions were performed, add them to response
            if actions_performed:
                actions_text = "\n\n".join(actions_performed)
                agent_response = f"{agent_response}\n\nğŸ“‹ Actions performed:\n{actions_text}"
            
            # Store in memory
            self.memory.add_interaction(user_message, agent_response)
            
            return agent_response
            
        except Exception as e:
            return f"I encountered an error: {str(e)}. Please try again."

print("âœ“ Productivity Agent created!")


# ============================================================================
# CAPABILITY 4: EVALUATION SYSTEM
# ============================================================================

class AgentEvaluator:
    """Evaluates agent performance and quality"""
    
    def __init__(self, agent: ProductivityAgent):
        self.agent = agent
        self.evaluation_results = []
    
    def run_test_scenarios(self) -> Dict[str, Any]:
        """Run test scenarios and evaluate responses"""
        
        test_cases = [
            {
                "input": "Add a task to buy groceries with high priority",
                "expected_keywords": ["task", "add", "groceries"],
                "category": "task_management"
            },
            {
                "input": "Show me all my tasks",
                "expected_keywords": ["task", "list"],
                "category": "task_management"
            },
            {
                "input": "Create a note about the meeting tomorrow",
                "expected_keywords": ["note", "creat", "meeting"],
                "category": "note_taking"
            }
        ]
        
        results = {
            "total_tests": len(test_cases),
            "passed": 0,
            "failed": 0,
            "details": []
        }
        
        for test in test_cases:
            response = self.agent.process_message(test["input"])
            
            # Check if response contains expected keywords
            passed = any(keyword.lower() in response.lower() for keyword in test["expected_keywords"])
            
            results["details"].append({
                "input": test["input"],
                "response": response[:150] + "..." if len(response) > 150 else response,
                "passed": passed,
                "category": test["category"]
            })
            
            if passed:
                results["passed"] += 1
            else:
                results["failed"] += 1
        
        return results
    
    def evaluate_response_quality(self, user_input: str, agent_response: str) -> Dict[str, float]:
        """Evaluate response quality metrics"""
        
        metrics = {
            "relevance": 0.0,
            "helpfulness": 0.0,
            "clarity": 0.0
        }
        
        # Simple heuristics
        if len(agent_response) > 20:
            metrics["clarity"] = 0.8
        
        if any(word in agent_response.lower() for word in ["task", "note", "help", "sure"]):
            metrics["relevance"] = 0.9
            metrics["helpfulness"] = 0.85
        
        return metrics

print("âœ“ Evaluator created!")


# ============================================================================
# MAIN EXECUTION AND DEMO
# ============================================================================

def main():
    """Main function to demonstrate the agent capabilities"""
    
    print("=" * 80)
    print("AI PRODUCTIVITY ASSISTANT - CAPSTONE PROJECT")
    print("Demonstrating: Memory, Tool Integration, Orchestration & Evaluation")
    print("=" * 80)
    print()
    
    # Initialize agent
    agent = ProductivityAgent()
    evaluator = AgentEvaluator(agent)
    
    # Demo interactions
    demo_messages = [
        "Hi! I need help organizing my work.",
        "Add a task to complete the project report with high priority",
        "Also create a note about the client meeting",
        "What tasks do I have?",
        "Complete task 1"
    ]
    
    print("ğŸ¤– INTERACTIVE DEMO\n")
    for i, message in enumerate(demo_messages, 1):
        print(f"User ({i}): {message}")
        response = agent.process_message(message)
        print(f"Agent: {response}\n")
        print("-" * 80)
    
    # Show statistics
    print("\nğŸ“Š TASK STATISTICS")
    stats = agent.task_manager.get_statistics()
    print(f"Total tasks: {stats['total']}")
    print(f"Completed: {stats['completed']}")
    print(f"Pending: {stats['pending']}")
    
    # Run evaluation
    print("\nğŸ”� AGENT EVALUATION")
    eval_results = evaluator.run_test_scenarios()
    print(f"Tests passed: {eval_results['passed']}/{eval_results['total_tests']}")
    print(f"Success rate: {(eval_results['passed']/eval_results['total_tests'])*100:.1f}%")
    
    print("\nâœ… Demonstrated Capabilities:")
    print("1. âœ“ Memory: Conversation history and context management")
    print("2. âœ“ Tool Integration: Task and note management tools")
    print("3. âœ“ Orchestration: Intelligent routing to appropriate tools")
    print("4. âœ“ Evaluation: Automated testing and quality metrics")
    
    return agent, evaluator

# Run the demo
agent, evaluator = main()

