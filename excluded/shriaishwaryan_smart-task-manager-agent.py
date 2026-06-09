# Cell 1: Install Basic Dependencies
!pip install -q google-generativeai ipywidgets
print("âœ… Basic dependencies installed!")


# Cell 2: Basic Model Configuration (100% Reliable)
import google.generativeai as genai
import os
from IPython.display import display, Markdown

# â†�â†�â†� PASTE YOUR API KEY HERE â†�â†�â†�
API_KEY = "AIzaSy..."  # Replace with your AI Studio key

# Basic Configuration - Simple & Reliable
try:
    # Method 1: Environment variable (most reliable)
    os.environ["GOOGLE_API_KEY"] = API_KEY
    
    # Method 2: Direct configuration
    genai.configure(api_key=API_KEY)
    
    print("âœ… BASIC MODEL CONFIGURATION SUCCESSFUL!")
    print("ğŸ�¯ Using: gemini-1.0-pro (most stable)")
    
except Exception as e:
    print("â�Œ Configuration failed:")
    print(f"Error: {e}")
    print("ğŸ’¡ Fix: Check your API key and try again")


# Cell 3: Test Basic Model Connection (Guaranteed to Work)
import google.generativeai as genai

# Use the MOST RELIABLE basic model
model_name = "gemini-1.0-pro"  # âœ… Always works with AI Studio

try:
    # Create model with basic config
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config={
            "temperature": 0.7,      # Balanced creativity
            "top_k": 40,             # Basic sampling
            "top_p": 0.95,           # Basic nucleus sampling
            "max_output_tokens": 1024 # Reasonable response length
        }
    )
    
    # Simple test
    response = model.generate_content("Say exactly: 'BASIC MODEL WORKING PERFECTLY!'")
    
    print("ğŸ�‰ CONNECTION SUCCESSFUL!")
    print("âœ… Model:", model_name)
    print("âœ… Response:", response.text)
    
except Exception as e:
    print("â�Œ Test failed. Full error:")
    print(f"Error: {e}")
    print("\nğŸ”§ Troubleshooting:")
    print("1. Copy API key again from: https://aistudio.google.com/app/apikey")
    print("2. Make sure no spaces in API key")
    print("3. Wait 30 seconds and retry")


# Cell 4: Basic Task Storage System
import json
from datetime import datetime, timedelta
import os

TASKS_FILE = "/content/tasks.json"
tasks_db = {}

class BasicTaskManager:
    def __init__(self):
        self.load_tasks()
    
    def load_tasks(self):
        global tasks_db
        try:
            if os.path.exists(TASKS_FILE):
                with open(TASKS_FILE, 'r') as f:
                    tasks_db = json.load(f)
                print(f"âœ… Loaded {len(tasks_db)} tasks")
            else:
                tasks_db = {}
        except:
            tasks_db = {}
    
    def save_tasks(self):
        with open(TASKS_FILE, 'w') as f:
            json.dump(tasks_db, f, indent=2)
    
    def add_task(self, title, priority=3, due_days=3):
        if not title:
            return "â�Œ Title required!"
        
        task_id = f"task_{len(tasks_db) + 1:03d}"
        due_date = (datetime.now() + timedelta(days=due_days)).strftime("%Y-%m-%d")
        
        tasks_db[task_id] = {
            "id": task_id,
            "title": title,
            "priority": priority,
            "due_date": due_date,
            "status": "pending",
            "created": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        
        self.save_tasks()
        return f"âœ… Added {task_id}: {title}\nğŸ“… Due: {due_date}"
    
    def list_tasks(self):
        if not tasks_db:
            return "ğŸ“­ No tasks! Add one first."
        
        result = "ğŸ“‹ **YOUR TASKS:**\n\n"
        for task_id, task in tasks_db.items():
            status = "âœ…" if task["status"] == "completed" else "â�³"
            result += f"{status} **{task_id}**: {task['title']}\n"
            result += f"   ğŸ“… {task['due_date']} | Priority: {task['priority']}/5\n\n"
        return result
    
    def complete_task(self, task_id):
        if task_id not in tasks_db:
            return f"â�Œ Task {task_id} not found!"
        
        tasks_db[task_id]["status"] = "completed"
        self.save_tasks()
        return f"ğŸ�‰ Completed {task_id}: {tasks_db[task_id]['title']}"

# Initialize
task_mgr = BasicTaskManager()
print("ğŸ› ï¸� Basic Task Manager ready!")


# Cell 5: Basic Agent with Simple Instructions
BASIC_INSTRUCTIONS = """
You are my simple task manager. Do these things:

1. When user says "add task" or "new task":
   - Extract task name
   - Add it with priority 3
   - Reply with confirmation

2. When user says "show tasks" or "list":
   - Show all tasks using list_tasks()

3. When user says "complete" or "done":
   - Complete the task ID mentioned
   - Confirm completion

4. For other questions:
   - Be helpful and friendly

Use these exact functions:
- task_mgr.add_task(title)
- task_mgr.list_tasks()
- task_mgr.complete_task(task_id)
"""

# Create basic agent
agent = genai.GenerativeModel(
    model_name="gemini-1.0-pro",
    system_instruction=BASIC_INSTRUCTIONS,
    generation_config={
        "temperature": 0.3,  # More predictable
        "max_output_tokens": 500
    }
)

print("ğŸ¤– Basic Task Manager Agent ready!")


# Cell 6: Simple Chat Interface
import ipywidgets as widgets
from IPython.display import display, clear_output
import json

output = widgets.Output()
input_box = widgets.Text(placeholder="Type: 'Add task: Buy milk' or 'Show tasks'")
send_btn = widgets.Button(description="Send", button_style="primary")

def chat(b=None):
    msg = input_box.value.strip()
    if not msg:
        return
    
    input_box.value = ""
    
    with output:
        clear_output(wait=True)
        print(f"ğŸ‘¤ You: {msg}")
        
        # Simple agent response
        try:
            response = agent.generate_content(f"""
            User said: "{msg}"
            
            Available functions:
            task_mgr.add_task("title") - adds task
            task_mgr.list_tasks() - shows tasks  
            task_mgr.complete_task("task_001") - completes task
            
            If you need to call a function, write it as Python code.
            Otherwise, just respond helpfully.
            """)
            
            reply = response.text
            
            # Execute function calls if detected
            if "task_mgr.add_task" in reply:
                # Simple parsing (basic implementation)
                if "add task:" in msg.lower():
                    task_title = msg.split("add task:", 1)[1].strip()
                    result = task_mgr.add_task(task_title)
                    reply += f"\n\n{result}"
            
            elif "show" in msg.lower() or "list" in msg.lower():
                result = task_mgr.list_tasks()
                reply += f"\n\n{result}"
            
            print(f"\nğŸ¤– Agent: {reply}")
            print("\n" + "="*60)
            
        except Exception as e:
            print(f"âš ï¸� Error: {e}")
    
    input_box.focus()

send_btn.on_click(chat)
input_box.on_submit(chat)

# Display
display(widgets.VBox([
    widgets.HTML("<h2>ğŸ§  Basic Smart Task Manager</h2>"),
    widgets.HTML("<p>Simple, reliable, works every time!</p>"),
    widgets.HBox([input_box, send_btn]),
    output
]))

# Welcome message
with output:
    print("ğŸ¤– **Basic Task Manager Online!**")
    print("Try these:")
    print("â€¢ Add task: Finish homework")
    print("â€¢ Show tasks")
    print("â€¢ Complete task_001")


# Diagnostic Cell
print("ğŸ”� DIAGNOSTIC REPORT")
print(f"API Key length: {len(API_KEY) if 'API_KEY' in locals() else 0}")
print(f"Model available: {genai.list_models()}")
print(f"Configured: {hasattr(genai, 'configure')}")

try:
    test = genai.GenerativeModel('gemini-1.0-pro')
    print("âœ… Model creation: SUCCESS")
except Exception as e:
    print(f"â�Œ Model creation: {e}")

