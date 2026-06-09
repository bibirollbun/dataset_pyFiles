# ==========================================================
# Neighborhood Helper AI Agent - Full 5-Day Project
# Simulated version for Kaggle Notebook
# ==========================================================

# --- Day 0: Setup ---
# Standard Python libraries
import random

# --- Day 1: Introduction to Agents ---
# Create simple Agent class
class Agent:
    def __init__(self, name):
        self.name = name
        self.tools = {}
        self.memory = {}
    
    def add_tool(self, name, func):
        self.tools[name] = func
    
    def save_memory(self, key, value):
        self.memory[key] = value
    
    def ask(self, query):
        response = []
        for name, func in self.tools.items():
            response.append(f"{name}: {func()}")
        return " | ".join(response)

# Create three agents
event_agent = Agent("Event Agent")
task_agent = Agent("Task Agent")
info_agent = Agent("Info Agent")

# Test Day 1
print("Day 1 Test:")
print(event_agent.ask("Hello Event Agent!"))
print(task_agent.ask("Hello Task Agent!"))
print(info_agent.ask("Hello Info Agent!"))

# --- Day 2: Tools & Interoperability ---
# Define tools (functions)
def get_local_events():
    return ["Book Club", "Yoga Class", "Farmers Market"]

def find_tasks():
    return ["Dog walking", "Grocery delivery", "Tutoring"]

def get_local_services():
    return ["Library", "Recycling Center", "Community Hall"]

# Add tools to agents
event_agent.add_tool("get_events", get_local_events)
task_agent.add_tool("find_tasks", find_tasks)
info_agent.add_tool("get_services", get_local_services)

# Test Day 2
print("\nDay 2 Test:")
print(event_agent.ask("Show me local events"))
print(task_agent.ask("Show me available tasks"))
print(info_agent.ask("Show me nearby services"))

# --- Day 3: Context & Memory ---
# Save long-term memory
event_agent.save_memory("user_preferences", ["Book Club"])
task_agent.save_memory("preferred_tasks", ["Tutoring"])

# Save session memory
event_agent.save_memory("last_event_query", "Yoga Class")
task_agent.save_memory("last_task_query", "Dog walking")

# Test Day 3
print("\nDay 3 Test:")
print(event_agent.ask("Suggest an event I would like"))
print(task_agent.ask("Suggest a task I would enjoy"))

# --- Day 4: Agent Quality (Logs) ---
# Simple logs to track queries
event_agent.save_memory("logs", ["List upcoming events", "Book Club inquiry"])
task_agent.save_memory("logs", ["List tasks", "Dog walking inquiry"])
info_agent.save_memory("logs", ["List services", "Library inquiry"])

# Display logs
print("\nDay 4 Test: Logs")
print("Event Agent Logs:", event_agent.memory["logs"])
print("Task Agent Logs:", task_agent.memory["logs"])
print("Info Agent Logs:", info_agent.memory["logs"])

# --- Day 5: Multi-Agent Collaboration & Demo ---
# Simulate A2A communication
tasks_related_to_events = ["Tutoring at Book Club", "Yoga Class volunteer"]  # Example

# Demo queries
print("\nDay 5 Test: Multi-Agent Interaction")
print("Event Agent:", event_agent.ask("Show me local events"))
print("Task Agent:", task_agent.ask("Show me available tasks"))
print("Info Agent:", info_agent.ask("Show me nearby services"))
print("Task Agent received from Event Agent:", tasks_related_to_events)

# Optional Interactive Demo
while True:
    user_input = input("\nAsk the Neighborhood Helper Agent something (type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    response = event_agent.ask(user_input) + " | " + task_agent.ask(user_input) + " | " + info_agent.ask(user_input)
    print("Agent Response:", response)


