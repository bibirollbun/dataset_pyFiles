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


import threading
import random
import time
from queue import Queue

# --- Shared Resources ---
# 1. Thread-safe Queue for A2A communication (e.g., Shopping Agent tells Meal Agent what it bought)
MESSAGE_QUEUE = Queue()
# 2. Lock to ensure clear, non-interleaved printing from multiple threads
PRINT_LOCK = threading.Lock()

# Base class for all agents
class Agent:
    def __init__(self, name):
        """Initializes the agent with a name."""
        self.name = name

    def _safe_print(self, message):
        """Prints a message using a lock to ensure atomic output."""
        with PRINT_LOCK:
            # Print the current thread name for visualization of concurrency
            print(f"[{threading.current_thread().name}] {self.name}: {message}")

    def execute_task(self, task, min_time=1, max_time=3):
        """Simulates the execution of a task with a random duration."""
        self._safe_print(f"Starting task: **{task}**")
        
        # Simulate task duration
        duration = random.randint(min_time, max_time)
        time.sleep(duration)
        
        self._safe_print(f"Completed task: {task} (Duration: {duration}s)")
        return duration

# Specialized agent for meal planning
class MealPlanningAgent(Agent):
    def plan_meal_workflow(self):
        """Plans a meal and waits for shopping results."""
        # Step 1: Initial Planning
        meal = random.choice(['Pasta', 'Salad', 'Steak', 'Sushi'])
        self.execute_task(f"Planning {meal} menu", 1, 2)
        
        # Step 2: Send Request to Shopping Agent (A2A Communication Outgoing)
        ingredients = {'Pasta': ['Noodles', 'Sauce'], 'Salad': ['Lettuce', 'Dressing'], 'Steak': ['Steak', 'Potatoes'], 'Sushi': ['Rice', 'Fish']}[meal]
        request = (self.name, f"Need to buy {', '.join(ingredients)} for {meal}")
        MESSAGE_QUEUE.put(request)
        self._safe_print(f"Requested shopping for **{meal}**. Waiting for updates...")

        # Step 3: Wait for A2A Response
        purchased_items = None
        start_time = time.time()
        timeout = 10 # Wait up to 10 seconds for a response
        
        while time.time() - start_time < timeout:
            if not MESSAGE_QUEUE.empty():
                sender, message = MESSAGE_QUEUE.get()
                if "Purchased" in message and sender == "Shopping Assistant":
                    purchased_items = message.split(': ')[1]
                    self._safe_print(f"Received update from **{sender}**: {message}")
                    break
                else:
                    # Put back any irrelevant message for other agents
                    MESSAGE_QUEUE.put((sender, message)) 
            time.sleep(0.5) # Polling interval

        # Step 4: Finalize Plan
        if purchased_items:
            self.execute_task(f"Finalizing {meal} recipe based on purchased items: {purchased_items}", 1, 1)
        else:
            self._safe_print("Shopping update timed out. Cannot finalize meal plan.")


# Specialized agent for shopping
class ShoppingAgent(Agent):
    def shop_workflow(self):
        """Checks the queue for requests, performs shopping, and sends a result."""
        self._safe_print("Ready to receive shopping requests...")
        
        # Step 1: Wait for A2A Request
        request_details = None
        while request_details is None:
            if not MESSAGE_QUEUE.empty():
                sender, message = MESSAGE_QUEUE.get()
                if "Need to buy" in message and sender == "Meal Planner":
                    request_details = message
                    self._safe_print(f"Received request from **{sender}**: {message}")
                else:
                    # Put back any irrelevant message
                    MESSAGE_QUEUE.put((sender, message)) 
            time.sleep(1) # Wait for request

        # Step 2: Perform Shopping Task
        item_list = request_details.split('Need to buy ')[1].split(' for ')[0].split(', ')
        purchased_items = random.sample(item_list, k=random.randint(1, len(item_list)))
        self.execute_task(f"Shopping for: {', '.join(item_list)}", 2, 4)
        
        # Step 3: Send Confirmation to Meal Agent (A2A Communication Outgoing)
        confirmation = (self.name, f"Purchased: {', '.join(purchased_items)}")
        MESSAGE_QUEUE.put(confirmation)
        self._safe_print(f"Sent purchase confirmation: {', '.join(purchased_items)}")


# Specialized agent for travel planning
class TravelAgent(Agent):
    def plan_trip_workflow(self):
        """Simulates a complex trip planning task."""
        # Step 1: Research Destination
        destination = random.choice(['Paris', 'New York', 'Tokyo', 'London'])
        self.execute_task(f"Researching travel restrictions for {destination}", 1, 2)
        
        # Step 2: Book Flights
        self.execute_task(f"Booking flights to {destination}", 2, 3)
        
        # Step 3: Book Accommodation (A self-contained critical step)
        if random.random() < 0.8: # 80% success rate
            self.execute_task(f"Booking accommodation in {destination} (Success)", 1, 2)
        else:
            self.execute_task(f"Accommodation booking failed. Re-trying...", 1, 1)
            self.execute_task(f"Booking accommodation in {destination} (Success)", 1, 2)
        
        self._safe_print(f"Trip planning to **{destination}** is complete.")


def run_agents():
    """Initializes agents, creates threads for each workflow, and waits for them to complete."""
    print("\n" + "="*50)
    print("--- Starting Enhanced Agent Simulation (A2A Communication) ---")
    print("="*50 + "\n")
    
    # List of agents
    agents = [
        MealPlanningAgent("Meal Planner"),
        ShoppingAgent("Shopping Assistant"),
        TravelAgent("Travel Coordinator")
    ]

    threads = []
    
    # Map agents to their complex workflow methods
    workflow_map = {
        MealPlanningAgent: 'plan_meal_workflow',
        ShoppingAgent: 'shop_workflow',
        TravelAgent: 'plan_trip_workflow',
    }
    
    # Iterate through agents, determine their task method, and start a new thread
    for agent in agents:
        for agent_type, method_name in workflow_map.items():
            if isinstance(agent, agent_type):
                thread = threading.Thread(
                    target=getattr(agent, method_name), # Get the method by name
                    name=f"{agent.name}-Thread"
                )
                threads.append(thread)
                thread.start() # Start execution of the thread
                break # Move to the next agent

    # Wait for all threads to finish their execution
    for thread in threads:
        thread.join()
        
    print("\n" + "="*50)
    print("--- All agents have completed their tasks. Simulation finished. ---")
    print("="*50 + "\n")


if __name__ == "__main__":
    run_agents()

