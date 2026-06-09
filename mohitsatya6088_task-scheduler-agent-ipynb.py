print("Kaggle Agents Project Started Successfully ðŸš€")


import json
import datetime

FILE = "tasks.json"

# Load tasks from file
def load_tasks():
    try:
        with open(FILE, "r") as f:
            return json.load(f)
    except:
        return []

# Save tasks to file
def save_tasks(tasks):
    with open(FILE, "w") as f:
        json.dump(tasks, f, indent=4)



# Add a new task
def add_task(task, date_time):
    tasks = load_tasks()
    tasks.append({"task": task, "time": date_time, "status": "pending"})
    save_tasks(tasks)
    return "Task added successfully!"

# View today's tasks
def view_today():
    tasks = load_tasks()
    today = datetime.date.today().strftime("%Y-%m-%d")
    today_list = [t for t in tasks if today in t["time"]]
    return today_list if today_list else "No tasks today."

# Mark task as completed
def complete_task(task_name):
    tasks = load_tasks()
    for t in tasks:
        if t["task"].lower() == task_name.lower():
            t["status"] = "done"
            save_tasks(tasks)
            return "Task marked completed!"
    return "Task not found."



# Test the agent

print(add_task("Gym workout", "2025-12-01 07:00"))
print(add_task("Work on Kaggle project", "2025-12-01 10:00"))

print("\nToday's tasks:")
print(view_today())

print("\nMarking task complete:")
print(complete_task("Gym workout"))

print("\nUpdated tasks:")
print(view_today())



def agent():
    print("ðŸ§  Task Agent Activated!")
    print("Commands:")
    print("1) add <task> <YYYY-MM-DD HH:MM>")
    print("2) today")
    print("3) complete <task>")

    while True:
        user = input("\nYou: ")

        if user.startswith("add"):
            parts = user.split(" ", 2)
            task = parts[1]
            time = parts[2]
            print("Agent:", add_task(task, time))

        elif user == "today":
            print("Agent:", view_today())

        elif user.startswith("complete"):
            task = user.replace("complete ", "")
            print("Agent:", complete_task(task))

        elif user in ["exit", "quit"]:
            print("Agent: Goodbye! ðŸ‘‹")
            break

        else:
            print("Agent: Invalid command.")



# Uncomment the next line only when you want to chat with the agent manually
# agent()



# Final automated execution for Kaggle Publishing (NO input required)

print("Running automated test...")

add_task("Demo task - Kaggle run", "2025-12-02 09:00")
print("Today's tasks:", view_today())

print("Agent test completed successfully âœ”")


