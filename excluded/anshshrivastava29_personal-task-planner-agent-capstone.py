# Install (only if needed in Kaggle environment)
!pip install --quiet google-generativeai


# Minimal multi-agent example for Capstone
# NOTE: Do NOT commit your API key. Use Kaggle secrets or set it manually for local runs.

# Example "tool"
def calculate_time_needed(task):
    return f"Estimated: 30 minutes for task - {task}"

# Simple in-notebook "session memory"
session_memory = {}

# Agent 1: Clean tasks
def clean_agent(tasks):
    # Simulate cleaning: strip and capitalize
    return [t.strip().capitalize() for t in tasks]

# Agent 2: Prioritize tasks
def priority_agent(cleaned_tasks):
    # Very simple prioritization: by length (demo only)
    prioritized = []
    for t in cleaned_tasks:
        level = "High" if len(t) <= 15 else "Medium"
        prioritized.append((t, level))
    return prioritized

# Agent 3: Schedule & save memory
def schedule_agent(prioritized_tasks, user_id="user_default"):
    schedule = []
    hour = 9
    for task, pr in prioritized_tasks:
        est = calculate_time_needed(task)
        schedule.append({"time": f"{hour}:00", "task": task, "priority": pr, "estimate": est})
        hour += 1
    session_memory[user_id] = {"priorities": prioritized_tasks, "schedule": schedule}
    return schedule

# Pipeline
def task_planner(tasks):
    step1 = clean_agent(tasks)
    step2 = priority_agent(step1)
    step3 = schedule_agent(step2)
    return step1, step2, step3

# Example usage
if __name__ == "__main__":
    tasks = ["finish report", "do laundry", "prepare slides for presentation"]
    cleaned, prioritized, schedule = task_planner(tasks)
    print("CLEANED:", cleaned)
    print("PRIORITIZED:", prioritized)
    print("SCHEDULE:", schedule)


