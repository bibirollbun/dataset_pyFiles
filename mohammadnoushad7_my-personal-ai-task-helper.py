# AI Personal Task Assistant – Starter Code
# Simple offline agent (No API keys, no external calls)

def personal_task_agent(task):
    """
    Simple agent that takes any personal task and returns:
    - Summary
    - Steps to do it
    - Tools needed (if any)
    - Time estimation
    """
    
    prompt = f"""
    You are a personal AI assistant.
    Task: {task}

    Give output in this format:
    - Summary
    - Steps to do it
    - Tools needed
    - Time estimate
    """

    # Agent logic (offline processing)
    summary = f"This task is about: {task}"
    steps = [
        "Understand what the task is",
        "Plan the task",
        "Start doing it",
        "Review and complete"
    ]
    tools = "No special tools needed"
    time_estimate = "Approx. 30 minutes"

    return {
        "summary": summary,
        "steps": steps,
        "tools": tools,
        "time_estimate": time_estimate
    }

# Test the agent
result = personal_task_agent("Plan a healthy diet for 1 week")
result

