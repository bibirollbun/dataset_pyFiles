# ==========================================
# Student Study Support & Productivity Agent
# Pure Python Version — Kaggle/Colab Ready
# ==========================================

# Step 1: Initialize memory
memory = {
    "tasks": [],
    "subjects": [],
    "goals": ""
}

# Step 2: Define Tools

def create_study_plan(subjects, days):
    """Create a simple study plan distributing subjects across days."""
    plan = {}
    index = 0
    for d in range(1, days+1):
        plan[f"Day {d}"] = subjects[index % len(subjects)]
        index += 1
    return plan

def summarize_topic(text):
    """Summarize input text (basic placeholder)."""
    if len(text) <= 100:
        return text
    return text[:120] + " ... (summary continues)"

def add_task(task):
    """Add a study task to memory."""
    memory["tasks"].append(task)
    return f"Task added: {task}"

def next_task():
    """Return the next task from memory."""
    if memory["tasks"]:
        return f"Next task: {memory['tasks'][0]}"
    return "No tasks remaining!"

# Step 3: Simulated Agent Function
def study_agent(input_text):
    """
    Simulates agent reasoning to choose tool based on keywords
    """
    text = input_text.lower()
    
    if "make" in text and "plan" in text:
        # Extract subjects and days
        import re
        subjects = re.findall(r'\[(.*?)\]', input_text)
        subjects = [s.strip() for s in subjects[0].split(',')] if subjects else ["Maths","Science","English"]
        days_search = re.search(r'(\d+)', input_text)
        days = int(days_search.group(1)) if days_search else 5
        return create_study_plan(subjects, days)
    
    elif "summarize" in text:
        return summarize_topic(input_text)
    
    elif "add task" in text:
        return add_task(input_text.split("Add task:")[-1].strip())
    
    elif "next task" in text:
        return next_task()
    
    else:
        return "Sorry, I don't understand. Please ask about planning, summarizing, or tasks."

# ==========================================
# Step 4: Demonstrate Agent Usage
# ==========================================

print("=== Study Plan ===")
plan = study_agent("I have [Maths, Science, English]. Make a 5 day plan")
for day, subject in plan.items():
    print(f"{day}: {subject}")

print("\n=== Topic Summary ===")
text = """
Operating system is a system software that manages hardware and software resources,
provides services for computer programs, and acts as an intermediary between the user
and computer hardware.
"""
summary = study_agent(f"Summarize this topic: {text}")
print(summary)

print("\n=== Adding Tasks ===")
task1 = study_agent("Add task: Complete Maths Chapter 3")
print(task1)
task2 = study_agent("Add task: Revise English grammar")
print(task2)

print("\n=== Next Task ===")
next_t = study_agent("What should I study next?")
print(next_t)

print("\n=== All Tasks in Memory ===")
print(memory["tasks"])


