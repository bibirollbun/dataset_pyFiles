import json, time, uuid
from concurrent.futures import ThreadPoolExecutor

# -----------------------
# MEMORY BANK
# -----------------------
MEMORY_FILE = "/kaggle/working/memory_productivity.json"

def load_memory():
    try:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    except:
        return {"tasks": {}, "jobs": {}, "logs": []}

def save_memory(mem):
    with open(MEMORY_FILE, "w") as f:
        json.dump(mem, f, indent=2)

memory = load_memory()


# -----------------------
# LOGGING TOOL
# -----------------------
def log_event(event_dict):
    memory["logs"].append(event_dict)
    print(json.dumps(event_dict))


# -----------------------
# AGENT 1: TASK CREATOR
# -----------------------
def create_task(user_id, title):
    task_id = str(uuid.uuid4())[:8]
    task = {
        "task_id": task_id,
        "user_id": user_id,
        "title": title,
        "created_at": time.time()
    }
    memory["tasks"][task_id] = task
    log_event({"event": "task_created", **task})
    return task


# -----------------------
# PARALLEL AGENTS (priority/category/effort)
# -----------------------
def priority_refine(task):
    return 5 if "urgent" in task["title"].lower() else 3

def category_refine(task):
    t = task["title"].lower()
    if "email" in t:
        return "communication"
    if "study" in t:
        return "learning"
    return "general"

def effort_refine(task):
    return 30 if "email" in task["title"].lower() else 90


def refine_task(task):
    with ThreadPoolExecutor() as exe:
        futures = [
            exe.submit(priority_refine, task),
            exe.submit(category_refine, task),
            exe.submit(effort_refine, task)
        ]
        priority, category, effort = [f.result() for f in futures]

    refined = {
        "priority": priority,
        "category": category,
        "effort": effort
    }
    
    log_event({"event": "task_refined", "task_id": task["task_id"], **refined})
    return refined


# -----------------------
# LOOP AGENT (Scheduler)
# -----------------------
def schedule_task(task_id):
    job_id = str(uuid.uuid4())[:8]
    memory["jobs"][job_id] = {"task_id": task_id, "status": "pending"}
    log_event({"event": "job_created", "job_id": job_id, "task_id": task_id})

    for attempt in range(1, 4):
        log_event({"event": "scheduling_attempt", "job_id": job_id, "task_id": task_id, "attempt": attempt})

        if attempt < 3:
            log_event({"event": "job_paused", "job_id": job_id, "reason": "simulated conflict"})
            time.sleep(0.2)
        else:
            memory["jobs"][job_id]["status"] = "completed"
            log_event({"event": "job_completed", "job_id": job_id, "slot": "tomorrow_18:00"})

    return job_id


# -----------------------
# COACH AGENT
# -----------------------
def coach_insight(user_id):
    tasks = [t for t in memory["tasks"].values() if t["user_id"] == user_id]
    jobs = [j for j in memory["jobs"].values() if memory["tasks"][j["task_id"]]["user_id"] == user_id]

    completed = sum(1 for j in jobs if j["status"] == "completed")

    insight = {
        "user_id": user_id,
        "total_tasks": len(tasks),
        "completed_or_scheduled": completed,
        "pending": len(tasks) - completed,
        "advice": "Try time-blocking high priority tasks in the morning."
    }

    print("\n--- Coach insight for", user_id, "---")
    print(json.dumps(insight, indent=2))
    return insight


# -----------------------
# PIPELINE EXECUTION
# -----------------------
def run_pipeline():
    # 4 sample tasks
    t1 = create_task("user_1", "Prepare project report")
    t2 = create_task("user_1", "Reply to client emails")
    t3 = create_task("user_2", "Study for ML exam")
    t4 = create_task("user_1", "Design slide deck")

    for t in [t1, t2, t3, t4]:
        refine_task(t)
        schedule_task(t["task_id"])

    insight = coach_insight("user_1")

    # Save memory
    save_memory(memory)
    print("\nMemory saved to:", MEMORY_FILE)

    # Final evaluation
    completed = sum(1 for j in memory["jobs"].values() if j["status"] == "completed")
    total = len(memory["jobs"])
    print(f"\nEvaluation: completion_rate = {completed}/{total} ({(completed/total)*100:.1f}%)")

# Run everything
run_pipeline()


