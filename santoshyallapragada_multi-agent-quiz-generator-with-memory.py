# ==========================
# Phase 1: Setup
# ==========================
import random
import re
import math
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




# Load your dataset
excel_df = pd.read_excel("/kaggle/input/programming-mcqs/mcq_dataset.csv.xlsx")
excel_df.to_csv("/kaggle/working/mcq_dataset.csv", index=False)

print("Converted Excel to CSV successfully!")
print(excel_df.head())
mcq_df = pd.read_csv("/kaggle/working/mcq_dataset.csv")
mcq_df.columns = mcq_df.columns.str.strip().str.lower()
print("Dataset loaded successfully!")
mcq_df.columns = mcq_df.columns.str.strip().str.lower()
print(mcq_df.head())
print(mcq_df["domain"].unique())
mcq_df.columns = (
    mcq_df.columns
    .astype(str)
    .str.strip()
    .str.replace(r'\s+', '', regex=True)   # remove spaces
    .str.lower()
)

print("Normalized columns:", mcq_df.columns.tolist())




# Topic library
TOPIC_LIBRARY = {
    "python": [
        "Basics", "Data Types", "Control Flow", "Functions",
        "Modules", "OOP", "File Handling", "Error Handling",
        "Libraries (NumPy, Pandas)", "APIs", "Projects"
    ],
    "sql": [
        "Basics", "SELECT Queries", "Joins", "Aggregations",
        "Subqueries", "Indexes", "Views", "Stored Procedures",
        "Optimization"
    ],
    "java":[
        "Basics", "Data Types", "Control Flow", "OOP Concepts",
        "Classes & Objects", "Inheritance", "Polymorphism",
        "Encapsulation", "Abstraction", "Interfaces",
        "Packages", "Exception Handling", "Collections Framework",
        "Multithreading", "File I/O", "Generics",
        "Java 8 Features (Lambdas, Streams)", "JVM & Memory Model",
        "Build Tools (Maven, Gradle)", "Projects"
    ],
    "ai_agents": [
        "Basics of Agents", "Types of Agents", "LLM Fundamentals",
        "RAG (Retrieval-Augmented Generation)", "Tools & ToolCalling",
        "Prompt Engineering", "Memory & Context Management",
        "Planning & Reasoning", "Agent Frameworks (LangChain, AutoGen, CrewAI)",
        "Building Multi-Agent Systems", "Deploying AI Agents"
    ],
    "machine_learning": [
        "Basics", "Supervised Learning", "Unsupervised Learning",
        "Regression", "Classification", "Clustering",
        "Feature Engineering", "Model Evaluation",
        "Neural Networks", "Deep Learning Basics",
        "TensorFlow/PyTorch", "ML Projects"
    ],
    "system_design": [
        "Basics", "Scalability", "Load Balancing",
        "Caching", "Database Design", "CAP Theorem",
        "Consistency Models", "Message Queues",
        "Microservices", "High-Level Design",
        "Low-Level Design", "System Design Case Studies"
    ],
    "operating_systems": [
        "Basics", "Processes", "Threads", "Scheduling",
        "Memory Management", "Deadlocks", "Virtual Memory",
        "File Systems", "System Calls", "CPU Management"
    ],
    "full_stack_development": [
        "HTML", "CSS", "JavaScript", "Frontend Frameworks (React/Angular/Vue)",
        "Backend Basics", "Node.js", "Express/Django/Flask",
        "REST APIs", "Authentication", "Databases (SQL/NoSQL)",
        "DevOps Basics", "Deployment", "Projects"
    ],
    "data_structures_and_algorithms": [
        "Arrays", "Strings", "Linked Lists", "Stacks & Queues",
        "Trees", "Graphs", "Heaps", "Sorting & Searching",
        "Greedy", "DP", "Recursion", "Time & Space Complexity",
        "Practice Problems"
    ],
    "git_and_github": [
        "Basics", "Branches", "Merging", "Pull Requests",
        "Version Control Workflows", "Rebasing", "Collaboration",
        "GitHub Actions (Basics)"
    ],
    "computer_networking": [
        "Basics", "OSI Model", "TCP/IP", "DNS", "HTTP/HTTPS",
        "Sockets", "Load Balancing", "CDNs"
    ],
    "aptitude_and_reasoning": [
        "Quantitative Aptitude", "Logical Reasoning",
        "Puzzles", "Interviews Practice"
    ]
}

# Domain aliases
DOMAIN_ALIASES = {
    # Python
    "python programming": "python",
    "py": "python",
    "python basics": "python",

    # SQL / Databases
    "database": "sql",
    "dbms": "sql",
    "data querying": "sql",
    "postgres": "sql",
    "mysql": "sql",

    # Java
    "core java": "java",
    "java programming": "java",
    "jvm": "java",

    # AI Agents
    "agents": "ai_agents",
    "ai agent": "ai_agents",
    "llm agents": "ai_agents",
    "rag": "ai_agents",

    # Machine Learning
    "ml": "machine_learning",
    "ai": "machine_learning",
    "artificial intelligence": "machine_learning",
    "deep learning": "machine_learning",

    # System Design
    "sd": "system_design",
    "architecture": "system_design",
    "software design": "system_design",
    "high level design": "system_design",

    # Operating Systems
    "os": "operating_systems",
    "linux": "operating_systems",
    "kernel": "operating_systems",

    # Full Stack Development
    "fullstack": "full_stack_development",
    "web development": "full_stack_development",
    "frontend": "full_stack_development",
    "backend": "full_stack_development",
    "mern": "full_stack_development",

    # DSA
    "coding": "data_structures_and_algorithms",
    "dsa": "data_structures_and_algorithms",
    "competitive programming": "data_structures_and_algorithms",
    "algorithms": "data_structures_and_algorithms",

    # Git & GitHub
    "git": "git_and_github",
    "github": "git_and_github",
    "version control": "git_and_github",

    # Networking
    "networking": "computer_networking",
    "computer networks": "computer_networking",
    "tcp ip": "computer_networking",

    # Aptitude
    "aptitude": "aptitude_and_reasoning",
    "reasoning": "aptitude_and_reasoning",
    "quantitative": "aptitude_and_reasoning",
}



# ==========================
# Phase 2: Goal Parser
# ==========================
def parse_goal(user_input: str):
    """
    Parse free-text goal to extract:
    - domain (python/sql/java) via keywords/aliases
    - duration_days (days inferred from 'weeks', 'months', etc.)
    """
    user_input = user_input.lower()
    duration_match = re.search(r'(\d+)\s*(days|day|weeks|week|month|months|year|years)', user_input)
    duration = None
    if duration_match:
        num = int(duration_match.group(1))
        unit = duration_match.group(2)
        if "day" in unit: duration = num
        elif "week" in unit: duration = num * 7
        elif "month" in unit: duration = num * 30
        elif "year" in unit: duration = num * 365

    domain = None
    # Direct domain match
    for key in TOPIC_LIBRARY.keys():
        if key in user_input:
            domain = key
            break
    # Alias match
    if not domain:
        for alias, mapped in DOMAIN_ALIASES.items():
            if alias in user_input:
                domain = mapped
                break

    return {"domain": domain, "duration_days": duration}

# ==========================
# Phase 3: Planner
# ==========================
def create_plan(domain: str, duration_days: int, hours_per_day: int = 2):
    """
    Create a weekly plan by distributing topics across weeks.
    Each week includes topics and estimated study hours.
    """
    topics = TOPIC_LIBRARY.get(domain)
    if not topics or duration_days is None or duration_days <= 0:
        return {"error": "Domain not found or invalid duration"}

    total_hours = duration_days * hours_per_day
    hours_per_topic = total_hours / len(topics)
    weeks = max(1, math.ceil(duration_days / 7))
    chunk_size = max(1, math.ceil(len(topics) / weeks))

    weekly_plan = []
    for i in range(weeks):
        start = i * chunk_size
        end = start + chunk_size
        weekly_topics = topics[start:end]
        weekly_plan.append({
            "week": i + 1,
            "topics": weekly_topics,
            "estimated_hours": round(len(weekly_topics) * hours_per_topic, 2)
        })
    return {"domain": domain, "duration_days": duration_days, "weekly_plan": weekly_plan}

# ==========================
# Phase 4: Quiz Generator (Local/Kaggle Dataset)
def generate_quiz(domain: str, num_questions: int = 5, seed: int = 42):
    domain_questions = mcq_df[mcq_df["domain"].astype(str).str.strip().str.lower() == domain.lower()]
    if domain_questions.empty:
        print(f"No quiz items found for domain: {domain}")
        return []
    sampled = domain_questions.sample(n=min(num_questions, len(domain_questions)), random_state=seed)
    quiz = []
    for _, row in sampled.iterrows():
        quiz.append({
            "question": row["question"],
            "options": [row["optiona"], row["optionb"], row["optionc"], row["optiond"]],
            "correct_answer": row["answer"],
            "explanation": row.get("explanation", "")
        })
    return quiz


 

# ==========================
# Phase 5: Evaluator
# ==========================
""" 
def generate_quiz(domain: str, num_questions: int = 5, seed: int = 42):
    domain_questions = mcq_df[mcq_df["domain"].astype(str).str.strip().str.lower() == domain.lower()]
    if domain_questions.empty:
        print(f"No quiz items found for domain: {domain}")
        return []
    sampled = domain_questions.sample(n=min(num_questions, len(domain_questions)), random_state=seed)
    quiz = []
    for _, row in sampled.iterrows():
        quiz.append({
            "question": row["question"],
            "options": [row["optiona"], row["optionb"], row["optionc"], row["optiond"]],
            "correct_answer": row["answer"],
            "explanation": row.get("explanation", "")
        })
    return quiz
    """
def evaluate_quiz(quiz, user_answers):
    """
    Evaluate a quiz by comparing user answers with correct answers.
    
    Parameters:
    quiz (list of dict): Quiz generated by generate_quiz
    user_answers (list): List of user-selected answers (same length as quiz)
    
    Returns:
    dict: Evaluation summary with score and detailed results
    """
    score = 0
    details = []
    
    for i, (q, user_ans) in enumerate(zip(quiz, user_answers)):
        correct = q["correct_answer"]
        is_correct = (str(user_ans).strip().lower() == str(correct).strip().lower())
        if is_correct:
            score += 1
        details.append({
            "question": q["question"],
            "options": q["options"],
            "user_answer": user_ans,
            "correct_answer": correct,
            "is_correct": is_correct,
            "explanation": q.get("explanation", "")
        })
    
    return {
        "score": score,
        "total": len(quiz),
        "details": details
    }
# ==========================
# Phase 6: Memory System
# ==========================
LEARNER_MEMORY = {"domain": None, "progress": {}, "mastery": {}, "weak_topics": []}

def update_memory(plan: dict, week_index: int, evaluation: dict):
    """
    Update learner memory:
    - progress (week -> score)
    - mastery (topic -> strong/weak/pending)
    - weak_topics list
    Aligns topics with evaluated questions; marks extras as pending.
    """
    if "weekly_plan" not in plan or week_index >= len(plan["weekly_plan"]):
        return
    LEARNER_MEMORY["domain"] = plan["domain"]
    LEARNER_MEMORY["progress"][week_index + 1] = evaluation["score"]
    topics = plan["weekly_plan"][week_index]["topics"]

    # Map correctness by index; if fewer questions than topics, mark remaining as pending
    for idx, topic in enumerate(topics):
        if idx < len(evaluation["details"]):
            detail = evaluation["details"][idx]
            if detail["is_correct"]:
                LEARNER_MEMORY["mastery"][topic] = "strong"
            else:
                LEARNER_MEMORY["mastery"][topic] = "weak"
                if topic not in LEARNER_MEMORY["weak_topics"]:
                    LEARNER_MEMORY["weak_topics"].append(topic)
        else:
            LEARNER_MEMORY["mastery"][topic] = LEARNER_MEMORY["mastery"].get(topic, "pending")

# ==========================
# Phase 7: Scheduler
# ==========================
def create_daily_schedule(plan: dict, hours_per_day: int = 2):
    """
    Create a day-by-day schedule chunking weekly topics across ~7 days/week.
    """
    if "weekly_plan" not in plan:
        return []
    daily_schedule = []
    day_counter = 1
    for week in plan["weekly_plan"]:
        topics = week["topics"]
        if len(topics) == 0:
            continue
        chunk_size = max(1, len(topics) // 7)
        for i in range(0, len(topics), chunk_size):
            day_topics = topics[i:i + chunk_size]
            daily_schedule.append({
                "day": day_counter,
                "week": week["week"],
                "topics": day_topics,
                "estimated_hours": round(len(day_topics) * (hours_per_day / chunk_size), 2),
                "revision": []
            })
            day_counter += 1
    return daily_schedule

def integrate_memory(daily_schedule: list, memory: dict):
    """
    Add weak topics from memory into each day's revision list if present in that day.
    """
    weak_topics = memory.get("weak_topics", [])
    for day in daily_schedule:
        for topic in weak_topics:
            if topic in day["topics"] and topic not in day["revision"]:
                day["revision"].append(topic)
    return daily_schedule

# ==========================
# Phase 8: Visualization
# ==========================
def plot_progress(memory: dict):
    """
    Plot weekly scores to visualize progress.
    """
    weeks = list(memory["progress"].keys())
    scores = list(memory["progress"].values())
    if not weeks:
        print("No progress yet to plot.")
        return
    plt.figure(figsize=(6, 3))
    plt.plot(weeks, scores, marker="o")
    plt.title("Weekly Progress")
    plt.xlabel("Week")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.3)
    plt.show()

# ==========================
# Phase 9: Example end-to-end run
# ==========================
if __name__ == "__main__":
    # 1) Parse goal
    goal_text = "I want to learn java in 28 days"
    parsed = parse_goal(goal_text)
    domain = parsed["domain"]
    duration_days = parsed["duration_days"]
    print("Parsed Goal ->", parsed)

    # 2) Create plan
    plan = create_plan(domain, duration_days, hours_per_day=2)
    if "error" in plan:
        print("Plan error:", plan["error"])
    else:
        print("\nWeekly Plan:")
        for w in plan["weekly_plan"]:
            print(f"Week {w['week']}: Topics={w['topics']}, Estimated Hours={w['estimated_hours']}")

        # 3) Generate quiz for current domain (Week 1 demo)
        quiz = generate_quiz(domain, num_questions=3)
        if not quiz:
            print("\nNo quiz items found for domain:", domain)
        else:
            print("\nQuiz:")
            for i, q in enumerate(quiz, 1):
                print(f"{i}. {q['question']}")
                print("   Options:", q["options"])

            # 4) Simulate user answers (replace with real input)
            simulated_answers = {i + 1: random.choice(["A", "B", "C", "D"]) for i in range(len(quiz))}

            # 5) Evaluate
            evaluation = evaluate_quiz(quiz, simulated_answers)
            print("\nEvaluation Summary:", evaluation["score"], "/", evaluation["total"])
            for d in evaluation["details"]:
                icon = "✅" if d["is_correct"] else "❌"
                print(f"{icon} Q: {d['question']} | Your: {d['user_answer']} | Correct: {d['correct_answer']} | Note: {d['explanation']}")

            # 6) Update memory
            update_memory(plan, week_index=0, evaluation=evaluation)
            print("\nMemory (weak topics):", LEARNER_MEMORY["weak_topics"])

            # 7) Build daily schedule and integrate memory
            daily = create_daily_schedule(plan, hours_per_day=2)
            daily = integrate_memory(daily, LEARNER_MEMORY)
            print("\nDaily Schedule (first 7 days):")
            for day in daily[:7]:
                print(f"Day {day['day']} (Week {day['week']}): Topics={day['topics']} | Hours={day['estimated_hours']} | Revision={day['revision']}")

            # 8) Plot progress
            plot_progress(LEARNER_MEMORY)


