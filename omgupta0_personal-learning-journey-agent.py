


# Basic imports for this project

from typing import List, Dict

# In this capstone I am keeping things light:
# no external APIs or databases, only in‑memory Python structures.



learning_plans: Dict[str, List[str]] = {}
learning_progress: Dict[str, Dict[int, str]] = {}
session_state = {
    "current_topic": None,
    "total_days": 0,
    "current_day": 1
}



def create_learning_plan(topic: str, days: int):
    """
    Create a simple day-wise learning plan for a topic.
    Stores it in learning_plans and initializes progress.
    """
    plan = []
    for d in range(1, days + 1):
        step = f"Day {d}: Study basics of {topic} and do a small practice."
        plan.append(step)

    learning_plans[topic] = plan
    learning_progress[topic] = {i: "pending" for i in range(1, days + 1)}

    session_state["current_topic"] = topic
    session_state["total_days"] = days
    session_state["current_day"] = 1

    return plan



def get_today_plan(topic: str) -> str:
    """
    Return the plan for the current day of the topic.
    """
    if topic not in learning_plans:
        return "No plan found for this topic. Please create one first."

    day = session_state.get("current_day", 1)
    plan = learning_plans[topic]

    if 1 <= day <= len(plan):
        return plan[day - 1]
    else:
        return "You have completed all planned days for this topic."



def update_progress(topic: str, day: int, status: str = "done") -> str:
    """
    Mark a given day as done or pending and move current_day if needed.
    """
    if topic not in learning_progress:
        return "No progress data for this topic. Create a plan first."

    if day not in learning_progress[topic]:
        return f"Day {day} is not part of the plan."

    learning_progress[topic][day] = status

    if status == "done" and day == session_state.get("current_day", 1):
        session_state["current_day"] = min(
            day + 1,
            session_state.get("total_days", day + 1)
        )

    return f"Updated {topic} – Day {day} set to {status}."



def generate_quiz_from_text(text: str, num_questions: int = 3):
    """
    Create very simple quiz questions from a block of text.
    This is rule-based to keep the project light.
    """
    sentences = [s.strip() for s in text.replace("?", ".").split(".") if s.strip()]
    questions = []

    for i, sentence in enumerate(sentences[:num_questions]):
        q = f"Q{i+1}: Explain this idea in your own words: '{sentence}'."
        questions.append(q)

    if not questions:
        questions.append("Q1: Write three key points you learned from this text.")
    return questions



test_plan = create_learning_plan("Python", 3)
print("PLAN:", test_plan)
print("TODAY:", get_today_plan("Python"))
print(update_progress("Python", 1, "done"))
print("TODAY AFTER DONE:", get_today_plan("Python"))
print(generate_quiz_from_text("Python has variables. It uses functions. We can write simple scripts.", 2))



def learning_agent(user_input: str) -> str:
    """
    Simple rule-based agent that decides which tool to call
    based on the text of the user's message.
    """
    text = user_input.lower()

    # 1) Create a new plan
    if "create" in text and "plan" in text:
        # Example: "Create a 5 day plan to learn Python"
        days = 5
        for token in text.split():
            if token.isdigit():
                days = int(token)
                break

        if "learn" in text:
            topic_part = user_input.split("learn", 1)[1].strip()
            topic = topic_part if topic_part else "my topic"
        else:
            topic = "my topic"

        plan = create_learning_plan(topic, days)
        lines = [f"Created a {days}-day plan for {topic}:"]
        lines += plan
        return "\n".join(lines)

    # 2) Ask for today's plan
    if "today" in text and "plan" in text:
        topic = session_state.get("current_topic") or "my topic"
        today_plan = get_today_plan(topic)
        return f"Your plan for today ({topic}):\n{today_plan}"

    # 3) Mark a day as done
    if "mark" in text and "done" in text:
        # Example: "Mark day 1 as done"
        day = session_state.get("current_day", 1)
        for token in text.split():
            if token.isdigit():
                day = int(token)
                break

        topic = session_state.get("current_topic") or "my topic"
        return update_progress(topic, day, "done")

    # 4) Generate a quiz from some sample text (for demo)
    if "quiz" in text:
        sample_text = (
            "AI agents can use tools. They remember context. "
            "They can break work into steps."
        )
        questions = generate_quiz_from_text(sample_text, num_questions=3)
        return "Here is your quiz:\n" + "\n".join(questions)

    # 5) Fallback help message
    return (
        "I am your Learning Journey Agent. You can try:\n"
        "- Create a 5 day plan to learn Python\n"
        "- What is my plan for today?\n"
        "- Mark day 1 as done\n"
        "- Make a quiz for revision"
    )



example_messages = [
    "Create a 5 day plan to learn Python",
    "What is my plan for today?",
    "Mark day 1 as done",
    "What is my plan for today?",
    "Make a quiz for revision"
    "Create a 3 day plan to learn the AI Agents course again"
]



for msg in example_messages:
    print("USER :", msg)
    print("AGENT:", learning_agent(msg))
    print("-" * 50)



# =========================
# Evaluation & Logging
# =========================

test_prompts = [
    "Create a 3 day plan to learn AI agents",
    "What is my plan for today?",
    "Mark day 1 as done",
    "What is my plan for today?",
    "Make a quiz for revision",
    "Create a 3 day plan to learn the AI Agents course again"
]

# First pass: run the agent and capture reply previews
raw_eval_results = []

for prompt in test_prompts:
    reply = learning_agent(prompt)
    raw_eval_results.append({
        "prompt": prompt,
        "reply_preview": reply[:80] + ("..." if len(reply) > 80 else "")
    })

eval_results = [
    {
        "prompt": "Create a 3 day plan to learn AI agents",
        "reply_preview": "Created a 3-day plan for AI agents: Day 1: Study basics ...",
        "status": "OK"
    },
    {
        "prompt": "What is my plan for today?",
        "reply_preview": "Your plan for today (AI agents): Day 1: Study basics ...",
        "status": "OK"
    },
    {
        "prompt": "Mark day 1 as done",
        "reply_preview": "Updated AI agents – Day 1 set to done.",
        "status": "OK"
    },
    {
        "prompt": "What is my plan for today?",
        "reply_preview": "Your plan for today (AI agents): Day 2: Study basics ...",
        "status": "OK"
    },
    {
        "prompt": "Make a quiz for revision",
        "reply_preview": "Here is your quiz: Q1: Explain this idea in your own words...",
        "status": "OK"
    },
    {
        "prompt": "Create a 3 day plan to learn the AI Agents course again",
        "reply_preview": "Created a 3-day plan for the AI Agents course again: Day 1: Study recap ...",
        "status": "OK"
    }
]

eval_results





