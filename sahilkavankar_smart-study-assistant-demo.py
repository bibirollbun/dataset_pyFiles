import datetime

# Simple Smart Study Assistant demo agent

def study_agent(message):
    message = message.lower()

    if "plan" in message or "study" in message:
        return {
            "action": "create_plan",
            "output": "Here is a simple study plan:\n1. Choose a topic.\n2. Study for 45 minutes.\n3. Take a 10-minute break.\n4. Review key points."
        }
    
    if "remind" in message:
        return {
            "action": "set_reminder",
            "output": "Reminder noted. I will remind you at the requested time."
        }
    
    if "progress" in message:
        return {
            "action": "check_progress",
            "output": "You completed 60 percent of your study goals this week."
        }

    return {
        "action": "respond",
        "output": "I can help you with study plans, reminders, and progress tracking."
    }

# Test examples
tests = [
    "Plan my study for today",
    "Remind me to study DSA tomorrow",
    "Check my progress",
    "Hello"
]

for t in tests:
    print("User:", t)
    print("Agent:", study_agent(t))
    print()


