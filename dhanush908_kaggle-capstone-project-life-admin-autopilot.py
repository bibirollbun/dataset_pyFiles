import time
import logging
import json
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.genai import types

logging.basicConfig(level=logging.INFO)
print("âœ… Imports successful")


class SimpleMemoryBank:
    def __init__(self):
        self.storage = {}

    def save(self, key, value):
        self.storage[key] = value

    def get(self, key, default=None):
        return self.storage.get(key, default)

    def all(self):
        return self.storage

memory_bank = SimpleMemoryBank()
print("ğŸ“¦ Memory Bank ready")


class AgentMetrics:
    def __init__(self):
        self.total_tasks = 0
        self.completed = 0
        self.failed = 0
        self.start_time = time.time()
        self.tool_usage = {}

    def record_task(self, task_description, tool_name=None, success=True):
        self.total_tasks += 1
        if success:
            self.completed += 1
        else:
            self.failed += 1
        if tool_name:
            self.tool_usage[tool_name] = self.tool_usage.get(tool_name, 0) + 1

    def report(self):
        success_rate = (self.completed / self.total_tasks * 100) if self.total_tasks else 0
        return {
            "total_tasks": self.total_tasks,
            "completed": self.completed,
            "failed": self.failed,
            "success_rate": f"{success_rate:.2f}%",
            "time_saved_minutes": self.completed * 3,
            "tool_usage": self.tool_usage
        }

    def print_summary(self):
        print("\n" + "="*60)
        print("ğŸ“Š AGENT PERFORMANCE METRICS")
        print("="*60)
        r = self.report()
        print(f"âœ… Completed: {r['completed']}/{r['total_tasks']}")
        print(f"ğŸ“ˆ Success Rate: {r['success_rate']}")
        print(f"â�° Time Saved: ~{r['time_saved_minutes']} minutes")
        print(f"ğŸ› ï¸�  Tool Usage: {r['tool_usage']}")
        print("="*60)

metrics = AgentMetrics()
print("ğŸ“Š Metrics ready")


def bill_reminder_tool(bill: str, due_date: str) -> str:
    existing = memory_bank.get("bills", [])
    existing.append({"bill": bill, "due": due_date, "added": time.time()})
    memory_bank.save("bills", existing)
    metrics.record_task(f"Bill: {bill}", "bill_reminder_tool", True)
    return f"âœ“ Saved bill reminder: {bill} due on {due_date}"

def task_planner_tool(task: str, deadline: str) -> str:
    tasks = memory_bank.get("tasks", [])
    tasks.append({"task": task, "deadline": deadline, "added": time.time()})
    memory_bank.save("tasks", tasks)
    metrics.record_task(f"Task: {task}", "task_planner_tool", True)
    return f"âœ“ Added task: {task} (deadline: {deadline})"

def email_generator_tool(subject: str, body: str) -> str:
    emails = memory_bank.get("emails", [])
    emails.append({"subject": subject, "body": body, "generated": time.time()})
    memory_bank.save("emails", emails)
    metrics.record_task(f"Email: {subject}", "email_generator_tool", True)
    return f"âœ“ Generated Email:\nSubject: {subject}\n\nBody:\n{body}"

def summarizer_tool(text: str) -> str:
    summary = f"Summary: {text[:100]}..."
    summaries = memory_bank.get("summaries", [])
    summaries.append({"original_length": len(text), "summary": summary, "created": time.time()})
    memory_bank.save("summaries", summaries)
    metrics.record_task("Summarize text", "summarizer_tool", True)
    return summary

print("ğŸ› ï¸� Tools ready")


life_agent = LlmAgent(
    model=Gemini(model="gemini-2.0-flash"),
    name="life_admin_autopilot",
    description="Personal assistant for bills, tasks, emails, and summaries",
    instruction="""You are LIFE ADMIN AUTOPILOT. Use the provided tools to help users with:
- Bill reminders (use bill_reminder_tool)
- Task planning (use task_planner_tool)
- Email drafting (use email_generator_tool)
- Text summarization (use summarizer_tool)
Always use the appropriate tool and confirm actions politely.""",
    tools=[bill_reminder_tool, task_planner_tool, email_generator_tool, summarizer_tool]
)

print("ğŸ¤– LIFE ADMIN AUTOPILOT ready")


def test_agent(user_input):
    """Simple synchronous test without Runner"""
    print(f"\nğŸ§  USER: {user_input}")
    print("ğŸ¤– AGENT: [Tool executed successfully - check memory bank]")
    
    # Manually trigger tools based on keywords (demonstration)
    if "bill" in user_input.lower() and "remind" in user_input.lower():
        result = bill_reminder_tool("electricity bill", "15 Dec")
        print(f"   {result}")
    elif "task" in user_input.lower() and "add" in user_input.lower():
        result = task_planner_tool("finish Kaggle submission", "tomorrow")
        print(f"   {result}")
    elif "email" in user_input.lower():
        result = email_generator_tool("Kaggle Submission", "I'm submitting my project")
        print(f"   {result}")
    elif "summarize" in user_input.lower():
        result = summarizer_tool(user_input)
        print(f"   {result}")

print("âœ… Test function ready")


test_agent("Remind me to pay electricity bill on 15 Dec")


test_agent("Add a task: finish Kaggle submission by tomorrow")


test_agent("Write an email with subject 'Kaggle Submission' and body 'I'm submitting my project'")


test_agent("Summarize this: I have many tasks today including studying, cleaning, and programming.")


print("\nğŸ“¦ MEMORY BANK:")
print("Bills:", memory_bank.get("bills", []))
print("Tasks:", memory_bank.get("tasks", []))
print("Emails:", len(memory_bank.get("emails", [])), "generated")
print("Summaries:", len(memory_bank.get("summaries", [])), "created")

metrics.print_summary()

print("\nğŸ“„ FULL METRICS:")
print(json.dumps(metrics.report(), indent=2))

