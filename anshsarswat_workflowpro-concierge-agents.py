import time
import logging
from concurrent.futures import ThreadPoolExecutor

# ------------------ Logging (Observability) ------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

class LocalLLM:
    """A lightweight rule-based model to simulate LLM responses offline."""

    def generate(self, role, prompt):
        logging.info(f"[LLM] Processing request as {role}...")

        if "break into steps" in prompt.lower():
            return "- Research topic\n- Write draft\n- Edit and finalize\n- Send or submit"
        
        if "extract event" in prompt.lower():
            return "{'title':'Team Meeting', 'time':'Tomorrow 3PM'}"
        
        if "summarize" in prompt.lower():
            return "You need to schedule a Team Meeting and complete a task today."
        
        return "I understand. Let me assist."


# ------------------ Tools ------------------
class CalendarTool:
    def create_event(self, title, time):
        logging.info("[CalendarTool] Scheduling event...")
        return f"ğŸ“… Event Scheduled: '{title}' at {time}"


class EmailTool:
    def draft_email(self, to, subject, body):
        logging.info("[EmailTool] Creating email draft...")
        return f"ğŸ“§ Draft Created:\nTo: {to}\nSubject: {subject}\nBody: {body}"


# ------------------ Memory System ------------------
class Memory:
    def __init__(self):
        self.store = []

    def add(self, entry):
        logging.info("[Memory] Storing entry...")
        self.store.append(entry)

    def get(self):
        return self.store


# ------------------ Agents ------------------
class CalendarAgent:
    def __init__(self, llm, tool):
        self.llm = llm
        self.tool = tool

    def handle(self, request):
        response = self.llm.generate("Calendar Agent", "Extract event details: " + request)
        return self.tool.create_event("Team Meeting", "Tomorrow at 3PM")


class TaskAgent:
    def __init__(self, llm):
        self.llm = llm

    def handle(self, request):
        response = self.llm.generate("Task Planner", "Break into steps: " + request)
        return f"ğŸ“� Task Breakdown:\n{response}"


class SummarizerAgent:
    def __init__(self, llm):
        self.llm = llm

    def summarize(self, logs):
        text = "\n".join(logs)
        return self.llm.generate("Summarizer", "Summarize: " + text)


# ------------------ Orchestrator (Multi-Agent System) ------------------
class ConciergeOrchestrator:
    def __init__(self):
        self.llm = LocalLLM()
        self.memory = Memory()
        self.calendar_agent = CalendarAgent(self.llm, CalendarTool())
        self.task_agent = TaskAgent(self.llm)
        self.summarizer = SummarizerAgent(self.llm)

    def process(self, user_request):
        logging.info("[Orchestrator] Starting multi-agent workflow...")

        outputs = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_calendar = executor.submit(self.calendar_agent.handle, user_request)
            future_summary = executor.submit(self.task_agent.handle, user_request)

            outputs.append(future_calendar.result())
            outputs.append(future_summary.result())

        summary = self.summarizer.summarize(outputs)
        outputs.append("ğŸ“Œ Summary: " + summary)

        for o in outputs:
            self.memory.add(o)

        return outputs


# ------------------ RUN TEST ------------------

agent = ConciergeOrchestrator()

test_request = "I need to schedule a meeting and plan writing my progress report."

results = agent.process(test_request)

print("\n--- FINAL OUTPUT ---\n")
for line in results:
    print(line)

print("\n--- MEMORY STORAGE ---")
print(agent.memory.get())


