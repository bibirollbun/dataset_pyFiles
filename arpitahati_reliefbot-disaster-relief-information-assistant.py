import os

base_dir = "project"
subdirs = [
    "agents",
    "tools",
    "memory",
    "core",
]

os.makedirs(base_dir, exist_ok=True)
for sd in subdirs:
    os.makedirs(os.path.join(base_dir, sd), exist_ok=True)

# Make packages importable
for path in [base_dir] + [os.path.join(base_dir, sd) for sd in subdirs]:
    init_path = os.path.join(path, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            f.write("# Package init\n")

print("Project structure created.")



%%writefile project/core/observability.py
import logging
from typing import Any, Dict


def get_logger(name: str = "disaster_agent") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def log_event(
    logger: logging.Logger,
    component: str,
    message: str,
    extra: Dict[str, Any] | None = None,
) -> None:
    if extra is None:
        extra = {}
    logger.info(f"{component}: {message} | extra={extra}")



%%writefile project/core/a2a_protocol.py
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Message:
    sender: str
    receiver: str
    kind: str
    content: Dict[str, Any]
    trace: List[str] = field(default_factory=list)

    def add_trace(self, note: str) -> None:
        self.trace.append(note)


def make_message(sender: str, receiver: str, kind: str, content: Dict[str, Any]) -> Message:
    return Message(sender=sender, receiver=receiver, kind=kind, content=content)



%%writefile project/core/context_engineering.py
from typing import Any, Dict, List


def build_planner_context(
    user_input: str,
    conversation_history: List[Dict[str, Any]],
    session_memory: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "user_input": user_input,
        "recent_messages": conversation_history[-5:],
        "session_memory": session_memory,
    }



%%writefile project/memory/session_memory.py
from typing import Any, Dict, List


class SessionMemory:
    """
    Simple in-memory session memory.
    Lives for as long as the Python kernel is running.
    """

    def __init__(self) -> None:
        self.conversation_history: List[Dict[str, Any]] = []
        self.state: Dict[str, Any] = {
            "preferred_location": None,
            "last_safe_spots": [],
        }

    def add_message(self, role: str, content: str) -> None:
        self.conversation_history.append({"role": role, "content": content})

    def set_preferred_location(self, location: str) -> None:
        self.state["preferred_location"] = location

    def get_preferred_location(self) -> str | None:
        return self.state.get("preferred_location")

    def set_safe_spots(self, spots: List[str]) -> None:
        self.state["last_safe_spots"] = spots

    def get_safe_spots(self) -> List[str]:
        return self.state.get("last_safe_spots", [])

    def snapshot(self) -> Dict[str, Any]:
        return {
            "conversation_history": list(self.conversation_history),
            "state": dict(self.state),
        }



%%writefile project/tools/tools.py
from typing import Any, Dict, List, Tuple


class DisasterTools:
    """
    Domain tools for disaster relief search, summarization, and simple utilities.
    In a real app this would query live APIs; here we use a small in-notebook dataset.
    """

    # Simple static dataset of relief resources
    DATASET: List[Dict[str, Any]] = [
        {
            "city": "kolkata",
            "disaster": "flood",
            "type": "shelter",
            "name": "Kolkata Municipal Flood Shelter - Salt Lake",
            "address": "Block A, Salt Lake Stadium Complex",
            "contact": "+91-33-1234-5678",
        },
        {
            "city": "kolkata",
            "disaster": "flood",
            "type": "medical",
            "name": "Flood Relief Medical Camp - Howrah",
            "address": "GT Road, near Howrah Maidan",
            "contact": "+91-33-2345-6789",
        },
        {
            "city": "mumbai",
            "disaster": "cyclone",
            "type": "shelter",
            "name": "Cyclone Safe Shelter - Dadar School",
            "address": "Dadar East Municipal School",
            "contact": "+91-22-1111-2222",
        },
        {
            "city": "mumbai",
            "disaster": "cyclone",
            "type": "food",
            "name": "Community Kitchen - Bandra",
            "address": "Near Bandra Station, West exit",
            "contact": "+91-22-3333-4444",
        },
        {
            "city": "bhubaneswar",
            "disaster": "cyclone",
            "type": "shelter",
            "name": "State Cyclone Relief Centre",
            "address": "Unit-9 High School Ground",
            "contact": "+91-674-555-6666",
        },
    ]

    @staticmethod
    def normalize(text: str) -> str:
        return text.strip().lower()

    @staticmethod
    def extract_location(user_input: str, default: str | None = None) -> str | None:
        text = user_input.lower()
        known_cities = ["kolkata", "mumbai", "bhubaneswar", "delhi", "chennai"]
        for city in known_cities:
            if city in text:
                return city
        return default

    @staticmethod
    def detect_need_type(user_input: str) -> str:
        text = user_input.lower()
        if any(k in text for k in ["shelter", "stay", "camp"]):
            return "shelter"
        if any(k in text for k in ["food", "water", "ration"]):
            return "food"
        if any(k in text for k in ["doctor", "medical", "hospital", "medicine"]):
            return "medical"
        return "general"

    @staticmethod
    def detect_disaster_type(user_input: str) -> str:
        text = user_input.lower()
        if "flood" in text:
            return "flood"
        if "cyclone" in text or "storm" in text:
            return "cyclone"
        if "earthquake" in text:
            return "earthquake"
        return "unknown"

    @classmethod
    def lookup_resources(
        cls,
        location: str | None,
        need_type: str,
        disaster: str,
    ) -> List[Dict[str, Any]]:
        if location is None:
            return []

        location = cls.normalize(location)
        disaster = cls.normalize(disaster)

        matches: List[Dict[str, Any]] = []
        for row in cls.DATASET:
            if row["city"] != location:
                continue
            if disaster != "unknown" and row["disaster"] != disaster:
                continue
            if need_type != "general" and row["type"] != need_type:
                continue
            matches.append(row)

        # If no strict match and need_type was not general, relax filtering
        if not matches and need_type != "general":
            for row in cls.DATASET:
                if row["city"] == location:
                    matches.append(row)
        return matches

    @staticmethod
    def summarize_resources(
        location: str | None,
        need_type: str,
        disaster: str,
        resources: List[Dict[str, Any]],
    ) -> Tuple[str, List[str]]:
        if not resources:
            base = "I could not find verified relief centres for your query."
            if not location:
                base += " Please mention your city (for example: Kolkata, Mumbai, Bhubaneswar)."
            else:
                base += f" I don't have stored data for {location.title()} in this demo."
            return base, []

        lines = []
        safe_spots = []
        for idx, r in enumerate(resources, start=1):
            line = (
                f"{idx}. {r['name']} — {r['type'].title()} centre\n"
                f"   Address: {r['address']}\n"
                f"   Contact: {r['contact']}"
            )
            lines.append(line)
            safe_spots.append(r["name"])

        intro = f"Here are verified relief options near you"
        if location:
            intro += f" in {location.title()}"
        intro += ".\n"

        details = "\n".join(lines)
        meta = f"(Need: {need_type}, Disaster: {disaster})"
        return intro + details + "\n" + meta, safe_spots



%%writefile project/agents/planner.py
from typing import Any, Dict

from project.core.context_engineering import build_planner_context
from project.tools.tools import DisasterTools


class Planner:
    def __init__(self, logger) -> None:
        self.logger = logger

    def plan(
        self,
        user_input: str,
        conversation_history: list[Dict[str, Any]],
        session_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        ctx = build_planner_context(user_input, conversation_history, session_state)

        preferred_location = session_state.get("preferred_location")
        location = DisasterTools.extract_location(user_input, default=preferred_location)
        need_type = DisasterTools.detect_need_type(user_input)
        disaster_type = DisasterTools.detect_disaster_type(user_input)

        task = {
            "location": location,
            "need_type": need_type,
            "disaster_type": disaster_type,
            "raw_user_input": user_input,
        }

        self.logger.info(f"Planner created task: {task}")
        return task



%%writefile project/agents/worker.py
from typing import Any, Dict, List

from project.tools.tools import DisasterTools


class Worker:
    def __init__(self, logger) -> None:
        self.logger = logger

    def execute(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        location = task.get("location")
        need_type = task.get("need_type", "general")
        disaster_type = task.get("disaster_type", "unknown")

        self.logger.info(
            f"Worker fetching resources for location={location}, "
            f"need={need_type}, disaster={disaster_type}"
        )
        resources = DisasterTools.lookup_resources(location, need_type, disaster_type)
        self.logger.info(f"Worker found {len(resources)} resources")
        return resources



%%writefile project/agents/evaluator.py
from typing import Any, Dict, List, Tuple

from project.tools.tools import DisasterTools


class Evaluator:
    def __init__(self, logger) -> None:
        self.logger = logger

    def evaluate(
        self,
        task: Dict[str, Any],
        resources: List[Dict[str, Any]],
    ) -> Tuple[str, Dict[str, Any]]:
        location = task.get("location")
        need_type = task.get("need_type", "general")
        disaster_type = task.get("disaster_type", "unknown")

        summary, safe_spots = DisasterTools.summarize_resources(
            location=location,
            need_type=need_type,
            disaster=disaster_type,
            resources=resources,
        )

        meta = {
            "location": location,
            "need_type": need_type,
            "disaster_type": disaster_type,
            "num_results": len(resources),
            "safe_spots": safe_spots,
        }

        self.logger.info(f"Evaluator produced response meta={meta}")
        return summary, meta



%%writefile project/main_agent.py
from typing import Any, Dict

from project.agents.planner import Planner
from project.agents.worker import Worker
from project.agents.evaluator import Evaluator
from project.core.observability import get_logger, log_event
from project.memory.session_memory import SessionMemory


class MainAgent:
    def __init__(self) -> None:
        self.logger = get_logger("DisasterReliefAgent")
        self.memory = SessionMemory()
        self.planner = Planner(self.logger)
        self.worker = Worker(self.logger)
        self.evaluator = Evaluator(self.logger)

    def handle_message(self, user_input: str) -> Dict[str, Any]:
        log_event(self.logger, "MainAgent", "Received user message", {"text": user_input})

        # Store user message
        self.memory.add_message("user", user_input)

        # Build and run plan
        snapshot = self.memory.snapshot()
        task = self.planner.plan(
            user_input=user_input,
            conversation_history=snapshot["conversation_history"],
            session_state=snapshot["state"],
        )

        resources = self.worker.execute(task)
        response_text, meta = self.evaluator.evaluate(task, resources)

        # Update memory with location and safe spots
        if task.get("location"):
            self.memory.set_preferred_location(task["location"])
        if meta.get("safe_spots"):
            self.memory.set_safe_spots(meta["safe_spots"])

        self.memory.add_message("assistant", response_text)

        result = {
            "response": response_text,
            "task": task,
            "meta": meta,
        }
        log_event(self.logger, "MainAgent", "Prepared response", {"meta": meta})
        return result


def run_agent(user_input: str):
    agent = MainAgent()
    result = agent.handle_message(user_input)
    return result["response"]



%%writefile project/app.py
from project.main_agent import run_agent


def chat_once():
    user_input = input("Describe your situation (type 'quit' to exit): ")
    if user_input.lower().strip() == "quit":
        return False
    response = run_agent(user_input)
    print("\n--- Assistant Response ---")
    print(response)
    print("--------------------------\n")
    return True


if __name__ == "__main__":
    print("Disaster Relief Information Assistant (demo)")
    while True:
        if not chat_once():
            break



%%writefile project/run_demo.py
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from project.main_agent import run_agent  # noqa: E402


if __name__ == "__main__":
    print(run_agent("Hello! This is a demo from Kolkata flood shelter."))



%%writefile project/requirements.txt
# All used libraries are from Python standard library.
# This file is here for completeness if you later deploy.



from project.main_agent import run_agent

print("Test 1:")
print(
    run_agent(
        "I am in Kolkata and there is a big flood. "
        "Please help me find a safe shelter to stay with my family."
    )
)

print("\nTest 2:")
print(
    run_agent(
        "Cyclone warning in Mumbai, where can I get food or relief centre details?"
    )
)

print("\nTest 3 (no location):")
print(run_agent("There is heavy flood, we need medical help urgently!"))


