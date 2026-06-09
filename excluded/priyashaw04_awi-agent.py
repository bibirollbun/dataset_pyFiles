# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Code Cell: Agent JSON Definitions
import json

# Task Understanding Agent
task_understanding_agent = json.dumps({
    "name": "TaskUnderstandingAgent",
    "role": "Extracts intent, priority, deadlines, and entities from unstructured tasks.",
    "tools": ["EmailParserTool", "EntityExtractorTool"],
    "prompt": "Analyze the incoming task text. Extract: intent (e.g., 'repair', 'process invoice'), priority (high/medium/low based on keywords like 'urgent'), deadlines (e.g., 'by tomorrow'), and entities (e.g., names, departments). Output in JSON format.",
    "behavior": {
        "input": "task_text (string)",
        "output": "structured_task (JSON: {intent, priority, deadline, entities})",
        "model": "gpt-4",
        "next_agent": "BusinessRulesAgent"
    }
})

# Business Rules & Compliance Agent
business_rules_agent = json.dumps({
    "name": "BusinessRulesAgent",
    "role": "Checks against Standard Operating Procedures (SOPs), validates requirements, and flags compliance risks.",
    "tools": ["RuleCheckerTool", "ComplianceFlagTool"],
    "prompt": "Given the structured task, check against internal rules (e.g., HR tasks require approval for interns, Finance tasks must have invoice numbers). Flag risks like data privacy violations. Output validation status and flags.",
    "behavior": {
        "input": "structured_task (JSON)",
        "output": "validated_task (JSON: {task, valid: true/false, flags: [list]})",
        "rules": ["SOP_HR_Onboarding", "SOP_Finance_Invoices", "Compliance_GDPR"],
        "next_agent": "RoutingDecisionAgent"
    }
})

# Routing & Decision Agent
routing_decision_agent = json.dumps({
    "name": "RoutingDecisionAgent",
    "role": "Assigns tasks to teams/tools using decision logic, incorporating memory for improvements.",
    "tools": ["RoutingLogicTool", "MemoryQueryTool"],
    "prompt": "Based on validated task, query memory for similar past routings. Apply logic: e.g., 'repair' -> IT, 'invoice' -> Finance. Route autonomously to team/tool.",
    "behavior": {
        "input": "validated_task (JSON)",
        "output": "routed_task (JSON: {task, route: {team, tool, action}})",
        "logic": "if intent=='repair' and priority=='high' -> route to IT; else query memory for patterns",
        "next_agent": "EvaluationComponent"
    }
})

# Evaluation Component
evaluation_agent = json.dumps({
    "name": "EvaluationAgent",
    "role": "Tracks routing correctness, accuracy, violations, and latency.",
    "tools": ["MetricsTrackerTool"],
    "prompt": "Compare routed task against ground truth (if available) or rules. Calculate accuracy %, log violations, measure latency.",
    "behavior": {
        "input": "routed_task (JSON)",
        "output": "metrics (JSON: {accuracy, violations, latency_ms})",
        "ground_truth": "predefined for demos",
        "logs_to": "memory_store"
    }
})

print("Agent JSONs defined. Example:", json.loads(task_understanding_agent)["name"])


# Code Cell: Install Dependencies and Define Tools
!pip install openai pandas matplotlib --quiet  # For AI, data, and plots

import re
import time
import os
from openai import OpenAI  # For AI; use mock if no key

# Mock OpenAI if no key
try:
    api_key = os.getenv('OPENAI_API_KEY')  # Add via Kaggle Secrets
    client = OpenAI(api_key=api_key) if api_key else None
except:
    client = None

class EmailParserTool:
    def parse_email(self, email_text: str) -> dict:
        entities = re.findall(r'\b[A-Z][a-z]+\b', email_text)
        if client:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": f"Extract intent, priority, deadline from: {email_text}"}]
            )
            extracted = json.loads(response.choices[0].message.content)
        else:
            # Mock extraction
            extracted = {"intent": "repair" if "laptop" in email_text.lower() else "invoice" if "invoice" in email_text.lower() else "onboarding" if "onboarding" in email_text.lower() else "update_plan",
                         "priority": "high" if "urgent" in email_text.lower() else "medium",
                         "deadline": "tomorrow" if "tomorrow" in email_text.lower() else None}
        return {"intent": extracted["intent"], "priority": extracted["priority"], "deadline": extracted["deadline"], "entities": entities}

class RuleCheckerTool:
    def check_rules(self, structured_task: dict) -> dict:
        flags = []
        valid = True
        if structured_task["intent"] == "onboarding" and "intern" in structured_task["entities"]:
            flags.append("HR Approval Required")
            valid = False
        if structured_task["intent"] == "invoice" and not any("INV" in str(e) for e in structured_task.get("entities", [])):
            flags.append("Missing Invoice ID")
            valid = False
        return {"valid": valid, "flags": flags}

class RoutingLogicTool:
    def route_task(self, validated_task: dict, memory: dict) -> dict:
        intent = validated_task["task"]["intent"]
        if intent in memory:
            route = memory[intent]
        else:
            routes = {"repair": "IT", "invoice": "Finance", "onboarding": "HR", "update_plan": "CRM"}
            route = routes.get(intent, "General")
        return {"task": validated_task["task"], "route": {"team": route, "tool": f"{route}_Tool", "action": "assign"}}

class MemoryStoreTool:
    def __init__(self):
        self.store = {}
    def store_decision(self, intent: str, route: str):
        self.store[intent] = route
    def query(self, intent: str) -> str:
        return self.store.get(intent, None)

class MetricsTrackerTool:
    def track(self, routed_task: dict, ground_truth: str, start_time: float) -> dict:
        latency = time.time() - start_time
        accuracy = 1.0 if routed_task["route"]["team"] == ground_truth else 0.0
        violations = len(routed_task.get("flags", []))
        return {"accuracy": accuracy, "violations": violations, "latency_ms": latency * 1000}

print("Tools defined successfully.")


# Code Cell: Routing Decision Logic
def route_decision(validated_task, memory_tool):
    task = validated_task["task"]
    intent = task["intent"]
    priority = task["priority"]
    
    past_route = memory_tool.query(intent)
    if past_route:
        route = past_route
    else:
        if intent == "repair" and priority == "high":
            route = "IT"
        elif intent == "invoice":
            route = "Finance"
        elif intent == "onboarding":
            route = "HR"
        elif intent == "update_plan":
            route = "CRM"
        else:
            route = "General"
    
    memory_tool.store_decision(intent, route)
    return {"task": task, "route": {"team": route, "tool": f"{route}_Tool", "action": "assign"}}

print("Routing logic defined.") 


# Code Cell: Sample Demos (Fully working, self-contained)

import time
import json
from typing import Dict, Any, List

# ----------------------------
# Tool Implementations (Mock)
# ----------------------------

class MemoryStoreTool:
    """
    Simple in-memory store for routing preferences or historical mappings.
    Provides dict-like helpers without requiring direct iteration on the object.
    """
    def __init__(self):
        # Example: learned or pre-configured intent -> department mappings
        self._store: Dict[str, str] = {
            "repair": "IT",
            "payment": "Finance",
            "onboarding": "HR",
            "plan_update": "CRM",
        }

    def get(self, key: str, default=None):
        return self._store.get(key, default)

    def set(self, key: str, value: Any):
        self._store[key] = value

    def has(self, key: str) -> bool:
        return key in self._store

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._store)


class EmailParserTool:
    """
    Naive parser to extract intent, priority, deadline, entities from a text task.
    This is illustrative; adapt rules for production usage.
    """
    def parse_email(self, text: str) -> Dict[str, Any]:
        lower = text.lower()

        # Intent detection (simple heuristics)
        intent = None
        if any(k in lower for k in ["not turning on", "repair", "broken", "fix"]):
            intent = "repair"
        elif any(k in lower for k in ["payment", "invoice", "process payment", "inv"]):
            intent = "payment"
        elif any(k in lower for k in ["onboarding", "new intern", "hire", "start onboarding"]):
            intent = "onboarding"
        elif any(k in lower for k in ["update my service plan", "add premium support", "plan", "upgrade"]):
            intent = "plan_update"
        else:
            intent = "general"

        # Priority detection
        priority = "normal"
        if any(k in lower for k in ["urgent", "asap", "immediately", "high priority"]):
            priority = "high"

        # Deadline detection (naive: look for 'tomorrow', 'today', etc.)
        deadline = None
        if "tomorrow" in lower:
            deadline = "tomorrow"
        elif "today" in lower:
            deadline = "today"

        # Entities extraction (very naive)
        entities: List[str] = []
        if "laptop" in lower:
            entities.append("Laptop")
        if "invoice" in lower or "inv" in lower:
            # Try to capture a code like #INV334
            # Simple token scan:
            tokens = text.replace(",", " ").split()
            inv_codes = [t for t in tokens if t.upper().startswith("#INV")]
            entities.extend(inv_codes if inv_codes else ["Invoice"])
        if "intern" in lower:
            entities.append("Intern")
        if "service plan" in lower or "plan" in lower:
            entities.append("Service Plan")
        if "premium support" in lower:
            entities.append("Premium Support")

        return {
            "intent": intent,
            "priority": priority,
            "deadline": deadline,
            "entities": entities,
        }


class RuleCheckerTool:
    """
    Validates structured task object against simple rules.
    """
    def check_rules(self, structured: Dict[str, Any]) -> Dict[str, Any]:
        flags: List[str] = []
        valid = True

        # Example rules
        if structured.get("intent") == "general":
            flags.append("unclear_intent")
        if structured.get("priority") not in {"low", "normal", "high"}:
            flags.append("invalid_priority")

        if flags:
            valid = False if "invalid_priority" in flags else True

        return {"valid": valid, "flags": flags}


class RoutingLogicTool:
    """
    Routes tasks to departments based on intent, priority, and available memory mappings.
    """
    def route_task(self, validated_task: Dict[str, Any], memory: MemoryStoreTool) -> Dict[str, Any]:
        # Expecting validated_task to have the original structured task under 'task'
        if "task" not in validated_task or not isinstance(validated_task["task"], dict):
            raise ValueError("validated_task must include a dict under key 'task'")

        structured = validated_task["task"]
        intent = structured.get("intent", "general")
        priority = structured.get("priority", "normal")

        # Use memory's helpers instead of treating it as an iterable/dict
        route = memory.get(intent, None)

        # Fallback routing when memory doesn't have a mapping
        if route is None:
            route = "Operations" if intent == "general" else "Operations"

        # Priority-based escalation example
        if priority == "high" and route not in {"IT", "Finance", "HR", "CRM"}:
            route = "Escalations"

        return {
            "department": route,
            "intent": intent,
            "priority": priority,
            "entities": structured.get("entities", []),
            "deadline": structured.get("deadline"),
            "flags": validated_task.get("flags", []),
            "valid": validated_task.get("valid", True),
        }


class MetricsTrackerTool:
    """
    Tracks routing accuracy and latency against ground truth.
    """
    def track(self, routed: Dict[str, Any], ground_truth: str, start_time: float) -> Dict[str, Any]:
        latency_ms = (time.time() - start_time) * 1000.0
        predicted = routed.get("department")
        correct = (predicted == ground_truth)
        return {
            "latency_ms": round(latency_ms, 2),
            "predicted": predicted,
            "ground_truth": ground_truth,
            "correct": correct,
        }


# ----------------------------
# Demo Execution
# ----------------------------

memory = MemoryStoreTool()
parser = EmailParserTool()
checker = RuleCheckerTool()
router = RoutingLogicTool()
tracker = MetricsTrackerTool()

test_cases = [
    {"task": "Laptop not turning on; I need urgent help.", "ground_truth": "IT"},
    {"task": "Please process payment #INV334.", "ground_truth": "Finance"},
    {"task": "Start onboarding for a new intern tomorrow.", "ground_truth": "HR"},
    {"task": "Please update my service plan and add premium support.", "ground_truth": "CRM"},
]

for i, case in enumerate(test_cases):
    print(f"\nDemo {i+1}: {case['task']}")
    start = time.time()

    structured = parser.parse_email(case["task"])
    print(f"  Structured: {structured}")

    validated = checker.check_rules(structured)
    validated["task"] = structured  # Attach the structured task for the router
    print(f"  Validated: {validated}")

    routed = router.route_task(validated, memory)
    print(f"  Routed: {routed}")

    metrics = tracker.track(routed, case["ground_truth"], start)
    print(f"  Metrics: {metrics}")



# Code Cell: Evaluation
import pandas as pd
import matplotlib.pyplot as plt

# Collect metrics from demos
results = []
for case in test_cases:
    start = time.time()
    structured = parser.parse_email(case["task"])
    validated = checker.check_rules(structured)
    validated["task"] = structured
    routed = router.route_task(validated, memory)
    metrics = tracker.track(routed, case["ground_truth"], start)
    results.append(metrics)

df = pd.DataFrame(results)

# Add derived columns
df["accuracy"] = df["correct"].astype(int)        # 1 if correct, 0 if not
df["violations"] = (~df["correct"]).astype(int)   # 1 if incorrect, 0 if correct

print("Evaluation Results:")
print(df)
print(f"Average Accuracy: {df['accuracy'].mean():.2%}")
print(f"Total Violations: {df['violations'].sum()}")
print(f"Average Latency: {df['latency_ms'].mean():.2f} ms")

# Plot accuracy over tests
df.plot(y="accuracy", kind="line", title="Routing Accuracy Over Tests")
plt.ylim(0, 1.1)  # keep accuracy scale between 0 and 1
plt.show()

# Save results
df.to_csv("/kaggle/working/evaluation_results.csv", index=False)
print("Results saved to /kaggle/working/evaluation_results.csv")


