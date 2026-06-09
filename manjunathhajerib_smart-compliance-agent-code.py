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


#install pdf reader from python 
!pip install --quiet PyPDF2
print("PyPDF2 installed successfully!")



# ==========================================
# Session Service 
# ==========================================

import json
from typing import Any, Dict, List

class SimpleSessionService:
    """
    Lightweight in-memory session manager for demo.
    Keeps:
      - history: list of events (strings)
      - preferences: arbitrary key/value settings
    """

    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(self, user_id: str = "default_user") -> Dict[str, Any]:
        """Create a session if missing and return it."""
        if user_id not in self.sessions:
            self.sessions[user_id] = {"history": [], "preferences": {}}
        return self.sessions[user_id]

    def append_event(self, user_id: str = "default_user", event: str = "") -> None:
        """Append an event (string) to the session history."""
        session = self.create_session(user_id)
        session["history"].append(event)

    def set_preference(self, user_id: str = "default_user", key: str = "", value: Any = None) -> None:
        """Set a preference key/value for the session."""
        session = self.create_session(user_id)
        session["preferences"][key] = value

    def get_preferences(self, user_id: str = "default_user") -> Dict[str, Any]:
        """Return preferences dict for a user session."""
        session = self.create_session(user_id)
        return session["preferences"]

    def get_history(self, user_id: str = "default_user") -> List[str]:
        """Return history list for a user session."""
        session = self.create_session(user_id)
        return session["history"]

# Create a global session_service instance for notebook use
session_service = SimpleSessionService()
print("Session service ready!")



#Testing session sevice
session_service.append_event("test_user", "Uploaded document")
session_service.set_preference("test_user", "compliance_mode", "strict")
print(session_service.get_history("test_user"))
print(session_service.get_preferences("test_user"))



# ==========================================
# Memory Service (Long-term)
# ==========================================

from typing import Any, Dict

class SimpleMemoryService:
    """
    Long-term memory store.
    Useful for saving:
      - preferred compliance level
      - custom rule profile
      - user-specific settings
    """

    def __init__(self):
        self.memory: Dict[str, Dict[str, Any]] = {}

    def save(self, user_id: str = "default_user", key: str = "", value: Any = None) -> None:
        """Save a key/value pair for a user."""
        if user_id not in self.memory:
            self.memory[user_id] = {}
        self.memory[user_id][key] = value

    def load(self, user_id: str = "default_user", key: str = "") -> Any:
        """Load a single stored value."""
        return self.memory.get(user_id, {}).get(key, None)

    def get_all(self, user_id: str = "default_user") -> Dict[str, Any]:
        """Return all long-term memory for a user."""
        return self.memory.get(user_id, {})

# Global memory instance
memory_service = SimpleMemoryService()

print("Memory service ready!")



#Testing service
memory_service.save("user1", "preferred_ruleset", "GDPR-Strict")
memory_service.save("user1", "language", "English")

print(memory_service.get_all("user1"))



# ==========================================
# STEP 4 — Document Reader Agent
# ==========================================

from PyPDF2 import PdfReader
from io import BytesIO

class DocumentReaderAgent:
    """
    Extracts text from:
      - PDF bytes
      - Plain text
    """

    def extract_text_from_pdf(self, file_bytes: bytes) -> str:
        try:
            reader = PdfReader(BytesIO(file_bytes))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)
        except Exception as e:
            return f"[ERROR] PDF extraction failed: {e}"

    def run(self, document_input):
        """
        Accepts:
          - file bytes (PDF)
          - plain text string
        Returns:
          - dict { 'text': extracted_text }
        """
        if isinstance(document_input, bytes):
            text = self.extract_text_from_pdf(document_input)
            return {"text": text}

        elif isinstance(document_input, str):
            return {"text": document_input}

        else:
            return {"text": "", "error": "Unsupported input type."}


# Global instance
document_reader = DocumentReaderAgent()
print("DocumentReaderAgent ready!")



document_reader.run("This is a test policy document.")



# ==========================================
# STEP 5 — Policy Checker Agent
# ==========================================

import re

class PolicyCheckerAgent:
    """
    Runs rule-based compliance checks on extracted document text.
    Each rule includes:
      - id
      - description
      - keywords
      - required flag
    """

    def __init__(self):
        # Compliance rules (expandable)
        self.RULES = [
            {
                "id": "data_retention",
                "description": "Document should mention data retention policy",
                "keywords": ["data retention", "retention period", "retain data"],
                "required": True
            },
            {
                "id": "gdpr",
                "description": "Should reference GDPR or data protection rights",
                "keywords": ["gdpr", "data subject", "right to access", "right to delete"],
                "required": False   # optional but good to have
            },
            {
                "id": "security_controls",
                "description": "Should mention security controls",
                "keywords": ["security", "encryption", "access control", "security policy"],
                "required": True
            },
            {
                "id": "contact_person",
                "description": "Should mention a contact or DPO",
                "keywords": ["DPO", "data protection officer", "contact", "responsible person"],
                "required": True
            }
        ]

    def check_rule(self, rule, text: str):
        """Check if rule keywords appear in the document text."""
        matches = 0
        for k in rule["keywords"]:
            if re.search(r"\b" + re.escape(k) + r"\b", text, re.IGNORECASE):
                matches += 1

        passed = matches > 0 or not rule["required"]

        return {
            "rule_id": rule["id"],
            "description": rule["description"],
            "required": rule["required"],
            "matches": matches,
            "passed": passed
        }

    def run(self, text: str):
        """Run all rules and return structured results."""
        results = []
        for rule in self.RULES:
            results.append(self.check_rule(rule, text))
        return {"checks": results}


# Global instance
policy_checker = PolicyCheckerAgent()
print("PolicyCheckerAgent ready!")



sample_text = """
This policy provides data retention guidelines.
We follow GDPR requirements.
Security controls include encryption.
Contact our DPO for data-related queries.
"""

policy_checker.run(sample_text)



# ==========================================
# STEP 6 — Risk Scoring Agent
# ==========================================

class RiskScoringAgent:
    """
    Computes a final compliance score based on rule check results.
    Score = (passed_rules / total_rules) * 100
    """

    def compute_score(self, check_results):
        total = len(check_results)
        passed = sum(1 for r in check_results if r["passed"])
        failed = total - passed
        
        score = int((passed / total) * 100)   # integer 0–100

        return {
            "total_rules": total,
            "passed_rules": passed,
            "failed_rules": failed,
            "compliance_score": score
        }

    def run(self, checks_dict):
        return self.compute_score(checks_dict["checks"])


# Global instance
risk_scorer = RiskScoringAgent()
print("RiskScoringAgent ready!")



checks = policy_checker.run("""
Data retention rules exist.
GDPR rights are included.
Security controls apply.
Contact the DPO for more info.
""")

risk_scorer.run(checks)



# ==========================================
# STEP 7 — Manager Agent (Pipeline Orchestrator)
# ==========================================

class ManagerAgent:
    """
    Orchestrates:
      document → reader → checker → scorer → final_report
    """

    def __init__(self, reader_agent, checker_agent, scorer_agent):
        self.reader = reader_agent
        self.checker = checker_agent
        self.scorer = scorer_agent

    def run(self, document_input, user_id="default_user"):
        # SESSION LOGGING
        session_service.append_event(user_id, "ManagerAgent: Started processing document")

        # 1. Extract document text
        reader_output = self.reader.run(document_input)
        text = reader_output.get("text", "")
        session_service.append_event(user_id, f"DocumentReader: Extracted {len(text)} characters")

        # 2. Perform rule checks
        check_output = self.checker.run(text)
        session_service.append_event(user_id, f"PolicyChecker: Checked {len(check_output['checks'])} rules")

        # 3. Compute final score
        score_output = self.scorer.run(check_output)
        session_service.append_event(user_id, f"RiskScorer: Score = {score_output['compliance_score']}")

        # 4. Return combined report
        final_report = {
            "text_preview": text[:500],
            "rule_checks": check_output,
            "risk_score": score_output
        }

        session_service.append_event(user_id, "ManagerAgent: Completed analysis")

        return final_report


# Global orchestrator instance
manager_agent = ManagerAgent(document_reader, policy_checker, risk_scorer)

print("ManagerAgent ready!")



test_text = """
Data retention rules apply.
GDPR rights are protected.
Security includes encryption and access control.
Contact the Data Protection Officer for info.
"""

result = manager_agent.run(test_text, "demo_user")
result



# ==========================================
# STEP 8 — Observability Layer
# ==========================================

import time

class Observability:
    """
    Simple observability layer:
      - logs (printed + stored)
      - traces (list of events)
      - metrics (documents processed, avg time)
    """

    def __init__(self):
        self.logs = []
        self.traces = []
        self.metrics = {
            "documents_processed": 0,
            "average_processing_time": 0.0,
        }

    def log(self, message):
        """Store timestamped log + print it."""
        entry = {
            "timestamp": time.time(),
            "message": message
        }
        self.logs.append(entry)
        print("[LOG]", message)

    def start_trace(self):
        """Start a new trace record."""
        trace = {"start": time.time(), "events": []}
        self.traces.append(trace)
        return trace

    def add_trace_event(self, trace, event):
        """Add event to trace."""
        trace["events"].append({
            "time": time.time(),
            "event": event
        })

    def end_trace(self, trace):
        """Close trace and compute duration + update metrics."""
        trace["end"] = time.time()
        trace["duration_seconds"] = trace["end"] - trace["start"]

        # update metrics
        self.metrics["documents_processed"] += 1
        count = self.metrics["documents_processed"]

        prev_avg = self.metrics["average_processing_time"]
        new_avg = ((prev_avg * (count - 1)) + trace["duration_seconds"]) / count
        self.metrics["average_processing_time"] = new_avg

        self.log(f"Trace completed in {trace['duration_seconds']:.4f}s")

    def get_logs(self):
        return self.logs

    def get_traces(self):
        return self.traces

    def get_metrics(self):
        return self.metrics


# Global observability instance
observability = Observability()

print("Observability system ready!")



trace = observability.start_trace()
observability.add_trace_event(trace, "test_event")
observability.end_trace(trace)

observability.get_traces()



# ==========================================
# STEP 9 — Manager Agent with Observability
# ==========================================

class ManagerAgent:
    """
    Full orchestrator with observability integration.
    Steps:
      - Start trace
      - Log events
      - Extract text → run checks → score
      - End trace
      - Return final audit report
    """

    def __init__(self, reader_agent, checker_agent, scorer_agent):
        self.reader = reader_agent
        self.checker = checker_agent
        self.scorer = scorer_agent

    def run(self, document_input, user_id="default_user"):
        
        # 1. Start trace
        trace = observability.start_trace()
        observability.log("ManagerAgent: Started compliance audit")
        observability.add_trace_event(trace, "audit_start")

        # 2. Extract text
        observability.log("DocumentReaderAgent: extracting text")
        reader_output = self.reader.run(document_input)
        text = reader_output.get("text", "")
        observability.add_trace_event(trace, f"text_extracted ({len(text)} chars)")

        # 3. Run rule checks
        observability.log("PolicyCheckerAgent: checking rules")
        check_output = self.checker.run(text)
        observability.add_trace_event(trace, f"rules_checked ({len(check_output['checks'])} rules)")

        # 4. Score the document
        observability.log("RiskScoringAgent: scoring document")
        score_output = self.scorer.run(check_output)
        observability.add_trace_event(trace, f"score_computed ({score_output['compliance_score']})")

        # 5. Build report
        final_report = {
            "trace_id": len(observability.traces),
            "text_preview": text[:500],
            "rule_checks": check_output,
            "risk_score": score_output
        }

        observability.log("ManagerAgent: audit complete")
        observability.add_trace_event(trace, "audit_end")

        # 6. End trace
        observability.end_trace(trace)

        return final_report


# Re-create manager instance with observability
manager_agent = ManagerAgent(document_reader, policy_checker, risk_scorer)

print("ManagerAgent updated with Observability!")



test_text = """
Data retention rules apply.
GDPR rights exist.
Security includes encryption.
Contact our DPO for details.
"""

result = manager_agent.run(test_text, "obs_test")
result



# ==================================================
# STEP 10 — Evaluation Suite (Scenarios + Results)
# ==================================================

import pandas as pd
import time

# -----------------------------------------------
# 1. Evaluation Scenarios
# -----------------------------------------------
evaluation_scenarios = [
    {
        "name": "Fully Compliant Document",
        "input_text": """
        Our data retention policy defines how long we retain data.
        We follow GDPR and support rights such as access and deletion.
        Security controls include encryption and access control.
        Contact our Data Protection Officer for any questions.
        """,
        "expected_min_score": 90
    },
    {
        "name": "Missing GDPR Reference",
        "input_text": """
        This policy includes data retention rules.
        Security controls include encryption.
        Contact the responsible person for information.
        """,
        "expected_min_score": 60
    },
    {
        "name": "Weak Security Mentions",
        "input_text": """
        Data retention rules apply.
        GDPR rights are guaranteed.
        Contact information is available.
        """,
        "expected_min_score": 50
    },
    {
        "name": "Poorly Written Document",
        "input_text": """
        This document is missing many required compliance pieces.
        No security, no retention, no GDPR, no DPO information.
        """,
        "expected_min_score": 10
    }
]


# -----------------------------------------------
# 2. Evaluation Runner
# -----------------------------------------------
def run_evaluation():
    results = []

    for scenario in evaluation_scenarios:
        start = time.time()

        # Run agent
        output = manager_agent.run(scenario["input_text"], user_id=scenario["name"])
        score = output["risk_score"]["compliance_score"]

        duration = round(time.time() - start, 3)

        success = score >= scenario["expected_min_score"]

        results.append({
            "Scenario": scenario["name"],
            "Compliance Score": score,
            "Expected Min Score": scenario["expected_min_score"],
            "Passed": success,
            "Processing Time (s)": duration
        })

    return pd.DataFrame(results)


# -----------------------------------------------
# 3. Run Evaluation + Show Table
# -----------------------------------------------
evaluation_results = run_evaluation()
evaluation_results



evaluation_results.to_csv("evaluation_results.csv", index=False)



# ==================================================
# STEP 11 — Final Demo (Text Version)
# ==================================================

demo_text = """
Our organization follows strict data retention and deletion guidelines.
GDPR rights such as right to access and right to delete are respected.
Security measures include encryption and access control across all systems.
Please contact our Data Protection Officer (DPO) for any data-related issues.
"""

print("=== Running Compliance Audit (Text Input) ===\n")

# Run the agent
final_output_demo = manager_agent.run(demo_text, user_id="demo_text")

# ------------------------
# 1. Display Final Score
# ------------------------
print("\n=== COMPLIANCE SCORE ===")
print(f"Score: {final_output_demo['risk_score']['compliance_score']} / 100\n")

# ------------------------
# 2. Rule Breakdown
# ------------------------
print("=== RULE BREAKDOWN ===")
for rule in final_output_demo["rule_checks"]["checks"]:
    print(f"[{rule['rule_id']}] Passed={rule['passed']} | Matches={rule['matches']}")

# ------------------------
# 3. Text Preview
# ------------------------
print("\n=== DOCUMENT PREVIEW (first 300 chars) ===")
print(final_output_demo["text_preview"][:300])

# ------------------------
# 4. Session Log (safe access)
# ------------------------
print("\n=== SESSION LOG ===")
print(session_service.get_history("demo_text"))

# ------------------------
# 5. Trace Details (safe access)
# ------------------------
print("\n=== TRACE DETAILS (Most Recent Trace) ===")
print(observability.get_traces()[-1] if observability.get_traces() else "No trace found")



# ==================================================
# Reusable PDF Demo Cell (Upload → Enter Filename → Run)
# ==================================================

# Ask for PDF file name
pdf_path = input("Enter the PDF file name (e.g., policy.pdf): ").strip()

# Load PDF file
try:
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
except Exception as e:
    print(f"[ERROR] Could not open PDF file: {e}")
    raise

print("\n=== Running PDF Compliance Audit ===\n")

# Run audit
pdf_output = manager_agent.run(pdf_bytes, user_id="demo_pdf")

# ------------------------
# Show Results
# ------------------------

print("\n=== COMPLIANCE SCORE ===")
print(pdf_output["risk_score"])

print("\n=== RULE BREAKDOWN ===")
for r in pdf_output["rule_checks"]["checks"]:
    print(f"[{r['rule_id']}] Passed={r['passed']} | Matches={r['matches']}")

print("\n=== TEXT PREVIEW (first 300 chars) ===")
print(pdf_output["text_preview"][:300])

print("\n=== TRACE DETAILS ===")
print(observability.get_traces()[-1] if observability.get_traces() else "No trace data found")


