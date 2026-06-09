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


import os
import logging
import json
import uuid
from datetime import datetime
from collections import Counter
from typing import List, Dict, Any, Optional
from kaggle_secrets import UserSecretsClient


import google.generativeai as genai


# Retrieve Gemini API key from Kaggle secrets for secure configuration
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("GEMINI_API_KEY")

# Option 2 (for local dev only): uncomment to enter interactively
# from getpass import getpass
# api_key = getpass("Enter your Google Generative AI API key: ")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. "
        "Set it as a Kaggle secret or environment variable before running."
    )


# Use a fast, cost-efficient Gemini model for multiple agent calls
genai.configure(api_key=api_key)
GEMINI_MODEL_NAME = "gemini-2.0-flash"

# Create a global model instance
gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)

# for m in genai.list_models():
#     if 'generateContent' in m.supported_generation_methods:
#         print(m.name)


# ---------------------------------------------------------
# Logging config & metrics 
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("incident_copilot")

METRICS = {
    "incidents_started": 0,
    "incidents_resolved": 0,
    "tool_calls": Counter(),
    "agent_calls": Counter(),
}


def call_llm(
    prompt: str,
    *,
    system_instruction: Optional[str] = None,
    max_output_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    """
    Unified LLM caller using Google Generative AI (Gemini).

    - Uses a global `gemini_model`.
    - Allows an optional system-style instruction to control behavior.
    """
    logger.info("[LLM] call_llm invoked")

    try:
        content_parts = []
        if system_instruction:
            # Many frameworks separate system vs user, but Gemini's API
            # lets us just prepend the system guidance into the prompt content.
            content_parts.append(f"System: {system_instruction}\n")
        content_parts.append(f"User: {prompt}")

        response = gemini_model.generate_content(
            "\n".join(content_parts),
            generation_config={
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
            },
        )

        # Basic safety: handle empty or blocked response cases
        if not response or not response.candidates:
            logger.warning("[LLM] Empty or missing response from Gemini")
            return "I could not generate a response for this request."

        text = response.text or ""
        return text.strip()

    except Exception as e:
        logger.exception(f"[LLM] Error while calling Gemini: {e}")
        # Fallback generic answer to keep the notebook robust
        return "There was an issue calling the language model. Please check logs or configuration."



RUNBOOKS = [
    {
        "id": "RB001",
        "title": "Web Service Connection Timeout Runbook",
        "keywords": ["timeout", "connection", "web service", "gateway"],
        "steps": [
            "Check if the web service pod is running.",
            "Verify network connectivity between web service and database.",
            "Inspect recent deployment or configuration changes.",
            "Restart the web service if needed and monitor logs."
        ],
    },
    {
        "id": "RB002",
        "title": "Database Authentication Failure Runbook",
        "keywords": ["authentication", "db", "database", "invalid password"],
        "steps": [
            "Verify DB credentials in the application configuration.",
            "Check DB user status and privileges.",
            "Rotate secrets if compromised.",
        ],
    },
]

SAMPLE_LOGS = [
    "2025-11-01T10:00:00Z web-service ERROR connection timeout to db:5432",
    "2025-11-01T10:01:00Z web-service WARN retrying connection",
    "2025-11-01T10:02:00Z web-service ERROR authentication failed for user 'app'",
]



def runbook_search_tool(query: str) -> List[Dict[str, Any]]:
    """
    Very simple keyword-matching runbook search.
    Scores runbooks by how many of their keywords appear in the query
    and returns them sorted by score (descending).
    """
    METRICS["tool_calls"]["runbook_search"] += 1
    query_lower = query.lower()
    results = []
    for rb in RUNBOOKS:
        score = sum(1 for kw in rb["keywords"] if kw in query_lower)
        if score > 0:
            results.append((score, rb))
    results.sort(key=lambda x: x[0], reverse=True)
    logger.info(f"[TOOL] runbook_search_tool query='{query}' results={len(results)}")
    return [rb for _, rb in results]


def log_search_tool(query: str, logs: Optional[List[str]] = None) -> List[str]:
    METRICS["tool_calls"]["log_search"] += 1
    if logs is None:
        logs = SAMPLE_LOGS
    query_lower = query.lower()
    matches = [line for line in logs if query_lower in line.lower()]
    logger.info(f"[TOOL] log_search_tool query='{query}' matches={len(matches)}")
    return matches



SESSIONS: Dict[str, Dict[str, Any]] = {}
INCIDENT_MEMORY: List[Dict[str, Any]] = []


def create_incident_session(
    description: str, logs: Optional[List[str]] = None
) -> str:
    """
    Create a new incident session and return its session_id.
    This acts as a lightweight session store for the orchestrator
    and coach to update during the lifecycle of an incident.
    """
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {
        "session_id": session_id,
        "description": description,
        "logs": logs or [],
        "created_at": datetime.utcnow().isoformat(),
        "status": "open",
        "actions": [],
        "selected_runbooks": [],
        "resolution_summary": None,
    }
    METRICS["incidents_started"] += 1
    logger.info(f"[SESSION] Created incident session: {session_id}")
    return session_id


def get_session(session_id: str) -> Dict[str, Any]:
    return SESSIONS[session_id]


def update_session(session_id: str, **kwargs) -> None:
    session = SESSIONS[session_id]
    session.update(kwargs)
    logger.info(f"[SESSION] Updated incident session: {session_id} with {list(kwargs.keys())}")


def add_incident_memory(record: Dict[str, Any]) -> None:
    INCIDENT_MEMORY.append(record)
    logger.info(f"[MEMORY] Added incident memory record id={record.get('incident_id')}")


def search_incident_memory(query: str) -> List[Dict[str, Any]]:
    results = []
    query_lower = query.lower()
    for rec in INCIDENT_MEMORY:
        text = (rec.get("summary") or "") + " " + " ".join(rec.get("tags", []))
        if query_lower in text.lower():
            results.append(rec)
    logger.info(f"[MEMORY] search_incident_memory for '{query}' -> {len(results)} matches")
    return results



class Agent:
    def __init__(self, name: str):
        self.name = name

    def log_call(self):
        METRICS["agent_calls"][self.name] += 1
        logger.info(f"[AGENT] {self.name} called")

    def act(self, **kwargs):
        raise NotImplementedError



class RetrievalAgent(Agent):
    """
    Agent responsible for all retrieval operations:
    - runbooks
    - past incident memory
    - relevant log lines
    """
    def __init__(self):
        super().__init__("RetrievalAgent")

    def act(self, incident_description: str, logs: List[str]) -> Dict[str, Any]:
        self.log_call()
        runbooks = runbook_search_tool(incident_description)
        memory_hits = search_incident_memory(incident_description)

        log_hits = []
        for kw in ["timeout", "authentication", "error"]:
            log_hits.extend(log_search_tool(kw, logs))

        return {
            "runbooks": runbooks,
            "memory_hits": memory_hits,
            "log_hits": list(set(log_hits)),
        }



class AnalysisAgent(Agent):
    def __init__(self):
        super().__init__("AnalysisAgent")

    def act(self, incident_description: str, retrieval_context: Dict[str, Any]) -> Dict[str, Any]:
        self.log_call()
        runbook_titles = ", ".join(rb["title"] for rb in retrieval_context["runbooks"])
        memory_summaries = ", ".join(
            rec.get("summary", "") for rec in retrieval_context["memory_hits"]
        )
        log_sample = "\n".join(retrieval_context["log_hits"][:5])

        base_prompt = f"""
Incident description:
{incident_description}

Relevant runbooks:
{runbook_titles or "None"}

Past incident summaries:
{memory_summaries or "None"}

Sample log lines:
{log_sample or "None"}
"""

        root_cause = call_llm(
            base_prompt + "\n\nIdentify the likely root cause in 2–3 sentences.",
            system_instruction="You are a senior SRE analyzing incidents.",
        )

        resolution_plan = call_llm(
            base_prompt + "\n\nPropose a concise, numbered resolution plan.",
            system_instruction="You are a senior SRE proposing concrete, actionable steps.",
        )

        raw_analysis = call_llm(
            base_prompt + "\n\nProvide a combined explanation of root cause and resolution steps.",
            system_instruction="Explain clearly but concisely.",
        )

        return {
            "raw_analysis": raw_analysis,
            "probable_root_cause": root_cause,
            "resolution_plan": resolution_plan,
        }



class CoachAgent(Agent):
    def __init__(self):
        super().__init__("CoachAgent")

    def act(
        self,
        session: Dict[str, Any],
        analysis: Dict[str, Any],
        auto_mode: bool = True,
    ) -> Dict[str, Any]:
        self.log_call()
        plan = analysis.get("resolution_plan", "")
        steps = [s.strip() for s in plan.split("\n") if s.strip()]

        actions_taken = []
        for step in steps:
            logger.info(f"[COACH] Suggesting step: {step}")
            if auto_mode:
                user_feedback = "yes"
            else:
                user_feedback = input(f"Execute: {step}\nDid it succeed? (yes/no): ").strip().lower()

            actions_taken.append({"step": step, "success": (user_feedback == "yes")})
            if user_feedback != "yes":
                logger.info("[COACH] Step reported as failed; would branch/escalate in a real system.")
                break

        return {"actions_taken": actions_taken}



class OrchestratorAgent(Agent):
    def __init__(self, retrieval_agent: RetrievalAgent, analysis_agent: AnalysisAgent, coach_agent: CoachAgent):
        super().__init__("OrchestratorAgent")
        self.retrieval_agent = retrieval_agent
        self.analysis_agent = analysis_agent
        self.coach_agent = coach_agent

    def act(self, session_id: str, auto_mode: bool = True) -> Dict[str, Any]:
        self.log_call()
        session = get_session(session_id)
        description = session["description"]
        logs = session["logs"]

        retrieval_context = self.retrieval_agent.act(description, logs)
        update_session(session_id, selected_runbooks=retrieval_context["runbooks"])

        analysis = self.analysis_agent.act(description, retrieval_context)
        coaching_result = self.coach_agent.act(session, analysis, auto_mode=auto_mode)

        summary_prompt = f"""
Summarize this incident and its resolution in 3–4 sentences for an incident postmortem.

Incident description:
{description}

Root cause:
{analysis['probable_root_cause']}

Resolution steps taken:
{coaching_result['actions_taken']}
"""
        resolution_summary = call_llm(
            summary_prompt,
            system_instruction="You are writing a concise incident postmortem summary.",
        )

        update_session(
            session_id,
            status="resolved",
            actions=session.get("actions", []) + coaching_result["actions_taken"],
            resolution_summary=resolution_summary,
        )
        METRICS["incidents_resolved"] += 1

        memory_record = {
            "incident_id": session_id,
            "summary": resolution_summary,
            "tags": ["demo", "incident", "runbook"],
            "created_at": datetime.utcnow().isoformat(),
        }
        add_incident_memory(memory_record)

        return {
            "retrieval_context": retrieval_context,
            "analysis": analysis,
            "coaching": coaching_result,
            "resolution_summary": resolution_summary,
        }


# Instantiate agents
retrieval_agent = RetrievalAgent()
analysis_agent = AnalysisAgent()
coach_agent = CoachAgent()
orchestrator = OrchestratorAgent(retrieval_agent, analysis_agent, coach_agent)



demo_description = "Our web service is throwing connection timeout errors when talking to the database."
demo_logs = SAMPLE_LOGS

session_id = create_incident_session(demo_description, demo_logs)
result = orchestrator.act(session_id, auto_mode=True)

print("=== RESOLUTION SUMMARY ===")
print(result["resolution_summary"])
print("\n=== ACTIONS TAKEN ===")
for action in result["coaching"]["actions_taken"]:
    print(f"- {action['step']} | success={action['success']}")

print("\n=== SELECTED RUNBOOKS ===")
for rb in result["retrieval_context"]["runbooks"]:
    print(f"- {rb['id']}: {rb['title']}")



print("=== METRICS ===")
print(f"Incidents started: {METRICS['incidents_started']}")
print(f"Incidents resolved: {METRICS['incidents_resolved']}")
print("Agent calls:", dict(METRICS["agent_calls"]))
print("Tool calls:", dict(METRICS["tool_calls"]))



EVAL_CASES = [
    {
        "id": "EVAL1",
        "description": "Users see frequent timeout errors when accessing the website.",
        "logs": SAMPLE_LOGS,
        "expected_runbook_id": "RB001",
        "expected_keyword": "timeout",
    },
    {
        "id": "EVAL2",
        "description": "The service fails to authenticate with the database due to invalid password.",
        "logs": SAMPLE_LOGS,
        "expected_runbook_id": "RB002",
        "expected_keyword": "authentication",
    },
]


def evaluate_system(eval_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    runbook_hits = 0
    keyword_hits = 0

    for case in eval_cases:
        logger.info(f"[EVAL] Evaluating case {case['id']}")
        session_id = create_incident_session(case["description"], case["logs"])
        result = orchestrator.act(session_id, auto_mode=True)

        selected_ids = [rb["id"] for rb in result["retrieval_context"]["runbooks"]]
        if case["expected_runbook_id"] in selected_ids:
            runbook_hits += 1

        if case["expected_keyword"].lower() in (result["resolution_summary"] or "").lower():
            keyword_hits += 1

    n = len(eval_cases)
    return {
        "runbook_match_rate": runbook_hits / n if n else 0.0,
        "keyword_match_rate": keyword_hits / n if n else 0.0,
        "total_cases": n,
    }


eval_results = evaluate_system(EVAL_CASES)
print("=== EVALUATION RESULTS ===")
print(json.dumps(eval_results, indent=2))



submission_df = pd.DataFrame(
    [
        {
            "runbook_match_rate": eval_results["runbook_match_rate"],
            "keyword_match_rate": eval_results["keyword_match_rate"],
            "total_cases": eval_results["total_cases"],
        }
    ]
)

# Write to the working directory.
submission_df.to_csv("submission.csv", index=False)

print("Saved submission.csv:")
print(submission_df)





