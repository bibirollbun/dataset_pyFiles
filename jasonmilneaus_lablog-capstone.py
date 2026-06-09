import os
from kaggle_secrets import UserSecretsClient

try:
    # Load Gemini API key from Kaggle user secrets
    user_secrets = UserSecretsClient()
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")

    # Expose for google-genai / ADK
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    # Stay on direct Gemini API in this notebook environment
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"

    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        "ğŸ”‘ Authentication Error: Please make sure you have added "
        "'GOOGLE_API_KEY' to your Kaggle secrets. "
        f"Details: {e}"
    )



# Cell 2: ADK core imports and retry configuration

# --- Standard library imports ------------------------------------------------
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# --- ADK imports (consistent with Day-4b agent evaluation) -------------------
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.adk.models.google_llm import Gemini

# --- Retry configuration (pattern from course notebooks) ---------------------
from google.genai import types

retry_config = types.HttpRetryOptions(
    attempts=5,            # retry several times
    exp_base=7,            # exponential backoff base
    initial_delay=1,       # initial delay in seconds
    http_status_codes=[
        429,  # rate limit
        500,  # internal server error
        503,  # service unavailable
        504,  # gateway timeout
    ],
)

print("âœ… ADK core modules imported. Retry configuration ready.")



# Cell 3: Experiment data model and helpers

from pathlib import Path

# ---------------------------------------------------------------------------
# Timestamp helper
# ---------------------------------------------------------------------------

def now_iso() -> str:
    """Return the current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()
# ---------- Section 3: Data model and filesystem helpers ----------

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any
import json

# Base directory and experiments folder
BASE_DIR = Path.cwd()
EXPERIMENTS_DIR = BASE_DIR / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)

@dataclass
class ExperimentEvent:
    timestamp: str
    type: str
    content: str
    tags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ExperimentLog:
    experiment_id: str
    metadata: Dict[str, Any]
    events: List[ExperimentEvent]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "metadata": self.metadata,
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentLog":
        """Reconstruct an ExperimentLog from a dict saved to JSON."""
        events_data = data.get("events", [])

        # Prefer ExperimentEvent.from_dict if it exists, otherwise build directly
        events = []
        for e in events_data:
            if hasattr(ExperimentEvent, "from_dict"):
                events.append(ExperimentEvent.from_dict(e))
            else:
                events.append(
                    ExperimentEvent(
                        timestamp=e.get("timestamp"),
                        type=e.get("type", "note"),
                        content=e.get("content", ""),
                        tags=e.get("tags", []),
                    )
                )

        return cls(
            experiment_id=data.get("experiment_id", ""),
            metadata=data.get("metadata", {}),
            events=events,
        )



def experiment_dir(experiment_id: str) -> Path:
    d = EXPERIMENTS_DIR / experiment_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def events_path(experiment_id: str) -> Path:
    return experiment_dir(experiment_id) / "events.json"

def report_path(experiment_id: str) -> Path:
    return experiment_dir(experiment_id) / "report.md"

def save(self, path: str) -> None:
    """Write the log to disk as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(self.to_dict(), f, indent=2)

def load_log(experiment_id: str) -> ExperimentLog:
    path = events_path(experiment_id)
    if not path.exists():
        raise FileNotFoundError(f"No events.json for {experiment_id}")
    data = json.loads(path.read_text())
    events = [
        ExperimentEvent(
            timestamp=e["timestamp"],
            type=e["type"],
            content=e["content"],
            tags=e.get("tags", []),
        )
        for e in data.get("events", [])
    ]
    return ExperimentLog(
        experiment_id=data["experiment_id"],
        metadata=data.get("metadata", {}),
        events=events,
    )

from pathlib import Path

def save_log(log: ExperimentLog) -> None:
    """
    Persist an ExperimentLog to experiments/<id>/events.json.

    Uses pathlib.Path so we can call write_text safely.
    """
    # events_path(...) currently returns a string, so wrap it as a Path
    path = Path(events_path(log.experiment_id))
    path.parent.mkdir(parents=True, exist_ok=True)

    data = json.dumps(log.to_dict(), indent=2)
    path.write_text(data, encoding="utf-8")

def init_experiment(experiment_id: str, title: str, user: str = "jason") -> ExperimentLog:
    """
    Create a fresh experiment directory and log file (if none exists),
    or load an existing one without wiping events.
    """
    exp_dir = experiment_dir(experiment_id)
    path = events_path(experiment_id)

    if path.exists():
        # Reuse existing log
        data = json.loads(path.read_text())
        events = [
            ExperimentEvent(
                timestamp=e["timestamp"],
                type=e["type"],
                content=e["content"],
                tags=e.get("tags", []),
            )
            for e in data.get("events", [])
        ]
        meta = data.get("metadata", {})
        return ExperimentLog(experiment_id=experiment_id, metadata=meta, events=events)

    # New experiment
    meta = {
        "title": title,
        "user": user,
        "experiment_id": experiment_id,
        "created_at": now_iso(),
    }
    log = ExperimentLog(experiment_id=experiment_id, metadata=meta, events=[])
    save_log(log)      # âœ… this is the correct way to persist
    return log

def save_experiment_events(experiment_id: str, new_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Append new events to the log and save."""
    try:
        log = load_log(experiment_id)
    except FileNotFoundError:
        log = init_experiment(experiment_id, title=experiment_id, user="jason")

    for e in new_events:
        ts = e.get("timestamp") or now_iso()
        log.events.append(
            ExperimentEvent(
                timestamp=ts,
                type=e["type"],
                content=e["content"],
                tags=e.get("tags", []),
            )
        )

    return save_log(log)

print("âœ… Experiment data model + filesystem helpers ready.")




# Cell 4: Filesystem persistence for experiments

# Core imports
import os
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any

# Base directory (Kaggle working dir)
BASE_DIR = Path.cwd()
EXPERIMENTS_DIR = BASE_DIR / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


BASE_DIR = "experiments"


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def exp_dir(experiment_id: str) -> str:
    """Return the directory path for the given experiment ID."""
    return os.path.join(BASE_DIR, experiment_id)


def events_path(experiment_id: str) -> str:
    """Return the full path to events.json for this experiment."""
    return os.path.join(exp_dir(experiment_id), "events.json")


def report_path(experiment_id: str) -> str:
    """Return the full path to report.md for this experiment."""
    return os.path.join(exp_dir(experiment_id), "report.md")


# ---------------------------------------------------------------------------
# Experiment log management
# ---------------------------------------------------------------------------

def init_experiment(experiment_id: str, title: str, user: str = "jason") -> ExperimentLog:
    """
    Create a fresh experiment directory and log file (if none exists),
    or load an existing one.

    Args:
        experiment_id: Unique identifier for the experiment.
        title: Human-readable experiment title.
        user: Name associated with the experiment run.

    Returns:
        An ExperimentLog instance ready for event accumulation.
    """
    os.makedirs(exp_dir(experiment_id), exist_ok=True)
    epath = events_path(experiment_id)

    # If a log already exists, load and update metadata
    if os.path.exists(epath):
        log = load_log(experiment_id)
        log.metadata.setdefault("title", title)
        log.metadata.setdefault("user", user)
        return log

    # Otherwise create a brand new log
    metadata = {
        "title": title,
        "created_at": now_iso(),
        "user": user,
    }

    log = ExperimentLog(
        experiment_id=experiment_id,
        metadata=metadata,
        events=[],
    )

    # Use the helper, not a method on the class
    save_log(log)
    return log



from pathlib import Path

def load_log(experiment_id: str) -> ExperimentLog:
    """
    Load an ExperimentLog from experiments/<id>/events.json.

    Uses ExperimentLog.from_dict(...) instead of a non-existent from_file().
    """
    path = Path(events_path(experiment_id))
    if not path.exists():
        raise FileNotFoundError(f"No log found for experiment {experiment_id}: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    return ExperimentLog.from_dict(data)



# ---------------------------------------------------------------------------
# Tool-ready functions (to be exposed to the root agent)
# ---------------------------------------------------------------------------

def save_experiment_events(experiment_id: str, events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Append structured ExperimentEvent dictionaries to an experiment log,
    filling timestamps if needed.

    This function is ADK-tool-safe: clear arguments and structured return.

    Args:
        experiment_id: Experiment ID.
        events: List of dicts with keys: 'type', 'content', 'tags',
                and optional 'timestamp'.

    Returns:
        Dictionary summarising the result.
    """
    log = load_log(experiment_id)

    for evt in events:
        if not evt.get("timestamp"):
            evt["timestamp"] = now_iso()
        log.events.append(ExperimentEvent(**evt))

    log.save(events_path(experiment_id))

    return {
        "status": "success",
        "experiment_id": experiment_id,
        "event_count": len(log.events),
    }


def save_experiment_report(experiment_id: str, markdown: str) -> Dict[str, Any]:
    """
    Save the experiment's markdown report to disk.

    Args:
        experiment_id: Experiment ID.
        markdown: Markdown-formatted report text.

    Returns:
        Dictionary summarising the result.
    """
    rpath = report_path(experiment_id)
    os.makedirs(os.path.dirname(rpath), exist_ok=True)

    with open(rpath, "w", encoding="utf-8") as f:
        f.write(markdown)

    return {
        "status": "success",
        "experiment_id": experiment_id,
        "report_path": rpath,
    }


print("âœ… Filesystem persistence initialized.")



# Cell 5: Ingestion Sub-Agent definition

ingestion_instruction = """
You are the Ingestion Agent for LabLog.

Your job:
Convert free-form experiment notes into a list of structured ExperimentEvent
objects that describe goals, setup, actions, observations, results, and next
steps.

You ALWAYS receive a JSON object with this shape:

{
  "experiment_context": {
    "experiment_id": "...",
    "title": "...",
    "goal_events": [ExperimentEvent, ...],
    "setup_events": [ExperimentEvent, ...]
  },
  "content_type": "...",
  "raw_content": "...",
  "question_topic": "... (optional)"
}

The user interface may send `content_type` equal to:

- "auto": a generic free-form note. You MUST infer the most appropriate
  ExperimentEvent type from the text itself.
- "note", "code", "data_summary", "file_description",
  "image_description", "audio_transcript", "gap_answer":
  these are hints about the kind of content, but you STILL choose the
  final ExperimentEvent.type based on the meaning of the text.

ExperimentEvent format:

{
  "timestamp": "",          // leave empty string, the system will fill it
  "type": "...",            // see allowed types below
  "content": "...",         // concise but informative
  "tags": ["...", ...]
}

Allowed ExperimentEvent.type values:
- "goal"            : when the note states what the experiment is trying to achieve
- "background"      : context, prior work, motivations
- "setup_detail"    : descriptions of hardware, wiring, mechanical setup, fixtures
- "firmware_change" : changes in firmware, code, configuration, parameters
- "procedure"       : step-by-step actions performed during the experiment
- "data_capture"    : details of how data was collected or processed
- "observation"     : what was observed or measured (including "results" style text)
- "next_step"       : proposed future actions or open questions
- "note"            : any other information that does not fit the above categories

Mapping guidelines for free-form notes (including content_type="auto"):

- If the user describes what they are trying to achieve or why:
  -> "goal" or "background"

- If they describe hardware, wiring, mounting, or physical arrangement:
  -> "setup_detail"

- If they describe changes to firmware, code, ODR, ranges, filters, etc:
  -> "firmware_change"

- If they describe what they actually did in sequence:
  -> "procedure"

- If they describe how data was logged, analysed, or processed:
  -> "data_capture"

- If they describe what happened, what was observed, or specific results:
  -> "observation"

- If they describe future plans, TODOs, or open questions:
  -> "next_step"

- If none of these are clearly applicable:
  -> "note"

Output:
Return ONLY a JSON object of the form:

{
  "events": [ ExperimentEvent, ... ]
}

Never include explanations, comments, or markdown. JSON ONLY.
"""


ingestion_agent = LlmAgent(
    name="lablog_ingestion_agent",
    description="Extracts structured experiment events from raw notes, code, or textual summaries.",
    instruction=ingestion_instruction,
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
)

# Runner for programmatic use (e.g., orchestrator wrapper functions)
ingestion_runner = InMemoryRunner(agent=ingestion_agent)

print("âœ… Ingestion sub-agent loaded.")



# Cell 6: Report Sub-Agent definition

report_instruction = """
You are the Report Agent for LabLog.

Your job:
Generate a clear, structured markdown report summarising the experiment.

Input:
You ALWAYS receive a JSON object:

{
  "experiment_metadata": {
    "experiment_id": "...",
    "title": "...",
    "user": "...",
    "created_at": "...",
    "start_date": "...",
    "end_date": "..."
  },
  "events": [ ExperimentEvent, ... ]
}

Event Format:
Each ExperimentEvent contains:
{
  "timestamp": "...",
  "type": "...",
  "content": "...",
  "tags": [...]
}

Output:
Return ONLY markdown text (no JSON, no commentary).

Start the markdown with a header that looks like this:

# <experiment title>

**Experiment ID:** <experiment_id>  
**Performed by:** <user>  
**Date Range:** <start_date> to <end_date>

Then include these sections:

## 1. Goal and Background

## 2. Setup
### 2.1 Hardware
### 2.2 Firmware and Software
### 2.3 Test Conditions

## 3. Procedure and Timeline

## 4. Observations

## 5. Outcomes and Conclusions

## 6. Next Steps and Open Questions

Guidelines:
- Use ONLY the provided event data. Never fabricate hardware,
  parameters, results, or conditions.
- Group related events into coherent paragraphs.
- Be concise and factual.
- Maintain consistent markdown formatting.
"""


report_agent = LlmAgent(
    name="lablog_report_agent",
    description="Produces structured markdown reports from experiment events.",
    instruction=report_instruction,
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
)

# Runner for the orchestrator to call programmatically
report_runner = InMemoryRunner(agent=report_agent)

print("âœ… Report sub-agent loaded.")



# Cell 7: Evaluator Sub-Agent definition

evaluator_instruction = """
You are the Evaluator Agent for LabLog.

Your purpose:
Critically evaluate a draft experiment report and identify missing information.

Inputs:
You ALWAYS receive one JSON object:

{
  "experiment_metadata": {
    "experiment_id": "...",
    "title": "...",
    "user": "...",
    "created_at": "...",
    "start_date": "...",
    "end_date": "..."
  },
  "events": [ ExperimentEvent, ... ],
  "report_markdown": "full report text generated by the Report Agent"
}

Your REQUIRED OUTPUT (JSON ONLY):

{
  "scores": {
    "goal_clarity": 1-5,
    "setup_completeness": 1-5,
    "decisions_captured": 1-5,
    "next_steps_clarity": 1-5
  },
  "overall_comment": "Short summary of report quality.",
  "missing_information": [
    "Description of a missing detail",
    "Another missing detail"
  ],
  "questions_for_user": [
    {
      "id": "q1",
      "topic": "hardware_setup" | "firmware_config" | "procedure" |
               "data" | "results" | "next_steps" | "other",
      "question": "Concrete question asking the user for clarification."
    }
  ],
  "should_regenerate_report": true | false
}

Scoring:
- 5 = excellent
- 4 = good but could be clearer
- 3 = adequate but incomplete
- 2 = poor / missing key information
- 1 = unclear or incorrect

Logic:
- If ANY score <= 3 OR if essential details are missing,
  'should_regenerate_report' MUST be true.
- DO NOT rewrite or modify the report. Only evaluate it.
- DO NOT generate markdown or explanations outside the JSON.
"""

evaluator_agent = LlmAgent(
    name="lablog_evaluator_agent",
    description="Evaluates draft experiment reports and identifies missing details.",
    instruction=evaluator_instruction,
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
)

evaluator_runner = InMemoryRunner(agent=evaluator_agent)

print("âœ… Evaluator sub-agent loaded.")


# Cell 8: Wrapper functions for calling sub-agents

import asyncio


# Helper: extract JSON object from LLM text output

def parse_json_from_text(text: str) -> dict:
    """
    Robustly parse a JSON object from an LLM text response.

    Handles:
    - leading/trailing whitespace
    - ```json ... ``` fenced blocks
    - extra prose before/after the JSON (keeps the first {...} block)
    """
    if text is None:
        raise ValueError("Agent returned None instead of text.")

    s = text.strip()
    if not s:
        raise ValueError("Agent returned an empty response; no JSON to parse.")

    # Strip ```...``` fences if present
    if s.startswith("```"):
        lines = s.splitlines()
        # Drop first line if it's a fence (``` or ```json)
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        # Drop last line if it's a closing fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()

    # If there is surrounding prose, try to isolate the first {...} block
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and first < last:
        candidate = s[first : last + 1]
    else:
        candidate = s

    # Let json.loads raise if it's still invalid
    return json.loads(candidate)



# ---------------------------------------------------------------------------
# Ingestion wrapper
# ---------------------------------------------------------------------------

async def run_ingestion(
    experiment_id: str,
    content_type: str,
    raw_content: str,
    question_topic: Optional[str] = None,
) -> dict:
    """
    Call the Ingestion Agent to convert raw user input into ExperimentEvent dicts.

    Args:
        experiment_id: Experiment identifier.
        content_type: One of:
            'note', 'code', 'data_summary', 'file_description',
            'image_description', 'audio_transcript', 'gap_answer'
        raw_content: Raw text to be parsed into structured events.
        question_topic: Optional topic if this is answering an evaluator question.

    Returns:
        dict: {"events": [...]} parsed JSON from the ingestion agent.
    """
    log = load_log(experiment_id)

    context = {
        "experiment_id": experiment_id,
        "title": log.metadata.get("title", ""),
        "goal_events": [
            evt.to_dict() for evt in log.events
            if evt.type in ("goal", "background")
        ],
        "setup_events": [
            evt.to_dict() for evt in log.events
            if evt.type == "setup_detail"
        ],
    }

    payload = {
        "experiment_context": context,
        "content_type": content_type,
        "raw_content": raw_content,
    }

    if question_topic is not None:
        payload["question_topic"] = question_topic

    # Call ingestion agent
    result = await ingestion_runner.run_debug(json.dumps(payload))
    text_output = result[-1].content.parts[0].text

    # Debug hook if things still go wrong:
    # print("INGESTION RAW OUTPUT:", repr(text_output))

    data = parse_json_from_text(text_output)
    return data


# ---------------------------------------------------------------------------
# Report wrapper
# ---------------------------------------------------------------------------

async def run_report(experiment_id: str) -> str:
    """
    Call the Report Agent to generate markdown for the full experiment report.

    Args:
        experiment_id: ID of the experiment.

    Returns:
        Markdown report string.
    """
    log = load_log(experiment_id)

    meta = log.metadata.copy()
    meta["experiment_id"] = log.experiment_id

    if log.events:
        timestamps = [evt.timestamp for evt in log.events]
        meta["start_date"] = min(timestamps)
        meta["end_date"] = max(timestamps)
    else:
        now = now_iso()
        meta["start_date"] = now
        meta["end_date"] = now

    payload = {
        "experiment_metadata": meta,
        "events": [evt.to_dict() for evt in log.events],
    }

    result = await report_runner.run_debug(json.dumps(payload))
    markdown = result[-1].content.parts[0].text
    return markdown


# ---------------------------------------------------------------------------
# Evaluation wrapper
# ---------------------------------------------------------------------------

async def run_evaluation(experiment_id: str, report_markdown: str) -> dict:
    """
    Call the Evaluator Agent to critique a draft report.

    Args:
        experiment_id: ID of the experiment.
        report_markdown: Markdown text produced by the report agent.

    Returns:
        dict containing:
            - scores
            - overall_comment
            - missing_information
            - questions_for_user
            - should_regenerate_report
    """
    log = load_log(experiment_id)

    meta = log.metadata.copy()
    meta["experiment_id"] = log.experiment_id

    payload = {
        "experiment_metadata": meta,
        "events": [evt.to_dict() for evt in log.events],
        "report_markdown": report_markdown,
    }

    result = await evaluator_runner.run_debug(json.dumps(payload))
    text_output = result[-1].content.parts[0].text

    # Debug hook if needed:
    # print("EVALUATOR RAW OUTPUT:", repr(text_output))

    eval_data = parse_json_from_text(text_output)
    return eval_data


print("âœ… Sub-agent wrapper functions ready.")



# Cell 9: Orchestrator â€“ end-to-end LabLog session

async def lablog_run_session(
    experiment_id: str,
    title: str,
    user: str,
    session_notes: List[Dict[str, Any]],
    gap_answers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Run a full LabLog session:

    1. Initialise or load an experiment log.
    2. Ingest a sequence of user-provided notes/actions and save events.
    3. Generate a draft report.
    4. Evaluate the report.
    5. Optionally ingest answers to evaluator questions and regenerate.
    6. Save the final report.

    Args:
        experiment_id:
            Unique identifier for the experiment (e.g. "exp_001").
        title:
            Human-readable experiment title.
        user:
            Name of the person running the experiment.
        session_notes:
            List of dicts, each with:
                - "content_type": one of the types supported by run_ingestion
                - "raw_content": free-form text describing goals, setup,
                                 actions, observations, etc.
        gap_answers:
            Optional mapping from evaluator question IDs (e.g. "q1") to
            answer text. If provided and the evaluator requests regeneration,
            these answers are ingested as additional events.

    Returns:
        dict with:
            - "experiment_id"
            - "draft_report"
            - "evaluation"
            - "final_report"
    """
    # 1. Initialise or load the experiment log
    log = init_experiment(experiment_id=experiment_id, title=title, user=user)
    print(f"ğŸ”¬ Experiment '{experiment_id}' initialised with title: {title}")

    # 2. Ingest notes and append events
    for i, note in enumerate(session_notes, start=1):
        content_type = note["content_type"]
        raw_content = note["raw_content"]
        print(f"\nğŸ“¥ Ingesting note {i} ({content_type}):\n{raw_content}\n")

        ingestion_result = await run_ingestion(
            experiment_id=experiment_id,
            content_type=content_type,
            raw_content=raw_content,
            question_topic=note.get("question_topic"),
        )
        events = ingestion_result.get("events", [])
        print(f"â�¡ï¸�  Ingestion produced {len(events)} event(s).")

        save_summary = save_experiment_events(experiment_id, events)
        print(
            f"ğŸ’¾ Saved events. Total event count now: "
            f"{save_summary['event_count']}"
        )

    # 3. Generate a draft report
    print("\nğŸ“� Generating draft report...")
    draft_report = await run_report(experiment_id)
    draft_result = save_experiment_report(experiment_id, draft_report)
    print(f"ğŸ’¾ Draft report saved at: {draft_result['report_path']}")

    # 4. Evaluate the draft report
    print("\nğŸ§ª Evaluating draft report...")
    evaluation = await run_evaluation(experiment_id, draft_report)
    scores = evaluation.get("scores", {})
    print("ğŸ“Š Evaluation scores:", scores)
    print("ğŸ’¬ Overall comment:", evaluation.get("overall_comment", ""))

    questions = evaluation.get("questions_for_user", [])
    should_regen = evaluation.get("should_regenerate_report", False)

    if questions:
        print("\nâ�“ Evaluator follow-up questions:")
        for q in questions:
            print(f"- {q['id']} ({q['topic']}): {q['question']}")
    else:
        print("\nâ„¹ï¸� Evaluator did not request additional details.")

    # 5. Optionally ingest gap answers and regenerate the report
    final_report = draft_report

    if should_regen and gap_answers:
        print("\nğŸ”„ Ingesting answers to evaluator questions and regenerating...")

        for q in questions:
            qid = q["id"]
            topic = q["topic"]
            if qid not in gap_answers:
                print(f"- Skipping {qid} (no answer provided).")
                continue

            answer_text = gap_answers[qid]
            print(f"\nğŸ“¥ Answer for {qid} ({topic}):\n{answer_text}\n")

            ingestion_result = await run_ingestion(
                experiment_id=experiment_id,
                content_type="gap_answer",
                raw_content=answer_text,
                question_topic=topic,
            )
            events = ingestion_result.get("events", [])
            print(f"â�¡ï¸�  Ingestion produced {len(events)} event(s) from answer.")
            save_summary = save_experiment_events(experiment_id, events)
            print(
                f"ğŸ’¾ Saved answer events. Total event count now: "
                f"{save_summary['event_count']}"
            )

        # Regenerate final report after gap-filling
        print("\nğŸ“� Regenerating final report...")
        final_report = await run_report(experiment_id)
        final_result = save_experiment_report(experiment_id, final_report)
        print(f"âœ… Final report saved at: {final_result['report_path']}")
    else:
        print("\nâœ… Draft report accepted as final (no regeneration).")

    return {
        "experiment_id": experiment_id,
        "draft_report": draft_report,
        "evaluation": evaluation,
        "final_report": final_report,
    }


print("âœ… Orchestrator function lablog_run_session() defined.")



# Cell 10: Helper to answer evaluator questions and regenerate the report

async def lablog_answer_questions_and_regenerate(
    experiment_id: str,
    gap_answers: Dict[str, str],
    evaluation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Ingest answers to evaluator questions for an experiment and
    regenerate the report.

    Args:
        experiment_id: ID of the experiment.
        gap_answers: Mapping from question_id (e.g. "q1") to answer text.
        evaluation: Optional existing evaluation dict. If not provided,
                    this function will generate and evaluate a draft
                    report internally.

    Returns:
        dict with:
            - "final_report"
            - "final_report_path"
            - "evaluation" (the evaluation that drove the questions)
    """
    # If we do not have an evaluation yet, generate one now
    if evaluation is None:
        print("â„¹ï¸� No evaluation provided; generating a fresh draft + evaluation...")
        draft_result = await lablog_run_session(
            experiment_id=experiment_id,
            title=load_log(experiment_id).metadata.get("title", experiment_id),
            user=load_log(experiment_id).metadata.get("user", "jason"),
            session_notes=[],  # no new notes here
            gap_answers=None,
        )
        evaluation = draft_result["evaluation"]

    questions = evaluation.get("questions_for_user", [])
    if not questions:
        print("â„¹ï¸� Evaluator had no questions; regenerating report anyway.")
    else:
        print("\nğŸ“� Ingesting answers to evaluator questions...")
        for q in questions:
            qid = q["id"]
            topic = q["topic"]
            if qid not in gap_answers:
                print(f"- Skipping {qid} (no answer provided).")
                continue

            answer_text = gap_answers[qid]
            print(f"\nğŸ“¥ Answer for {qid} ({topic}):\n{answer_text}\n")

            ingestion_result = await run_ingestion(
                experiment_id=experiment_id,
                content_type="gap_answer",
                raw_content=answer_text,
                question_topic=topic,
            )
            events = ingestion_result.get("events", [])
            print(f"â�¡ï¸�  Ingestion produced {len(events)} event(s) from answer.")
            save_summary = save_experiment_events(experiment_id, events)
            print(
                f"ğŸ’¾ Saved answer events. Total event count now: "
                f"{save_summary['event_count']}"
            )

    # Regenerate final report after gap-filling
    print("\nğŸ“� Regenerating final report after answering questions...")
    final_report = await run_report(experiment_id)
    final_result = save_experiment_report(experiment_id, final_report)
    print(f"âœ… Final report saved at: {final_result['report_path']}")

    return {
        "final_report": final_report,
        "final_report_path": final_result["report_path"],
        "evaluation": evaluation,
    }



# Cell 11: Interactive helpers â€“ add a note, generate+evaluate report, generate pdf

# Helper: run async coroutines from normal notebook cells (Kaggle/Jupyter safe)

import asyncio
import nest_asyncio

# Allow re-entrancy into the Jupyter event loop
nest_asyncio.apply()

def run_async(coro):
    """
    Run an async coroutine from a synchronous notebook cell.

    This is used by:
      - ui_add_note(...) in the chat UI callbacks
      - lablog_generate_and_evaluate(...)
      - the demo / test cells

    In Kaggle notebooks the event loop is already running, so we use
    nest_asyncio + run_until_complete on the existing loop.
    """
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


# PDF export dependencies
!pip install -q reportlab

from textwrap import wrap
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from pathlib import Path
from datetime import datetime

# ==========================
# Improved PDF Export Helper (wrapped text)
# ==========================

import markdown
import re
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# PDF support flag (must run BEFORE export_report_to_pdf)
from pathlib import Path
from datetime import datetime
import shutil

# Try to import reportlab for simple PDF generation
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False


def _markdown_to_lines(md_text: str) -> list[str]:
    """
    Very simple markdown â†’ plain text converter:
    - strips leading '#' from headings
    - keeps blank lines
    - leaves everything else as-is
    """
    lines: list[str] = []
    for raw in md_text.splitlines():
        line = raw.rstrip()
        stripped = line.lstrip()
        if stripped.startswith("#"):
            # drop leading hash markers
            stripped = stripped.lstrip("#").strip()
            lines.append(stripped)
        else:
            lines.append(line)
    return lines


from pathlib import Path

# Root directory for all experiments
EXPERIMENTS_ROOT = Path("experiments")
EXPERIMENTS_ROOT.mkdir(exist_ok=True)

import textwrap

# --- precise wrapping helpers for ReportLab ---------------------------

def _wrap_line_for_canvas(line: str,
                          c: canvas.Canvas,
                          font_name: str,
                          font_size: int,
                          max_width: float) -> list[str]:
    """
    Wrap a single logical line so each returned sub-line fits into max_width
    according to the actual font metrics.
    """
    words = line.split()
    if not words:
        return [""]

    wrapped: list[str] = []
    current = words[0]

    for w in words[1:]:
        candidate = current + " " + w
        if c.stringWidth(candidate, font_name, font_size) <= max_width:
            current = candidate
        else:
            wrapped.append(current)
            current = w

    wrapped.append(current)

    # Handle very long single words (e.g. long URLs) that still overflow
    final_lines: list[str] = []
    for part in wrapped:
        if c.stringWidth(part, font_name, font_size) <= max_width:
            final_lines.append(part)
            continue

        # Hard-wrap within the word
        chunk = ""
        for ch in part:
            candidate = chunk + ch
            if c.stringWidth(candidate, font_name, font_size) <= max_width:
                chunk = candidate
            else:
                if chunk:
                    final_lines.append(chunk)
                chunk = ch
        if chunk:
            final_lines.append(chunk)

    return final_lines


def export_report_to_pdf(experiment_id: str) -> Path:
    """
    Convert experiments/<id>/report.md into a simple text-style PDF.

    The PDF is written to experiments/<id>/Report_<id>_<timestamp>.pdf
    and a copy is placed in /kaggle/working/ so Kaggle exposes it in Outputs.

    Returns the /kaggle/working path for use with FileLink.
    """
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError(
            "reportlab is not available in this environment; "
            "cannot export PDF. The markdown report is still saved."
        )

    exp_dir = EXPERIMENTS_ROOT / experiment_id
    md_file = exp_dir / "report.md"
    if not md_file.exists():
        raise FileNotFoundError(f"Report not found: {md_file}")

    md_text = md_file.read_text(encoding="utf-8")
    lines = _markdown_to_lines(md_text)

    # Timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"Report_{experiment_id}_{timestamp}.pdf"

    # Path inside experiment folder
    pdf_path = exp_dir / filename

    # Basic PDF layout
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    width, height = A4
    margin = 50
    x = margin
    y = height - margin

    font_name = "Helvetica"
    font_size = 10
    line_spacing = 14
    c.setFont(font_name, font_size)

    max_text_width = width - 2 * margin

    for line in lines:
        # Preserve blank lines
        if not line.strip():
            y -= line_spacing
            if y < margin:
                c.showPage()
                c.setFont(font_name, font_size)
                y = height - margin
            continue

        wrapped_lines = _wrap_line_for_canvas(
            line, c, font_name, font_size, max_text_width
        )

        for wline in wrapped_lines:
            if y < margin:
                c.showPage()
                c.setFont(font_name, font_size)
                y = height - margin
            c.drawString(x, y, wline)
            y -= line_spacing

    c.showPage()
    c.save()

    # Copy to /kaggle/working so Kaggle will surface it
    top_level_copy = Path("/kaggle/working") / filename
    shutil.copy(pdf_path, top_level_copy)

    return top_level_copy






import asyncio
from typing import Dict, Any, Optional

async def lablog_add_note(
    experiment_id: str,
    content_type: str,
    raw_content: str,
    question_topic: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Ingest a single free-form note into the experiment log.

    Args:
        experiment_id: Experiment identifier.
        content_type: One of the types supported by run_ingestion:
            'note', 'code', 'data_summary', 'file_description',
            'image_description', 'audio_transcript', 'gap_answer'.
        raw_content: Free-form text describing goals, setup, actions,
                     observations, etc.
        question_topic: Optional topic if this is answering an evaluator question.

    Returns:
        dict with:
            - "events_added": number of events created
            - "total_events": event count after saving
    """
    # Ensure experiment exists
    log = init_experiment(
        experiment_id=experiment_id,
        title=load_log(experiment_id).metadata.get("title", experiment_id)
        if experiment_id in os.listdir("experiments")
        else experiment_id,
        user="jason",
    )

    print(f"\nğŸ“¥ Ingesting input for '{experiment_id}' ({content_type}):")
    print(raw_content)
    print()

    ingestion_result = await run_ingestion(
        experiment_id=experiment_id,
        content_type=content_type,
        raw_content=raw_content,
        question_topic=question_topic,
    )

    events = ingestion_result.get("events", [])
    print(f"â�¡ï¸�  Ingestion produced {len(events)} event(s).")

    save_summary = save_experiment_events(experiment_id, events)
    print(
        f"ğŸ’¾ Saved events. Total event count now: {save_summary['event_count']}"
    )

    return {
        "events_added": len(events),
        "total_events": save_summary["event_count"],
    }


async def lablog_generate_and_evaluate(
    experiment_id: str,
) -> Dict[str, Any]:
    """
    Generate a draft report for the given experiment and evaluate it.

    Returns:
        dict with:
            - "draft_report"
            - "evaluation"
    """
    print(f"\nğŸ“� Generating draft report for '{experiment_id}'...")
    draft_report = await run_report(experiment_id)
    draft_result = save_experiment_report(experiment_id, draft_report)
    print(f"ğŸ’¾ Draft report saved at: {draft_result['report_path']}")

    print("\nğŸ§ª Evaluating draft report...")
    evaluation = await run_evaluation(experiment_id, draft_report)
    print("ğŸ“Š Evaluation scores:", evaluation.get("scores", {}))
    print("ğŸ’¬ Overall comment:", evaluation.get("overall_comment", ""))

    questions = evaluation.get("questions_for_user", [])
    if questions:
        print("\nâ�“ Evaluator follow-up questions:")
        for q in questions:
            print(f"- {q['id']} ({q['topic']}): {q['question']}")
    else:
        print("\nâœ… Evaluator did not request additional details.")

    return {
        "draft_report": draft_report,
        "evaluation": evaluation,
    }


print("âœ… Interactive helpers lablog_add_note() and lablog_generate_and_evaluate() ready.")



# Chat-style UI for LabLog:
# - Scrollable log
# - Chatty replies
# - Evaluator questions inline as tasks (dropdown shows question text)
# - PDF export with clickable download link

import ipywidgets as widgets
from IPython.display import display, HTML
from pathlib import Path

# Track latest evaluator questions by ID
pending_questions = {}  # id -> question dict


# -------------------------------------------------------------
# Experiment ID helper
# -------------------------------------------------------------
def generate_new_experiment_id() -> str:
    base = Path("experiments")
    base.mkdir(exist_ok=True)
    existing = [
        p.name for p in base.iterdir()
        if p.is_dir() and p.name.startswith("exp_")
    ]
    n = 1
    while f"exp_{n:03d}" in existing:
        n += 1
    return f"exp_{n:03d}"


# -------------------------------------------------------------
# Widgets
# -------------------------------------------------------------
new_exp_button = widgets.Button(
    description="Start New Experiment",
    button_style="success",
    layout={"width": "220px"},
)

status_label = widgets.Label(value="No experiment active.")

exp_id_box = widgets.Text(
    value="",
    description="Experiment ID:",
    placeholder="exp_001",
    layout={"width": "300px"},
)

pdf_button = widgets.Button(
    description="Export report to PDF",
    button_style="warning",
    layout={"width": "220px"},
)

# Raw output widget
output = widgets.Output(layout={"white_space": "pre-wrap"})

# Scrollable log container
log_panel = widgets.VBox(
    [output],
    layout=widgets.Layout(
        border="1px solid #ccc",
        padding="8px",
        height="260px",
        overflow_y="auto",
    ),
)

# Dropdown for answering evaluator questions
question_dropdown = widgets.Dropdown(
    options=[("None (normal mode: no open questions)", "")],
    value="",
    description="Question:",
    disabled=True,  # becomes enabled when questions exist
    layout={"width": "100%"},
)

input_box = widgets.Textarea(
    value="",
    description="Content:",
    placeholder="Describe goal, setup, actions, observations, or answer a question...",
    layout={"width": "100%", "height": "120px"},
)

send_button = widgets.Button(
    description="Send to LabLog",
    button_style="primary",
    layout={"width": "180px"},
)

report_button = widgets.Button(
    description="Generate + Evaluate report",
    button_style="info",
    layout={"width": "260px"},
)


# -------------------------------------------------------------
# Small helpers
# -------------------------------------------------------------
def scroll_log_to_bottom():
    """Best-effort scroll of the log panel; never raises if unsupported."""
    try:
        # Some widget versions support this
        log_panel.scroll_to_bottom()
    except Exception:
        # Fallback: do nothing rather than crash
        pass


def update_status(exp_id: str):
    try:
        log = load_log(exp_id)
        n = len(log.events)
        status_label.value = f"Experiment {exp_id}: {n} event(s) logged."
    except FileNotFoundError:
        status_label.value = f"Experiment {exp_id}: (no log yet)."


def _refresh_question_dropdown():
    """Update dropdown from current pending_questions dict."""
    if not pending_questions:
        question_dropdown.options = [("None (normal mode: no open questions)", "")]
        question_dropdown.value = ""
        question_dropdown.disabled = True
        return

    opts = [("None (normal mode)", "")]
    for qid, q in pending_questions.items():
        qtext = q["question"].strip().replace("\n", " ")
        if len(qtext) > 80:
            qtext = qtext[:77] + "..."
        label = f"{qid} ({q['topic']}): {qtext}"
        opts.append((label, qid))
    question_dropdown.options = opts
    question_dropdown.value = ""
    question_dropdown.disabled = False


def set_question_options(questions):
    """Set pending questions from evaluator output and refresh dropdown."""
    global pending_questions
    pending_questions = {q["id"]: q for q in questions}
    _refresh_question_dropdown()


async def ui_add_note(exp_id: str, text: str, question_topic: str | None = None):
    """Ingest a note or an answer to an evaluator question.

    This helper:
      - ensures the experiment exists
      - calls the ingestion agent
      - saves new events
      - returns how many events were added and the new total
    """
    # Ensure experiment exists
    init_experiment(experiment_id=exp_id, title=exp_id, user="jason")

    # Call ingestion agent with the right content type
    if question_topic:
        ingestion_result = await run_ingestion(
            experiment_id=exp_id,
            content_type="gap_answer",
            raw_content=text,
            question_topic=question_topic,
        )
    else:
        ingestion_result = await run_ingestion(
            experiment_id=exp_id,
            content_type="auto",
            raw_content=text,
        )

    events = ingestion_result.get("events", [])

    # Save events (this may return None in the current implementation)
    save_experiment_events(exp_id, events)

    # Re-load log to get the true total event count
    log_after = load_log(exp_id)
    total_events = len(log_after.events)

    return {
        "events": events,
        "events_added": len(events),
        "total_events": total_events,
    }



# -------------------------------------------------------------
# Callbacks
# -------------------------------------------------------------
def on_new_exp_clicked(b):
    """Create a fresh experiment ID, initialise its folder and log to the UI."""
    set_question_options([])  # clear any old questions

    with output:
        print("â”€" * 60)
        print("ğŸ§ª Creating a new experiment...")

    try:
        exp_id = generate_new_experiment_id()

        # Make sure the experiment is initialised on disk
        init_experiment(
            experiment_id=exp_id,
            title=f"Experiment {exp_id}",
            user="jason",
        )

        # Update widgets
        exp_id_box.value = exp_id
        update_status(exp_id)

        with output:
            print(f"ğŸ�‰ New experiment created: {exp_id}")
            print(f"Folder: experiments/{exp_id}")
    except Exception:
        import traceback
        with output:
            print("âš ï¸� Error while creating a new experiment:")
            traceback.print_exc()

    scroll_log_to_bottom()


def on_send_clicked(b):
    exp_id = exp_id_box.value.strip()
    if not exp_id:
        with output:
            print("âš ï¸� Please create or enter an experiment ID first.")
        scroll_log_to_bottom()
        return

    text = input_box.value.strip()
    if not text:
        with output:
            print("âš ï¸� Content is empty.")
        scroll_log_to_bottom()
        return

    # Are we answering a specific evaluator question?
    selected_qid = question_dropdown.value or ""
    question_topic = None
    if selected_qid:
        q = pending_questions.get(selected_qid)
        if q:
            question_topic = q["topic"]

    # Show user message in log
    with output:
        print("â”€" * 60)
        if question_topic:
            print(f"ğŸ‘¤ You [{exp_id}] (answer to '{question_topic}'):")
        else:
            print(f"ğŸ‘¤ You [{exp_id}]:")
        print(text)
    scroll_log_to_bottom()

    # Run ingestion + save
    try:
        result = run_async(ui_add_note(exp_id, text, question_topic=question_topic))
    except Exception:
        import traceback
        with output:
            print("âš ï¸� Error while sending note to LabLog:")
            traceback.print_exc()
        scroll_log_to_bottom()
        return  # keep content box intact on error

    added = result["events_added"]
    total = result["total_events"]
    events = result["events"]
    unique_types = sorted(set(e["type"] for e in events))

    # Chat-style reply
    with output:
        if added == 0:
            print("ğŸ¤– LabLog: I could not confidently turn that into a structured event.")
            print("   Try making the goal, setup change, observation, or answer more explicit.")
        else:
            if len(unique_types) == 1:
                etype = unique_types[0]
                if etype == "goal":
                    msg = "Got it, I have stored this as a goal for the experiment."
                elif etype == "background":
                    msg = "Stored as background context for the experiment."
                elif etype == "setup_detail":
                    msg = "OK, stored as a setup / hardware detail."
                elif etype == "firmware_change":
                    msg = "OK, logged as a firmware / configuration change."
                elif etype == "procedure":
                    msg = "Noted, stored as part of the experimental procedure."
                elif etype == "data_capture":
                    msg = "Captured, stored as a data capture / analysis step."
                elif etype == "observation":
                    msg = "Thanks, stored as an observation / result."
                elif etype == "next_step":
                    msg = "Logged as a next step / TODO."
                else:
                    msg = f"Stored as a general '{etype}' event."
                print(f"ğŸ¤– LabLog: {msg}")
            else:
                msg_types = ", ".join(unique_types)
                print(f"ğŸ¤– LabLog: I split that into several events: {msg_types}.")

        if question_topic:
            print(f"ğŸ§© Answer recorded for evaluator topic: {question_topic}")

        print(f"ğŸ“¦ Total events in {exp_id}: {total}")
    scroll_log_to_bottom()

    # After answering a question, remove it from pending list and refresh dropdown
    if selected_qid:
        pending_questions.pop(selected_qid, None)
        _refresh_question_dropdown()

    # Clear content and reset dropdown selection
    input_box.value = ""
    question_dropdown.value = ""
    update_status(exp_id)


def on_report_clicked(b):
    exp_id = exp_id_box.value.strip()
    if not exp_id:
        with output:
            print("âš ï¸� Please create or enter an experiment ID first.")
        scroll_log_to_bottom()
        return

    with output:
        print("=" * 60)
        print(f"ğŸ“„ Generating and evaluating report for '{exp_id}'...")
    scroll_log_to_bottom()

    try:
        result = run_async(lablog_generate_and_evaluate(exp_id))
    except Exception:
        import traceback
        with output:
            print("âš ï¸� Error while generating/evaluating report:")
            traceback.print_exc()
        scroll_log_to_bottom()
        return

    evaluation = result["evaluation"]
    scores = evaluation.get("scores", {})
    comment = evaluation.get("overall_comment", "")
    questions = evaluation.get("questions_for_user", [])
    regen = evaluation.get("should_regenerate_report", False)

    # Update global questions + dropdown
    set_question_options(questions)

    with output:
        print("ğŸ“Š Scores:", scores)
        print("ğŸ’¬ Comment:", comment)
        print("ğŸ”� Should regenerate:", regen)

        if questions:
            print("â�“ Evaluator questions (answer via the Question dropdown + Content box):")
            for q in questions:
                print(f"  - {q['id']} ({q['topic']}): {q['question']}")
            print("ğŸ’¡ To answer: choose a question in 'Question:' below,")
            print("   type your response in Content, then click 'Send to LabLog'.")
            print("   After answering, click 'Generate + Evaluate report' again to refresh the report.")
        else:
            print("âœ… No follow-up questions.")
        print(f"ğŸ“Œ Report saved to: experiments/{exp_id}/report.md")
    scroll_log_to_bottom()

    update_status(exp_id)


from IPython.display import FileLink
from pathlib import Path

def on_pdf_clicked(b):
    exp_id = exp_id_box.value.strip()
    if not exp_id:
        with output:
            print("âš ï¸� Please enter an experiment ID first.")
        scroll_log_to_bottom()
        return

    try:
        pdf_path = export_report_to_pdf(exp_id)   # returns a Path in /kaggle/working
        pdf_path = Path(pdf_path)

        with output:
            print("â”€" * 60)
            print(f"ğŸ“„ Report PDF exported for {exp_id}: {pdf_path}")
            print("ğŸ”— Download PDF:")

            # IMPORTANT: use the filename relative to /kaggle/working
            display(FileLink(pdf_path.name))

    except Exception:
        import traceback
        with output:
            print("âš ï¸� Error while exporting PDF:")
            traceback.print_exc()

    scroll_log_to_bottom()



# Wire up buttons
new_exp_button.on_click(on_new_exp_clicked)
send_button.on_click(on_send_clicked)
report_button.on_click(on_report_clicked)
pdf_button.on_click(on_pdf_clicked)


# -------------------------------------------------------------
# Layout: log above, input + controls below
# -------------------------------------------------------------
ui = widgets.VBox(
    [
        widgets.HBox([new_exp_button, status_label]),
        exp_id_box,
        widgets.HTML("<hr>"),
        log_panel,
        widgets.HTML("<hr>"),
        question_dropdown,
        input_box,
        widgets.HBox([send_button, report_button, pdf_button]),
    ]
)

ui


