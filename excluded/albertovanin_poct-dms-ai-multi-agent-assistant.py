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


# 5. Setup & Imports

!pip install --quiet google-adk google-genai nest_asyncio

import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import nest_asyncio
nest_asyncio.apply()

# ADK core
from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent
from google.adk.tools.agent_tool import AgentTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

# Observability / events
from google.adk.events import Event, EventActions

# Gen AI SDK (Gemini)
from google import genai
from google.genai import types as genai_types

# Basic utilities
import textwrap
import json
import time


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = api_key


# 6. Environment configuration & Gemini client

API_KEY = os.environ.get("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is not set. "
        "Please configure it using Kaggle Secrets."
    )

# Create a Gen AI client (using the GenAI SDK)
client = genai.Client(api_key=API_KEY)

DEFAULT_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "gemini-embedding-001"

print("ðŸŸ¢ Gemini Client initialized correctly")


# Inspect available input datasets and files
import os

print("Dirs in /kaggle/input:")
print(os.listdir("/kaggle/input"))

print("\nRecursive listing:")
for root, dirs, files in os.walk("/kaggle/input"):
    print(root)
    for f in files:
        print("  -", f)


!pip install --quiet pypdf

from pypdf import PdfReader

def load_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

POCT_MANUAL_PATH = "/kaggle/input/poct-manual/poct_manual.pdf"
DMS_MANUAL_PATH  = "/kaggle/input/poct-dms-manuals2/dms_manual.pdf"

poct_manual_text = load_pdf_text(POCT_MANUAL_PATH)
dms_manual_text  = load_pdf_text(DMS_MANUAL_PATH)

print("Length POCT:", len(poct_manual_text))
print("Length DMS:", len(dms_manual_text))


# 8. Chunking manuals & creating embeddings with Gemini

def chunk_text(text: str, max_tokens: int = 512) -> List[str]:
    # simple character-based chunking as baseline
    chunks = []
    step = max_tokens * 4  # rough approximation chars vs tokens
    for i in range(0, len(text), step):
        chunks.append(text[i:i+step])
    return chunks

poct_chunks = chunk_text(poct_manual_text)
dms_chunks = chunk_text(dms_manual_text)

print(f"POCT chunks: {len(poct_chunks)}, DMS chunks: {len(dms_chunks)}")

def embed_chunks(chunks: List[str]) -> List[List[float]]:
    embeddings: List[List[float]] = []
    for ch in chunks:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=ch,
        )
        # Newer google-genai returns a list of embeddings
        if hasattr(response, "embedding"):
            # older style (just in case)
            vec = response.embedding.values
        elif hasattr(response, "embeddings"):
            # newer style: response.embeddings[0].values
            vec = response.embeddings[0].values
        else:
            raise ValueError(f"Unexpected embedding response format: {response}")
        embeddings.append(vec)
    return embeddings

# In a real project you may want to cache these to disk for speed.
poct_embeddings = embed_chunks(poct_chunks)
dms_embeddings = embed_chunks(dms_chunks)

len(poct_embeddings), len(dms_embeddings)


# 9. Simple in-memory vector store and retrieval

import math

@dataclass
class DocumentChunk:
    text: str
    embedding: List[float]
    source: str  # "POCT" or "DMS"

def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

vector_store: List[DocumentChunk] = []

for ch, emb in zip(poct_chunks, poct_embeddings):
    vector_store.append(DocumentChunk(text=ch, embedding=emb, source="POCT"))

for ch, emb in zip(dms_chunks, dms_embeddings):
    vector_store.append(DocumentChunk(text=ch, embedding=emb, source="DMS"))

def retrieve_relevant_chunks(query: str, k: int = 5) -> List[DocumentChunk]:
    q_emb = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
    ).embedding.values

    scored = [
        (cosine_similarity(q_emb, doc.embedding), doc)
        for doc in vector_store
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored[:k]]


# 10. Custom tools: mocked POCT / DMS operations

def get_device_status(device_id: str) -> str:
    """Mock tool: returns a simulated device status."""
    # In a real scenario, this would call a DMS/POCT API.
    return f"Device {device_id} is connected and reports status: OK, last QC passed."

def list_recent_poct_errors(device_id: str) -> str:
    """Mock tool: returns recent errors for a POCT device."""
    return f"Recent errors for {device_id}: E-452 (no QC available), E-101 (sample clotting)."

def list_operator_flags(operator_id: str) -> str:
    """Mock tool: returns DMS flags associated with an operator."""
    return f"Operator {operator_id} has 1 active flag: 'Password expired policy'."

def check_qc_status(device_group: str) -> str:
    """Mock tool: returns QC status for a device group."""
    return f"Device group {device_group}: last daily QC passed, weekly QC pending for 2 devices."

def list_dms_alerts() -> str:
    """Mock tool: returns open DMS alerts."""
    return "Open DMS alerts: 3 connectivity alerts, 1 QC overdue, 1 operator locked out."

def check_device_connection(device_id: str) -> str:
    """Mock tool: returns connectivity status."""
    return f"Device {device_id} is ONLINE via TCP/IP, last heartbeat 2 minutes ago."


# 11. RAG documentation tool for ADK

def rag_lookup(query: str, max_chunks: int = 5) -> str:
    """Simple RAG lookup over POCT + DMS manuals."""
    relevant = retrieve_relevant_chunks(query, k=max_chunks)
    joined = "\n\n---\n\n".join(
        [f"[{doc.source}] {doc.text[:1000]}" for doc in relevant]
    )
    return joined


# 12. Memory & session services

session_service = InMemorySessionService()

# For this notebook we will simulate memory with a simple dict keyed by session_id
GLOBAL_MEMORY: Dict[str, Dict[str, Any]] = {}

def get_memory(session_id: str) -> Dict[str, Any]:
    return GLOBAL_MEMORY.setdefault(session_id, {})

def update_memory(session_id: str, key: str, value: Any) -> None:
    mem = GLOBAL_MEMORY.setdefault(session_id, {})
    mem[key] = value


# 13. Observability: simple logging & metrics

OBSERVABILITY_LOGS: List[Dict[str, Any]] = []

def log_event(session_id: str, agent_name: str, event_type: str, payload: Dict[str, Any]) -> None:
    OBSERVABILITY_LOGS.append({
        "ts": time.time(),
        "session_id": session_id,
        "agent": agent_name,
        "event_type": event_type,
        "payload": payload,
    })

def summarize_observability() -> Dict[str, Any]:
    total_events = len(OBSERVABILITY_LOGS)
    by_agent: Dict[str, int] = {}
    for ev in OBSERVABILITY_LOGS:
        by_agent[ev["agent"]] = by_agent.get(ev["agent"], 0) + 1
    return {
        "total_events": total_events,
        "events_by_agent": by_agent,
    }


# 14. Define RAG Documentation Agent (LlmAgent using rag_lookup tool)

def documentation_tool(query: str) -> str:
    """Tool wrapper around RAG lookup."""
    return rag_lookup(query)

documentation_agent = LlmAgent(
    model=DEFAULT_MODEL,
    name="documentation_agent",
    description="Retrieves and explains documentation from POCT and DMS manuals.",
    instruction=textwrap.dedent("""
        You are a documentation assistant for POCT instruments and a DMS platform.
        When the user or another agent asks a question, you should:
        1. Use the `documentation_tool` to retrieve relevant manual sections.
        2. Read and synthesize the content.
        3. Answer clearly, referencing POCT vs DMS context when relevant.
    """),
    tools=[documentation_tool],
)


# 15. POCT Troubleshooting Agent

def poct_status_tool(device_id: str) -> str:
    return get_device_status(device_id)

def poct_errors_tool(device_id: str) -> str:
    return list_recent_poct_errors(device_id)

poct_agent = LlmAgent(
    model=DEFAULT_MODEL,
    name="poct_agent",
    description="Helps troubleshoot POCT device errors and alarms.",
    instruction=textwrap.dedent("""
        You are a POCT troubleshooting expert.
        When the user describes an error or alarm on a POCT device:
        1. Clarify which device and error code are involved.
        2. Use tools like `poct_status_tool`, `poct_errors_tool`, and `documentation_tool`.
        3. Propose clear, step-by-step troubleshooting instructions.
        4. When useful, reference QC status or connectivity and suggest escalation criteria.
    """),
    tools=[
        poct_status_tool,
        poct_errors_tool,
        documentation_tool,
    ],
)


# 16. DMS Workflow Agent

def dms_operator_flags_tool(operator_id: str) -> str:
    return list_operator_flags(operator_id)

def dms_qc_status_tool(device_group: str) -> str:
    return check_qc_status(device_group)

def dms_alerts_tool() -> str:
    return list_dms_alerts()

dms_workflow_agent = LlmAgent(
    model=DEFAULT_MODEL,
    name="dms_workflow_agent",
    description="Supports daily DMS workflows for POCT operations.",
    instruction=textwrap.dedent("""
        You are a DMS workflow assistant for a POCT management system.
        You help with:
        - operator management
        - QC plans and schedules
        - flags and alerts
        Use tools to inspect operator flags, QC status, and current DMS alerts.
        Always respond with clear operational guidance suitable for nurses and technicians.
    """),
    tools=[
        dms_operator_flags_tool,
        dms_qc_status_tool,
        dms_alerts_tool,
        documentation_tool,
    ],
)


# 17. DMS Configuration Agent

def dms_connection_tool(device_id: str) -> str:
    return check_device_connection(device_id)

dms_config_agent = LlmAgent(
    model=DEFAULT_MODEL,
    name="dms_config_agent",
    description="Helps with DMS configuration issues (connectivity, profiles, LIS mappings).",
    instruction=textwrap.dedent("""
        You are a configuration expert for a POCT DMS platform.
        You help users diagnose and resolve configuration issues like:
        - device connectivity
        - device groups
        - profiles and policies
        - LIS mappings
        Use tools to check connectivity and look up documentation.
        Provide clear, step-by-step configuration guidance.
    """),
    tools=[
        dms_connection_tool,
        documentation_tool,
    ],
)


# 18. Orchestrator Agent (router-style LlmAgent)

# Wrap sub-agents as tools so that Orchestrator can delegate via Agent-as-a-Tool pattern
poct_agent_tool = AgentTool(agent=poct_agent)
dms_workflow_agent_tool = AgentTool(agent=dms_workflow_agent)
dms_config_agent_tool = AgentTool(agent=dms_config_agent)
documentation_agent_tool = AgentTool(agent=documentation_agent)

orchestrator_agent = LlmAgent(
    model=DEFAULT_MODEL,
    name="orchestrator_agent",
    description="Routes healthcare POCT/DMS questions to specialized agents.",
    instruction=textwrap.dedent("""
        You are the main orchestrator for a POCT & DMS multi-agent system.
        The user is a nurse, technician, or POCT coordinator.

        Your responsibilities:
        1. Understand if the request is about:
           - POCT device troubleshooting
           - DMS workflow
           - DMS configuration
           - Documentation / manuals
        2. Call the appropriate agent tool:
           - `poct_agent_tool`
           - `dms_workflow_agent_tool`
           - `dms_config_agent_tool`
           - `documentation_agent_tool`
        3. Summarize the final answer for the user.

        Always keep responses concise, safe, and clinically oriented.
    """),
    tools=[
        poct_agent_tool,
        dms_workflow_agent_tool,
        dms_config_agent_tool,
        documentation_agent_tool,
    ],
)


# 19. Runner & helper to talk to the multi-agent system

import asyncio
from google.genai import types as genai_types

# Define a fixed app / user / session for this notebook demo
APP_NAME = "poct_dms_multi_agent_assistant"
USER_ID = "demo_user"
SESSION_ID = "session_poct_dms_demo"

# Use existing event loop (works in Kaggle/Colab)
loop = asyncio.get_event_loop()

# Create a session once, using the InMemorySessionService (async API)
try:
    session = loop.run_until_complete(
        session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID,
        )
    )
    print("Session created:", SESSION_ID)
except Exception as e:
    # If it already exists, we just reuse it
    print("Session already exists or could not be created, continuing anyway:", e)
    # We don't strictly need the session object for the rest of this notebook
    session = None

# Create the Runner with root agent + session service
runner = Runner(
    agent=orchestrator_agent,
    app_name=APP_NAME,
    session_service=session_service,
)

def run_interaction(user_text: str) -> str:
    """
    Send user_text through the orchestrator and capture observability + memory.
    Uses a single logical session (SESSION_ID) for the notebook demo.
    """
    # Build GenAI content
    content = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=user_text)]
    )

    # Log user message
    log_event(SESSION_ID, "orchestrator_agent", "user_message", {"text": user_text})

    # Run the agent via ADK Runner â€“ this returns a stream (generator) of Events
    events = runner.run(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=content,
    )

    answer_text = ""

    # Extract final response text from the event stream
    for ev in events:
        try:
            if ev.is_final_response():
                if getattr(ev, "content", None) and getattr(ev.content, "parts", None):
                    for part in ev.content.parts:
                        if hasattr(part, "text") and part.text:
                            answer_text += part.text
        except Exception:
            # Best-effort parsing, ignore unexpected event shapes
            pass

    if not answer_text:
        answer_text = "No response generated."

    # Store simple memory
    update_memory(SESSION_ID, "last_query", user_text)
    update_memory(SESSION_ID, "last_answer", answer_text)

    # Log agent response
    log_event(SESSION_ID, "orchestrator_agent", "agent_response", {"text": answer_text})

    return answer_text


# 21. Demo 1 â€“ POCT error troubleshooting

user_message = (
    "Our cobas b123 shows error E-452 saying no QC is available. "
    "What should I do to resolve this on the device?"
)

answer = run_interaction(user_message)
print("=== Agent answer (POCT error) ===\n")
print(answer)


# 22. Demo 2 â€“ DMS workflow issue

user_message = (
    "An operator cannot log in to the DMS because their credentials are blocked. "
    "How can I check operator flags and fix this?"
)

answer = run_interaction(user_message)
print("=== Agent answer (DMS workflow) ===\n")
print(answer)


# 23. Demo 3 â€“ DMS configuration issue

user_message = (
    "One POCT device is not sending results to the DMS. "
    "How can I check the connection and what configuration should I verify?"
)

answer = run_interaction(user_message)
print("=== Agent answer (DMS configuration) ===\n")
print(answer)


# 24. Observability & Analytics

print("=== Observability summary ===")
print(json.dumps(summarize_observability(), indent=2))

print("\n=== Example of raw logs (first 5 events) ===")
for ev in OBSERVABILITY_LOGS[:5]:
    print(ev)

