"""
Multi-Agent AI Competition Assistant — FastAPI server (single-file)

Features:
- Receives user messages via /message
- Sends context to OpenAI (chat completions) and supports function-calling style tools
- Implements two example "tools": task_manager and calendar_stub
- Simple SQLite memory for short-term conversation memory
- Includes an SSL compatibility shim so the module can import even in environments without the built-in ssl module

Install:
pip install fastapi uvicorn openai aiosqlite python-dotenv

Run:
export OPENAI_API_KEY="sk-..."
uvicorn ai_agent_server:app --reload --port 8000

Notes:
- Replace or extend the tool implementations with real integrations (Google Calendar API, SMTP, CRMs, etc.)
- This is a minimal, educational starter — production requires auth, rate-limiting, error handling, secure key management.
- In restricted sandboxes without an ssl module, network access (including real OpenAI calls) will not work; the shim only allows imports and local logic to run.
"""

import os
import json
import asyncio
import sys
import types
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

# ---------------------------------------------------------------------------
# SSL compatibility shim
# ---------------------------------------------------------------------------
# Some sandboxed environments (like the one where you saw
# `ModuleNotFoundError: No module named 'ssl'`) don't ship the builtin
# `ssl` module. Libraries like FastAPI / AnyIO import `ssl` at module import
# time, which causes the crash.
#
# Here we ensure a minimal "ssl" module is present so imports succeed.
# In real deployments, Python will provide a real ssl implementation and
# this shim is effectively a no-op.
try:  # pragma: no cover - environment dependent
    import ssl as _ssl  # type: ignore[assignment]  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - sandbox-only path
    _ssl = types.ModuleType("ssl")
    sys.modules["ssl"] = _ssl

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import aiosqlite
import openai
from dotenv import load_dotenv

# API key loaded directly without environment requirement
openai.api_key = "AIzaSyBuePB3psFPtkaZMHyJBUDpxtTN0fz10ZU"

DB_PATH = "agent_memory.db"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler: initialize DB on startup."""
    try:
        await init_db()
    except Exception as e:
        # For sandbox/notebook environments, fail softly.
        print(f"[WARN] DB init failed: {e}")
    yield
    # Add any teardown logic here if needed.

app = FastAPI(title="Multi-Agent AI Competition Assistant", lifespan=lifespan)

# -----------------------------
# Simple DB helpers
# -----------------------------
async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await db.commit()


# Initialize DB at FastAPI startup instead of import time.
# This avoids event-loop issues in notebooks / restricted sandboxes and
# ensures the coroutine is awaited properly in a running server.




async def save_message(role: str, content: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO messages (role, content) VALUES (?, ?)",
            (role, content),
        )
        await db.commit()


async def get_recent_messages(limit: int = 10) -> List[Dict[str, str]]:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = await cur.fetchall()
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]


# -----------------------------
# Models
# -----------------------------
class UserMessage(BaseModel):
    user_id: Optional[str] = "default_user"
    message: str


class AgentResponse(BaseModel):
    reply: str
    tool_used: Optional[str] = None
    tool_result: Optional[Dict[str, Any]] = None


# -----------------------------
# Tool implementations (stubs)
# -----------------------------

def tool_task_manager_create(args: Dict[str, Any]) -> Dict[str, Any]:
    """Create a task in a (simulated) task manager.

    Expected args:
        {
            "title": "Write report",
            "due": "2025-11-30"  # optional
        }
    """
    title = args.get("title")
    due = args.get("due")
    if not title:
        return {"status": "error", "error": "title is required"}
    task_id = abs(hash(title + str(due))) % 100_000
    return {"status": "created", "task_id": task_id, "title": title, "due": due}


def tool_calendar_create_event(args: Dict[str, Any]) -> Dict[str, Any]:
    """Create an event in a calendar (stub).

    Expected args:
        {
            "title": "Standup",
            "start": "2025-11-24T09:00:00Z",
            "end": "2025-11-24T09:15:00Z",
            "location": "Zoom"  # optional
        }
    """
    required = ["title", "start", "end"]
    missing = [key for key in required if key not in args or args[key] in (None, "")]
    if missing:
        return {"status": "error", "error": f"Missing fields: {', '.join(missing)}"}
    event_id = abs(hash(json.dumps(args, sort_keys=True))) % 100_000
    return {"status": "event_created", "event_id": event_id, "details": args}


# Tool registry maps "function" name to executor
TOOL_REGISTRY: Dict[str, Any] = {
    "task_manager.create_task": tool_task_manager_create,
    "calendar.create_event": tool_calendar_create_event,
}


# -----------------------------
# OpenAI interaction helpers
# -----------------------------

def build_function_schemas() -> List[Dict[str, Any]]:
    """Return function schema list for OpenAI function-calling.

    Keep schemas small and explicit so the model can call them when needed.
    """
    return [
        {
            "name": "task_manager.create_task",
            "description": "Create a task with a title and optional due date",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short task title"},
                    "due": {
                        "type": "string",
                        "description": "Due date in YYYY-MM-DD format (optional)",
                    },
                },
                "required": ["title"],
            },
        },
        {
            "name": "calendar.create_event",
            "description": "Create a calendar event with title, start and end times",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "start": {"type": "string", "description": "ISO timestamp"},
                    "end": {"type": "string", "description": "ISO timestamp"},
                    "location": {"type": "string"},
                },
                "required": ["title", "start", "end"],
            },
        },
    ]


def call_openai_with_functions(
    system_prompt: str,
    user_input: str,
    recent_messages: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Synchronous helper that calls OpenAI with function schemas.

    This is run in a thread pool from the async FastAPI handler so it
    doesn't block the event loop.
    """
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]

    # append recent messages from memory as assistant/user context
    for m in recent_messages:
        messages.append({"role": m["role"], "content": m["content"]})

    messages.append({"role": "user", "content": user_input})

    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        functions=build_function_schemas(),
        function_call="auto",
        max_tokens=600,
        temperature=0.2,
    )
    return resp


# -----------------------------
# API endpoint: message -> agent
# -----------------------------

@app.post("/message", response_model=AgentResponse)
async def handle_message(payload: UserMessage) -> AgentResponse:
    # save user message
    await save_message("user", payload.message)

    recent = await get_recent_messages(limit=8)

    system_prompt = (
        "You are a helpful AI agent assistant. You can call external tools to create "
        "tasks or calendar events. When appropriate, call the tool via the function "
        "schema. Otherwise, answer directly."
    )

    # call OpenAI with function schemas enabled (off the main event loop)
    loop = asyncio.get_event_loop()
    try:
        raw = await loop.run_in_executor(
            None,
            lambda: call_openai_with_functions(system_prompt, payload.message, recent),
        )
    except Exception as e:  # pragma: no cover - external service
        raise HTTPException(status_code=500, detail=str(e))

    # raw is an OpenAI response object; navigate its content
    choices = raw.get("choices", [])
    if not choices:
        raise HTTPException(status_code=500, detail="no choices from model")

    first = choices[0]
    message = first.get("message", {}) or {}

    # check for function call
    function_call = message.get("function_call")
    if function_call:
        fn_name = function_call.get("name")
        raw_args = function_call.get("arguments") or "{}"
        try:
            args = json.loads(raw_args)
        except Exception:
            args = {}

        # Execute the local tool if available
        tool_exec = TOOL_REGISTRY.get(fn_name or "")
        if not tool_exec:
            tool_result: Dict[str, Any] = {"error": f"Tool {fn_name} not found"}
        else:
            tool_result = tool_exec(args)

        # Save tool result into memory
        await save_message("assistant", f"[tool:{fn_name}] {json.dumps(tool_result)}")

        # send the tool result back to the model so it can craft a final user-visible reply
        followup_messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]
        for m in recent:
            followup_messages.append({"role": m["role"], "content": m["content"]})
        followup_messages.append({"role": "user", "content": payload.message})
        followup_messages.append(
            {"role": "function", "name": fn_name, "content": json.dumps(tool_result)}
        )

        # final model response (the model will see the tool output and produce a reply)
        final_resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=followup_messages,
            max_tokens=400,
            temperature=0.1,
        )
        final_text = final_resp["choices"][0]["message"]["content"]

        await save_message("assistant", final_text)

        return AgentResponse(reply=final_text, tool_used=fn_name, tool_result=tool_result)

    # No function call — just reply
    reply_text = message.get("content") or ""
    await save_message("assistant", reply_text)
    return AgentResponse(reply=reply_text)


# -----------------------------
# Simple demo endpoints
# -----------------------------

@app.get("/memory/recent")
async def memory_recent(limit: int = 10) -> List[Dict[str, str]]:
    return await get_recent_messages(limit=limit)


@app.get("/tools")
async def list_tools() -> Dict[str, List[str]]:
    return {"tools": list(TOOL_REGISTRY.keys())}


# -----------------------------
# Lightweight internal tests / self-checks
# -----------------------------

# These are simple test cases you can run manually (they are *not* executed
# automatically on import, so they won't interfere with your server).


def _test_tool_task_creation() -> None:
    """Basic test for task_manager.create_task tool."""
    ok = tool_task_manager_create({"title": "Test Task", "due": "2025-11-30"})
    assert ok["status"] == "created"
    assert ok["title"] == "Test Task"

    missing_title = tool_task_manager_create({})
    assert missing_title["status"] == "error"


def _test_calendar_event_creation() -> None:
    """Basic test for calendar.create_event tool."""
    ok = tool_calendar_create_event(
        {
            "title": "Demo Event",
            "start": "2025-11-24T09:00:00Z",
            "end": "2025-11-24T09:30:00Z",
            "location": "Online",
        }
    )
    assert ok["status"] == "event_created"
    assert ok["details"]["title"] == "Demo Event"

    missing = tool_calendar_create_event({"title": "Incomplete"})
    assert missing["status"] == "error"


def _test_tool_registry() -> None:
    """Ensure tools are correctly registered."""
    assert "task_manager.create_task" in TOOL_REGISTRY
    assert "calendar.create_event" in TOOL_REGISTRY


def _run_self_tests() -> None:
    """Run a subset of quick, local tests.

    This does *not* hit external services like OpenAI; it's safe to run
    even in offline or sandboxed environments.
    """
    _test_tool_task_creation()
    _test_calendar_event_creation()
    _test_tool_registry()
    print("Self-tests passed.")


if __name__ == "__main__":  # pragma: no cover - manual invocation
    _run_self_tests()


