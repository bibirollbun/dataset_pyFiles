import os
import uuid
import logging
from datetime import datetime
import asyncio
import nest_asyncio
import json
from kaggle_secrets import UserSecretsClient
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types

# Suppress warnings
logging.getLogger("google_genai.types").setLevel(logging.ERROR)
nest_asyncio.apply()




LOG_FILE = "jarvis_logs.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)

logging.info("JARVIS v0.3 Notebook starting up...")
print("JARVIS - Google ADK Demo Config Started\n")

GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"

MODEL_NAME = "gemini-2.5-flash"
MEMORY_STORE = {"entries": []}
USER_ID = "demo_user"
APP_NAME = "jarvis_demo"

logging.info(f"Model set to: {MODEL_NAME}")
logging.info(f"User: {USER_ID}, App: {APP_NAME}")
SESSION_ID = f"session_{uuid.uuid4().hex}"
logging.info(f"Generated new session_id: {SESSION_ID}")




def save_to_memory(key: str, value: str, category: str = "general") -> dict:
    """Store information in memory for later recall."""
    logging.info(f"[TOOL] save_to_memory(key={key}, category={category})")
    MEMORY_STORE["entries"].append(
        {
            "key": key,
            "value": value,
            "category": category,
            "timestamp": datetime.now().isoformat(),
        }
    )
    result = {"status": "success", "message": f"Stored: {key}={value}"}
    logging.info(f"[TOOL RESULT] save_to_memory -> {result}")
    return result


def recall_from_memory(query: str, limit: int = 5) -> dict:
    """Recall memories by keyword matching."""
    logging.info(f"[TOOL] recall_from_memory(query={query}, limit={limit})")
    entries = MEMORY_STORE.get("entries", [])
    if not entries:
        result = {"memories": [], "message": "No memories yet"}
        logging.info(f"[TOOL RESULT] recall_from_memory -> {result}")
        return result

    query_lower = query.lower()
    matches = [
        e for e in entries if query_lower in f"{e['key']} {e['value']}".lower()
    ]
    subset = matches[-limit:] or entries[-limit:]
    result = {"memories": subset, "count": len(matches)}
    logging.info(f"[TOOL RESULT] recall_from_memory -> {result}")
    return result


def execute_python_code(code: str, description: str = "") -> dict:
    """Execute Python code and return the output."""
    logging.info(f"[TOOL] execute_python_code(description={description})")
    try:
        from io import StringIO
        import sys
        import math

        old_stdout, sys.stdout = sys.stdout, StringIO()
        try:
            exec(code, {"__builtins__": __builtins__, "math": math})
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        result = {"status": "success", "output": output or "Executed"}
        logging.info(f"[TOOL RESULT] execute_python_code -> {result}")
        return result
    except Exception as e:
        result = {"status": "error", "error": str(e)}
        logging.exception("[TOOL ERROR] execute_python_code failed")
        return result


def calculate_expression(expression: str) -> dict:
    """Calculate a mathematical expression safely."""
    logging.info(f"[TOOL] calculate_expression(expression={expression})")
    try:
        import math

        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        allowed.update({"abs": abs, "round": round, "pow": pow})
        result_value = eval(expression, {"__builtins__": {}}, allowed)
        result = {"result": result_value}
        logging.info(f"[TOOL RESULT] calculate_expression -> {result}")
        return result
    except Exception as e:
        result = {"error": str(e)}
        logging.exception("[TOOL ERROR] calculate_expression failed")
        return result


def get_current_datetime() -> dict:
    """Get the current date and time."""
    logging.info("[TOOL] get_current_datetime()")
    now = datetime.now()
    result = {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "formatted": now.strftime("%A, %B %d, %Y at %I:%M %p"),
    }
    logging.info(f"[TOOL RESULT] get_current_datetime -> {result}")
    return result


TOOLS = [
    save_to_memory,
    recall_from_memory,
    execute_python_code,
    calculate_expression,
    get_current_datetime,
]

print("Loaded Tools", TOOLS)
logging.info("All tools loaded")


jarvis_worker = Agent(
    name="JARVIS",
    model=MODEL_NAME,
    description="Personal AI Assistant and executor agent.",
    instruction=(
        "You are JARVIS - a helpful AI assistant.\n"
        "You are a WORKER agent that executes tasks, calls tools when needed, "
        "and returns concise and helpful answers.\n"
        "You have access to tools: save_to_memory, recall_from_memory, "
        "execute_python_code, calculate_expression, get_current_datetime."
    ),
    tools=TOOLS,
)
planner_agent = Agent(
    name="Planner",
    model=MODEL_NAME,
    description="High-level planner and router for the JARVIS system.",
    instruction=(
        "You are the PLANNER for the JARVIS system.\n"
        "Your job is to understand the user request and decide whether it:\n"
        "- requires computation or code (then delegate to JARVIS),\n"
        "- involves memory operations (delegate to JARVIS),\n"
        "- or just needs a direct natural language answer.\n"
        "Use your sub-agent JARVIS whenever tools or detailed reasoning are helpful.\n"
        "Keep responses concise and user-friendly."
    ),
    sub_agents=[jarvis_worker],
    tools=[],
)

runner = InMemoryRunner(agent=planner_agent, app_name=APP_NAME)

print(f"Planner Agent and Runner initialised with session id {SESSION_ID}")
logging.info(f"Planner agent created with session id {SESSION_ID}")


def extract_text(content) -> str:
    """Extract text from a Content object."""
    if content and hasattr(content, "parts"):
        return " ".join(
            p.text for p in content.parts if hasattr(p, "text") and p.text
        ).strip()
    return ""


async def init_session():
    """Create a session if it does not already exist (idempotent-ish)."""
    try:
        await runner.session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID,
        )
        logging.info(f"Created new session: {SESSION_ID}")
    except Exception as e:
        # If session already exists, just log and continue
        logging.warning(f"Session may already exist ({SESSION_ID}): {e}")


async def chat(message: str) -> str:
    """Chat with the Planner/JARVIS multi-agent system."""
    logging.info(f"[USER] {message}")
    try:
        content = types.Content(role="user", parts=[types.Part(text=message)])
        response_text = ""

        async for event in runner.run_async(
            user_id=USER_ID, session_id=SESSION_ID, new_message=content
        ):
            if hasattr(event, "content"):
                text = extract_text(event.content)
                if text:
                    response_text = text

        logging.info(f"[ASSISTANT] {response_text}")
        return response_text or "Done."
    except Exception as e:
        logging.exception("Error during chat()")
        return f"Error: {e}"

print("Chat Functions Loaded")
logging.info("Chat helper functions loaded.")


async def run_demo():
    with open("jarvis_output.txt", "w", encoding="utf-8") as f:

        def log_out(text):
            print(text)
            f.write(text + "\n")

        log_out("=" * 60)
        log_out("JARVIS v0.3 - Google ADK Demo Results")
        log_out(f"Timestamp: {datetime.now()}")
        log_out("=" * 60)

        demo_tests = [
            ("Greeting", "Hello JARVIS! Say hi."),
            (
                "Memory",
                "Remember I'm an AI Engineer who loves Kaggle and my name is Amlan.",
            ),
            (
                "Code",
                "Calculate factorial of 10 using Python and give the code as well.",
            ),
            ("Math", "What is 2 to the power of 16?"),
            ("Recall", "What do you remember about me?"),
            ("Time", "What time is it right now?"),
        ]

        for title, query in demo_tests:
            log_out(f"\n[{title}]")
            log_out(f"Q: {query}")
            response = await chat(query)
            log_out(f"A: {response}")
            await asyncio.sleep(0.5)

        log_out("\n" + "=" * 60)
        log_out("Demo complete!")
        log_out("Saved: jarvis_output.txt")

    print("\n Saved: jarvis_output.txt")
    logging.info("Demo run complete.")


EVAL_TESTS = [
    {
        "id": "math_power",
        "prompt": "What is 2 to the power of 10?",
        "expected_contains": "1024",
    },
    {
        "id": "memory_store",
        "prompt": "Remember that my favorite language is Python.",
        "expected_contains": "remember",
    },
    {
        "id": "memory_recall",
        "prompt": "What is my favorite language?",
        "expected_contains": "Python",
    },
    {
        "id": "datetime_format",
        "prompt": "Tell me the current date in YYYY-MM-DD format only.",
        "expected_contains": "-",
    },
]


async def run_evaluation():
    """Run the evaluation tests and log results."""
    logging.info("Starting evaluation tests...")
    results = []
    passed = 0

    for test in EVAL_TESTS:
        prompt = test["prompt"]
        expected = test["expected_contains"]

        logging.info(f"[EVAL] Running test: {test['id']} | prompt={prompt}")
        response = await chat(prompt)
        success = expected.lower() in response.lower()

        result = {
            "id": test["id"],
            "prompt": prompt,
            "expected_contains": expected,
            "response": response,
            "success": success,
            "timestamp": datetime.now().isoformat(),
        }
        results.append(result)
        if success:
            passed += 1

    eval_file = "jarvis_eval_results.jsonl"
    with open(eval_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(EVAL_TESTS)
    summary = f"Evaluation complete. Passed {passed}/{total} tests."
    print("\n" + "=" * 60)
    print(summary)
    print(f"Detailed results saved to {eval_file}")
    logging.info(summary)
    logging.info(f"Eval results saved to {eval_file}")


async def main():
    await init_session()
    await run_demo()
    await run_evaluation()


if __name__ == "__main__":
    asyncio.run(main())

