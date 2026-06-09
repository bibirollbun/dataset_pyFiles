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


# Configure Gemini API Key
import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google.adk.agents import LlmAgent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent, AGENT_CARD_WELL_KNOWN_PATH
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

import warnings
warnings.filterwarnings("ignore")

print("âœ… ADK components imported successfully.")


retry_config = types.HttpRetryOptions(
    attempts=5,        # Maximum retry attempts
    exp_base=7,        # Delay multiplier (exponential backoff)
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

print("âœ… Retry configuration loaded.")


from typing import List, Dict, Any
import time
import random

# ---------------------------------------------------------
# Simple in-notebook "test registry" to simulate real tests
# In a real project, these could be pytest / Playwright / API tests, etc.
# ---------------------------------------------------------
TEST_REGISTRY: Dict[str, Dict[str, Any]] = {
    "login_smoke": {
        "name": "Login Smoke Test",
        "description": "Verify that a user can log in with valid credentials.",
        "tags": ["smoke", "auth"],
    },
    "checkout_flow": {
        "name": "Checkout Flow Test",
        "description": "Simulate user checkout from cart to payment confirmation.",
        "tags": ["e2e", "checkout"],
    },
    "search_results_relevance": {
        "name": "Search Results Relevance Test",
        "description": "Ensure that top results are relevant to the search query.",
        "tags": ["ml", "search"],
    },
}


def discover_tests() -> List[Dict[str, Any]]:
    """
    Discover available test cases.

    Returns:
        A list of test metadata dictionaries with id, name, description, and tags.
    """
    tests = []
    for test_id, meta in TEST_REGISTRY.items():
        tests.append(
            {
                "id": test_id,
                "name": meta["name"],
                "description": meta["description"],
                "tags": meta.get("tags", []),
            }
        )
    return tests


def run_test_case(test_id: str) -> Dict[str, Any]:
    """
    Execute a single test case by ID.

    Args:
        test_id: Identifier of the test to run.

    Returns:
        A structured result with status, duration, and logs.
    """
    if test_id not in TEST_REGISTRY:
        return {
            "test_id": test_id,
            "status": "ERROR",
            "duration_sec": 0.0,
            "logs": [f"Test '{test_id}' not found in registry."],
        }

    start = time.time()
    # ğŸ”§ Here you would call the real test runner (pytest, Playwright, etc.)
    # For now we simulate pass/fail randomly.
    time.sleep(0.2)  # simulate execution time
    passed = random.random() > 0.1  # ~90% pass rate

    duration = round(time.time() - start, 3)
    status = "PASSED" if passed else "FAILED"

    logs = [
        f"Executing test: {test_id}",
        f"Description: {TEST_REGISTRY[test_id]['description']}",
        f"Result: {status}",
    ]

    return {
        "test_id": test_id,
        "status": status,
        "duration_sec": duration,
        "logs": logs,
    }


def run_test_suite(test_ids: List[str] | None = None) -> Dict[str, Any]:
    """
    Run a suite of tests (all or a selected subset).

    Args:
        test_ids: Optional list of test IDs to run. If None, runs all tests.

    Returns:
        Aggregated suite results including per-test results and summary stats.
    """
    if test_ids is None:
        test_ids = list(TEST_REGISTRY.keys())

    results = []
    passed = 0
    failed = 0
    errored = 0

    for tid in test_ids:
        result = run_test_case(tid)
        results.append(result)

        if result["status"] == "PASSED":
            passed += 1
        elif result["status"] == "FAILED":
            failed += 1
        else:
            errored += 1

    summary = {
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "errored": errored,
    }

    return {
        "suite_test_ids": test_ids,
        "summary": summary,
        "results": results,
    }


def generate_report(suite_result: Dict[str, Any]) -> str:
    """
    Generate a human-readable test report from a suite result.

    Args:
        suite_result: Output of run_test_suite().

    Returns:
        A text report summarizing the execution.
    """
    summary = suite_result.get("summary", {})
    lines = [
        "AI Test Automation Agent â€“ Execution Report",
        "-----------------------------------------",
        f"Total tests : {summary.get('total', 0)}",
        f"Passed      : {summary.get('passed', 0)}",
        f"Failed      : {summary.get('failed', 0)}",
        f"Errored     : {summary.get('errored', 0)}",
        "",
        "Per-test results:",
    ]

    for r in suite_result.get("results", []):
        lines.append(
            f"- {r['test_id']}: {r['status']} ({r['duration_sec']}s)"
        )

    return "\n".join(lines)


print("âœ… Test automation tools defined (discover_tests, run_test_case, run_test_suite, generate_report).")


# Create Runner, Session, and Interaction Helper

from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

# Initialize session service
session_service = InMemorySessionService()

# Application + user identifiers
APP_NAME = "ai-testautomation-agent"
USER_ID = "demo_user"

async def run_agent_query(agent, query: str):
    """
    Sends a query to the AI Test Automation Agent and streams its response.
    """
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
    )

    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    print(f"\nğŸ‘¤ User: {query}")
    print("ğŸ¤– Agent Response:\n")

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session.id,
        new_message=types.Content(parts=[types.Part(text=query)]),
    ):
        if event.is_final_response():
            # Safely handle missing/empty content
            if not event.content or not getattr(event.content, "parts", None):
                # Optional: print debug info
                # print("âš ï¸� Final event has no text content:", event)
                continue

            for part in event.content.parts:
                if getattr(part, "text", None):
                    print(part.text)


# Agent Definition
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini

test_automation_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="ai_testautomation_agent",
    description="LLM agent that orchestrates automated tests over a web application.",
    instruction="""
You are the AI Test Automation Agent.

Your responsibilities:
- Discover available automated tests by calling the discover_tests tool.
- Select appropriate tests based on the user's request (area, feature, smoke vs regression).
- Run those tests via run_selected_tests.
- Summarize the outcome using summarize_test_results.
- Clearly explain what was run and what passed/failed.

Always:
- Use the tools instead of guessing results.
- Return concise, structured explanations that a QA engineer can understand.
""",
    tools=[discover_tests, run_selected_tests, summarize_test_results],
)

print("âœ… AI Test Automation Agent created.")


# Test 1 â€“ Discover Tests
await run_agent_query(
    test_automation_agent,
    "List the available automated tests and group them by area."
)


# Test 2 â€“ Smoke Tests
await run_agent_query(
    test_automation_agent,
    "Run a small smoke suite for authentication and checkout, then summarize the results."
)


# Test 3 â€“ Regression-style Run
await run_agent_query(
    test_automation_agent,
    "Run all tests related to checkout flows and give me a concise failure report."
)

