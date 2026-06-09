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


"""
Capstone multi-agent script (capstone_script.py)

This is a single-file demonstration of a minimal multi-agent system that
implements at least three course concepts:

1. Multi-Agent System (sequential agents: PlannerAgent -> ResearchAgent -> WriterAgent)
2. Tools (a custom WebScraperTool stub and a CodeExecutionTool stub)
3. Sessions & Memory (InMemorySessionService + MemoryBank on disk)
4. Observability (structured logging)
5. Long-running operation support (pause/resume via saved session state)

Note: This script is intentionally standalone and uses simulated tool results
so it can run without network access or paid APIs. Replace stubs with real
implementations (e.g., actual web scraping, LLM clients) when integrating.

How to use:
    python capstone_script.py

It will run a demo workflow and save a session JSON file named `session_state.json`.
To simulate pause/resume, run once to create the session, then run again â€” the
script will detect and resume the saved session.

"""

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any

# --------------------------- Observability ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("capstone")

# --------------------------- Session & Memory ---------------------------
SESSION_FILE = "session_state.json"
MEMORY_FILE = "memory_bank.json"

class InMemorySessionService:
    """Simple session service that can persist to disk (pause/resume)."""
    def __init__(self, session_file=SESSION_FILE):
        self.session_file = session_file
        self.state = {
            "session_id": str(uuid.uuid4()),
            "created_at": time.time(),
            "stage": "init",
            "planner_output": None,
            "research_output": [],
            "writer_output": None,
        }
        if os.path.exists(self.session_file):
            logger.info("Found existing session file. Loading to resume...")
            self.load()

    def save(self):
        with open(self.session_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)
        logger.info("Session saved to %s", self.session_file)

    def load(self):
        with open(self.session_file, "r", encoding="utf-8") as f:
            self.state = json.load(f)
        logger.info("Session loaded: stage=%s", self.state.get("stage"))


class MemoryBank:
    """A tiny long-term memory implemented on disk as JSON."""
    def __init__(self, memory_file=MEMORY_FILE):
        self.memory_file = memory_file
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            self.data = {"user_preferences": {}, "past_topics": []}

    def remember_topic(self, topic: str):
        if topic not in self.data["past_topics"]:
            self.data["past_topics"].append(topic)
            self._persist()
            logger.info("Remembered new topic: %s", topic)

    def set_preference(self, key: str, value: Any):
        self.data["user_preferences"][key] = value
        self._persist()

    def _persist(self):
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)


# --------------------------- Tools (real integrations + stubs) ---------------------------
# Example integration point for REAL Google Search tool
# Replace with your API call (SerpAPI, Google CSE, Bing API, or allowed built-in search tool)
#
# Example usage:
#   from external_tools import google_search
#   results = google_search(query)
#
# For now, we keep the stub but show where to plug real search.
class WebScraperTool:
    """A stubbed web scraper tool. Replace this with real HTTP + parsing logic.

    The real tool would fetch webpages and return text or structured data.
    Here we simulate by returning canned results based on the query.
    """
    @staticmethod
    def scrape(query: str) -> List[Dict[str, Any]]:
        logger.info("WebScraperTool.scrape called with query: '%s'", query)
        # Simulated results
        time.sleep(0.5)  # simulate latency
        return [
            {"title": f"Intro to {query}", "url": f"https://example.com/{query}-intro", "snippet": f"This page explains {query}..."},
            {"title": f"Advanced {query}", "url": f"https://example.com/{query}-advanced", "snippet": f"Deep dive into {query}..."},
        ]


class CodeExecutionTool:
    """Executes small Python code snippets in a sandboxed way. Very limited.
    For real use, integrate a safe execution environment or external service.
    """
    @staticmethod
    def run(code_str: str) -> Dict[str, Any]:
        logger.info("CodeExecutionTool: executing code snippet")
        # Very small sandbox: only allow arithmetic and simple list/dict expressions via eval with empty globals
        try:
            result = eval(code_str, {"__builtins__": {}}, {})
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}


# --------------------------- Agent Definitions ---------------------------
@dataclass
class PlannerAgent:
    """Breaks a user request into tasks."""
    def plan(self, user_request: str) -> Dict[str, Any]:
        logger.info("PlannerAgent planning for request: %s", user_request)
        # Very simple planner rules
        tasks = [
            {"id": "research", "action": "gather_sources", "query": user_request},
            {"id": "synthesize", "action": "summarize_and_write", "format": "short_report"},
        ]
        plan = {"user_request": user_request, "tasks": tasks}
        logger.info("PlannerAgent produced plan with %d tasks", len(tasks))
        return plan


@dataclass
class ResearchAgent:
    """Uses tools to gather and verify information. Includes a loop agent pattern
    that keeps searching until a minimum number of sources is collected or a max
    iteration count is reached.
    """
    min_sources: int = 2
    max_iterations: int = 4

    def gather(self, query: str) -> List[Dict[str, Any]]:
        logger.info("ResearchAgent starting gather for query: %s", query)
        results: List[Dict[str, Any]] = []
        iteration = 0
        while len(results) < self.min_sources and iteration < self.max_iterations:
            iteration += 1
            logger.info("ResearchAgent iteration %d", iteration)
            scraped = WebScraperTool.scrape(query + f"+iter{iteration}")
            # naive dedup
            for r in scraped:
                if r["url"] not in {x["url"] for x in results}:
                    results.append(r)
            logger.info("ResearchAgent collected %d sources so far", len(results))
            # simulate evaluation step (in real system, check credibility, recency, etc.)
            time.sleep(0.2)
        logger.info("ResearchAgent finished with %d results", len(results))
        return results


@dataclass
# Example integration point for REAL LLM model call
# Replace with your actual LLM SDK (OpenAI, Google, Anthropic, etc.)
#
# Example:
#   from openai import OpenAI
#   client = OpenAI()
#   response = client.chat.completions.create(
#       model="gpt-4o-mini",
#       messages=[{"role": "user", "content": prompt}]
#   )
#   return response.choices[0].message.content
#
# This block below shows where you plug LLM summarization.
class WriterAgent:
    """Composes final output. May call the CodeExecutionTool to produce small
    derived artifacts (e.g. counts, simple stats).
    """
    def write(self, plan: Dict[str, Any], sources: List[Dict[str, Any]], memory: MemoryBank) -> str:
        logger.info("WriterAgent composing final output")
        # Use memory preferences
        pref = memory.data.get("user_preferences", {}).get("style", "concise")
        lines = []
        lines.append(f"Report for: {plan['user_request']}")
        lines.append(f"Style preference: {pref}")
        lines.append("")
        lines.append("Sources collected:")
        for i, s in enumerate(sources, 1):
            lines.append(f"{i}. {s['title']} â€” {s['url']}")
        lines.append("")
        # Example of using CodeExecutionTool to compute a trivial stat
        code_snippet = str(len(sources))
        exec_result = CodeExecutionTool.run(code_snippet)
        if exec_result.get("success"):
            lines.append(f"(Generated statistic) Number of sources: {exec_result['result']}")
        else:
            lines.append("(Statistic generation failed)")
        # Short summary simulated
        lines.append("")
        lines.append("Summary:")
        lines.append(f"Based on {len(sources)} sources, we summarize that '{plan['user_request']}' has the following themes: ... (simulated)")
        return "\n".join(lines)


# --------------------------- Main Workflow / Orchestrator ---------------------------
class CapstoneOrchestrator:
    def __init__(self):
        self.session = InMemorySessionService()
        self.memory = MemoryBank()
        self.planner = PlannerAgent()
        self.researcher = ResearchAgent()
        self.writer = WriterAgent()

    def run(self, user_request: str, force_restart=False):
        # If we found an existing session and user doesn't want restart, resume
        if not force_restart and self.session.state.get("stage") != "init":
            logger.info("Resuming existing session at stage: %s", self.session.state.get("stage"))
            return self._resume_flow()

        # Start a fresh flow
        logger.info("Starting new orchestrator flow for request: %s", user_request)
        plan = self.planner.plan(user_request)
        self.session.state["planner_output"] = plan
        self.session.state["stage"] = "planned"
        self.session.save()

        # Research step (loop agent)
        sources = self.researcher.gather(plan["user_request"])
        self.session.state["research_output"] = sources
        self.session.state["stage"] = "researched"
        self.session.save()

        # Remember topic in memory bank
        self.memory.remember_topic(plan["user_request"])

        # Simulate a long-running operation: allow user to "pause" here by exiting early.
        logger.info("Simulating potential long-running operation. You can re-run the script to resume.")

        # Writer step
        report = self.writer.write(plan, sources, self.memory)
        self.session.state["writer_output"] = report
        self.session.state["stage"] = "written"
        self.session.save()

        logger.info("Workflow complete. Report written to session state.")
        return report

    def _resume_flow(self):
        stage = self.session.state.get("stage")
        if stage == "planned":
            plan = self.session.state.get("planner_output")
            sources = self.researcher.gather(plan["user_request"])
            self.session.state["research_output"] = sources
            self.session.state["stage"] = "researched"
            self.session.save()
            report = self.writer.write(plan, sources, self.memory)
            self.session.state["writer_output"] = report
            self.session.state["stage"] = "written"
            self.session.save()
            return report
        elif stage == "researched":
            plan = self.session.state.get("planner_output")
            sources = self.session.state.get("research_output", [])
            report = self.writer.write(plan, sources, self.memory)
            self.session.state["writer_output"] = report
            self.session.state["stage"] = "written"
            self.session.save()
            return report
        elif stage == "written":
            logger.info("Session already completed. Returning existing report.")
            return self.session.state.get("writer_output")
        else:
            logger.info("Unknown session state, restarting fresh flow")
            return None


# --------------------------- Demo Entrypoint ---------------------------
if __name__ == "__main__":
    orchestrator = CapstoneOrchestrator()

    # Example user request. In a real system, collect from CLI or web input.
    example_request = "benefits of microservices architecture"

    # Optionally let user force restart by deleting the session file first or passing a flag.
    result = orchestrator.run(example_request)

    if result:
        print("\n--- FINAL REPORT (from session) ---\n")
        print(result)
    else:
        print("No report generated. Run the script again to resume or start fresh by deleting session_state.json.")


