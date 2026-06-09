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


# Full Multi-Agent System Code (Error-Free)
# Minimal Multi-Agent System (Option A)
# - Planner, Thinker, Coder, Evaluator
# - Tools: Calculator, Search, Runner
# - Memory (short-term + long-term)
# - A2A message bus
# - Orchestrator with example usage at the bottom

import math
import textwrap
import time
import uuid
from typing import Any, Dict, List, Optional

# ---------------------------
# Mock LLM (offline deterministic)
# ---------------------------
class MockLLM:
    def __init__(self, name: str = "MockLLM"):
        self.name = name

    def generate(self, prompt: str, max_len: int = 256) -> str:
        header = f"[{self.name}]"
        snippet = " ".join(prompt.strip().split())[:200]
        return f"{header} {snippet}"

# ---------------------------
# Tools
# ---------------------------
class CalculatorTool:
    def run(self, expr: str) -> Dict[str, Any]:
        try:
            # restricted eval environment using math functions
            result = eval(expr, {"__builtins__": None}, math.__dict__)
            return {"success": True, "output": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

class SearchTool:
    def run(self, query: str) -> Dict[str, Any]:
        # Offline mock search: returns a single simulated hit
        hits = [{"title": f"Mock result for '{query}'", "snippet": "This is an offline simulated search result."}]
        return {"success": True, "query": query, "hits": hits}

class RunnerTool:
    def run_code(self, code: str, func: Optional[str] = None, args: Optional[list] = None) -> Dict[str, Any]:
        local: Dict[str, Any] = {}
        try:
            exec(code, {}, local)
            out = None
            if func and func in local and callable(local[func]):
                if args is None:
                    args = []
                out = local[func](*args)
            return {"success": True, "result": out, "locals": list(local.keys())}
        except Exception as e:
            return {"success": False, "error": str(e)}

# ---------------------------
# Agent base class
# ---------------------------
class Agent:
    def __init__(self, name: str):
        self.name = name
        self.llm = MockLLM(name)
        self.memory: List[str] = []

    def think(self, prompt: str) -> str:
        res = self.llm.generate(prompt)
        self.memory.append(res)
        return res

# ---------------------------
# Planner Agent
# ---------------------------
class PlannerAgent(Agent):
    def plan(self, objective: str) -> Dict[str, Any]:
        prompt = f"Planner: create a clear 3-step plan for: {objective}"
        raw = self.think(prompt)
        plan = [
            {"step": 1, "title": "Analyze & Decompose", "desc": "Break the task into sub-tasks and success criteria."},
            {"step": 2, "title": "Prototype & Implement", "desc": "Create prototype code or actions for each subtask."},
            {"step": 3, "title": "Test & Report", "desc": "Validate results and prepare a final report."},
        ]
        return {"raw": raw, "plan": plan}

# ---------------------------
# Thinker Agent
# ---------------------------
class ThinkerAgent(Agent):
    def analyze(self, objective: str, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt = f"Thinker: analyze objective '{objective}' with plan '{plan}'. Provide risks and edge-cases."
        raw = self.think(prompt)
        notes = {
            "risks": ["misinterpreting requirements", "insufficient testing", "unhandled edge-cases"],
            "edge_cases": ["empty input", "very large inputs", "wrong data types"],
            "recommendations": ["add unit tests", "validate inputs", "log important events"]
        }
        return {"raw": raw, "notes": notes}

# ---------------------------
# Coder Agent
# ---------------------------
class CoderAgent(Agent):
    def __init__(self, name: str = "Coder"):
        super().__init__(name)
        self.runner = RunnerTool()

    def generate_code(self, spec: str) -> Dict[str, Any]:
        prompt = f"Coder: produce Python code for: {spec}"
        raw = self.think(prompt)
        # Heuristic: if 'fibonacci' in spec, produce fibonacci implementation
        if "fibonacci" in spec.lower():
            code = textwrap.dedent('''\
                def fibonacci(n: int) -> list:
                    """Return list of first n fibonacci numbers."""
                    if n <= 0:
                        return []
                    seq = [0, 1]
                    while len(seq) < n:
                        seq.append(seq[-1] + seq[-2])
                    return seq[:n]

                def test_fibonacci():
                    assert fibonacci(0) == []
                    assert fibonacci(1) == [0]
                    assert fibonacci(6) == [0,1,1,2,3,5]
            ''')
            return {"raw": raw, "code": code, "entry": "fibonacci", "tests": ["test_fibonacci"]}
        else:
            code = textwrap.dedent(f'''\
                def solution(*args, **kwargs):
                    """Placeholder solution for: {spec}"""
                    return None

                def test_solution():
                    assert solution() is None
            ''')
            return {"raw": raw, "code": code, "entry": "solution", "tests": ["test_solution"]}

    def run_generated(self, code_obj: Dict[str, Any]) -> Dict[str, Any]:
        code = code_obj["code"]
        entry = code_obj.get("entry")
        exec_result = self.runner.run_code(code, func=entry, args=[10] if entry == "fibonacci" else None)

        # Run tests (if any)
        tests_report: Dict[str, Any] = {}
        try:
            local: Dict[str, Any] = {}
            exec(code, {}, local)
            for t in code_obj.get("tests", []):
                if t in local and callable(local[t]):
                    try:
                        local[t]()
                        tests_report[t] = {"success": True}
                    except Exception as e:
                        tests_report[t] = {"success": False, "error": str(e)}
                else:
                    tests_report[t] = {"success": False, "error": "test not found"}
        except Exception as e:
            tests_report = {"error": str(e)}

        return {"execution": exec_result, "tests": tests_report}

# ---------------------------
# Evaluator Agent
# ---------------------------
class EvaluatorAgent(Agent):
    def evaluate(self, run_result: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"Evaluator: summarize run_result {run_result}"
        raw = self.think(prompt)
        ok = bool(run_result.get("success"))
        summary = {
            "ok": ok,
            "notes": "Execution succeeded" if ok else "Execution failed or errored",
            "details": run_result
        }
        return {"raw": raw, "summary": summary}

# ---------------------------
# Memory Agent
# ---------------------------
class MemoryAgent:
    def __init__(self):
        self.short_term: Dict[str, Any] = {}
        self.long_term: List[Dict[str, Any]] = []

    def store_short(self, key: str, value: Any):
        self.short_term[key] = value

    def persist_long(self, record: Dict[str, Any]) -> Dict[str, Any]:
        # attach an id and timestamp and return the stored record
        rec = dict(record)
        rec["_id"] = str(uuid.uuid4())
        rec["_ts"] = time.time()
        self.long_term.append(rec)
        return rec

    def recall(self, filter_fn=None) -> List[Dict[str, Any]]:
        if filter_fn is None:
            return self.long_term
        return [r for r in self.long_term if filter_fn(r)]

# ---------------------------
# A2A Message Bus
# ---------------------------
class A2ABus:
    def __init__(self):
        self.messages: List[Dict[str, Any]] = []

    def send(self, sender: str, receiver: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        msg = {"from": sender, "to": receiver, "payload": payload, "time": time.time()}
        self.messages.append(msg)
        return msg

    def fetch_for(self, receiver: str) -> List[Dict[str, Any]]:
        return [m for m in self.messages if m["to"] == receiver]

# ---------------------------
# MultiAgentSystem Orchestrator
# ---------------------------
class MultiAgentSystem:
    def __init__(self):
        self.planner = PlannerAgent("Planner")
        self.thinker = ThinkerAgent("Thinker")
        self.coder = CoderAgent("Coder")
        self.evaluator = EvaluatorAgent("Evaluator")
        self.memory = MemoryAgent()
        self.bus = A2ABus()

    def run(self, objective: str, iterations: int = 2) -> Dict[str, Any]:
        # 1. Plan
        plan_out = self.planner.plan(objective)
        self.memory.store_short("last_plan", plan_out)

        # 2. Analyze
        analysis = self.thinker.analyze(objective, plan_out["plan"])
        self.memory.store_short("last_analysis", analysis)

        artifacts: List[Dict[str, Any]] = []
        runs: List[Dict[str, Any]] = []

        for i in range(iterations):
            spec = f"Iteration {i+1}: implement for objective: {objective}. Notes: {analysis['notes']}"
            # choose code spec; prefer fibonacci if objective mentions it
            code_spec = "create fibonacci" if "fibonacci" in objective.lower() else spec
            code_obj = self.coder.generate_code(code_spec)
            run_out = self.coder.run_generated(code_obj)

            # Evaluate
            eval_out = self.evaluator.evaluate(run_out["execution"])

            # Persist memory (and get record back with _id)
            record = {
                "iteration": i+1,
                "spec": code_spec,
                "code_entry": code_obj.get("entry"),
                "run": run_out,
                "eval": eval_out["summary"]
            }
            stored = self.memory.persist_long(record)

            # send a bus message with stored record id
            self.bus.send("MultiAgentSystem", "Logger", {"event": "iteration_complete", "record_id": stored["_id"]})

            artifacts.append(code_obj)
            runs.append({"run": run_out, "eval": eval_out})

            # stop early on success
            if run_out.get("execution", {}).get("success"):
                break

        # Reporter-like summary
        summary = {
            "objective": objective,
            "plan": plan_out,
            "analysis": analysis,
            "artifacts": artifacts,
            "runs": runs,
            "memory_snapshot": self.memory.recall()[-5:],
            "messages": self.bus.messages[-10:],
        }
        return summary

# ---------------------------
# Example usage (safe to run as-is)
# ---------------------------
if __name__ == "__main__":
    mas = MultiAgentSystem()

    print("=== Example: Fibonacci Task ===")
    res_fib = mas.run("Create a Fibonacci generator with tests", iterations=3)
    print("Plan:", res_fib["plan"]["plan"])
    print("Latest run evaluation:", res_fib["runs"][-1]["eval"]["summary"])
    print("Memory (last record):", res_fib["memory_snapshot"][-1])

    print("\n=== Example: Generic Task ===")
    res_gen = mas.run("Produce a performance summary report for dataset X", iterations=2)
    print("Plan:", res_gen["plan"]["plan"])
    print("Latest run evaluation:", res_gen["runs"][-1]["eval"]["summary"])
    print("Messages on bus:", res_gen["messages"])

    print("\nRun complete. Inspect the returned dictionaries for full details.")


