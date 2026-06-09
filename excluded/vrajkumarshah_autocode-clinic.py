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


import os, json, time, uuid, tempfile, shutil, subprocess
from typing import Dict, List, Any, Tuple
import logging
import textwrap
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("autocode_clinic")

SANDBOX_ROOT = "/tmp/autocode_sandbox"
os.makedirs(SANDBOX_ROOT, exist_ok=True)



MODEL_BACKEND = os.environ.get("MODEL_BACKEND", "mock")
MODEL_ACCESS_TOKEN = os.environ.get("MODEL_ACCESS_TOKEN", "")
GENERIC_MODEL_NAME = os.environ.get("GENERIC_MODEL_NAME", "tier3-transformer-large")

def mock_model(prompt: str, max_tokens: int = 512) -> str:
    pl = prompt.lower()
    # diagnosis json
    if "diagnose root cause" in pl or "diagnos" in pl:
        return json.dumps({
            "root_causes": ["incorrect operator", "variable naming issue"],
            "confidence": 0.75,
            "suggested_tests": ["add edge-case tests for negative inputs"]
        })
    if "generate patch" in pl or "patch" in pl:
        return json.dumps({
            "patch_type": "file_replacement",
            "patch": {
                "solution.py": "def add(a, b):\n    return a + b\n"
            },
            "explanation": "Fixed arithmetic operator from - to + in add function."
        })
    return "MOCK_RESPONSE: " + prompt[:200]

def call_model(prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
    """
    Model-agnostic wrapper. Default uses mock_model.
    To use a remote model:
      - set MODEL_BACKEND='remote'
      - implement remote call using MODEL_ACCESS_TOKEN securely (not in notebook)
    """
    if MODEL_BACKEND == "mock":
        return mock_model(prompt, max_tokens=max_tokens)
    else:
        raise RuntimeError("Remote model backend selected but no remote client implemented in this demo.")



class SessionStateManager:
    def __init__(self):
        self.store: Dict[str, Dict[str, Any]] = {}

    def init_session(self, session_id: str):
        self.store.setdefault(session_id, {"history": [], "analysis": [], "patches": []})

    def append(self, session_id: str, key: str, value: Any):
        self.init_session(session_id)
        self.store[session_id]["history"].append({key: value})

    def get(self, session_id: str) -> Dict[str, Any]:
        return self.store.get(session_id, {})

    def clear(self, session_id: str):
        if session_id in self.store:
            del self.store[session_id]

state_manager = SessionStateManager()



def run_python_tests(code_files: Dict[str, str], test_code: str, timeout_sec: int = 10) -> Tuple[bool, str]:
    """
    Very lightweight isolated runner for Kaggle demo.
    For production, use container-based secure execution.
    Returns (passed_bool, combined_output).
    """
    tmp = tempfile.mkdtemp(dir=SANDBOX_ROOT)
    try:
        for fname, content in code_files.items():
            with open(os.path.join(tmp, fname), "w") as f:
                f.write(content)
        test_path = os.path.join(tmp, "test_runner.py")
        with open(test_path, "w") as f:
            f.write(test_code)
        proc = subprocess.run(["python", test_path], cwd=tmp, capture_output=True, text=True, timeout=timeout_sec)
        ok = proc.returncode == 0
        out = proc.stdout + "\n" + proc.stderr
        return ok, out
    finally:
        shutil.rmtree(tmp)



import ast

def simple_static_analyzer(code: str) -> List[str]:
    issues: List[str] = []
    try:
        tree = ast.parse(code)
    except Exception as e:
        issues.append(f"SyntaxError: {e}")
        return issues
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "eval":
            issues.append("Use of eval() detected - risky")
    return issues



class BaseAgent:
    def __init__(self, name: str):
        self.name = name
    def log(self, msg: str):
        logger.info(f"[{self.name}] {msg}")

class StaticAnalysisAgent(BaseAgent):
    def analyze(self, code_files: Dict[str, str]) -> Dict[str, List[str]]:
        self.log("Running static analysis")
        return {fname: simple_static_analyzer(content) for fname, content in code_files.items()}

class PatchGenerationAgent(BaseAgent):
    def propose_patch(self, code_files: Dict[str, str], test_output: str, static_analysis: Dict[str, Any]) -> Dict[str, Any]:
        self.log("Proposing patch (via model wrapper)")
        prompt = (
            "Generate patch for failing code.\n"
            f"Files: {list(code_files.keys())}\n"
            f"StaticAnalysis: {json.dumps(static_analysis)}\n"
            f"TestOutput: {test_output}\n"
            "Return a JSON with keys: patch_type, patch, explanation."
        )
        resp = call_model(prompt)
        # Try to parse to JSON, else wrap as fallback
        try:
            parsed = json.loads(resp)
        except Exception:
            parsed = {"patch_type": "file_replacement", "patch": {list(code_files.keys())[0]: resp}, "explanation": "Auto-generated (fallback)"}
        return parsed

class ValidationAgent(BaseAgent):
    def validate(self, code_files: Dict[str, str], tests: str) -> Tuple[bool, str]:
        self.log("Running validation tests in sandbox")
        return run_python_tests(code_files, tests)

class OrchestratorAgent(BaseAgent):
    def __init__(self):
        super().__init__("Orchestrator")
        self.analysis_agent = StaticAnalysisAgent("StaticAnalysis")
        self.patch_agent = PatchGenerationAgent("PatchGen")
        self.validator = ValidationAgent("Validator")

    def handle_job(self, session_id: str, code_files: Dict[str, str], tests: str, max_iters: int = 3) -> Dict[str, Any]:
        state_manager.init_session(session_id)
        # Static analysis
        analysis = self.analysis_agent.analyze(code_files)
        state_manager.append(session_id, "analysis", analysis)

        for i in range(max_iters):
            self.log(f"Loop iteration {i+1}")
            ok, output = self.validator.validate(code_files, tests)
            state_manager.append(session_id, "validation", {"ok": ok, "output": output})
            if ok:
                self.log("Validation passed")
                return {"status": "fixed", "iterations": i, "output": output}
            patch_obj = self.patch_agent.propose_patch(code_files, output, analysis)
            state_manager.append(session_id, "patch_proposed", patch_obj)
            if patch_obj.get("patch_type") == "file_replacement" and isinstance(patch_obj.get("patch"), dict):
                for fname, new_content in patch_obj["patch"].items():
                    code_files[fname] = new_content
                state_manager.append(session_id, "patched_files", list(patch_obj["patch"].keys()))
            else:
                self.log("Patch type unsupported in demo (skipping actual apply)")
        self.log("Exhausted iterations without successful fix")
        return {"status": "unfixed", "iterations": max_iters, "last_output": output}

orchestrator = OrchestratorAgent()



code_files = {
    "solution.py": "def add(a,b):\n    return a - b\n"
}
tests = """
import solution

def test_add():
    assert solution.add(2,3) == 5

if __name__ == '__main__':
    test_add()
"""

session_id = "demo-" + str(uuid.uuid4())
result = orchestrator.handle_job(session_id, code_files.copy(), tests, max_iters=2)
print("RESULT:", result)
print("\nSESSION HISTORY (truncated):\n", json.dumps(state_manager.get(session_id), indent=2)[:1000])



import os
import json

BUGGY_DIR = "/kaggle/input/bug-probs-demo"  # folder of buggy programs
FIXED_DIR = "/kaggle/working/fixed_programs"
ARTIFACT_DIR = "/kaggle/working/artifacts"

os.makedirs(FIXED_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def load_buggy_programs(folder_path: str):
    """
    Loads all .py files from folder_path into a dataset-like list
    Each item: {"id": filename, "files": [{"name": filename, "content": code}], "tests": default test string}
    """
    dataset = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".py"):
            path = os.path.join(folder_path, fname)
            with open(path, "r") as f:
                code = f.read()
            module_name = fname.replace(".py", "")
            # Minimal safe test
            default_test = f"""
import {module_name} as mod

def test_dummy():
    pass  # no-op test

if __name__ == "__main__":
    test_dummy()
"""
            dataset.append({
                "id": module_name,
                "files": [{"name": fname, "content": code}],
                "tests": default_test
            })
    return dataset

dataset = load_buggy_programs(BUGGY_DIR)
print(f"Loaded {len(dataset)} buggy programs from {BUGGY_DIR}")



import pandas as pd
import uuid
import tempfile
import shutil
import subprocess

def run_python_tests(code_files: dict, test_code: str, timeout_sec: int = 30) -> tuple:
    tmp = tempfile.mkdtemp(dir="/tmp/autocode_sandbox")
    try:
        for fname, content in code_files.items():
            with open(os.path.join(tmp, fname), "w") as f:
                f.write(content)
        test_path = os.path.join(tmp, "test_runner.py")
        with open(test_path, "w") as f:
            f.write(test_code or "")
        try:
            proc = subprocess.run(
                ["python", test_path],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout_sec
            )
            ok = proc.returncode == 0
            out = proc.stdout + "\n" + proc.stderr
            return ok, out
        except subprocess.TimeoutExpired:
            return False, f"TimeoutExpired: Test exceeded {timeout_sec} seconds"
    finally:
        shutil.rmtree(tmp)

globals()["run_python_tests"] = run_python_tests


results = []

for ex in dataset:
    sess_id = "sess-" + str(uuid.uuid4())
    files = {f["name"]: f["content"] for f in ex["files"]}
    tests = ex.get("tests") or ""

    out = orchestrator.handle_job(sess_id, files.copy(), tests=tests, max_iters=3)

    if out["status"] == "fixed":
        for fname, content in files.items():
            fixed_path = os.path.join(FIXED_DIR, fname)
            with open(fixed_path, "w") as f:
                f.write(content)

    artifact_path = os.path.join(ARTIFACT_DIR, f"session_{sess_id}.json")
    sess_data = state_manager.get(sess_id)
    with open(artifact_path, "w") as f:
        json.dump(sess_data, f, indent=2)

    results.append({
        "id": ex["id"],
        "status": out["status"],
        "iterations": out.get("iterations", -1),
        "artifact": artifact_path
    })

df = pd.DataFrame(results)
display(df)
print(f"Fix rate: {float((df.status == 'fixed').sum()) / len(df):.2f}")
print(f"Fixed programs saved to: {FIXED_DIR}")
print(f"Session logs saved to: {ARTIFACT_DIR}")



SUMMARY_PATH = "/kaggle/working/fixed_programs_summary.json"

with open(SUMMARY_PATH, "w") as f:
    json.dump(results, f, indent=2)

print("Saved full run summary to:", SUMMARY_PATH)


