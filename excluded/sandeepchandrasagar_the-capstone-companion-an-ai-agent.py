import os
from typing import Optional

from google.adk.agents import LlmAgent
from google.adk.tools.tool_context import ToolContext


# ========== TOOL DEFINITION ==========

def audit_capstone_project(
    project_path: str,
    tool_context: Optional[ToolContext] = None
) -> str:
    """
    Run the full Capstone Companion pipeline on a project directory.
    Returns markdown report.
    """
    if not os.path.exists(project_path):
        return f"â�Œ ERROR: Path not found:\n`{project_path}`"

    try:
        result = run_capstone_companion(project_path)
        return result
    except Exception as e:
        return f"ğŸ”¥ Critical Failure:\n{str(e)}"


# ========== AGENT DEFINITION ==========

agent = LlmAgent(
    name="CapstoneCompanion",
    model="gemini-2.5-flash-lite",
    instruction=(
        "You are the Capstone Companion â€” an expert code reviewer and hackathon score maximizer. "
        "You audit Kaggle capstone projects, check docs, evaluate against rubric, "
        "detect missing rubric criteria, generate scorecards, scripts, and deployment plans.\n\n"
        "When the user gives you a folder path, ALWAYS call the tool "
        "`audit_capstone_project(path)` instead of guessing."
    ),
    tools=[audit_capstone_project],
    description="An agent that audits Kaggle hackathon capstone projects using a multi-agent pipeline."
)



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


# Install required libraries
!pip install -q google-generativeai python-dotenv
print("Libraries installed.")


# Imports and Gemini configuration

import os
import glob
import re
import textwrap
import json
from typing import Dict, Any, List

import google.generativeai as genai

# 1) Load API key (Kaggle Secrets -> .env fallback)
GEMINI_API_KEY = None

try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY")
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
    print("Gemini API Key loaded from Kaggle Secrets.")
except Exception as e:
    print("Kaggle Secrets not available or failed:", e)
    print("Trying .env...")
    from dotenv import load_dotenv
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not found. "
        "Set it as a Kaggle Secret or in a .env file."
    )

# 2) Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# You can change model name if needed (depends on what is available)
MODEL_NAME = "gemini-2.0-flash"
model = genai.GenerativeModel(MODEL_NAME)

print(f"Gemini client configured with model: {MODEL_NAME}")


# Robust Agent constructor that auto-detects the right "system/instruction" field
import os
import asyncio
import inspect
from pydantic import ValidationError

from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner

# ensure API key present
if not os.environ.get("GEMINI_API_KEY") and "GEMINI_API_KEY" in globals():
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

# create LLM (no retry options to avoid earlier version issues)
llm = Gemini(model="gemini-2.5-flash-lite")

# candidate names we've seen across ADK docs / examples
candidates = [
    "instruction",       # common in newer ADK docs
    "instructions",      # sometimes plural
    "instruction_text",  # occasional variants
    "description",       # used in some examples
    "system_message",    # older examples (your original)
    "system",            # variations
    "prompt",            # fallback
]

# the text we want the agent to use as system prompt / instructions
instr_text = (
    "You are Capstone Companion â€” an agent that helps run and report the capstone pipeline. "
    "When asked, run the pipeline tool and return a concise summary."
)

# 1) Try introspection to show what's allowed (helpful for debugging)
print("\n--- Inspecting Agent for accepted fields ---")
try:
    sig = inspect.signature(Agent)
    print("Agent __init__ signature:", sig)
except Exception:
    # fallback: try pydantic model fields (v2: model_fields; v1: __fields__)
    if hasattr(Agent, "model_fields"):
        print("Agent.model_fields keys:", list(Agent.model_fields.keys()))
    elif hasattr(Agent, "__fields__"):
        print("Agent.__fields__ keys:", list(Agent.__fields__.keys()))
    else:
        print("Couldn't introspect Agent signature cleanly. Proceeding with trial instantiation.")

# 2) Try to construct Agent using candidate param names
root_agent = None
used_field = None
for field in candidates:
    try:
        kwargs = {field: instr_text, "name": "capstone_companion_agent", "model": llm}
        # Some ADK versions expect 'tools' or other args; keep minimal
        root_agent = Agent(**kwargs)
        used_field = field
        print(f"âœ… Agent created using field: '{field}'")
        break
    except ValidationError as ve:
        # create failed - print short message and continue trying next field
        print(f"âœ– '{field}' rejected by Agent (ValidationError). Trying next...")
    except TypeError as te:
        # signature mismatch or unexpected args -> show and continue
        print(f"âœ– '{field}' TypeError: {te}. Trying next...")
    except Exception as e:
        print(f"âœ– '{field}' other error: {e}. Trying next...")

if root_agent is None:
    # last resort: create a minimal agent by inheriting BaseAgent (works if LlmAgent is finicky)
    print("\n!!! Couldn't create Agent with common fields. Falling back to BaseAgent subclass as a last-resort.")
    try:
        from google.adk.agents import BaseAgent

        class MinimalAgent(BaseAgent):
            # Implement required abstract/override methods minimally
            def __init__(self, name="capstone_companion_agent"):
                super().__init__(name=name)

        root_agent = MinimalAgent()
        used_field = "BaseAgent-fallback"
        print("âœ… Created MinimalAgent fallback (BaseAgent subclass).")
    except Exception as e:
        raise RuntimeError("Failed to create any kind of Agent. Paste full traceback here so I can debug further.") from e

# 3) create runner if agent exists
runner = InMemoryRunner(agent=root_agent)
print("\n--- DONE ---")
print("Agent object:", type(root_agent), "used_field:", used_field)


# Helper functions: reading files, scanning code, snapshot

def read_text_file(path: str) -> str:
    """Read a text file safely and return its contents (or error string)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading {path}: {e}"

def collect_project_files(project_dir: str) -> Dict[str, str]:
    """
    Collect key files from a project directory.
    - README.md (if exists)
    - requirements.txt (if exists)
    - All .py files (top-level)
    Returns a dict {relative_path: content}.
    """
    project_dir = os.path.abspath(project_dir)
    file_map: Dict[str, str] = {}

    # README
    readme_path = os.path.join(project_dir, "README.md")
    if os.path.exists(readme_path):
        file_map["README.md"] = read_text_file(readme_path)

    # requirements.txt
    req_path = os.path.join(project_dir, "requirements.txt")
    if os.path.exists(req_path):
        file_map["requirements.txt"] = read_text_file(req_path)

    # Top-level .py files
    py_files = glob.glob(os.path.join(project_dir, "*.py"))
    for p in py_files:
        rel = os.path.relpath(p, project_dir)
        file_map[rel] = read_text_file(p)

    return file_map

def scan_for_keywords(file_map: Dict[str, str], keywords: List[str]) -> Dict[str, List[str]]:
    """
    Very simple keyword scan across all files.
    Returns {filename: [keywords_found]}.
    """
    report: Dict[str, List[str]] = {}
    for filename, content in file_map.items():
        found = []
        for kw in keywords:
            if re.search(rf"\\b{re.escape(kw)}\\b", content):
                found.append(kw)
        if found:
            report[filename] = found
    return report

print("Project helper functions ready.")


#Define Hackathon scoring rubric

RUBRIC = {
    "Core Concept & Value": {
        "max_points": 15,
        "description": "Clarity of idea, usefulness, and uniqueness."
    },
    "Multi-Agent Design": {
        "max_points": 15,
        "description": "Proper multi-agent architecture, separation of concerns."
    },
    "Agent Evaluation & Autonomy": {
        "max_points": 20,
        "description": "How well agents are evaluated, tested, and able to act autonomously."
    },
    "Technical Depth & Implementation": {
        "max_points": 20,
        "description": "Quality of code, use of tools, robustness."
    },
    "Docs & README Quality": {
        "max_points": 15,
        "description": "README completeness, clarity, rubric mapping."
    },
    "Pitch & Presentation": {
        "max_points": 15,
        "description": "Ability to communicate the project: storytelling, clarity, impact."
    },
    "Bonus Creativity / Polish": {
        "max_points": 20,
        "description": "Delight factor, extra details, creative use-cases."
    },
}

TOTAL_MAX_POINTS = sum(v["max_points"] for v in RUBRIC.values())
print(f"Rubric defined. Total max points = {TOTAL_MAX_POINTS}")


# Gemini helper and multi-agent functions

import json, os

def call_gemini(system_instruction: str, user_prompt: str) -> str:
    full_prompt = system_instruction.strip() + "\n\n" + user_prompt.strip()
    response = model.generate_content(full_prompt)
    return getattr(response, "text", str(response))


# AUDITOR AGENT

def auditor_agent(project_dir: str, file_map: dict, keyword_report: dict) -> str:
    system_instruction = (
        "You are the Auditor Agent. Review code quality, structure, maintainability, "
        "and multi-agent related implementations. Return markdown."
    )

    code_blocks = []
    for filename, content in file_map.items():
        if filename.endswith(".py"):
            block = f"[{filename}]\n```python\n{content[:8000]}\n```"
            code_blocks.append(block)

    if code_blocks:
        files_str = "\n\n".join(code_blocks)
    else:
        files_str = "_No Python files found._"

    user_prompt = (
        f"Project: {project_dir}\n\n"
        f"Keyword scan:\n```json\n{json.dumps(keyword_report, indent=2)}\n```\n\n"
        f"Code files:\n{files_str}"
    )
    return call_gemini(system_instruction, user_prompt)


# DOCUMENTATION AGENT

def doc_agent(project_dir: str, file_map: dict) -> str:
    system_instruction = (
        "You are the Documentation Agent. Evaluate README clarity, completeness, "
        "structure, and rubric mapping. Return markdown."
    )

    readme = file_map.get("README.md", "_No README.md found._")

    user_prompt = (
        f"Project: {project_dir}\n\n"
        f"README content:\n```markdown\n{readme}\n```"
    )
    return call_gemini(system_instruction, user_prompt)


# VALIDATOR AGENT 

def validator_agent(project_dir: str, file_map: dict, rubric: dict) -> str:
    system_instruction = (
        "You are the Validator Agent. Score based on rubric, return a markdown "
        "scorecard table + total score + verdict."
    )

    readme = file_map.get("README.md", "")
    code_blocks = []
    for filename, content in file_map.items():
        if filename.endswith(".py"):
            block = f"[{filename}]\n```python\n{content[:4000]}\n```"
            code_blocks.append(block)

    code_str = "\n\n".join(code_blocks) if code_blocks else "_No code found._"

    user_prompt = (
        f"Rubric:\n```json\n{json.dumps(rubric, indent=2)}\n```\n\n"
        f"README:\n```markdown\n{readme}\n```\n\n"
        f"Code:\n{code_str}"
    )
    return call_gemini(system_instruction, user_prompt)


# SCRIPTER AGENT

def scripter_agent(project_dir: str, file_map: dict) -> str:
    system_instruction = (
        "You are the Scripter Agent. Create a short 2â€“3 minute spoken presentation script. "
        "Return clean markdown."
    )

    readme = file_map.get("README.md", "")

    user_prompt = (
        f"Project: {project_dir}\n\n"
        f"Use README for context:\n```markdown\n{readme[:5000]}\n```"
    )
    return call_gemini(system_instruction, user_prompt)


# DEPLOYMENT AGENT 

def deployment_agent(project_dir: str, file_map: dict) -> str:
    system_instruction = (
        "You are the Deployment Agent. Provide run instructions, Dockerfile, "
        "and environment variable notes."
    )

    file_list = "\n".join(f"- {name}" for name in file_map.keys())

    user_prompt = (
        f"Project: {project_dir}\n\n"
        f"Detected files:\n{file_list}"
    )
    return call_gemini(system_instruction, user_prompt)

print("Cell 5 loaded successfully â€” No syntax errors!")


# Orchestrator for the multi-agent system

import json
import os

KEYWORDS = [
    "SequentialAgent", "LoopAgent", "ParallelAgent",
    "tool", "tools", "session", "InMemorySessionService",
    "evaluation", "memory", "deploy", "docker", "cloud"
]

def run_capstone_companion(project_dir: str) -> str:
    """
    Pipeline:
    - Load files
    - Scan keywords
    - Run all agents
    - Combine into one markdown report
    """
    project_dir = os.path.abspath(project_dir)
    print(f"Scanning project at: {project_dir}")

    # Load files + keyword scan
    file_map = collect_project_files(project_dir)
    if not file_map:
        return f"No files found in project root: {project_dir}"

    keyword_report = scan_for_keywords(file_map, KEYWORDS)

    # Run agents sequentially
    print("Running Auditor Agent...")
    auditor_md = auditor_agent(project_dir, file_map, keyword_report)

    print("Running Documentation Agent...")
    doc_md = doc_agent(project_dir, file_map)

    print("Running Validator / Rubric Agent...")
    validator_md = validator_agent(project_dir, file_map, RUBRIC)

    print("Running Scripter Agent...")
    scripter_md = scripter_agent(project_dir, file_map)

    print("Running Deployment Agent...")
    deploy_md = deployment_agent(project_dir, file_map)

    # Build final markdown
    parts = []

    parts.append("# Capstone Companion Report\n")
    parts.append(f"**Project directory:** `{project_dir}`\n")

    parts.append("## Keyword Scan Summary\n")
    parts.append("```json\n" + json.dumps(keyword_report, indent=2) + "\n```")

    parts.append("\n---\n")
    parts.append("## 1. Code Audit (Auditor Agent)\n")
    parts.append(auditor_md)

    parts.append("\n---\n")
    parts.append("## 2. Documentation Review (Documentation Agent)\n")
    parts.append(doc_md)

    parts.append("\n---\n")
    parts.append("## 3. Hackathon Scorecard (Validator Agent)\n")
    parts.append(validator_md)

    parts.append("\n---\n")
    parts.append("## 4. Video Script (Scripter Agent)\n")
    parts.append(scripter_md)

    parts.append("\n---\n")
    parts.append("## 5. Deployment Plan (Deployment Agent)\n")
    parts.append(deploy_md)

    return "\n".join(parts)

print("Orchestrator (run_capstone_companion) ready.")


# Create a small mock project for testing

import pathlib

mock_root = pathlib.Path("mock_capstone_project")
mock_root.mkdir(exist_ok=True)

readme_text = (
    "# The Capstone Companion: An AI Agent to Win This Hackathon\n\n"
    "A multi-agent system that audits, validates, and improves Kaggle Capstone projects\n"
    "by enforcing the official scoring rubric.\n\n"
    "## Problem\n"
    "Hackathon participants struggle to track complex rubrics while building.\n\n"
    "## Solution\n"
    "A local tool that scans the repo and produces a Hackathon Scorecard,\n"
    "plus scripts and deployment hints.\n"
)

main_py = (
    "import os\n\n"
    "def main():\n"
    "    print('Hello from Capstone Companion demo!')\n\n"
    "if __name__ == '__main__':\n"
    "    main()\n"
)

requirements_txt = "google-generativeai\n"

(mock_root / "README.md").write_text(readme_text, encoding="utf-8")
(mock_root / "main.py").write_text(main_py, encoding="utf-8")
(mock_root / "requirements.txt").write_text(requirements_txt, encoding="utf-8")

print(f" Mock project created at: {mock_root.resolve()}")


# Run on the mock project

report = run_capstone_companion("mock_capstone_project")
print(report)


# Cell 9 â€“ Save report to a markdown file

output_dir = "mock_capstone_project"  # change to your real project dir when needed
output_path = os.path.join(output_dir, "capstone_companion_report.md")

with open(output_path, "w", encoding="utf-8") as f:
    f.write(report)

print(f"âœ… Report saved to: {output_path}")



import os
from IPython.display import Markdown, display

# Verify the path (using the variable you defined in Cell 9)
print(f"Reading from: {output_path}")

# Check if file exists first to avoid errors
if os.path.exists(output_path):
    # Read the file content
    with open(output_path, "r", encoding="utf-8") as f:
        md_content = f.read()
    
    # Render and display the markdown
    display(Markdown(md_content))
else:
    print("File not found. Please run Cell 9 to generate the report first.")

