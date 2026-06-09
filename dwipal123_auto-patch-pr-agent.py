# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Dataset reference (not directly used, but included for context)
print("/kaggle/input/agents-intensive-capstone-project/Hackathon dataset.txt")



# Core imports for agent functionality
import os
import json
import shutil
import asyncio
import git
import uuid
import subprocess
import re
import logging  # Added for observability: logging and tracing
from typing import List, Optional
from urllib.parse import quote, urlparse, urlunparse
from datetime import datetime

# ADK imports for multi-agent system
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService  # Sessions & state management
from google.genai import types

# Additional imports for tools and evaluation
from pydantic import BaseModel, Field
import requests  # For OpenAPI-like interactions in PR creation

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Constants
MODEL_NAME = "gemini-2.0-flash"  # Effective use of Gemini for LLM-powered agents
APP_NAME = "auto-patch-pr-agent"
TEMP_REPOS_DIR = "./temp_repos"

# Global stores
ARTIFACT_STORE = {}  # In-memory artifact store for issues
MEMORY_BANK = {}  # Long-term memory bank for adaptive learning (sessions & memory concept)




# # Setup authentication (security: no hardcoded keys)
# try:
#     from kaggle_secrets import UserSecretsClient
#     GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
#     os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
#     print("âœ… Setup and authentication complete.")
# except Exception as e:
#     print(f"ðŸ”‘ Authentication Error: Add 'GOOGLE_API_KEY' to Kaggle secrets. Details: {e}")

# # Configure logging for observability (tracing and metrics)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from dotenv import load_dotenv
import os
# Load environment variables from .env file
# By default, it looks for a file named '.env' in the current directory
load_dotenv()

os.getenv("GOOGLE_API_KEY")


# Utility functions with detailed comments
from pydantic import BaseModel, Field

class file_fixing_status(BaseModel):
    is_file_updated: bool = Field(description="status of file update")
    massege: str = Field(description="details about file update")


def ensure_dir(path: str) -> None:
    """Ensure directory exists; creates if not. Used for temp repo storage."""
    os.makedirs(path, exist_ok=True)

# async runner helper
async def run_agent(runner: Runner, session_id: str, prompt: str) -> str:
    content = types.Content(role="user", parts=[types.Part(text=prompt)])
    final_text = ""
    async for event in runner.run_async(session_id=session_id, user_id="user_1", new_message=content):
        try:
            if getattr(event, "content", None) and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        final_text += part.text
        except Exception:
            continue
    return final_text

def clone_repository(repo_url: str, github_token: str = "", branch: str = "") -> str:
    """Clone a GitHub repository to a local temp dir and optionally checkout a branch.

    Key concept: Custom tool for repo management.
    Returns local path on success or error string on failure.
    """
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    local_path = os.path.abspath(os.path.join(TEMP_REPOS_DIR, repo_name))
    if os.path.exists(local_path):
        shutil.rmtree(local_path)
    ensure_dir(os.path.dirname(local_path))

    try:
        if github_token:
            auth_url = repo_url.replace("https://", f"https://{github_token}@")
            git.Repo.clone_from(auth_url, local_path)
        else:
            git.Repo.clone_from(repo_url, local_path)
    except Exception as e:
        return f"Error cloning repo: {e}"

    # Branch handling for flexibility
    if branch:
        try:
            repo = git.Repo(local_path)
        except Exception as e:
            return f"Cloned to {local_path} but failed to open repo: {e}"

        local_branch_names = [b.name for b in repo.branches]
        if branch in local_branch_names:
            try:
                repo.git.checkout(branch)
                return local_path
            except Exception as e:
                return f"Cloned to {local_path} but failed to checkout local branch '{branch}': {e}"

        try:
            origin = repo.remotes.origin
            origin.fetch()
            origin_ref = f"origin/{branch}"
            remote_refs = [r.name for r in repo.refs]
            if origin_ref in remote_refs:
                try:
                    repo.create_head(branch, repo.refs[origin_ref]).set_tracking_branch(repo.refs[origin_ref])
                    repo.git.checkout(branch)
                    return local_path
                except Exception as e:
                    return f"Cloned to {local_path} but failed to create/track branch '{branch}': {e}"
            else:
                return f"Error: Branch '{branch}' not found locally or on origin after clone."
        except Exception as e:
            return f"Cloned to {local_path} but failed to fetch origin to find branch '{branch}': {e}"

    return local_path

def scan_files(local_path: str) -> List[str]:
    """Return list of .py files under local_path (ignores common dirs).

    Supports file discovery for linting.
    """
    files_found = []
    ignore = {'.git', '.venv', '__pycache__', 'node_modules'}
    for root, dirs, files in os.walk(local_path):
        dirs[:] = [d for d in dirs if d not in ignore]
        for f in files:
            if f.endswith(".py"):
                files_found.append(os.path.join(root, f))
    return files_found

def run_linter_and_store(local_path: str) -> str:
    """Run 'ruff check' on the given path. Store issues in ARTIFACT_STORE and return a reference ID.

    Key concept: Custom tool for linting; integrates with memory for issue tracking.
    Returns human-readable error strings on failure.
    """
    try:
        proc = subprocess.run(
            ["ruff", "check", local_path, "--output-format=json"],
            capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return "Error: Ruff linter not installed."
    except Exception as e:
        return f"Error running linter: {e}"

    raw = proc.stdout.strip() or "[]"
    try:
        issues = json.loads(raw)
    except json.JSONDecodeError:
        return "Error: Linter output not valid JSON."

    if not issues:
        return "No issues found."

    report_id = str(uuid.uuid4())
    ARTIFACT_STORE[report_id] = {"issues": issues, "count": len(issues), "repo_path": local_path}
    # Update memory bank for long-term learning
    MEMORY_BANK["last_issues"] = issues
    return f"Issues stored. Reference ID: {report_id}"

def fetch_issue_batch(report_id: str, batch_size: int = 3) -> str:
    """Fetch a batch of issues from ARTIFACT_STORE.

    Supports incremental processing.
    """
    data = ARTIFACT_STORE.get(report_id)
    if not data:
        return "Error: Report ID not found."
    batch = data.get("issues", [])[:batch_size]
    return json.dumps(batch) if batch else "No more issues to fix."

def write_file(file_path: str, content: str) -> str:
    """Write content to a file.

    Key concept: Custom tool for file modifications.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return f"Updated {file_path}"
    except Exception as e:
        return f"Error: {e}"

def display_artifact_changes(report_id: str) -> None:
    """Print a concise, human-friendly summary of planned changes grouped by filename.

    Enhances observability with formatted output.
    """
    data = ARTIFACT_STORE.get(report_id)
    if not data:
        print(f"\n\033[1;31mNo artifact found for ID: {report_id}\033[0m\n")
        return
    issues = data.get("issues", [])
    if not issues:
        print("\n\033[1;32mNo issues recorded in artifact.\033[0m\n")
        return

    print(f"\n\033[1;36mPlanned changes (report: {report_id})\033[0m\n")
    files = {}
    for it in issues:
        fname = it.get("filename") or it.get("path") or "<unknown>"
        files.setdefault(fname, []).append(it)

    for fname, its in files.items():
        print(f"\033[1;35m--- {fname} ---\033[0m")
        for i, issue in enumerate(its, start=1):
            code = issue.get("code") or issue.get("rule") or issue.get("type") or ""
            msg = issue.get("message") or issue.get("description") or ""
            line = issue.get("line") or (issue.get("location") or {}).get("start", {}).get("line")
            col = issue.get("col") or (issue.get("location") or {}).get("start", {}).get("col")
            suggestion = issue.get("fix") or issue.get("suggestion") or issue.get("replacement")

            loc = f" (line:{line}" + (f", col:{col}" if col else "") + ")" if line else ""
            head = f"[{code}]{loc}" if code or loc else f"[{i}]"
            print(f" {head} {msg}")
            if suggestion:
                if isinstance(suggestion, dict):
                    s = suggestion.get("content") or suggestion.get("patch") or suggestion.get("replacement") or str(suggestion)
                else:
                    s = str(suggestion)
                print(f"    \033[1;32mSuggestion:\033[0m {s}")
        print("")
    print("")


def _parse_github_owner_repo(url: str) -> Optional[tuple]:
    m = re.search(r"github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/.]+)", url or "")
    if not m:
        return None
    return m.group("owner"), m.group("repo").replace(".git", "")



def create_github_pr(local_path: str, repo_url: str, github_token: str = "") -> str:
    """
    Create a branch, push it, and open a PR. Returns PR URL or an error message.
    This is a simplified but functional rework of the original logic with clearer steps.
    """
    import requests

    try:
        repo = git.Repo(local_path)
    except Exception as e:
        return f"Error opening repo at {local_path}: {e}"

    try:
        current_branch = repo.active_branch.name
    except TypeError:
        return "Error: repository in detached HEAD; cannot determine current branch."
    except Exception as e:
        return f"Error determining current branch: {e}"

    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    safe_branch = re.sub(r'[^A-Za-z0-9._-]+', '-', current_branch).strip('-')
    new_branch = f"fix-issues-{safe_branch}-{timestamp}"

    try:
        new_ref = repo.create_head(new_branch, commit=repo.head.commit)
        new_ref.checkout()
        if repo.is_dirty(untracked_files=True):
            repo.git.add("--all")
            repo.index.commit(f"Auto-fixes committed by tool on {timestamp}")
    except Exception as e:
        return f"Error creating/checking out branch '{new_branch}': {e}"

    try:
        origin = repo.remotes.origin
    except Exception:
        return "Error: remote 'origin' not found."

    owner_repo = _parse_github_owner_repo(repo_url) or _parse_github_owner_repo(origin.url)
    owner, repo_name = (owner_repo + (None,))[:2] if owner_repo else (None, None)

    auth_user = None
    headers = {"Accept": "application/vnd.github+json"}
    if github_token and owner and repo_name:
        headers["Authorization"] = f"token {github_token}"
        try:
            uresp = requests.get("https://api.github.com/user", headers=headers, timeout=10)
            if uresp.status_code == 200:
                auth_user = uresp.json().get("login")
            else:
                return f"Error: token authentication failed: {uresp.status_code} {uresp.text}"
            # quick permission check
            repo_api = f"https://api.github.com/repos/{owner}/{repo_name}"
            resp = requests.get(repo_api, headers=headers, timeout=10)
            if resp.status_code == 200:
                perms = resp.json().get("permissions", {})
                if not perms.get("push", False):
                    private = resp.json().get("private", False)
                    scope_hint = "repo (full 'repo' scope required for private repositories)" if private else "public_repo (or 'repo')"
                    return (f"Token authenticated as '{auth_user}' but does NOT have push access to {owner}/{repo_name}.\n"
                            f"Likely causes: token scope/membership issues. Repo permissions: {perms}")
        except Exception as e:
            return f"Error checking token/permissions: {e}"

    pushed = False
    try:
        origin.push(refspec=f"{new_branch}:{new_branch}", set_upstream=True)
        pushed = True
    except Exception:
        # try authenticated remote URL if token provided
        if github_token and origin.url and origin.url.startswith("http"):
            orig_url = origin.url
            try:
                token_enc = quote(github_token, safe='')
                parsed = urlparse(orig_url)
                host = parsed.hostname or ''
                port = f":{parsed.port}" if parsed.port else ''
                new_netloc = f"x-access-token:{token_enc}@{host}{port}"
                authed = parsed._replace(netloc=new_netloc)
                origin.set_url(urlunparse(authed))
                repo.git.push("--set-upstream", "origin", new_branch)
                pushed = True
            except Exception as e:
                user_hint = f" (token user: '{auth_user}')" if auth_user else ""
                return (f"Error pushing branch with token-authenticated remote: {e}{user_hint}")
            finally:
                try:
                    origin.set_url(orig_url)
                except Exception:
                    pass
        else:
            try:
                repo.git.push("--set-upstream", "origin", new_branch)
                pushed = True
            except Exception as e:
                user_hint = f" (token user: '{auth_user}')" if auth_user else ""
                return f"Error pushing branch '{new_branch}': {e}{user_hint}"

    if not pushed:
        return f"Branch '{new_branch}' created locally but push failed."

    if not github_token:
        return f"Branch created and pushed: {new_branch}. No token provided to create PR. Base: {current_branch}"

    if not owner or not repo_name:
        return "Error: cannot parse owner and repo name."

    # prefer current_branch as base, fallback to repo default if needed
    base_branch = current_branch
    try:
        br_resp = requests.get(f"https://api.github.com/repos/{owner}/{repo_name}/branches/{base_branch}", headers=headers, timeout=10)
        if br_resp.status_code != 200:
            repo_meta = requests.get(f"https://api.github.com/repos/{owner}/{repo_name}", headers=headers, timeout=10)
            if repo_meta.status_code == 200:
                base_branch = repo_meta.json().get("default_branch") or base_branch
    except Exception:
        pass

    head_branch = new_branch
    try:
        if auth_user and auth_user != owner:
            check_upstream = requests.get(f"https://api.github.com/repos/{owner}/{repo_name}/branches/{new_branch}", headers=headers, timeout=10)
            if check_upstream.status_code != 200:
                head_branch = f"{auth_user}:{new_branch}"
    except Exception:
        pass

    api_url = f"https://api.github.com/repos/{owner}/{repo_name}/pulls"
    payload = {"title": f"Auto-fix: {new_branch}", "head": head_branch, "base": base_branch, "body": "Automated fixes created by tool."}
    try:
        resp = requests.post(api_url, json=payload, headers=headers, timeout=15)
    except Exception as e:
        return f"Error calling GitHub API to create PR: {e}"

    if resp.status_code in (200, 201):
        pr = resp.json()
        pr_url = pr.get("html_url", str(pr))
        try:
            repo.git.checkout(current_branch)
        except Exception as e:
            return f"{pr_url} (PR created; failed to checkout back to '{current_branch}': {e})"
        return pr_url
    elif resp.status_code == 422:
        return "Error creating PR: 422 Validation Failed. Ensure base/head are correct."
    else:
        return f"Error creating PR: {resp.status_code} {resp.text}"


# Main workflow (cleaner and easier to follow)
async def main():
    print("=== Google ADK Scalable Multi-Agent Tool ===")
    repo_url = input("Repo URL: ").strip()
    gh_token = input("GitHub Token: ").strip()
        # Prompt branch before invoking the clone tool so we can pass it in the prompt when present
    branch = input("Branch to analyze (leave empty to use default/current branch): ").strip()

    session_service = InMemorySessionService()
    session_id = "session_main"
    await session_service.create_session(session_id=session_id, user_id="user_1", app_name=APP_NAME)

    repo_cloner = Agent(name="RepoCloner", model=MODEL_NAME, tools=[clone_repository], instruction="Clone the repo. return the local path.")
    analyzer = Agent(name="Analyzer", model=MODEL_NAME, tools=[run_linter_and_store], instruction="Run linter. It returns a Reference ID. Output ONLY that ID.")
    fixer = Agent(
        name="Fixer", model=MODEL_NAME, tools=[write_file],
        instruction=("Use the current file content and issues. Update the content according to issues. "
                     "When done, set is_file_updated true/false and add your massege."
                     "Note: if you feel updating file can cause problem then set is_file_updated to false and explain why in massege."),
        output_schema=file_fixing_status
    )
    publisher = Agent(name="Publisher", model=MODEL_NAME, tools=[create_github_pr], instruction="Create a GitHub PR.")

    runner_scan = Runner(agent=repo_cloner, app_name=APP_NAME, session_service=session_service)
    runner_analyze = Runner(agent=analyzer, app_name=APP_NAME, session_service=session_service)
    runner_fix = Runner(agent=fixer, app_name=APP_NAME, session_service=session_service)
    runner_pub = Runner(agent=publisher, app_name=APP_NAME, session_service=session_service)

    print("\n[repo_cloner]: Cloning...")
    # Build prompt â€” include branch only if provided
    clone_prompt = f"Clone {repo_url} (token: {gh_token})"
    if branch:
        clone_prompt += f" branch: {branch}"
    # clone_prompt += " and scan."
    scan_out = await run_agent(runner_scan, session_id, clone_prompt)

    # The repo_cloner tool returns the local path directly in normal flow. Fall back to predictable path.
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    local_path = scan_out if os.path.exists(scan_out) else os.path.abspath(os.path.join(TEMP_REPOS_DIR, repo_name))

    print("\n[Analyzer]: Analyzing...")
    analysis_resp = await run_agent(runner_analyze, session_id, f"Run linter on {local_path}")
    print(f"Agent Output: {analysis_resp}")

    match = re.search(r"([a-f0-9\-]{36})", analysis_resp)
    if not match:
        print("No Reference ID returned.")
        return

    report_id = match.group(1)
    print(f"Artifact ID: {report_id}")
    display_artifact_changes(report_id)

    if input("Fix issues? (y/n): ").lower() != 'y':
        print("User opted not to fix issues. Exiting.")
        return

    print("\n[Fixer]: Fixing...")
    artifact = ARTIFACT_STORE.get(report_id)
    if not artifact:
        print(f"report id {report_id} not found. No issues to fix.")
        return

    # Process a small number of issues for demo (preserve original behavior of slicing)
    for issue in artifact.get("issues", []):
        filename = issue.get("filename")
        if not filename or not os.path.exists(filename):
            print(f"Skipping invalid file: {filename}")
            continue
        with open(filename, "r", encoding="utf-8") as fh:
            current_content = fh.read()
        prompt = (
            "Here is the current file content and suggestion by ruff. Fix the code according to suggestions.\n\n"
            f"File Content:\n{current_content}\n\nsuggestion:\n{json.dumps(issue, default=str)}"
        )
        fix_resp = await run_agent(runner_fix, session_id, prompt)
        # print("Fixer response:", fix_resp)
        print("====")

    print("\n[Publisher]: Publishing...")
    pub_resp = await run_agent(runner_pub, session_id, f"Create PR for {local_path} repo {repo_url}. use git token {gh_token} if required")
    print("Publish result:", pub_resp)


await main()




