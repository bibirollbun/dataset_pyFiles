!pip install python-dotenv google-adk litellm mcp asyncio


import asyncio
import os
import sys
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.genai import types
from mcp import StdioServerParameters


DEFAULT_MODEL = "openai/gpt-4o"
DEFAULT_INSTRUCTION = """
You are a background web automation agent. 
Execute the steps provided faithfully. 
If a step fails, report the error.
"""


@dataclass
class Job:
    id: str
    name: str
    status: str = "PENDING"  # PENDING, RUNNING, COMPLETED, FAILED
    logs: List[str] = field(default_factory=list)
    task: Optional[asyncio.Task] = None
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


class WorkflowManager:
    """Manages saved workflows."""
    def __init__(self):
        self.workflows: Dict[str, List[str]] = {
            "demo_insurance": [
                "navigate to 'https://www.royalsundaram.in/MOPIS/Login.jsp'",
                "Enter username 'invictus' and password 'Secret123', click sign in",
                "Click 'Rating Calculator' -> 'New Business' -> 'Private Car'",
                "Enter vehicle MH 02 FR 1294 and click get started",
                "download the generated pdf"
            ],
            "google_check": [
                "navigate to google.com",
                "search for 'Google ADK python'",
                "summarize the first result"
            ]
        }

    def get_workflow(self, name: str) -> List[str]:
        return self.workflows.get(name, [])

    def add_workflow(self, name: str, steps: List[str]):
        self.workflows[name] = steps

    def list_workflows(self) -> List[str]:
        return list(self.workflows.keys())


class WebAgent:
    """A self-contained agent instance."""
    def __init__(self, job_id: str, log_callback):
        self.job_id = job_id
        self.log_callback = log_callback
        self.agent = None
        self.runner = None

    def log(self, message: str):
        """Helper to send logs to the job storage."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {message}"
        self.log_callback(formatted_msg)

    async def setup(self):
        """Initializes the LLM and MCP tools specific to this agent."""
        self.log("Initializing Agent & Tools...")
        load_dotenv()
        
        # 1. Setup Retry Logic
        retry_config = types.HttpRetryOptions(
            attempts=3, exp_base=2, initial_delay=1, max_delay=10, 
            http_status_codes=[429, 500, 503]
        )

        # 2. Setup Playwright Tool
        server_params = StdioServerParameters(command="npx", args=["@playwright/mcp@latest"])
        playwright_tool = McpToolset(
            connection_params=StdioConnectionParams(server_params=server_params, timeout=60)
        )

        # 3. Setup LLM
        model = LiteLlm(
            model=DEFAULT_MODEL,
            temperature=0.01,
            max_retries=3
        )

        # 4. Create Agent
        self.agent = LlmAgent(
            model=model,
            name=f"agent_{self.job_id}",
            instruction=DEFAULT_INSTRUCTION,
            tools=[playwright_tool],
        )
        self.runner = InMemoryRunner(agent=self.agent, app_name=f"runner_{self.job_id}")
        self.log("Agent Setup Complete.")

    async def run_steps(self, steps: List[str]):
        """Executes the workflow steps."""
        if not self.agent:
            await self.setup()

        self.log(f"Starting execution of {len(steps)} steps.")
        for i, step in enumerate(steps, 1):
            self.log(f"Step {i}: {step}")
            try:
                response = await self.runner.run_debug(step, verbose=False)
                self.log(f"Result: {response}")
            except Exception as e:
                self.log(f"â�Œ Error on step {i}: {str(e)}")
                raise e
        
        self.log("âœ… Workflow completed successfully.")


class JobManager:
    """Orchestrates background jobs."""
    def __init__(self):
        self.jobs: Dict[str, Job] = {}

    def create_job(self, name: str, steps: List[str]) -> str:
        job_id = str(uuid.uuid4())[:8]
        job = Job(id=job_id, name=name)
        self.jobs[job_id] = job
        
        task = asyncio.create_task(self._run_job(job, steps))
        job.task = task
        return job_id

    async def _run_job(self, job: Job, steps: List[str]):
        """Internal runner that handles the lifecycle."""
        job.status = "RUNNING"
        
        def job_logger(msg):
            job.logs.append(msg)

        agent = WebAgent(job.id, job_logger)
        
        try:
            await agent.setup()
            await agent.run_steps(steps)
            job.status = "COMPLETED"
        except asyncio.CancelledError:
            job.status = "CANCELLED"
            job_logger("âš ï¸� Job was cancelled by user.")
        except Exception as e:
            job.status = "FAILED"
            job_logger(f"ğŸ’¥ Critical Failure: {str(e)}")

    def list_jobs(self):
        return self.jobs.values()

    def get_job(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job and job.task and not job.task.done():
            job.task.cancel()
            return True
        return False


# Initialize the workflow and job managers
workflow_mgr = WorkflowManager()
job_mgr = JobManager()

print("âœ… Managers initialized!")
print("\nAvailable commands:")
print("  - workflow_mgr.list_workflows()")
print("  - job_id = job_mgr.create_job(name, steps)")
print("  - job_mgr.list_jobs()")
print("  - job = job_mgr.get_job(job_id)")
print("  - job_mgr.cancel_job(job_id)")


workflows = workflow_mgr.list_workflows()
print("Available workflows:")
for w in workflows:
    print(f"  - {w}")


# Example: Start the google_check workflow
workflow_name = "google_check"
steps = workflow_mgr.get_workflow(workflow_name)

if steps:
    job_id = job_mgr.create_job(workflow_name, steps)
    print(f"âœ… Job started with ID: {job_id}")
else:
    print(f"â�Œ Workflow '{workflow_name}' not found")


# List all jobs
import pandas as pd

jobs = list(job_mgr.list_jobs())
if jobs:
    job_data = [{"ID": j.id, "Name": j.name, "Status": j.status, "Started": j.created_at} for j in jobs]
    df = pd.DataFrame(job_data)
    display(df)
else:
    print("No jobs found")


# Replace with your actual job_id
job_id_to_check = "your_job_id_here"

job = job_mgr.get_job(job_id_to_check)
if job:
    print(f"Logs for Job {job.id} ({job.name}):")
    print("=" * 60)
    for log in job.logs:
        print(log)
else:
    print(f"Job {job_id_to_check} not found")


# Create a custom workflow
new_workflow_name = "my_custom_workflow"
new_steps = [
    "navigate to example.com",
    "click on 'More information'",
    "take a screenshot"
]

workflow_mgr.add_workflow(new_workflow_name, new_steps)
print(f"âœ… Workflow '{new_workflow_name}' created!")


# Replace with the job_id you want to cancel
job_id_to_cancel = "your_job_id_here"

success = job_mgr.cancel_job(job_id_to_cancel)
if success:
    print(f"âš ï¸� Job {job_id_to_cancel} cancelled")
else:
    print(f"â�Œ Could not cancel job {job_id_to_cancel}")


# Wait for a specific job to complete
async def wait_for_job(job_id: str, poll_interval: int = 2):
    """Poll job status until completion."""
    while True:
        job = job_mgr.get_job(job_id)
        if not job:
            print(f"Job {job_id} not found")
            break
        
        print(f"Job {job_id} status: {job.status}")
        
        if job.status in ["COMPLETED", "FAILED", "CANCELLED"]:
            print(f"\nFinal status: {job.status}")
            print("\nFinal logs:")
            for log in job.logs:
                print(log)
            break
        
        await asyncio.sleep(poll_interval)

# Example usage:
# await wait_for_job("your_job_id_here")

