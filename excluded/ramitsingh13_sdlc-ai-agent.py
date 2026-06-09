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


import os
import uuid
from typing import Dict, Any
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google.adk.agents import Agent, LlmAgent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.function_tool import FunctionTool
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.tools import AgentTool
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.adk.tools.tool_context import ToolContext


retry_config = types.HttpRetryOptions(
    attempts=4,
    exp_base=4,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)


def run_python_code(code: str) -> Dict[str, Any]:
    """
    Simulate running user-generated python code.
    In ADK we will delegate to a CalculationAgent / BuiltInCodeExecutor - here we provide a simple wrapper
    that returns a dict with status/result. In production you'd use ADK's code_executor tooling.
    """
    # Simple sandbox: write code to a temp file and run with python - but Kaggle may restrict subprocesses.
    # We'll just return the code for the example. Replace this with BuiltInCodeExecutor use in ADK context.
    return {"status": "ok", "stdout": f"--- simulated run ---\n{code[:300]}...", "code": code}


def deploy_artifact(artifact: Dict[str, Any]) -> Dict[str, Any]:
    # artifact: dict with 'service', 'package', 'version' keys
    service = artifact.get("service", "unknown-service")
    version = artifact.get("version", "v0.0.0")
    return {"status": "deployed", "service": service, "version": version, "message": f"Deployed {service}:{version}"}



run_code_tool = FunctionTool(func=run_python_code)
deploy_tool = FunctionTool(func=deploy_artifact)


requirements_agent = Agent(
    name="RequirementsAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
You are the Requirements engineer. Given a project short description (user input), produce:
1) A short project goal (1 sentence)
2) A prioritized list of functional & non-functional requirements (3-6 items)
3) A basic acceptance criteria section (2-4 bullets)
Output as a JSON object with keys: goal, requirements, acceptance_criteria
""",
    output_key="requirements",
    # no external tools needed
)


design_agent = Agent(
    name="DesignAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
You are a System Designer. Use the requirements available as {requirements}.
Produce:
1) High-level architecture (3-5 components)
2) Data model / main entities (3-6 items)
3) Interface contract / API stubs (minimal example)
Output as JSON: architecture, entities, api_stubs
""",
    output_key="design",
    # will receive {requirements} from previous agent via session state
)


implementation_agent = Agent(
    name="ImplementationAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
You are a Developer. Using {design}, generate:
1) Minimal skeleton code for service 'app' (a Python module with a main function)
2) Minimal requirements.txt content
3) Brief README instructions for running locally
Return JSON keys: skeleton_code, requirements_txt, readme
""",
    output_key="implementation",
)


testing_agent = Agent(
    name="TestingAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
You are the QA/Test engineer. Based on {implementation} and {requirements}, produce:
1) 3 unit test cases in pytest format (provide code)
2) A simple test-run checklist
Return JSON: tests_code, checklist
""",
    output_key="testing",
    tools=[run_code_tool],  # testing agent may call the run_code tool to simulate running tests
)


debugger_agent = Agent(
    name="DebuggerAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
You are an automated debugger. Inputs: failing test output and the code (from {implementation}).
Your job:
1) Identify probable cause(s) in 2-3 bullet points
2) Provide a corrected code patch (full updated code)
3) Return JSON: analysis, patched_code
""",
    output_key="debugger",
    tools=[run_code_tool]
)


deployer_agent = Agent(
    name="DeployerAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
You are the Deployment/DevOps assistant. Given a built artifact (from {implementation}) and a version tag,
1) Provide deployment steps (3-6 bullets)
2) Call the deploy tool with artifact={'service': 'app','package': 'app.tar.gz','version': version}
Return JSON: steps, deploy_result
""",
    output_key="deployment",
    tools=[deploy_tool],
)



maintenance_agent = Agent(
    name="MaintenanceAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
You are the Maintenance engineer. Given the project state ({requirements},{design},{implementation},{testing},{deployment}),
produce:
1) Monitoring checklist (3 bullets)
2) Simple incident runbook (3 steps)
3) Suggestions for next improvements (3 items)
Return JSON: monitoring, incident_runbook, improvements
""",
    output_key="maintenance",
)


root_sequential = SequentialAgent(
    name="PlanAndBuild",
    sub_agents=[requirements_agent, design_agent, implementation_agent],
)


parallel_work = ParallelAgent(
    name="TestDebugDeploy",
    sub_agents=[testing_agent, debugger_agent, deployer_agent, maintenance_agent],
)


sdlc_pipeline = SequentialAgent(
    name="SDLCPipeline",
    sub_agents=[root_sequential, parallel_work],
)


runner = InMemoryRunner(agent=sdlc_pipeline)

# ---- Example: Run the pipeline with a short user prompt ----
async def run_sdlc_example(project_prompt: str, version_tag: str = "v0.1.0"):
    """
    Runs the SDLC pipeline end-to-end. This function streams events and
    demonstrates how the ADK pipeline would orchestrate phase agents.
    """
    # Build initial user message
    user_message = types.Content(parts=[types.Part(text=project_prompt)])
    # The ADK runner will populate session state as agents run
    print("ðŸš€ Starting SDLC pipeline for:", project_prompt)
    async for event in runner.run_async(user_id="user_demo", session_id=str(uuid.uuid4()), new_message=user_message):
        # Print final text outputs as they arrive (simplified)
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(part.text)
        # If the event contains function responses (deploy_result etc), show the structured info
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "function_response") and part.function_response:
                    print("ðŸ”§ Function response:", part.function_response.response)


!pip install gradio==4.19.2 --no-deps --quiet


import gradio as gr
import asyncio
import nest_asyncio
nest_asyncio.apply()


async def run_sdlc_ui(project_description, version_tag="v0.1.0"):

    user_message = types.Content(parts=[types.Part(text=project_description)])
    session_id = str(uuid.uuid4())
    output_text = "ðŸš€ **Starting SDLC pipeline...**\n\n"

    async for event in runner.run_async(
        user_id="ui_user",
        session_id=session_id,
        new_message=user_message
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    output_text += part.text + "\n\n"

        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "function_response") and part.function_response:
                    output_text += "ðŸ”§ **Tool Output:**\n"
                    output_text += str(part.function_response.response) + "\n\n"

    return output_text


def gradio_wrapper(project_description, version_tag):
    return asyncio.run(run_sdlc_ui(project_description, version_tag))


with gr.Blocks(title="AI Multi-Agent SDLC Assistant") as demo:

    gr.Markdown("""
    # ðŸ¤– AI Multi-Agent SDLC Assistant  
    Enter a short project idea, and this app will generate:
    - Requirements  
    - System Design  
    - Skeleton Code  
    - Tests  
    - Debugging Suggestions  
    - Deployment Steps  
    - Maintenance Plan  

    All phases use your ADK Multi-Agent pipeline.
    """)

    with gr.Row():
        project_input = gr.Textbox(
            label="Project Description",
            placeholder="e.g., Build a simple task-tracking web app",
            lines=3,
        )

    version_input = gr.Textbox(
        label="Version Tag",
        value="v0.1.0",
    )

    run_btn = gr.Button("ðŸš€ Run SDLC Pipeline")

    output_box = gr.Markdown(
        "Results will appear here...",
    )

    run_btn.click(
        gradio_wrapper,
        inputs=[project_input, version_input],
        outputs=output_box,
    )


demo.launch()

