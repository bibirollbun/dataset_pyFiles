!pip install "gradio[mcp]"

from google.adk.agents import Agent, LlmAgent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types

from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from google.adk.apps.app import App, ResumabilityConfig
from google.adk.tools.function_tool import FunctionTool

from google.adk.sessions import DatabaseSessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import load_memory, preload_memory

from google.adk.runners import Runner
from google.adk.plugins.logging_plugin import (
    LoggingPlugin,
)
from google.adk.code_executors import BuiltInCodeExecutor

print("âœ… ADK components imported successfully.")


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


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)


# function to read a file from a given location and return its content
def read_file(file_path: str) -> str:
    with open(file_path, 'r') as file:
        content = file.read()
    return content


# function to write content to a file at a given location
def write_file(file_path: str, content: str) -> None:
    """Writes the given content to a file at a given location"""
    with open(file_path, 'w') as file:
        file.write(content)


#function to fetch a URL and return the content
def fetch_url(url: str) -> str:
    import requests
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses
    return response.text


async def auto_save_to_memory(callback_context):
    """Automatically save session to memory after each agent turn."""
    await callback_context._invocation_context.memory_service.add_session_to_memory(
        callback_context._invocation_context.session
    )

print("âœ… Callback created.")


APP_NAME = "default"  # Application
USER_ID = "default"  # User
SESSION = "default"  # Session

# Consolidated agent definitions with explicit input_key and output_key
# Root coordinator agent
root_agent = Agent(
    name="requirement_assistant",
    model="gemini-2.5-flash-lite",
    description="Coordinate tasks and collect requirements.",
    instruction=""""
        Collect the requirement and state the problem clearly.
        """,
    tools=[fetch_url, read_file],
    after_agent_callback=auto_save_to_memory,  # Saves after each turn!
    output_key="root_agent_requirements",
)
print("âœ… root_agent defined.")

# Summarizer agent: summarizes research/findings into key points
summarizer_agent = Agent(
    name="SummarizerAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="Read the provided research findings and return 3-5 bulleted key points. (Requirements: {root_agent_requirements})",
    after_agent_callback=auto_save_to_memory,  # Saves after each turn!
    output_key="final_summary",
)
print("âœ… summarizer_agent created.")

# Coding agent: writes candidate implementation based on findings and requirements
coding_agent = Agent(
    name="CodingAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
        Write concise, well-tested code and nothing else based on the provided requirements and research findings. 
        (Final Summary: {final_summary}, Root Agent Requirements: {root_agent_requirements})
        Write the code to a file called 'candidate_code.py'
        Do not attempt to run the file.
        """,
     tools=[write_file],
    output_key="candidate_code",
)
print("âœ… coding_agent created.")

# Unit test agent: generates unit tests for the candidate code
test_agent = Agent(
    name="TestAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    Generate unit tests that validate behaviors described by the requirements given here: {root_agent_requirements} 
    Use the read_file to read the candidate code named 'candidate_code.py'
    (Root Agent Requirements: {root_agent_requirements})
    Write the tests to a file called  'unit_tests.py'
    Do not attempt to run the file.
    """,
    tools=[preload_memory, write_file, read_file],
    output_key="unit_tests",
)
print("âœ… test_agent created.")

# Functional test agent: produces high-level end-to-end tests
functional_test_agent = Agent(
    name="FunctionalTestAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    Produce Playwright tests for the requirements given here: {root_agent_requirements} 
    Use the read_file to read the candidate code named 'candidate_code.py'
    user's request for Playwright tests
    Do not attempt to implement the requirement yourself.
    Write the Playwright tests  to a file called 'functional_tests.py
    Do not attempt to run the file.
    """,
    tools=[write_file, read_file],
    output_key="functional_tests",
)
print("âœ… functional_test_agent created.")

# Mutation test agent: proposes mutation tests to evaluate test-suite strength
mutation_test_agent = Agent(
    name="MutationTestAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    Write mutation tests using mutpy based on the candidate code. 
    Use the read_file tool to read the candidate code named 'candidate_code.py'
    Write the mutation tests to a file called 'mutation_tests.py'
    Do not attempt to run the file.
    """,
    tools=[preload_memory, write_file, read_file],
    output_key="mutation_test_findings",
)
print("âœ… mutation_test_agent created.")

# Property-based test agent: generates QuickCheck-style properties
property_based_test_agent = Agent(
    name="PropertyBasedTestAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    Generate property-based tests  and nothing else using pytest from deterministic functions and functional tests. 
    Use the read_file tool to read the candidate code named 'candidate_code.py'
    (Functional Tests: {functional_tests})
    Write the property based tests to a file called 'property_based_tests.py'
    Do not attempt to run the file.
    """,
    tools=[preload_memory, write_file, read_file],
    output_key="property_based_test_findings",
)
print("âœ… property_based_test_agent created.")

# Chaos engineering agent: designs resilience experiments
chaos_engineering_test_agent = Agent(
    name="ChaosEngineeringTestAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""Write Tests for safe chaos experiments (network latency, resource pressure) and recovery checks  and nothing else. 
    Use the read_file tool to read the candidate code named 'candidate_code.py'
    Read the Unit Tests here and avoid witing those tests again: {unit_tests}, 
    Read the functional tests here: {functional_tests},
    Read the root agent requirements here: {root_agent_requirements}
    Write the chaos engineering tests to a file called 'chaos_engg_tests.py'
    Do not attempt to run the file.
    """,
    tools=[preload_memory, write_file, read_file],
    output_key="chaos_engineering_tests",
)
print("âœ… chaos_engineering_test_agent created.")

# Adversarial/robustness agent: creates defensive adversarial tests and mitigations
adversarial_test_agent = Agent(
    name="AdversarialTestAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    Generate defensive adversarial test cases (fuzzing, malformed inputs, boundary checks) and suggest mitigations. 
    Do NOT generate exploit code. Also avoid repeating existing unit tests and functional tests. 
    Use the read_file tool to read the candidate code named 'candidate_code.py'
    Read the Unit Tests here and avoid witing those tests again: {unit_tests}, 
    Read the functional tests here: {functional_tests},
    Read the root agent requirements here: {root_agent_requirements})
    Write the adversarial tests to a file called 'adversarial_tests.py'
    Do not attempt to run the file.
    """,
    tools=[preload_memory, write_file, read_file],
    output_key="adversarial_tests",
)
print("âœ… adversarial_test_agent created.")


# The ParallelAgent runs all its sub-agents simultaneously.
parallel_research_team = ParallelAgent(
    name="ParallelResearchTeam",
    sub_agents=[ chaos_engineering_test_agent, adversarial_test_agent, property_based_test_agent, mutation_test_agent ], # TODO Fix hallucinations, currently this breaks randomly. 
)

# This SequentialAgent defines the high-level workflow: run the parallel team first, then run the aggregator.
sequential_agent = SequentialAgent(
    name="ResearchSystem",
    sub_agents=[root_agent, summarizer_agent,  coding_agent, functional_test_agent, test_agent, parallel_research_team ],
)

print("âœ… Parallel and Sequential Agents created.")


# DatabaseSessionService
# SQLite database will be created automatically
db_url = "sqlite:///my_agent_data.db"  # Local SQLite file
session_service = DatabaseSessionService(db_url=db_url)

memory_service = (
    InMemoryMemoryService()
)  # ADK's built-in Memory Service for development and testing


# Step 3: Create a new runner with persistent storage
runner = Runner(
            agent=sequential_agent, 
            app_name=APP_NAME, 
            session_service=session_service,
            memory_service=memory_service, 
            plugins=[
                LoggingPlugin()
            ]
         )

print("âœ… Upgraded to persistent sessions!")
print(f"   - Database: my_agent_data.db")
print(f"   - Sessions will survive restarts!")


# Define run_session helper function that will be reused throughout the notebook
async def run_session(
    runner_instance: Runner,
    user_queries: list[str] | str = None,
    session_name: str = "default",
):
    print(f"\n ### Session: {session_name}")

    # Get app name from the Runner
    app_name = runner_instance.app_name

    # Attempt to create a new session or retrieve an existing one
    try:
        session = await session_service.create_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
    except:
        session = await session_service.get_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )

    # Process queries if provided
    if user_queries:
        # Convert single query to list for uniform processing
        if type(user_queries) == str:
            user_queries = [user_queries]

        # Process each query in the list sequentially
        for query in user_queries:
            print(f"\nUser > {query}")

            # Convert the query string to the ADK Content format
            query = types.Content(role="user", parts=[types.Part(text=query)])

            # Stream the agent's response asynchronously
            async for event in runner_instance.run_async(
                user_id=USER_ID, session_id=session.id, new_message=query
            ):
                # Check if the event contains valid content
                if event.content and event.content.parts:
                    # Filter out empty or "None" responses before printing
                    if (
                        event.content.parts[0].text != "None"
                        and event.content.parts[0].text
                    ):
                        print(f"{MODEL_NAME} > ", event.content.parts[0].text)
    else:
        print("No queries!")


print("âœ… Helper functions defined.")


MODEL_NAME = "gemini-2.5-flash-lite"
await run_session(
    runner,
    ["""
    Write a simple calculator script that accepts an input from globals with the key 'input_expression' and does addition, subtraction, multiplication and division based on the expression. 
    The script shall not have a function or use eval directly.
    """],
    "stateful-agentic-session",
)
print("âœ… Generation completed.")


# Define ADK helper functions that will be reused throughout the notebook

from IPython.core.display import display, HTML
from jupyter_server.serverapp import list_running_servers

# Gets the proxied URL in the Kaggle Notebooks environment
def get_adk_proxy_url():
    PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
    ADK_PORT = "8000"

    servers = list(list_running_servers())
    if not servers:
        raise Exception("No running Jupyter servers found.")

    baseURL = servers[0]['base_url']

    try:
        path_parts = baseURL.split('/')
        kernel = path_parts[2]
        token = path_parts[3]
    except IndexError:
        raise Exception(f"Could not parse kernel/token from base URL: {baseURL}")

    url_prefix = f"/k/{kernel}/{token}/proxy/proxy/{ADK_PORT}"
    url = f"{PROXY_HOST}{url_prefix}"

    styled_html = f"""
    <div style="padding: 15px; border: 2px solid #f0ad4e; border-radius: 8px; background-color: #fef9f0; margin: 20px 0;">
        <div style="font-family: sans-serif; margin-bottom: 12px; color: #333; font-size: 1.1em;">
            <strong>âš ï¸� IMPORTANT: Action Required</strong>
        </div>
        <div style="font-family: sans-serif; margin-bottom: 15px; color: #333; line-height: 1.5;">
            The ADK web UI is <strong>not running yet</strong>. You must start it in the next cell.
            <ol style="margin-top: 10px; padding-left: 20px;">
                <li style="margin-bottom: 5px;"><strong>Run the next cell</strong> (the one with <code>!adk web ...</code>) to start the ADK web UI.</li>
                <li style="margin-bottom: 5px;">Wait for that cell to show it is "Running" (it will not "complete").</li>
                <li>Once it's running, <strong>return to this button</strong> and click it to open the UI.</li>
            </ol>
            <em style="font-size: 0.9em; color: #555;">(If you click the button before running the next cell, you will get a 500 error.)</em>
        </div>
        <a href='{url}' target='_blank' style="
            display: inline-block; background-color: #1a73e8; color: white; padding: 10px 20px;
            text-decoration: none; border-radius: 25px; font-family: sans-serif; font-weight: 500;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.2s ease;">
            Open ADK Web UI (after running cell below) â†—
        </a>
    </div>
    """

    display(HTML(styled_html))

    return url_prefix

print("âœ… Helper functions defined.")


session = await session_service.get_session(
    app_name=APP_NAME, user_id=USER_ID, session_id="conversation-01"
)

# Search for color preferences
search_response = await memory_service.search_memory(
    app_name=APP_NAME, user_id=USER_ID, query="What are the operations supported ?"
)

print("ğŸ”� Search Results:")
print(f"  Found {len(search_response.memories)} relevant memories")


from google.adk.agents import LlmAgent
# Code execution agent and runner
from google.adk.code_executors import BuiltInCodeExecutor

code_agent = LlmAgent(
    name="CodeExecutionAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    code_executor=BuiltInCodeExecutor(),
    instruction="""
        You are an agent that reads and executes 'candidate_code.py' with the given inputs   
    """,
    tools=[read_file],
    description="Executes Python code to return output.",
)

runner = Runner(agent=code_agent, app_name=APP_NAME,
                plugins=[
                    LoggingPlugin()
                ],
                session_service=session_service)

print("âœ… CodeExecutionAgent created.")



# Agent Interaction (Async)
async def call_agent_async(query):
    content = types.Content(role="user", parts=[types.Part(text=query)])
    print(f"\n--- Running Query: {query} ---")
    final_response_text = "No final text response captured."
    try:
        # Use run_async
        async for event in runner.run_async(
            user_id=USER_ID, session_id=SESSION, new_message=content
        ):
            print(f"Event ID: {event.id}, Author: {event.author}")

            # --- Check for specific parts FIRST ---
            has_specific_part = False
            if event.content and event.content.parts:
                for part in event.content.parts:  # Iterate through all parts
                    if part.executable_code:
                        # Access the actual code string via .code
                        print(
                            f"  Debug: Agent generated code:\n```python\n{part.executable_code.code}\n```"
                        )
                        has_specific_part = True
                    elif part.code_execution_result:
                        # Access outcome and output correctly
                        print(
                            f"  Debug: Code Execution Result: {part.code_execution_result.outcome} - Output:\n{part.code_execution_result.output}"
                        )
                        has_specific_part = True
                    # Also print any text parts found in any event for debugging
                    elif part.text and not part.text.isspace():
                        print(f"  Text: '{part.text.strip()}'")
                        # Do not set has_specific_part=True here, as we want the final response logic below

            # --- Check for final response AFTER specific parts ---
            # Only consider it final if it doesn't have the specific code parts we just handled
            if not has_specific_part and event.is_final_response():
                if (
                    event.content
                    and event.content.parts
                    and event.content.parts[0].text
                ):
                    final_response_text = event.content.parts[0].text.strip()
                    print(f"==> Final Agent Response: {final_response_text}")
                else:
                    print(
                        "==> Final Agent Response: [No text content in final event]")

    except Exception as e:
        print(f"ERROR during agent run: {e}")
    print("-" * 30)



#pip install "gradio[mcp]"

# use mcp to create Gradio web interface. The page accepts an input text and responds with the model output


import gradio as gr
import importlib.util
import sys
import json

async def execute_candidate_code(input_text: str) -> str:
    """
    Executes a code and returns the computed value

    Args:
        input_text (str): The input text expression

    Returns:
        string: The computed value from the code
    """
    spec = importlib.util.spec_from_file_location("candidate_code", "candidate_code.py")
    mod = importlib.util.module_from_spec(spec)
    # populate module globals (so the file sees input_expression in its globals())
    mod.__dict__['input_expression'] = input_text
    result = spec.loader.exec_module(mod)
    return result

# Create a standard Gradio interface
demo = gr.Interface(
    fn=execute_candidate_code,
    inputs=["textbox"],
    outputs="text",
    title="Code execution",
    description="Enter inputs for the code execution."
)

# Launch both the Gradio web interface and the MCP server
if __name__ == "__main__":
    demo.launch(mcp_server=False, inline=True)

