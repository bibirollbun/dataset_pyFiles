# Install the necessary libraries
!pip install -q google-adk




import os
import logging
from kaggle_secrets import UserSecretsClient
from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search, FunctionTool
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.genai import types


# --- 1. CONFIGURATION ---
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… API Key loaded.")
except:
    print("âš ï¸� Please set GOOGLE_API_KEY in Kaggle Secrets.")


# Configure basic retry logic
retry_config = types.HttpRetryOptions(
    attempts=3,
    exp_base=2,
    initial_delay=1,
    http_status_codes=[429, 500, 503]
)


def read_paper_abstract(paper_title: str) -> str:
    """
    Simulates reading the abstract of a specific paper.
    In production, this would use MCP to read a real PDF.
    """
    print(f"ğŸ“š Reading paper: {paper_title}...")
    # Mock data for demonstration
    return f"""
    ABSTRACT for {paper_title}:
    We propose a new method for Efficient RLHF using LoRA.
    Key finding: Our method reduces VRAM usage by 40% while maintaining accuracy.
    Experiment: We plotted the Loss vs Epochs curve showing a rapid descent.
    """


scout_agent = Agent(
    name="ScoutAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    You are a Ph.D. research scout.
    1. Search for 2 recent papers on the user's topic.
    2. Return a list of paper titles.
    """,
    tools=[google_search],
    output_key="paper_list" # Save output to session state
)


analyst_agent = Agent(
    name="AnalystAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    You are a Senior Researcher.
    1. Look at the 'paper_list' found by the Scout.
    2. Use the 'read_paper_abstract' tool to read the first paper.
    3. Summarize the key experiment mentioned in the abstract.
    """,
    tools=[FunctionTool(read_paper_abstract)],
    output_key="research_summary"
)


engineer_agent = Agent(
    name="EngineerAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    You are a Research Engineer.
    1. Read the 'research_summary'.
    2. Write a Python script to simulate or visualize the experiment described.
    3. Use 'matplotlib' to plot dummy data that mimics the paper's findings (e.g., Loss vs Epochs).
    4. Execute the code to generate the graph.
    """,
    code_executor=BuiltInCodeExecutor(),
    output_key="replication_result"
)


research_team = SequentialAgent(
    name="ScholarAgent_Team",
    sub_agents=[scout_agent, analyst_agent, engineer_agent]
)

runner = InMemoryRunner(
    agent=research_team,
    plugins=[LoggingPlugin()]
)


async def run_demo():
    print("ğŸš€ ScholarAgent Initialized. Starting Research Task...")
    topic = "Optimizing RLHF for LLMs"
    
    response = await runner.run_debug(
        f"Find and replicate research on: {topic}"
    )
    
    # The output will show the trace of all 3 agents working together
    print("\nâœ… Mission Complete. Check the logs above for the 'Thought-Action' trace.")

# Run the async function in Jupyter
await run_demo()




