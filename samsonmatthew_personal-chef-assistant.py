


pip install google-adk


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")


import uuid
from google.genai import types

from google.adk.agents import LlmAgent, Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner 
from google.adk.sessions import InMemorySessionService
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from google.adk.runners import InMemoryRunner

from google.adk.apps.app import App, ResumabilityConfig
from google.adk.tools.function_tool import FunctionTool

print("âœ… ADK components imported successfully.")


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


recipe_finder = Agent(
    name="RecipeFinderAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
You search for the best recipe for the user's request.

Return:
- Full ingredient list
- Estimated quantities
- Optional variations
""",
tools=[google_search],
)


formula_maker = Agent(
    name="FormulaMakerAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
You convert the ingredient list into a structured cooking formula.
Include:
- Precise quantities
- Ordering of steps
- Optional variations
"""
)


calorie_estimator = Agent(
    name="CalorieEstimatorAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
You estimate calories for the recipe.
Output:
- Total calorie count
- Per-ingredient calories
- Per-serving calories
"""
)


stepwise_guide = Agent(
    name="StepwiseGuideAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
You convert a structured formula into a clear step-by-step cooking procedure.
Include:
- Prep steps
- Cooking steps
- Temperatures and timing
- Serving notes
"""
)


chef_coordinator = Agent(
    name="ChefCoordinator",
    model="gemini-2.5-flash-lite",
    instruction="""
You are the Personal Chef Coordinator Agent.

Your workflow MUST follow these steps in order:

1. Call the `RecipeFinderAgent` tool to generate an initial recipe.    
2. Call the `FormulaMakerAgent` tool to structure the recipe formula.  
3. Call the `CalorieEstimatorAgent` tool to estimate calories.  
4. Call the `StepwiseGuideAgent` tool to generate step-by-step instructions.  
5. Combine everything into a final polished response for the user.

DO NOT do any sub-agent work yourself.  
You ONLY orchestrate the workflow.
""",
    tools=[
        AgentTool(recipe_finder),
        AgentTool(formula_maker),
        AgentTool(calorie_estimator),
        AgentTool(stepwise_guide),
    ],
)

print("âœ… ChefCoordinator agent created successfully.")


runner = InMemoryRunner(agent=chef_coordinator)
response = await runner.run_debug("make eba and egusi soup")


!adk create sample-agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY


# Define helper functions that will be reused throughout the notebook

#from IPython.core.display import display, HTML
from IPython.display import display, HTML
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




url_prefix = get_adk_proxy_url()


!adk web --url_prefix {get_adk_proxy_url()}

