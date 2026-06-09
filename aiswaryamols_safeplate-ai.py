import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.sessions import InMemorySessionService

print("âœ… ADK components imported successfully.")


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)


# Define helper functions that will be reused throughout the notebook

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


# Nutrients Agent: Its job is to use the google_search tool and present findings.
nutrients_agent = Agent(
    name="NutrientsAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a specialized research agent. Your only job is to use the
    google_search tool to find all nutrition information on the given food item and present the findings with citations.""",
    tools=[google_search],
    output_key="nutrients_findings",  # The result of this agent will be stored in the session state with this key.
)

print("âœ… nutrients_agent created.")


# Compatibility Agent: Its job is to use the google_search tool and present findings.
compatibility_agent = Agent(
    name="CompatibilityAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a specialized research agent. Your only job is to use the
    google_search tool to find the given food item is wrost for which all health conditions and present the findings with citations.""",
    tools=[google_search],
    output_key="compatibility_findings",  # The result of this agent will be stored in the session state with this key.
)

print("âœ… compatibility_agent created.")


# Checker Agent: Its job is to use the google_search tool and present findings.
checker_agent = Agent(
    name="CheckerAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a specialized research agent. Your only job is to use the google_search tool to find the given food item is good or bad for the provided health condition and present the findings with citations.""",
    tools=[google_search],
    output_key="checker_findings",  # The result of this agent will be stored in the session state with this key.
)

print("âœ… checker_agent created.")


# Weight Gain Recipe Agent: Its job is to use the google_search tool and present findings.
weightgainrecipe_agent = Agent(
    name="WeightgainRecipeAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a specialized research agent. Your only job is to use the
    google_search tool to find a single weight gain recipe for the given food item.""",
    tools=[google_search],
    output_key="weightgainrecipe_findings",  # The result of this agent will be stored in the session state with this key.
)

print("âœ… weightgainrecipe_agent created.")


# Weight Loss Recipe Agent: Its job is to use the google_search tool and present findings.
weightlossrecipe_agent = Agent(
    name="WeightlossRecipeAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a specialized research agent. Your only job is to use the
    google_search tool to find a single weight loss recipe for the given food item.""",
    tools=[google_search],
    output_key="weightlossrecipe_findings",  # The result of this agent will be stored in the session state with this key.
)

print("âœ… weightlossrecipe_agent created.")


# Recipe Agent: Its job is to use the google_search tool and present findings.
recipe_agent = Agent(
    name="RecipeAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a specialized research agent. Your only job is to use the
    weightgainrecipe_agent and weightlossrecipe_agent for getting the recipe for the given food item and present it in catching way with heading of Weight gain recipe and weight loss recipe.""",
    tools=[
        AgentTool(agent=weightlossrecipe_agent),
        AgentTool(agent=weightgainrecipe_agent),
    ],
    output_key="recipe_findings",  # The result of this agent will be stored in the session state with this key.
)

print("âœ… recipe_agent created.")


# The AggregatorAgent runs *after* the parallel step to synthesize the results.
aggregator_agent = Agent(
    name="AggregatorAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    # It uses placeholders to inject the outputs from the parallel agents, which are now in the session state.
    instruction="""Combine these three research findings into a brief summary:

    **Nutrition Information:**
    {nutrients_agent}
    
    **Compatibility Information:**
    {compatibility_agent}

     **Health Information:**
    {checker_agent}
    
    **Best Recipe:**
    {recipe_agent}
    
    Your should highlight all nutrients in the food item as a table, wrost for which all health conditions, is it good or bad for the provided health condition and the healthy recipe with the provided food item. The nutrients information should be a table and rest of the informations as a brief text which should be around 150 words.""",
    output_key="executive_summary",  # This will be the final output of the entire system.
)

print("âœ… aggregator_agent created.")


# The ParallelAgent runs all its sub-agents simultaneously.
parallel_health_team = ParallelAgent(
    name="ParallelHealthTeam",
    sub_agents=[nutrients_agent, compatibility_agent, checker_agent, recipe_agent],
)

# This SequentialAgent defines the high-level workflow: run the parallel team first, then run the aggregator.
root_agent = SequentialAgent(
    name="HealthSystem",
    sub_agents=[parallel_health_team, aggregator_agent],
)

print("âœ… Parallel and Sequential Agents created.")


runner = InMemoryRunner(agent=root_agent)

print("âœ… Runner created.")


response = await runner.run_debug("apple for kidney disease")

