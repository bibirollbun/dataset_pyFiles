# Taken from previous Days
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


# Taken from previous Days
import logging
import os

# Clean up any previous logs
for log_file in ["logger.log", "web.log", "tunnel.log"]:
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"ğŸ§¹ Cleaned up {log_file}")

# Configure logging with DEBUG log level.
logging.basicConfig(
    filename="logger.log",
    level=logging.DEBUG,
    format="%(filename)s:%(lineno)s %(levelname)s:%(message)s",
)

print("âœ… Logging configured")


# Taken from previous Days
from IPython.core.display import display, HTML
from jupyter_server.serverapp import list_running_servers


# Gets the proxied URL in the Kaggle Notebooks environment
def get_adk_proxy_url():
    PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
    ADK_PORT = "8000"

    servers = list(list_running_servers())
    if not servers:
        raise Exception("No running Jupyter servers found.")

    baseURL = servers[0]["base_url"]

    try:
        path_parts = baseURL.split("/")
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


# Creates the initial agent - will be overwritten in the next cell. Includes 'yes Y' to confirm the replacement if the agent already exists.
!yes Y | adk create law-agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY


%%writefile law-agent/agent.py

from typing import Any, Dict
from google.adk.agents import Agent, SequentialAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import AgentTool, google_search, FunctionTool
from google.genai import types

# This is the function that the ReferenceAgent will call to exit the loop.
def exit_loop():
    """Call this function ONLY when the country check is 'OK', indicating the response is correctly referenced and no more changes are needed."""
    return {"status": "approved", "message": "Response approved. Exiting refinement loop."}

# RulingAgent: Its job is to use the google_search tool to find rulings for a given case
# while taking the country of origin into consideration
ruling_agent = Agent(
    name="RulingAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
    You are a specialized law agent. Your only job is to find relevant information to the given case and country's legal system.
    Formulate your response as a bulletpoint for each finding including references for each point using google_search.
    """,
    tools=[google_search],
    output_key="case_rulings", # The result of this agent will be stored in the session state with this key.
)

# Country Check Agent: Its job is to check the case_rulings and their references and confirm whether they are applicable to the given country.
country_check_agent = Agent(
    name="CountryCheckAgent",
    model="gemini-2.5-flash-lite",
    instruction="""Read and analyze the provided information: {case_rulings}.
    Check the references and confirm if they are related to the given country and cover all the information in case_rulings.
    - If ALL references are correctly provided, respond with the exact phrase 'OK' and nothing else.
    - Otherwise respond with feedback on which parts are still missing references and need to be updated""",
    tools=[google_search],
    output_key="correct_rulings",
)

#Reference Agent: Its job is to change the case_rulings based on the CountryCheckAgent Feeback.
reference_agent = Agent(
    name="ReferenceAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
    You are a specialized law-reference search agent, you have case rulings and feedback regarding the references.
    Case Rulings: {case_rulings}
    Feedback: {correct_rulings}

    Your task is to analyze the feedback.
    - IF the feedback is exactly 'OK', you must call the `exit_loop` function and nothing else.
    - OTHERWISE consider {correct_rulings}, adapt the Case Rulings to only include referenced content.
    
    """,
    tools=[FunctionTool(exit_loop)],
    output_key="case_rulings",
)

# The LoopAgent contains the agents that will run repeatedly: Country Checker -> Referencer.
reference_refinement_loop = LoopAgent(
    name="ReferenceLoop",
    sub_agents=[country_check_agent, reference_agent],
    max_iterations=2, # Prevents infinite loops
)

# The root agent is a SequentialAgent that defines the overall workflow: Initial Rulings -> Reference Fixing Loop.
root_agent = SequentialAgent(
    name="LawHelpPipeline",
    sub_agents=[ruling_agent, reference_refinement_loop],
)



url_prefix = get_adk_proxy_url()


!adk web --log_level DEBUG --url_prefix {url_prefix}

