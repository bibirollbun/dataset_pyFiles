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


!rm -rf job_search_agent


!adk create job_search_agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY


%%writefile job_search_agent/agent.py

from google.genai import types
from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import google_search, AgentTool

MODEL_NAME = "gemini-2.5-flash-lite"

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

# Agent 1: The Gatherer: extracts professional goals from the user's request.
gatherer_agent = LlmAgent(
    name="CareerAgent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    output_key="job_parameters",
    instruction="""
    You are a Career Coach Specialist.
    Analyze the user's request.
    Extract the following information and return it as a JSON object:
    - target_role (e.g., "Product Manager", "Software Engineer")
    - target_location (e.g., "New York", "Remote", "London")
    - industries (list, e.g., ["Fintech", "Healthtech"])
    - experience_level (Entry, Senior, Manager, Executive)
    """
)

# Agent 2: Market & Company Researcher
market_agent = LlmAgent(
    name="MarketResearchAgent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    output_key="market_data",
    instruction="""
    You are a Job Market Analyst.
    Read the 'job_parameters' from the context.
    
    Identify 3-5 top companies or startups in the target location/industry that hire for this role.
    For each, provide:
    - Company Name
    - Brief culture/product summary
    - Why it's a good fit for this role

    Use the Google Search tools to get the most recent information for your research.
    
    Return the result as a JSON object under the key 'target_companies'.
    """,
    tools=[google_search]
)

# Agent 3: Location & Lifestyle Researcher
location_agent = LlmAgent(
    name="LocationAgent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    output_key="location_data",
    instruction="""
    You are a Relocation & Lifestyle Specialist.
    Read the 'job_parameters' from the context.
    
    Based on the 'target_location', provide:
    1. Average salary range for this role in this city.
    2. Cost of living rating (High/Medium/Low).
    3. Key tech/business hubs or districts in that city to look for offices.

    Use the Google Search tools to get the most recent information for your research.
    
    Return the result as a JSON object.
    """,
    tools=[google_search]
)

# Agent 4: Interview Coach
interview_agent = LlmAgent(
    name="InterviewAgent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    output_key="interview_data",
    instruction="""
    You are an Interview Coach.
    Read the 'job_parameters' from the context.
    
    Based on the 'target_role' and 'experience_level', provide:
    1. Full overview of typical the interview process for that role.
    2. Expected duration for the job search process.
    3. List of preparation resources for the interviews.

    Use the Google Search tools to get the most recent information for your research.
    
    Return the result as a JSON object.
    """,
    tools=[google_search]
)

# Agent 5: Strategist
strategy_agent = LlmAgent(
    name="StrategyAgent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    output_key="final_plan",
    instruction="""
    You are a Senior Career Strategist.
    You have access to:
    - 'job_parameters' (Candidate goals)
    - 'market_data' (Target companies)
    - 'location_data' (Salary & Hubs)
    - 'interview_data' (Interview Preparation)
    
    Your goal is to write a cohesive **Job Search Routine**.
    
    Structure:
    1. Strategy Overview (Summary of the market, salary expectations and expected job search duration for the role)
    2. Top Targets (List the best companies found)
    3. Available Preparation Material for Interviews
    4. The Weekly Routine (Create a Monday-Friday schedule for applying, networking, and upskilling based on the preparation resources)
    5. Closing Statament (motivational and encouragement)
    
    Use formatting to make it actionable.
    """
)

# Market research, Location analysis and Interview coach run in parallel
parallel_research_group = ParallelAgent(
    name="ParallelJobResearch",
    sub_agents=[market_agent, location_agent, interview_agent]
)

# Create the root agent for the Job Search  Workflow 
# Example: Gather -> (Market|Location|Interview) Research -> Strategy
root_agent = SequentialAgent(
    name="JobSearchWorkflow",
    description="Orchestrates parallel execution of Job Search tasks",
    sub_agents=[
        gatherer_agent,
        parallel_research_group,
        strategy_agent
    ]
)


#url_prefix = get_adk_proxy_url()


#!adk web --log_level DEBUG --url_prefix {url_prefix}


from job_search_agent.agent import root_agent
from google.adk.runners import InMemoryRunner
from IPython.display import Markdown


# Test the job search root agent
job_search_runner = InMemoryRunner(agent=root_agent)
result = await job_search_runner.run_debug(
    "I am a Senior Python Developer looking for work in Vancouver, Canada in the tech sector."
)


display(Markdown(result[-1].content.parts[0].text))




