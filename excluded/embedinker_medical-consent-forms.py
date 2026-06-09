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


!mkdir Consent


%%writefile Consent/agent.py

import warnings
warnings.filterwarnings("ignore")
from google.adk.agents import Agent
from dataclasses import dataclass
from google.genai import types
print("âœ… ADK components imported successfully.")

@dataclass
class config:
    """Configuration for models"""
    """
    Attributes:
        critic_model
        worker_model
        max_search_iterations
    """
    critic_model: str = "gemini-2.5-pro"
    worker_model: str = "gemini-2.5-flash"
    max_search_iterations: int = 5


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

tests_generator = Agent(
    name = "tests_generator",
    model=config.worker_model,
    description="This model acts like a ER doctor, triaging a list of diagnostics and distilling it into tests needed",
    instruction = """You are an ER doctor. You receive a list of diagnostics and transform it in a list of batches of tests to be performed:
        1. **Exclusionary:** decisive checks to exclude possible diagnostics upfront.
        2. **Confirmatory:** checks that increase the likelyhood of certain diagnostics from candidate list
        3. **Exploratory:** that allows exploring diagnostics not yet listed - that are compatible with symptoms or other diagnostics in the list
        You add layperson explanations for each test - as to what it means and why is asked. Label each bullet in the outline with a number - so it can be referred later into questions.
        You return the lists of tests to the calling agent.
    """,    
)
diagnostics_generator = Agent(
    name = "diagnostics_generator",
    model=config.worker_model,
    description="This model acts like a doctor, generating a list of candidate diagnostics",
    instruction="""You are a diagnostician doctor - that generates a list of candidate diagnostics from either:
        1. a list of symptoms
        2. a list of diagnostics compatible with the input list - by augmenting the input.
        You return the list of diagnostics - and the list of tests obtained to the calling agent.
    """,
    sub_agents=[]
)

interactive_consent_agent  = Agent(
    name = "informed_consent_agent",
    model=config.worker_model,
    description="The consent management assistant.It collaborates with doctor and patient to request granular consent on medical tests.",
    instruction="""You are a hospital staff member, in charge with getting explicit consent for medical data retrieval and tests performed from patient.
    Your workflow is as follows:
    1. **Get input:** You get a list of candidate diagnostics from the doctor - or a list of symptom descriptions from the patient
    2. **Propose:** You generate a list of candidate diagnostics. You use `diag_generator` tool for this list.
    3. **Generate tests:** You generate a list of tests from the list of diagnostics received. You use `tests_generator` tool for this, without waiting for more user input.
    4. You present the candidate diagnostics and lists of tests to the Patient, followed by asking for consent to perform the tests. The conversation would be saved for later reference and legal reasons.
    """,
    sub_agents=[diagnostics_generator,tests_generator],
    tools=[],
)   

root_agent = interactive_consent_agent

print("âœ… Multiagent structure defined and instantiated.")


try:
    url_prefix = get_adk_proxy_url()
    print("âœ… Creating dev UI start button")
except Exception as e:
    print("It appears it's a saving session: %s"%e)


!adk web  --log_level DEBUG  --url_prefix {url_prefix}

