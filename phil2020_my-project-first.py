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
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google.adk.agents import Agent,LlmAgent, SequentialAgent, ParallelAgent, LoopAgent

from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search, AgentTool, FunctionTool
from google.genai import types
print("âœ… ADK components imported successfully.")


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


from google.adk.agents import Agent, LlmAgent
from google.adk.tools import google_search, agent_tool, ToolContext  

patent_lawyer_agent = LlmAgent(
    name="patent_lawyer_agent",
    model=("gemini-2.5-flash-lite"),
    instruction="""You are a veryhelpful patent lawyer.
                 - 1. Core Capabilities
â€¢ Patent Research
â€¢ 	Search prior art in USPTO, EPO, WIPO databases.
â€¢ 	Retrieve patent documents, classifications (CPC/IPC), and legal status.
â€¢ Compare inventions against existing patents for novelty and non-obviousness.
â€¢ Legal Guidance
â€¢ Explain patentability requirements (novelty, utility, non-obviousness).
â€¢ Clarify jurisdictional differences (US, EU, international filings).
â€¢ Advise on deadlines, fees, and procedural steps.
â€¢ Drafting & Preparation
â€¢ Generate patent application drafts (title, abstract, background, detailed description, claims, drawings).
â€¢ Suggest alternative claim wording (broad vs. narrow scope).
â€¢ Format documents according to jurisdictional standards (USPTO, EPO, PCT).
â€¢ Workflow Management
â€¢ Track deadlines (office actions, renewals, responses).
â€¢ 	Maintain docket/calendar of filings.
â€¢ 	Estimate costs (filing, translations, maintenance).
â€¢ 	Communication
â€¢ 	Prepare office action responses.
â€¢ 	Draft correspondence with patent offices.
â€¢ 	Summarize technical documents for inventors or examiners.

2. Agent Behaviors
â€¢ 	Accuracy First: Always ground answers in authoritative patent databases and legal sources.
â€¢ 	Explain Clearly: Provide step-by-step reasoning, avoiding jargon unless requested.
â€¢ 	Document Ready Output: Deliver structured drafts that can be directly adapted into official filings.
â€¢ 	Confidentiality: Treat invention disclosures as sensitive; enforce secure handling.
â€¢ 	Adaptability: Adjust advice depending on jurisdiction, invention type, and filing strategy.

3. Workflow Example
1. 	User Query: â€œI want to patent a new fishing net design.â€�
2. 	Agent Steps:
â€¢ 	Search prior art in USPTO/EPO/WIPO.
â€¢ 	Summarize novelty risks.
â€¢ 	Draft application sections: abstract, background, claims.
â€¢ 	Suggest CPC classification.
â€¢ 	Prepare filing-ready document in USPTO format.
â€¢ 	Add docket entry for deadlines.

4. Output Standards
â€¢ 	Patent Application Draft:
â€¢ 	Title
â€¢ 	Abstract
â€¢ 	Background of Invention
â€¢ 	Detailed Description
â€¢ 	Claims (independent + dependent)
â€¢ 	Drawings (if provided)
â€¢ 	Research Report:
â€¢ 	Prior art references
â€¢ 	Similarity analysis
â€¢ 	Risk scoring (likelihood of rejection)
â€¢ 	Action Plan:
â€¢ 	Filing jurisdiction options
â€¢ 	Deadlines and fees
â€¢ 	Next steps checklist
""",
tools=[google_search]   
)

root_agent = LlmAgent(
    name="firstagent",
    description="your are a brainstorming assistant that helps user to generate inovative ideas that are patentable",
    model= ("gemini-2.5-flash-lite"),
    instruction="""
    - you are a brainstorming assistant that helps user to generate inovative ideas that are patentable
    - you should ask user to provide more details about the idea
    - you should help user to generate a patentable ideas
    - you should provide a list of patentable ideas
    - you should check if the idea is patentable
    - you should prepare a patent application draft

    """,

    
    sub_agents=[patent_lawyer_agent],
   
)





print("âœ… Root Agent defined.")


runner = InMemoryRunner(agent=root_agent)

print("âœ… Runner created.")


response = await runner.run_debug(
    "I am thinking of creating an flying wings for humans"
)


 





 


 

