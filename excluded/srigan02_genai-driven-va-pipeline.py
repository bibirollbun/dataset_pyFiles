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
    print(f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")


from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types

print("âœ… ADK components imported successfully.")


proposal_acceptance_agent = Agent(
    name="ProposalAcceptanceAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
    You are the Proposal Acceptance Agent.
    Responsibilities:
    1. Validate engagement details (PT or VA, testing type, grey/black/white box).
    2. Confirm timelines, scope and initial assumptions.
    3. Identify risks and missing information.

    Reporting Workflow:
    - You MUST generate output formatted for the ManagerAgent and ReviewAgent.
    - Include sections: (a) Engagement Summary, (b) Gaps, (c) Risks, (d) Recommendations.
    
    Output MUST be manager-review ready.
    """,
    output_key="proposal_acceptance_summary",
)
print("âœ… ProposalAcceptanceAgent created.")


data_requirement_agent = Agent(
    name="DataRequirementAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
    You are the Data Requirement Agent.
    Responsibilities:
    1. Collect input data: IPs, server info, env type, hosting, port scan scope, OS, status, groups.
    2. Validate data completeness.
    3. Flag missing or inconsistent information.

    Reporting Workflow:
    - Your results MUST be formatted for ManagerAgent only.
    - Include: (a) Completed Checklist, (b) Missing Inputs, (c) Data Validity Score.

    Output should be manager-ready.
    """,
    output_key="validated_data_requirements",
)
print("âœ… DataRequirementAgent created.")


testing_agent = Agent(
    name="TestingAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
    You are the Testing Agent.
    Responsibilities:
    1. Select appropriate tools for project type.
    2. Run VA/PT methodology steps.
    3. Provide raw findings (highly technical, JSON-like structure).

    Reporting Workflow:
    - You MUST report only to the ManagerAgent.
    - Output format: (a) Tools Used, (b) Technical Findings, (c) Evidence Summary.

    Do NOT create user-friendly descriptions here.
    """,
    output_key="raw_findings",
)

print("âœ… TestingAgent created.")


helper_agent = Agent(
    name="HelperAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
    You are the Helper Agent.
    Responsibilities:
    1. Track missing data and follow up with clients.
    2. Support TestingAgent and ReportingAgent with inputs.
    3. Alert ManagerAgent if data is still missing at deadline.
    4. Maintain operational logs.

    This agent DOES NOT report directly to Manager or Review.
    Output: helper summary + pending tasks + alerts.
    """,
    output_key="helper_status",
)
print("âœ… HelperAgent created.")


reporting_agent = Agent(
    name="ReportingAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
    You are the Reporting Agent.
    Responsibilities:
    1. Convert raw findings to readable VA/PT report.
    2. Classify vulnerabilities with severity, impact & remediation.
    3. Prepare PDF/Word/Excel style content.

    Reporting Workflow:
    - You MUST prepare a report structure specifically for ManagerAgent.
    - Include: (a) Executive Summary, (b) Detailed Findings, (c) Recommendations.

    Output is manager-ready, not yet review-approved.
    """,
    output_key="final_report_content",
)
print("âœ… ReportingAgent created.")


analytics_agent = Agent(
    name="AnalyticsEngineAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
    You are the Analytics Engine Agent.
    Responsibilities:
    1. Perform EDA on vulnerabilities and assets.
    2. Generate risk heatmaps (values only), patterns, severity distribution.
    3. Provide insights for dashboards.

    Reporting Workflow:
    - You MUST report to BOTH ManagerAgent and ReviewAgent.
    - Output must contain: (a) Analytics Summary, (b) Risk Patterns, (c) Insight Notes.

    Output must be review-quality analytics.
    """,
    output_key="analytics_output",
)
print("âœ… AnalyticsEngineAgent created.")


manager_agent = Agent(
    name="ManagerAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
    You are the Manager Agent.
    Responsibilities:
    1. Review outputs from Proposal, Data, Testing, Reporting and Analytics agents.
    2. Validate operational completeness and accuracy.
    3. Request corrections when required.
    4. Approve for ReviewAgent.

    Output includes: manager approval notes or corrective actions.
    """,
    output_key="manager_review",
)
print("âœ… ManagerAgent created.")


review_agent = Agent(
    name="ReviewAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
    You are the Review Agent.
    Responsibilities:
    1. Perform independent QA of Proposal and Analytics outputs.
    2. Validate content quality, correctness, scoring, severity, clarity.
    3. Approve or request rework via ManagerAgent.

    Output includes final review comments + approval status.
    """,
    output_key="review_output",
)
print("âœ… ReviewAgent created.")


root_agent = SequentialAgent(
    name="VA_Pipeline",
    sub_agents=[proposal_acceptance_agent,data_requirement_agent,testing_agent,reporting_agent,analytics_agent,manager_agent,review_agent],
)

print("âœ… Sequential Agent created.")


# Create runner for the root sequential agent
runner = InMemoryRunner(agent=root_agent)

# Run the pipeline with an input prompt
response = await runner.run_debug(
    "Start a Vulnerability Assessment for the client's web application. "
    "Begin with proposal acceptance and follow the sequential workflow."
)

print("\n\nğŸ”¥ FINAL PIPELINE OUTPUT ğŸ”¥\n")
print(response)

