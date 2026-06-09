import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    BRIGHT_DATA_API_KEY = UserSecretsClient().get_secret("BRIGHT_DATA_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["BRIGHT_DATA_API_KEY"] = BRIGHT_DATA_API_KEY
    print("âœ… API keys setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' and 'BRIGHT_DATA_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.genai import types

from mcp import StdioServerParameters
from pydantic import BaseModel, Field
from typing import List

print("âœ… ADK components imported successfully.")


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)


class CyberThreatReport(BaseModel):
    title: str = Field(description="The title of the cyber threat report")
    url: str = Field(description="The URL of the cyber threat report")

class ResearchOutput(BaseModel):
    articles: List[CyberThreatReport]


# Research Agent
# - Role: use the Bight Data MCP tools to find articles related to a malware campaign.
research_agent = Agent(
    name="ResearchAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a cybersecurity specialist research agent.
    Your role is to use the google_search tool to find 3-5 recent articles about the given malware campaign, or security vulnerability, or cyber threat, etc.
    Respond with the relevant articles' URLs. Make sure they are real (i.e., not 404s).
    """,
    tools=[
        McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command="npx",
                    args=[
                        "@brightdata/mcp",
                    ],
                    env={
                        "API_TOKEN": BRIGHT_DATA_API_KEY,
                        "PRO_MODE": "false"
                    }
                ),
                timeout=300,
            )
        )
    ],
    output_schema=ResearchOutput,
    output_key="articles_urls"
)


runner = InMemoryRunner(agent=research_agent)
response = await runner.run_debug("Lumma Stealer")


# Summarization Agent
# - Role: use the Bright Data MCP tools to read a given article and summarize it.
summarization_agent = Agent(
    name="SummarizationAgent",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=retry_config
    ),
    instruction="""You are a cybersecurity specialist agent.
    You will be given an URL as input. At that URL you can find a cyber threat report or research article about a malware campaign, security vulnerability, or cyber threat, etc.
    Your task is to read that article and create an executive summary of 3-5 key findings.
    Write them in text/markdown format. Include only text in the summary.
    """,
    tools=[
        McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command="npx",
                    args=[
                        "@brightdata/mcp",
                    ],
                    env={
                        "API_TOKEN": BRIGHT_DATA_API_KEY,
                        "PRO_MODE": "false"
                    }
                ),
                timeout=300,
            )
        )
    ],
    output_key="key_findings"
)


runner = InMemoryRunner(agent=summarization_agent)
response = await runner.run_debug("https://www.mcafee.com/blogs/other-blogs/mcafee-labs/lumma-stealer-on-the-rise-how-telegram-channels-are-fueling-malware-proliferation/")


# IOCs Extractor Agent
# - Role: use the Bright Data MCP tools to read a given article and extract IOCs from it.
iocs_extractor_argent = Agent(
    name="IOCsExtractorAgent",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=retry_config
    ),
    instruction="""You are a cybersecurity specialist agent.
    You will be given an URL as input. At that URL you can find a cyber threat report or research article about a malware campaign, security vulnerability, or cyber threat, etc.
    Your task is to read that article and extract the various indicators of compromise (IOCs) that can be found in that article/report.
    Write a text/markdown document as output with the IOCs identified, grouped by their relevant category (file hashes, C2 domains, IP addresses, MITRE ATT&CK techniques, etc.).
    Include **all** IOCs that are found in the article.
    If there are domains / URLs, sanitize them by replacing `.` with `[.]` and `http` with `hxxp`.
    """,
    tools=[
        McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command="npx",
                    args=[
                        "@brightdata/mcp",
                    ],
                    env={
                        "API_TOKEN": BRIGHT_DATA_API_KEY,
                        "PRO_MODE": "false"
                    }
                ),
                timeout=300,
            )
        )
    ],
    output_key="iocs"
)


runner = InMemoryRunner(agent=iocs_extractor_argent)
# response = await runner.run_debug("https://blog.qualys.com/vulnerabilities-threat-research/2024/10/20/unmasking-lumma-stealer-analyzing-deceptive-tactics-with-fake-captcha")
# response = await runner.run_debug("https://news.sophos.com/en-us/2025/05/09/lumma-stealer-coming-and-going/")
response = await runner.run_debug("https://www.mcafee.com/blogs/other-blogs/mcafee-labs/lumma-stealer-on-the-rise-how-telegram-channels-are-fueling-malware-proliferation/")


coordinator_agent = Agent(
    name="ResearchCoordinator",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a cybersecurity specialist research agent.
    Your goal is to research the latest insights about a malware campaign, security vulnerability, or cyber threat, etc.
    To achive your goal, you will apply the following workflow:
    1. Firstly, you **MUST** call `ResearchAgent` that will find URLs of cyber threat reports related to a topic provided by the user.
    2. Next, **FOR EACH URL**, you will call `SummarizationAgent` and `IOCsExtractorAgent` in order to extract _key findings and _IOCs_ from each article.
    3. Finally, you are going to combine the findings and create 1 single summary with key findings and IOCs across these articles.
    """,
    tools=[
        AgentTool(research_agent),
        AgentTool(summarization_agent),
        AgentTool(iocs_extractor_argent),
    ]
)


runner = InMemoryRunner(agent=coordinator_agent)
response = await runner.run_debug("Lumma Stealer")


runner = InMemoryRunner(agent=coordinator_agent)
response = await runner.run_debug("PIKABOT")







