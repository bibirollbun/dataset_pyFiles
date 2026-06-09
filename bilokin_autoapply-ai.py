from google.adk.agents.llm_agent import Agent, LlmAgent
from google.adk.models.lite_llm import LiteLlm

import uuid
from google.genai import types

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from google.adk.apps.app import App, ResumabilityConfig
from google.adk.runners import InMemoryRunner


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


mcp_browser_server = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="mcp-server-browsermcp",  # Run MCP server via npx
            args=[],
        ),
        timeout=30,
    )
)


api_base_url = 'http://192.168.0.125:3333/v1'
model_name_at_endpoint = "lm_studio/qwen/qwen-coder-30b"
model = LiteLlm(
        model=model_name_at_endpoint,
        api_base=api_base_url,
        extra_headers=None,
        api_key="YOUR_ENDPOINT_API_KEY"
    )
root_agent = Agent(
    model=model,
    name='root_agent',
    description='An expert job finder, who can find suitable jobs on different websites upon user request.',
    instruction='Answer user questions to the best of your knowledge',
    tools=[mcp_browser_server]
)



runner = InMemoryRunner(agent=root_agent)


response = await runner.run_debug("Find Software Dev jobs on LinkedIn", verbose=True)

