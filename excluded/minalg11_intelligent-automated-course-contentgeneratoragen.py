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
    print("Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


# ADK imports
from google.adk.agents import Agent, ParallelAgent, LlmAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner, Runner
from google.adk.tools import google_search, FunctionTool, AgentTool
from google.genai import types

# MCP-related imports (use the ADK MCP session manager)
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams, StdioServerParameters

from google.adk.sessions import InMemorySessionService

print("ADK components imported successfully.")


# Optional helper to show ADK web UI button in Kaggle (useful only if you start ADK web server)
from IPython.core.display import display, HTML
from jupyter_server.serverapp import list_running_servers

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
            The ADK web UI can be launched using `!adk web` if you wish to use the UI.
        </div>
        <a href='{url}' target='_blank' style="
            display: inline-block; background-color: #1a73e8; color: white; padding: 10px 20px;
            text-decoration: none; border-radius: 25px; font-family: sans-serif; font-weight: 500;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.2s ease;">
            Open ADK Web UI (if started) â†—
        </a>
    </div>
    """

    display(HTML(styled_html))
    return url_prefix

print("Helper functions loaded.")



retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)



# Use the ADK-provided google_search tool
# If your ADK exposes `google_search` as function above, use it directly
# In some SDK versions it's a class; adapt accordingly.
try:
    gs_tool = google_search  # use the imported google_search
    print("Google Search tool is ready.")
except Exception:
    gs_tool = None
    print("google_search tool not available in this environment.")  


# Define the Python function
def word_count_fn(text: str) -> dict:
    """
    Count the number of words in a given text and provide a preview.

    Purpose:
        Designed for AI agents (e.g., ReviewAgent) to evaluate content length 
        and provide a snippet for review or quality checks.

    Parameters:
        text (str): Input text to analyze. If None or not a string, it will be safely converted.

    Returns:
        dict: A dictionary with:
            - "word_count" (int): Total number of words.
            - "preview" (str): First 10 words joined as a preview snippet.
    """
    # Safety: handle missing or invalid inputs
    if text is None:
        return {"word_count": 0, "preview": ""}
    
    if not isinstance(text, str):
        text = str(text)

    # Split text into words
    words = text.strip().split()

    # Prepare result
    return {
        "word_count": len(words),
        "preview": " ".join(words[:10])
    }

# Function ready
print("word_count_fn created successfully")


# Test the function
sample_text = "This is a sample text to test the word_count_fn function for AI agent usage."
result = word_count_fn(sample_text)
print("Test result:", result)


# Creating ResearchAgent (base)
research_agent = Agent(
    name="ResearchAgent",
    model=Gemini(model="gemini-2.5-flash-lite", 
    retry_options=retry_config,
              
),
    
    instruction="""

Provide output in JSON with keys:
- topic_summary
- outline
- definitions
- concept_map
- important_points

Responsibilities:
- Research the topic
- Summarize core concepts
- Adapt for Grade 4
- Extract definitions, examples, rules
- Break topic into sub-topics
""",
    tools=[gs_tool] if gs_tool else [],
    output_key="research",
)
print("research_agent created.")


# Test ResearchAgent
runner = InMemoryRunner(agent=research_agent)
response = await runner.run_debug("Explain the Solar System for Grade 4.")
print("InMemoryRunner for research agent created.")


# creating content agent
content_agent = Agent(
    name="ContentAgent",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    instruction="""
You are the Content Generation Agent.

You can generate educational content directly from the user's input topic,
without relying on research_output or another agent.

Given ANY topic provided by the user, generate:

1. Lesson Plan  
2. Slide Content  
3. Quiz Bank  
4. Examples & Analogies  
5. Activities  

Return ONLY JSON:
{
 "lesson_plan": "",
 "slides": "",
 "quiz_bank": "",
 "examples": "",
 "activities": ""
}
""",
    tools=[gs_tool] if gs_tool else [],
    output_key="content_output",
)
print("Independent content_agent created.")



# Test ContentAgent
runner = InMemoryRunner(agent=content_agent)
response = await runner.run_debug("Photosynthesis for Grade 4")
print("InMemoryRunner for content agent created.")


# ReviewAgent (base) created with custom function tools
review_agent = LlmAgent(
    name="ReviewAgent",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    instruction="""
You are the Review Agent.

Your responsibilities:
- Review content for correctness.
- Improve clarity, grammar, explanations.
- Ensure Grade 4 level language.
- Fix structure, remove any complexity.
- Ensure content is teacher-friendly.

Return ONLY JSON:
{
 "clean_content": ""
}
""",
    tools=[word_count_fn],
        output_key="reviewed_output",
)
print("review_agent created with custom function tools")


# Test reviewAgent
runner = InMemoryRunner(agent=review_agent)

response = await runner.run_debug(
    "Photosynthesis is the process by which green plants make their food using sunlight."
)

print(response)


review_agent.tools = [word_count_fn]
     
print("Review agent created calls custom function tools")


# Test the review agent

runner = InMemoryRunner(agent=review_agent)
test_prompt = (
    "Review the following text and also count the number of words using the custom word_count tool: "
    "à¤¸à¤‚à¤—à¥�à¤¯à¤¾ à¤µà¤¹ à¤¶à¤¬à¥�à¤¦ à¤¹à¥ˆ à¤œà¥‹ à¤•à¤¿à¤¸à¥€ à¤µà¥�à¤¯à¤•à¥�à¤¤à¤¿, à¤µà¤¸à¥�à¤¤à¥�, à¤¸à¥�à¤¥à¤¾à¤¨ à¤¯à¤¾ à¤­à¤¾à¤µ à¤•à¤¾ à¤¨à¤¾à¤® à¤¬à¤¤à¤¾à¤¤à¤¾ à¤¹à¥ˆà¥¤ "
    "à¤‰à¤¦à¤¾à¤¹à¤°à¤£: à¤°à¤¾à¤®, à¤µà¤¿à¤¦à¥�à¤¯à¤¾à¤²à¤¯, à¤–à¥�à¤¶à¥€, à¤ªà¥�à¤¸à¥�à¤¤à¤•à¥¤ "
    "Provide quality feedback and the word count in JSON if possible."
)

print("ğŸ”� Running ReviewAgent test (this will show debug logs):")
response = await runner.run_debug(test_prompt)
print("\n--- Runner Debug Output (response) ---")
print(response)



# LocalizationAgent (base)
localization_agent = Agent(
    name="LocalizationAgent",
    model=Gemini(model="gemini-2.5-flash",
    retry_options=retry_config,),
    instruction="""
You are the Localization & Adaptation Agent.

Your responsibilities:
- Translate content into the target language (Hindi by default).
- Simplify or increase difficulty based on instructions.
- Add culturally relevant examples.
- Make content suitable for Grade 4 in India.
- Adjust tone, vocabulary, and examples.

Return ONLY JSON:
{
 "localized_content": ""
}
""",
    tools=[gs_tool] if gs_tool else [],
    output_key="localized_output",
)
print("localization_agent created.")



# Running test 

runner_loc = InMemoryRunner(agent=localization_agent)
loc_test_prompt = (
    "Simplify and adapt this explanation for Grade 4 Hindi students: "
    "Nouns are words that name a person, place, thing, or idea. Provide one short activity."
)

response_loc = await runner_loc.run_debug(loc_test_prompt)
print("\n--- LocalizationAgent Debug Output ---")
print(response_loc)


# MCP integration with Everything Server
mcp_image_server = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",  # Run MCP server via npx
            args=[
                "-y",  # Argument for npx to auto-confirm install
                "@modelcontextprotocol/server-everything",
            ],
            tool_filter=["getTinyImage"],
        ),
        timeout=30,
    )
)

print("MCP Tool created")


# MCP tool 
try:
    mcp_image_server  
    print(" Using existing mcp_image_server object.")
except NameError:
    
    mcp_image_server = McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-everything"],
                tool_filter=["getTinyImage"],
            ),
            timeout=30,
        )
    )
    print(" mcp_image_server created now (attempted).")

# Note: starting the underlying npx server may fail in Kaggle.
print("MCP tool ready (if environment permits starting the @modelcontextprotocol/server-everything server).")



# Attempt direct MCP invocation (best-effort)
if not mcp_image_server:
    print("MCP tool object is not available; skip direct MCP call.")
else:
    print("Attempting a direct MCP getTinyImage invocation (may fail in Kaggle).")
    try:
        # ADK MCP APIs vary; try common methods safely.
        result = None
        # Try .invoke or .call style first (some versions name them differently)
        if hasattr(mcp_image_server, "call"):
            result = mcp_image_server.call("getTinyImage", {"prompt": "Sangya Hindi Grade 4 tiny icon"})
        elif hasattr(mcp_image_server, "invoke"):
            result = mcp_image_server.invoke("getTinyImage", {"prompt": "Sangya Hindi Grade 4 tiny icon"})
        elif hasattr(mcp_image_server, "run"):
            result = mcp_image_server.run("getTinyImage", {"prompt": "Sangya Hindi Grade 4 tiny icon"})
        else:
            # Fallback: print available attributes for debugging
            print("MCP toolset methods:", [m for m in dir(mcp_image_server) if not m.startswith("_")][:40])
            raise RuntimeError("No standard invoke/call/run method found on mcp_image_server object.")
        print("MCP result:", result)
    except Exception as e:
        print("MCP direct call failed (likely environment restriction). Error:", repr(e))
        print("Include this error in your submission; run MCP locally for full functionality.")



TOOL_USAGE_INSTRUCTIONS = """
When the user asks for an image, tiny image, icon, or any picture,
you MUST call the MCP tool 'getTinyImage'.
Do NOT respond normally. ALWAYS call the MCP tool.
"""

# Inject instructions
research_agent.instruction += TOOL_USAGE_INSTRUCTIONS
content_agent.instruction += TOOL_USAGE_INSTRUCTIONS
review_agent.instruction += TOOL_USAGE_INSTRUCTIONS
localization_agent.instruction += TOOL_USAGE_INSTRUCTIONS

print("Tool usage instructions added to research_agent.")
print("Tool usage instructions added to content_agent.")
print("Tool usage instructions added to review_agent.")
print("Tool usage instructions added to localization_agent.")


# Inject MCP tool usage instructions into base agents (idempotent)
TOOL_USAGE_INSTRUCTIONS = """
When the user asks for an image, tiny image, icon, or any picture,
you MUST call the MCP tool 'getTinyImage'. 
Do NOT answer normally. ALWAYS call the tool.
"""

for a in (research_agent, content_agent, review_agent, localization_agent):
    if TOOL_USAGE_INSTRUCTIONS.strip() not in a.instruction:
        a.instruction += "\n" + TOOL_USAGE_INSTRUCTIONS

print("Tool usage instructions injected into base agents (if not already present).")



#Create fresh parallel instances cloning instruction & tools from base agents
research_agent_p = Agent(
    name="ResearchAgentParallel",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction=research_agent.instruction,
    tools=research_agent.tools.copy() if research_agent.tools else [],
    output_key=research_agent.output_key,
)

content_agent_p = Agent(
    name="ContentAgentParallel",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    instruction=content_agent.instruction,
    tools=content_agent.tools.copy() if content_agent.tools else [],
    output_key=content_agent.output_key,
)

review_agent_p = Agent(
    name="ReviewAgentParallel",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    instruction=review_agent.instruction,
    tools=review_agent.tools.copy() if review_agent.tools else [],
    output_key=review_agent.output_key,
)

localization_agent_p = Agent(
    name="LocalizationAgentParallel",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    instruction=localization_agent.instruction,
    tools=localization_agent.tools.copy() if localization_agent.tools else [],
    output_key=localization_agent.output_key,
)

print("Parallel agent instances created (cloned).")
print("ResearchAgentParallel created.")
print("ContentAgentParallel created.")
print("ReviewAgentParallel created.")
print("LocalizationAgentParallel created.")



# The ParallelAgent runs all its sub-agents simultaneously.
parallel_research_team = ParallelAgent(
    name="ParallelResearchTeam",
    sub_agents=[
        research_agent_p,
        content_agent_p,
        review_agent_p,
        localization_agent_p
    ],
    )

print("Parallel team created successfully!")


runner = InMemoryRunner(agent=parallel_research_team)
response = await runner.run_debug(
    "Generate a complete learning package for the following: "
    "Subject: Hindi, Topic: Sangya, Grade Level: 4. "
    "Use all agents and produce research summary, lesson plan, quizzes, activities, review findings, and localized version."
)


runner_par = InMemoryRunner(agent=parallel_research_team)
resp = await runner_par.run_debug("Create a short lesson outline for Sangya (Hindi nouns) for Grade 4.")
print(resp)


# Ensure gs_tool exists
if gs_tool:
    runner = InMemoryRunner(agent=research_agent)
    resp = await runner.run_debug("Search: Importance of nouns (Sangya) for Grade 4 students. Provide short bullets.")
    print(resp)
else:
    print("gs_tool not available in this environment.")


# Create session service (even if runner won't use it directly)
session_service = InMemorySessionService()
print("InMemorySessionService created")

# Create runner for parallel agent (NO session_service argument!)
runner_par = InMemoryRunner(
    app_name="ParallelApp",
    agent=parallel_research_team
)

print("InMemoryRunner configured successfully (parallel)")


# Create session service
session_service = InMemorySessionService()
print("InMemorySessionService created")

# Create runner
runner_par = InMemoryRunner(
    app_name="ParallelApp",
    agent=parallel_research_team
)
print("InMemoryRunner configured successfully (parallel)")

# Create session with REQUIRED arguments
session = await session_service.create_session(
    app_name="ParallelApp",
    user_id="default"  # User
)
print("Session created:", session.id)
print(f"\n ### Created new session: {session.id}")
# Send a test message
print("\nUser > Give two points about Sangya (Hindi nouns).")

response = await runner_par.run_debug(
    "Give two points about Sangya (Hindi nouns).",
    session_id=session.id
)

print("\nAssistant >", response)



# Direct test: call MCP tool programmatically (works if underlying server is running)
if mcp_image_server:
    try:
        # The exact invocation API can differ slightly. Many McpToolset objects provide .call or .invoke
        # We attempt a safe generic call pattern. Adjust if your ADK version uses different method names.
        result = mcp_image_server.call("getTinyImage", {"prompt": "Sangya Hindi Grade 4 tiny icon"})
        print("MCP getTinyImage result:", result)
    except Exception as e:
        print("MCP call failed (likely due to server not started or network restrictions):", e)
else:
    print("MCP tool not available to test directly.")



!adk create sample-agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY


url_prefix = get_adk_proxy_url()


!adk web --url_prefix {url_prefix}

