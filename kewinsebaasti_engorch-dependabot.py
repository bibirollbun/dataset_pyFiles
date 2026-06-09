pip install google-adk


# CELL 1: Setup
import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("✅ Gemini API key set.")
except Exception as e:
    print("❌ Add GOOGLE_API_KEY to Kaggle secrets. Error:", e)

# ADK imports
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.tools import AgentTool
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Retry config
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

print("✅ ADK imports ready.")



# CELL 2: Load & sanitize policy text
policy_text = ""
policy_path = "/kaggle/input/engineering-policy-docs/policies.md"

try:
    with open(policy_path, "r", encoding="utf-8") as f:
        raw = f.read()
    # Escape curly braces so ADK instruction templating won't try to inject variables
    policy_text = raw.replace("{", "{{").replace("}", "}}")
    print("✅ Policy loaded and sanitized (first 300 chars):")
    print(policy_text[:300] + "...")
except Exception as e:
    print("❌ Could not load policy file at", policy_path, "Error:", e)



# CELL 3 — CodeReviewAgent

code_review_agent = Agent(
    name="CodeReviewAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),

    instruction="""
You are CodeReviewAgent.

Your job is to review the user's code and provide a simple, readable explanation.

Your response MUST:
- Be plain text only
- Contain NO JSON
- Contain NO structured objects
- Contain NO function_call or tool instructions

Return:
- A short list of issues (bullet points)
- Suggested improvements
- A readable corrected version of the code, if relevant
""",

    output_key="code_review",
)

print("✅ CodeReviewAgent updated.")



# CELL 4 — CodeFixAgent

code_fix_agent = Agent(
    name="CodeFixAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),

    instruction="""
You are CodeFixAgent.

Your job is to fix or improve the user's code.

Your response MUST:
- Be plain text
- Not contain JSON, arrays, or dictionaries
- Not use function_call or tool syntax

Return:
- A short explanation of what was wrong
- A corrected version of the code in a code block
""",

    output_key="code_fix",
)

print("✅ CodeFixAgent updated.")



# CELL 5 — CommunicationAgent (Plain Text)

communication_agent = Agent(
    name="CommunicationAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),

    instruction="""
You are CommunicationAgent.

Your job is to write or rewrite professional emails or messages.

Your response MUST:
- Be plain text
- Not contain JSON
- Not use function_call
- Contain a clear subject line and message body

Format:

Subject: <subject line>

<message body>
""",

    output_key="communication",
)

print("✅ CommunicationAgent updated.")



# CELL 6 — PolicyAgent (Enterprise Safe + Knowledge Base)

policy_agent = Agent(
    name="PolicyAgent",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),

    instruction=f"""
You are PolicyAgent.

You answer ONLY based on the official company policies below.
This document is authoritative and complete.

---------------- POLICY DOCUMENT START ----------------
{policy_text}
---------------- POLICY DOCUMENT END ------------------

MANDATORY RULES:
1. Always answer in the format:
   Answer: <Allowed / Not allowed / Unclear>
   Explanation: <short explanation using the policy text>
   Next step: <guidance>

   Your response MUST:
- Be plain text
- Not contain JSON
- Not use function_call

2. If the documentation does not contain the rule:
   Answer: Unclear

3. NEVER guess or hallucinate.

4. Respond in plain English only.
    """,

    output_key="policy",
)

print("✅ PolicyAgent ready.")



# CELL 7 — JiraAgent

jira_agent = Agent(
    name="JiraAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),

    instruction="""
You are JiraAgent.

Your job is to convert the user's request into a plain-text Jira ticket.

Your response MUST:
- Be plain text only
- Not contain JSON or dictionary-like structures
- Not use function_call

Format your ticket like:

Title: <short summary>  
Description: <longer explanation>  
Priority: High / Medium / Low
""",

    output_key="jira",
)

print("✅ JiraAgent updated.")



# CELL 8 — OrchestratorAgent

orchestrator_agent = Agent(
    name="OrchestratorAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),

    instruction="""
You are OrchestratorAgent.

Your job is to ROUTE the user's message to the correct specialist agent.

You must choose based ONLY on natural language patterns:
- If the message contains code OR says "review", call CodeReviewAgent.
- If the user asks to fix, debug, or improve code, call CodeFixAgent.
- If the user wants an email/message, call CommunicationAgent.
- If the user asks about rules or policies, call PolicyAgent.
- If the user mentions ticket, Jira, bug report, issue, task, or story, call JiraAgent.

After receiving the result from the chosen agent:
- Produce a clean, HUMAN-READABLE summary
- No JSON
- No function calls
- No tool_call syntax
- Respond only in plain text
""",

    tools=[
        AgentTool(code_review_agent),
        AgentTool(code_fix_agent),
        AgentTool(communication_agent),
        AgentTool(policy_agent),
        AgentTool(jira_agent),
    ],

    output_key="final_output",
)

print("✅ OrchestratorAgent updated.")



# CELL 9 — Runner & Session Service Setup

APP_NAME = "MultiAgentDemo"
USER_ID = "test-user"

session_service = InMemorySessionService()

runner = Runner(
    agent=orchestrator_agent,
    app_name=APP_NAME,
    session_service=session_service
)

print("✅ Runner initialized with InMemorySessionService")



# CELL 9.5 — Helper for Multi-turn Interaction

async def run_session(
    runner_instance: Runner,
    user_queries: list[str] | str,
    session_name: str,
):

    print(f"\n### Session: {session_name}")

    try:
        session = await session_service.create_session(
            app_name=runner_instance.app_name,
            user_id=USER_ID,
            session_id=session_name
        )
    except:
        session = await session_service.get_session(
            app_name=runner_instance.app_name,
            user_id=USER_ID,
            session_id=session_name
        )

    if isinstance(user_queries, str):
        user_queries = [user_queries]

    for query in user_queries:
        print(f"\nUser > {query}")

        msg = types.Content(
            role="user",
            parts=[types.Part(text=query)]
        )

        async for event in runner_instance.run_async(
            user_id=USER_ID,
            session_id=session.id,
            new_message=msg
        ):
            if event.content and event.content.parts:
                text = event.content.parts[0].text
                if text and text != "None":
                    print(f"Assistant > {text}")



# CELL 10 — Fully Working Multi-Agent Tests

examples = [
    ("review", "Review this code:\nint main(){ return 0; }"),
    ("fix", "Fix this code: int x == 5;"),
    ("communicate", "Write a polite message asking for a meeting extension."),
    ("policy", "Is it allowed to deploy after 9 PM?"),
    ("jira", "Create a ticket for login failing on production."),
]

import asyncio

async def test_all():
    for task, prompt in examples:
        print("\n==============================")
        print(f"TASK: {task.upper()}")
        print("==============================")
        await run_session(
            runner_instance=runner,
            user_queries=prompt,
            session_name=f"session-{task}"
        )

await test_all()



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
            <strong>⚠️ IMPORTANT: Action Required</strong>
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
            Open ADK Web UI (after running cell below) ↗
        </a>
    </div>
    """

    display(HTML(styled_html))

    return url_prefix


print("✅ Helper functions defined.")


!adk create sample-agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY


url_prefix = get_adk_proxy_url()


!adk web --url_prefix {url_prefix}

