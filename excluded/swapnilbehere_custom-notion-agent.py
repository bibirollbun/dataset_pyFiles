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


# As we will use Web UI :
!mkdir -p /kaggle/working/notion_agent
!echo "from . import agent" > /kaggle/working/notion_agent/__init__.py


!pip install -q "google-adk[a2a]" "a2a-sdk" uv httpx
#clone the fun fact A2A 
!git clone https://github.com/a2aproject/a2a-samples.git /kaggle/working/a2a-samples


!pip install -r /kaggle/working/a2a-samples/samples/python/agents/adk_facts/requirements.txt


import subprocess, time
from kaggle_secrets import UserSecretsClient
import requests
import sys

# Paths
adk_facts_dir = "/kaggle/working/a2a-samples/samples/python/agents/adk_facts"

# 1) Make sure deps are installed (only needs to be done once per session)
!pip install -q -r /kaggle/working/a2a-samples/samples/python/agents/adk_facts/requirements.txt

# 2) Write .env for the facts agent
GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
with open(os.path.join(adk_facts_dir, ".env"), "w") as f:
    f.write(f"GOOGLE_API_KEY={GOOGLE_API_KEY}\n")

# 3) Start uvicorn in the background, with the *correct working directory*
facts_log = open("/kaggle/working/adk_facts_server.log", "w")
facts_proc = subprocess.Popen(
    ["uvicorn", "agent:a2a_app", "--host", "127.0.0.1", "--port", "8001"],
    cwd=adk_facts_dir,
    stdout=facts_log,
    stderr=subprocess.STDOUT,
)

time.sleep(30)
print("Waiting to Initialize Remote Agent....")
time.sleep(40) # give it a moment to start

print("facts_proc pid:", facts_proc.pid, "returncode:", facts_proc.poll())



# !tail -n 60 /kaggle/working/adk_facts_server.log
# To check if the Remote agent is working


time.sleep(20)
resp = requests.get("http://127.0.0.1:8001/.well-known/agent-card.json", timeout=5)
sys.path.append("/kaggle/working")


%%writefile /kaggle/working/notion_agent/agent.py

# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------

import os
from kaggle_secrets import UserSecretsClient
from datetime import datetime
from typing import Any, Dict, Optional
import logging
import requests
from google.adk.agents import LlmAgent,LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from dateutil import parser as dateparser
from datetime import datetime, date, timedelta
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.genai.types import HttpRetryOptions
from google.adk.tools.tool_context import ToolContext
from google.adk.agents.remote_a2a_agent import (
    RemoteA2aAgent,
    AGENT_CARD_WELL_KNOWN_PATH,
)


# -------------------------------------------------------------------
# Env + logging
# -------------------------------------------------------------------

# This Agent works for all the notion versions above 2022-06-28

NOTION_VERSION = "2022-06-28"
NOTION_BASE_URL = "https://api.notion.com/v1"


# Importing all the API Keys
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    NOTION_API_KEY = UserSecretsClient().get_secret("NOTION_API_KEY")
    os.environ["NOTION_API_KEY"] = NOTION_API_KEY
    print("âœ… Gemini API key setup complete.")
    NOTION_PARENT_PAGE = UserSecretsClient().get_secret("NOTION_PARENT_PAGE")
    os.environ["NOTION_PARENT_PAGE"] = NOTION_PARENT_PAGE
    print(f"Notion parent page: {NOTION_PARENT_PAGE}")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


# -------------------------------------------------------------------
# Notion REST helpers as tools
# -------------------------------------------------------------------

def _notion_headers() -> dict:
    """Internal helper to build Notion REST headers."""
    return {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }


gemini_retry = HttpRetryOptions(
    attempts=5,      # instead of max_retries
    initial_delay=5, # this one is allowed in recent versions
    max_delay=60,
    exp_base=2.0,    # instead of multiplier
    # retry_on is NOT supported in HttpRetryOptions -> remove it
)


# Notion is usually build with Databases, Pages and Blocks.
# So getting the right Database/Page requires its unique ID

def get_notion_ids() -> dict:
    """
    List all Notion databases this integration can see.

    Return shape:
    {
      "databases": [
        {"id": "...", "title": "...", "url": "..."},
        ...
      ]
    }
    """
    logging.info("get_notion_ids called")
    headers = _notion_headers()
    url = f"{NOTION_BASE_URL}/search"
    start_cursor = None
    databases = []

    while True:
        payload = {
            "page_size": 100,
            "filter": {"property": "object", "value": "database"},
        }
        if start_cursor:
            payload["start_cursor"] = start_cursor

        resp = requests.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

        for db in data.get("results", []):
            if db.get("object") != "database":
                continue
            title_rich = db.get("title", []) or []
            title = "".join(rt.get("plain_text", "") for rt in title_rich) or "Untitled database"
            databases.append(
                {"id": db["id"], "title": title, "url": db.get("url")}
            )

        if not data.get("has_more"):
            break
        start_cursor = data.get("next_cursor")

    logging.info("get_notion_ids returning %d databases", len(databases))
    return {"databases": databases}


def create_database_item(database_id: str, properties: Dict[str, Any]) -> dict:
    """
    Create a new page in ANY Notion database.

    Args:
      database_id: Notion database id.
      properties: simple dict like:
        {
          "Name": "Daily Habit",
          "Date": "2025-11-20",
          "Journal": True,
          "Exercise": False,
          ...
        }

    This function will:
      1) Fetch DB schema.
      2) Coerce values into correct Notion property objects.
      3) POST /v1/pages.
    """
    logging.info("create_database_item: db=%s, properties=%s", database_id, properties)
    headers = _notion_headers()

    # ---- 1. Fetch schema ----
    def _get_database_schema(db_id: str) -> dict:
        url = f"{NOTION_BASE_URL}/databases/{db_id}"
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return {
            "id": data.get("id"),
            "title": "".join(
                t.get("plain_text", "") for t in (data.get("title", []) or [])
            ),
            "properties": data.get("properties", {}),
        }

    # ---- 2. Coerce simple values to correct property types ----
    def _coerce_property_value(prop_schema: dict, value: Any) -> dict:
        ptype = prop_schema.get("type")

        def _as_str(v: Any) -> str:
            return v if isinstance(v, str) else str(v)

        if ptype == "title":
            return {
                "title": [
                    {
                        "text": {
                            "content": _as_str(value)
                        }
                    }
                ]
            }
        if ptype in ("rich_text", "text"):
            return {
                "rich_text": [
                    {
                        "text": {
                            "content": _as_str(value)
                        }
                    }
                ]
            }

        if ptype == "date":
            # Accept natural language: "2025-11-20", "20th nov 2025", "today", etc.
            if isinstance(value, (datetime, date)):
                dt = value
            elif isinstance(value, str):
                text = value.strip().lower()

                if text in {"today", "now"}:
                    dt = datetime.now()
                elif text == "tomorrow":
                    dt = datetime.now() + timedelta(days=1)
                elif text == "yesterday":
                    dt = datetime.now() - timedelta(days=1)
                else:
                    try:
                        dt = dateparser.parse(value, fuzzy=True)
                    except Exception:
                        dt = datetime.now()
            else:
                dt = datetime.now()

            date_str = dt.date().isoformat()
            return {"date": {"start": date_str}}

        if ptype == "checkbox":
            if isinstance(value, bool):
                checked = value
            else:
                vlower = _as_str(value).strip().lower()
                checked = vlower in {"true", "yes", "y", "1", "checked", "done"}
            return {"checkbox": checked}

        if ptype == "select":
            return {"select": {"name": _as_str(value)}}

        if ptype == "multi_select":
            if isinstance(value, (list, tuple, set)):
                names = [_as_str(v) for v in value]
            else:
                names = [_as_str(value)]
            return {"multi_select": [{"name": n} for n in names]}

        if ptype == "number":
            try:
                num = float(value)
            except Exception:
                num = None
            return {"number": num}

        if ptype == "url":
            return {"url": _as_str(value)}

        # Fallback: treat as rich_text
        return {
            "rich_text": [
                {
                    "text": {
                        "content": _as_str(value)
                    }
                }
            ]
        }

    schema = _get_database_schema(database_id)
    schema_props: dict = schema.get("properties", {})
    final_props: Dict[str, Any] = {}

    for prop_name, simple_value in properties.items():
        schema_entry = schema_props.get(prop_name)
        if not schema_entry:
            logging.warning("Property '%s' not in schema; using rich_text fallback.", prop_name)
            final_props[prop_name] = {
                "rich_text": [{"text": {"content": str(simple_value)}}]
            }
            continue
        final_props[prop_name] = _coerce_property_value(schema_entry, simple_value)

    # Ensure at least one title property is set
    if not any(
        pschema.get("type") == "title" and pname in final_props
        for pname, pschema in schema_props.items()
    ):
        for pname, pschema in schema_props.items():
            if pschema.get("type") == "title":
                final_props.setdefault(
                    pname,
                    {"title": [{"text": {"content": "New item"}}]},
                )
                break

    body = {"parent": {"database_id": database_id}, "properties": final_props}
    url = f"{NOTION_BASE_URL}/pages"
    resp = requests.post(url, headers=headers, json=body)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print("Error creating database item in Notion:")
        print(resp.status_code, resp.text)
        raise e

    data = resp.json()
    logging.info("create_database_item: created page %s", data.get("id"))
    return data


def create_page_anywhere(
    title: str,
    parent_page_id: Optional[str] = None,
    content: str = "",
) -> dict:
    """
    Create a new page under ANY existing Notion page (not in a database).

    This uses POST /v1/pages with a page parent. When the parent is a page,
    the only valid property is the title.

    If parent_page_id is None, the default NOTION_PARENT_PAGE is used.
    """
    if parent_page_id is None:
        parent_page_id = NOTION_PARENT_PAGE

    logging.info(
        "create_page_anywhere: parent_page_id=%s, title=%s, has_content=%s",
        parent_page_id,
        title,
        bool(content),
    )
    headers = _notion_headers()
    url = f"{NOTION_BASE_URL}/pages"

    properties = {
        "title": {
            "title": [
                {
                    "text": {
                        "content": title,
                    }
                }
            ]
        }
    }

    body: Dict[str, Any] = {
        "parent": {"page_id": parent_page_id},
        "properties": properties,
    }

    if content:
        body["children"] = [
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {"content": content},
                        }
                    ]
                },
            }
        ]

    resp = requests.post(url, headers=headers, json=body)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print("Error creating standalone page in Notion:")
        print(resp.status_code, resp.text)
        raise e

    data = resp.json()
    logging.info("create_page_anywhere: created page %s", data.get("id"))
    return data



# -------------------------------------------------------------------
# Notion MCP toolset (for search/fetch/etc. via MCP server)
# -------------------------------------------------------------------

def notion_mcp_toolset(notion_api_key: str):
    """
    Notion MCP via @notionhq/notion-mcp-server (Model Context Protocol).

    Exposes tools such as:
      - notion-search
      - notion-fetch
      - notion-create-pages
      - notion-update-page
      - notion-move-pages
    """
    tools = [
        "notion-search",
        "notion-fetch",
        "notion-create-pages",
        "notion-update-page",
        "notion-move-pages",
    ]
    server = McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="npx",
                args=["-y", "@notionhq/notion-mcp-server"],
                tool_filter=tools,
                env={"NOTION_TOKEN": notion_api_key},
            ),
            timeout=30,
        )
    )
    print("Notion MCP Tool Created!")
    return server


notion_mcp_server = notion_mcp_toolset(NOTION_API_KEY)


# -------------------------------------------------------------------
# Dedicated agents
# -------------------------------------------------------------------

# 1. Dedicated search agent
search_agent = LlmAgent(
    model="gemini-2.5-flash-lite",
    name="SearchAgent",
    instruction=(
        "You are a specialist in Google Search grounding.\n"
        "When asked a question that requires up-to-date or external information, "
        "use GoogleSearchTool to search the web and return a concise, well-sourced answer."
    ),
    tools=[GoogleSearchTool()],
)
search_tool = AgentTool(agent=search_agent)


# 2. Dedicated Notion agent
notion_agent = LlmAgent(
    model="gemini-2.5-flash-lite",
    name="NotionAgent",
    instruction=(
        "You are a specialist in managing my Notion workspaces.\n"
        "\n"
        "NOTION RULES:\n"
        "- To find databases by name (e.g. 'Habit Tracker', 'My Links', 'Laliga'), "
        "  ALWAYS call get_notion_ids and match on the 'title' field. "
        "  Do NOT ask the user for raw Notion IDs.\n"
        "- To create a new row/item in a database, call create_database_item with:\n"
        "    â€¢ database_id: the ID you found via get_notion_ids\n"
        "    â€¢ properties: a simple JSON object mapping property name â†’ value\n"
        "  This tool fetches the schema and coerces values into the correct Notion "
        "  property types (title, date, checkbox, etc.).\n"
        "- To create a new page under a normal Notion page (not a database row), "
        "  call create_page_anywhere with:\n"
        "    â€¢ parent_page_id: the page ID to create the child under\n"
        "    â€¢ title: the page title\n"
        "    â€¢ content: optional plain-text content that becomes a paragraph block.\n"
        "- Treat Notion MCP tools (notion-search, notion-fetch) as READ-ONLY. "
        "  Do NOT use low-level MCP write tools like API-post-page, API-create-a-database, "
        "  or API-update-a-database. All writes must go through create_database_item "
        "  or create_page_anywhere.\n"
        "- When the user asks to VIEW, FILTER, or SORT a database, use Notion MCP "
        "  query/search tools (notion-search, notion-fetch), not create_database_item.\n"
        "\n"
        "WEB PAGES:\n"
        "- When the user gives you a URL and it is relevant to Notion context, "
        "  call load_web_page with that URL to fetch its content, then answer based "
        "  on what you read.\n"
        "Always return a text based response after everything is done.\n"
        "NEVER up any ID inside text response. "
    ),
    tools=[
        notion_mcp_server,     # MCP tools like notion-search/notion-fetch (read-only)
        get_notion_ids,        # Python function tool
        create_database_item,  # Database writes
        create_page_anywhere,  # Page-under-page writes
    ],
)

notion_tool = AgentTool(agent=notion_agent)

def exit_notion_loop(tool_context: ToolContext):
    """
    Tool to signal the LoopAgent that Notion data is correct and
    no further retries are needed.
    """
    print(f"[Tool Call] exit_notion_loop triggered by {tool_context.agent_name}")
    tool_context.actions.escalate = True
    return {"status": "ok"}


# 3. Verify if the Notion agent Did its job correctly
notion_verifier_agent = LlmAgent(
    name="NotionVerifierAgent",
    model="gemini-2.5-flash-lite",
    instruction=(
        "You validate that the last Notion page or database row created by the NotionAgent "
        "matches the user's intent.\n\n"
        "Behavior:\n"
        "- NEVER call Notion write tools (create/update). Only use MCP read tools such "
        "  as 'notion-fetch' and the 'exit_notion_loop' tool.\n"
        "- Only inspect the specific page or row you are given.\n"
        "- Do not ask follow-up questions.\n"
        "- If everything matches, call exit_notion_loop and include a short natural-language "
        "  confirmation for the user.\n"
        "- If something is wrong, DO NOT call exit_notion_loop; instead, output a short, "
        "  machine-readable description of what is wrong.\n"
        "- Use as few tool calls as possible.\n"
        "You validate that the last Notion page or database row created by the NotionAgent "
        "matches the user's intent.\n"
        "- You need to use minimal time and resources, check only what your task is."
        "Behavior:\n"
        "- Use Notion MCP tools (e.g. `notion-fetch`) or other Notion tools to read the page "
        "  using the provided page ID.\n"
        "- Compare the stored 'expected' properties (like Name, URL, Date, Tags, etc.) with "
        "  what is actually in Notion.\n"
        "- If everything matches and the entry looks correct, CALL the `exit_notion_loop` tool "
        "  and do not return any extra text.\n"
        "- If something is wrong or missing, DO NOT call the tool. Instead, output a short, "
        "  machine-readable description of what needs to be fixed (e.g. which fields differ).\n"
        "-When you are finished and call exit_notion_loop, you must ALSO include a natural-language final message to the user summarizing what you did."
    ),
    tools=[
        notion_mcp_server,   # so it can call notion-fetch / notion-search
        exit_notion_loop,    # to signal loop completion
    ],
    output_key="notion_validation_feedback",
)

# 4. Loop Agent handles all the wrong-doings of Notion Agent
notion_loop_agent = LoopAgent(
    name="NotionTasks",
    sub_agents=[
        notion_agent,          # writes / fixes Notion entries
        notion_verifier_agent  # checks and optionally calls exit_notion_loop
    ],
    max_iterations=3,
)

notion_loop_tool = AgentTool(agent=notion_loop_agent)

# 5. Remote A2A Fun Facts Agent
facts_agent = RemoteA2aAgent(
    name="adk_facts_remote",
    description="Remote ADK Facts search agent",
    agent_card=f"http://127.0.0.1:8001{AGENT_CARD_WELL_KNOWN_PATH}",
)
facts_tool = AgentTool(agent=facts_agent)

# -------------------------------------------------------------------
# root_agent for ADK (used by `adk run` / `adk web`)
# -------------------------------------------------------------------

root_agent = LlmAgent(
    name="RootAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=gemini_retry,
    ),
    instruction=(
        "You are an orchestrator assistant.\n"
        "\n"
        "- For simple Notion actions (e.g. 'add one task', 'create one bookmark', "
        "  'log today's workout'), call the NotionAgent tool directly.\n"
        "- Only use the notion_loop_tool (the loop agent) when the user explicitly "
        "  asks for high reliability or multi-step edits (e.g. 'update many rows', "
        "  'fix inconsistent entries', 'make sure everything matches these rules').\n"
        "- The notion_loop_tool is more expensive; avoid it unless really needed.\n"
        "ROUTING HINTS:\n"
        "- If the user mentions Notion, databases, pages, templates, or workspace, "
        "  prefer the NotionAgent tool.\n"
        "- If the user asks for 'latest', 'current', 'today', 'trending', or "
        "  anything that clearly needs the live web, prefer the SearchAgent tool.\n"
        "\n"
        "Facts Tool:\n"
        "- Whenever user says give me facts, you need to use the facts_tool"
        "- If user asks to write facts somewhere, you need to use the facts_tool"
        "GENERAL:\n"
        "- Make sure you use minimum number of tool-calls, check the user input carefully and then decide which tool to use."
        "- Prefer a small number of well-chosen tool calls over many redundant ones.\n"
        "- Clearly explain your reasoning and what each specialized agent is doing "
        "  when helpful to the user.\n"
    ),
    tools=[
        search_tool,# Google Search (agent-as-tool)
        notion_tool,
        notion_loop_tool, # Notion management (agent-as-tool)
        facts_tool,
    ],
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


#NOTE: UNCOMMENT AND RUN THIS PART FOR WEB UI
#url_prefix = get_adk_proxy_url()


# NOTE : UNCOMMENT AND RUN THIS PART FOR WEB UI
#!adk web --url_prefix {url_prefix} /kaggle/working/


# Building a JSON structure that documents this run
import datetime
import json
metadata = {
    "environment": {
        "kaggle_working_dir": "/kaggle/working",
        "kaggle_outputs_dir": "/kaggle/outputs",
    },
    "notion": {
        "api_base_url": "https://api.notion.com/v1",
        "api_version": "XXXX",
        "parent_page_id_source": "Kaggle UserSecrets (NOTION_PARENT_PAGE)",
        "parent_page_id": {
            "present": "notion_parent_page" != "UNKNOWN_PARENT_PAGE"
        },
    },
    "agents": {
        "root_agent": {
            "name": "RootAgent",
            "type": "LlmAgent",
            "model": "gemini-2.5-flash-lite",
            "tools": [
                "SearchAgent (AgentTool)",
                "NotionAgent (AgentTool)",
                "NotionTasks LoopAgent (AgentTool)",
                "adk_facts_remote (RemoteA2aAgent via AgentTool)"
            ],
        },
        "notion_agent": {
            "name": "NotionAgent",
            "model": "gemini-2.5-flash-lite",
            "tools": [
                "notion_mcp_server (MCP: notion-search, notion-fetch, ...)",
                "get_notion_ids (Python tool)",
                "create_database_item (Python tool)",
                "create_page_anywhere (Python tool)",
            ],
            "write_policy": {
                "database_writes": "create_database_item",
                "page_writes": "create_page_anywhere",
                "mcp_usage": "read-only (search/fetch/view)"
            }
        },
        "notion_verifier_agent": {
            "name": "NotionVerifierAgent",
            "model": "gemini-2.5-flash-lite",
            "tools": [
                "notion_mcp_server (read-only verification)",
                "exit_notion_loop"
            ],
            "role": "Verifies Notion writes created by NotionAgent inside LoopAgent"
        },
        "notion_loop_agent": {
            "name": "NotionTasks",
            "type": "LoopAgent",
            "sub_agents": ["NotionAgent", "NotionVerifierAgent"],
            "max_iterations": 3,
            "usage_hint": "Use only for high-reliability or multi-step Notion edits"
        },
        "search_agent": {
            "name": "SearchAgent",
            "model": "gemini-2.5-flash-lite",
            "tools": ["GoogleSearchTool"],
            "role": "Grounded web search for up-to-date info"
        },
        "facts_agent": {
            "name": "adk_facts_remote",
            "type": "RemoteA2aAgent",
            "agent_card_url": "URL/.well-known/agent-card.json",
            "description": "Remote ADK Facts search agent (A2A protocol)"
        }
    }
}

working_path = "/kaggle/working/notion_agent_run_metadata.json"
with open(working_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"âœ… JSON metadata written to: {working_path}")

# OUTPUT
output_dir = "/kaggle/outputs"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "notion_agent_run_metadata.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"ğŸ“� JSON metadata also saved to: {output_path}")




