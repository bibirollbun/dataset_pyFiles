import os
from getpass import getpass

import logging
import warnings

logging.getLogger("google_genai.types").setLevel(logging.ERROR)
asyncio_logger = logging.getLogger("asyncio")
asyncio_logger.setLevel(logging.CRITICAL)
asyncio_logger.propagate = False
asyncio_logger.disabled = True

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

import pandas as pd

from google import genai
from google.adk.apps import App


from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )



DATA_DIR = "/kaggle/input/project"  # change if you use a different dataset

SALES_CSV   = f"{DATA_DIR}/sales.csv"
LEADS_CSV   = f"{DATA_DIR}/leads.csv"
CAMPAIGNS_CSV = f"{DATA_DIR}/campaigns.csv"  # optional

sales_df = pd.read_csv(SALES_CSV)
leads_df = pd.read_csv(LEADS_CSV)
campaigns_df = pd.read_csv(CAMPAIGNS_CSV)

print("âœ… Data loaded.")
print("Sales shape:", sales_df.shape)
print("Leads shape:", leads_df.shape)
print("Campaigns shape:", campaigns_df.shape)
print()

display(sales_df.head())
display(leads_df.head())
display(campaigns_df.head())



# ===== 3.1 Imports & global DataFrames =====
from typing import Literal, Optional, Dict, Any


# ===== 3.2 Helper: simple date-range filtering =====

def _filter_by_period(
    df: pd.DataFrame,
    date_col: str = "date",
    period: Literal["all", "last_30_days", "last_7_days"] = "all",
) -> pd.DataFrame:
    """Utility to filter a dataframe by a simple relative period."""
    if period == "all":
        return df

    if date_col not in df.columns:
        raise ValueError(f"Expected date column '{date_col}' in dataframe.")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    today = pd.Timestamp.today().normalize()

    if period == "last_30_days":
        start = today - pd.Timedelta(days=30)
    elif period == "last_7_days":
        start = today - pd.Timedelta(days=7)
    else:
        start = df[date_col].min()

    return df[df[date_col] >= start]


# ===== 3.3 Tool: analyze_sales_data_tool =====

def analyze_sales_data_tool(
    period: Literal["all", "last_30_days", "last_7_days"] = "all",
    group_by: Literal["day", "week", "month"] = "month",
) -> Dict[str, Any]:
    """
    Analyze the global `sales_df` and compute basic KPIs.

    Args:
        period: Time range filter for the analysis.
        group_by: Time granularity for trend aggregation.

    Returns:
        A dictionary with:
        - total_revenue: float
        - num_orders: int
        - avg_order_value: float
        - revenue_by_period: list of {period_label, revenue}
    """
    global sales_df
    if "sales_df" not in globals():
        raise RuntimeError("sales_df is not defined. Make sure Section 1 loaded the CSV.")

    df = _filter_by_period(sales_df, date_col="date", period=period)

    if df.empty:
        return {
            "total_revenue": 0.0,
            "num_orders": 0,
            "avg_order_value": 0.0,
            "revenue_by_period": [],
        }

    if "revenue" not in df.columns:
        raise ValueError("Expected column 'revenue' in sales_df.")
    if "order_id" not in df.columns:
        raise ValueError("Expected column 'order_id' in sales_df.")

    total_revenue = float(df["revenue"].sum())
    num_orders = int(df["order_id"].nunique())
    avg_order_value = float(total_revenue / num_orders) if num_orders > 0 else 0.0

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if group_by == "day":
        df["period_label"] = df["date"].dt.strftime("%Y-%m-%d")
    elif group_by == "week":
        df["period_label"] = df["date"].dt.to_period("W").astype(str)
    else:  # "month"
        df["period_label"] = df["date"].dt.to_period("M").astype(str)

    revenue_by_period = (
        df.groupby("period_label")["revenue"]
        .sum()
        .reset_index()
        .rename(columns={"revenue": "total_revenue"})
        .to_dict(orient="records")
    )

    return {
        "total_revenue": round(total_revenue, 2),
        "num_orders": num_orders,
        "avg_order_value": round(avg_order_value, 2),
        "revenue_by_period": revenue_by_period,
    }


# ===== 3.4 Tool: analyze_leads_tool =====

def analyze_leads_tool(
    period: Literal["all", "last_30_days", "last_7_days"] = "all",
) -> Dict[str, Any]:
    """
    Analyze the lead funnel based on the global `leads_df`.

    Assumptions:
      - leads_df has columns: 'created_at', 'lead_id', 'status'
      - typical statuses: 'new', 'contacted', 'won', 'lost' (you can adapt to your dataset)

    Args:
        period: Time range filter for created_at.

    Returns:
        A dictionary with:
        - total_leads: int
        - leads_by_status: list[{status, count}]
        - simple_conversion_rate: float (won / total if statuses present)
    """
    global leads_df
    if "leads_df" not in globals():
        raise RuntimeError("leads_df is not defined. Make sure Section 1 loaded the CSV.")

    df = _filter_by_period(leads_df, date_col="created_at", period=period)

    if df.empty:
        return {
            "total_leads": 0,
            "leads_by_status": [],
            "simple_conversion_rate": 0.0,
        }

    if "lead_id" not in df.columns:
        raise ValueError("Expected column 'lead_id' in leads_df.")
    if "status" not in df.columns:
        raise ValueError("Expected column 'status' in leads_df.")

    total_leads = int(df["lead_id"].nunique())

    leads_by_status = (
        df.groupby("status")["lead_id"]
        .nunique()
        .reset_index()
        .rename(columns={"lead_id": "count"})
        .to_dict(orient="records")
    )

    # Simple conversion rate: leads with status == "won" / total_leads
    won_count = int(df[df["status"].str.lower() == "won"]["lead_id"].nunique())
    simple_conversion_rate = float(won_count / total_leads) if total_leads > 0 else 0.0

    return {
        "total_leads": total_leads,
        "leads_by_status": leads_by_status,
        "simple_conversion_rate": round(simple_conversion_rate, 3),
    }


# ===== 3.5 Tool: campaign_performance_tool =====

def campaign_performance_tool(
    group_by: Literal["campaign", "channel"] = "campaign",
) -> Dict[str, Any]:
    """
    Analyze marketing campaigns based on the global `campaigns_df`.

    Assumptions:
      - campaigns_df has columns:
          'campaign_id', 'name', 'channel', 'spend', 'revenue'
        (adapt names if you used something different)

    Args:
        group_by: 'campaign' or 'channel' for aggregation.

    Returns:
        A dictionary with:
        - rows: list of {label, spend, revenue, roi}
    """
    global campaigns_df
    if "campaigns_df" not in globals():
        raise RuntimeError("campaigns_df is not defined. Make sure Section 1 loaded the CSV.")

    df = campaigns_df.copy()

    required = ["spend_pln", "revenue_pln"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' in campaigns_df.")

    if group_by == "channel":
        if "channel" not in df.columns:
            raise ValueError("Expected column 'channel' in campaigns_df.")
        df["label"] = df["channel"]
    else:
        # default: group by campaign
        if "name" in df.columns:
            df["label"] = df["name"]
        elif "campaign_id" in df.columns:
            df["label"] = df["campaign_id"]
        else:
            raise ValueError("Expected 'name' or 'campaign_id' in campaigns_df.")

    agg = (
        df.groupby("label")[["spend_pln", "revenue_pln"]]
        .sum()
        .reset_index()
    )

    agg["roi"] = agg.apply(
        lambda row: (row["revenue_pln"] - row["spend_pln"]) / row["spend_pln"]
        if row["spend_pln"] > 0
        else 0.0,
        axis=1,
    )

    rows = [
        {
            "label": str(row["label"]),
            "spend": round(float(row["spend_pln"]), 2),
            "revenue": round(float(row["revenue_pln"]), 2),
            "roi": round(float(row["roi"]), 3),
        }
        for _, row in agg.iterrows()
    ]

    return {"rows": rows}


# ===== 3.6 Tool: task_store_tool (simple in-notebook task store) =====

# Very simple in-memory task store.
TASK_STORE: list[Dict[str, Any]] = []


def task_store_tool(
    action: Literal["add", "list", "clear"] = "list",
    description: Optional[str] = None,
    priority: Literal["low", "medium", "high"] = "medium",
) -> Dict[str, Any]:
    """
    Minimal task store that lives inside the notebook.

    Args:
        action: One of:
            - "add": add a new task with description + priority.
            - "list": return all current tasks.
            - "clear": remove all tasks.
        description: Text of the task (required for "add").
        priority: 'low' | 'medium' | 'high'.

    Returns:
        A dictionary with:
        - tasks: current list of tasks (each {id, description, priority})
        - info: short message about what happened
    """
    global TASK_STORE

    if action == "add":
        if not description:
            raise ValueError("description is required when action='add'.")
        task_id = len(TASK_STORE) + 1
        TASK_STORE.append(
            {"id": task_id, "description": description, "priority": priority}
        )
        info = f"Added task #{task_id}."
    elif action == "clear":
        TASK_STORE = []
        info = "All tasks cleared."
    else:  # "list"
        info = f"Listing {len(TASK_STORE)} tasks."

    return {"tasks": TASK_STORE, "info": info}





analyze_sales_data_tool()


analyze_leads_tool()


campaign_performance_tool()


# ===== 4. Specialized Agents â€“ definitions with ADK =====

from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool


# --- 4.1 Analytics Agent ---

analytics_agent = LlmAgent(
    name="analytics_agent",
    model="gemini-2.5-flash",  # or another model available in your environment
    instruction=(
        "You are an Analytics Agent helping a small business owner understand their data.\n"
        "- Use the available tools (sales, leads, campaigns) instead of guessing.\n"
        "- Always explain which KPIs you used and what they mean.\n"
        "- Be concise, but include at least 3 concrete insights.\n"
        "- When possible, reference exact numbers (e.g. total revenue, conversion rate, best channel).\n"
    ),
    tools=[
        analyze_sales_data_tool,
        analyze_leads_tool,
        campaign_performance_tool,
    ],
)


# --- 4.2 Planner Agent ---

planner_agent = LlmAgent(
    name="planner_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a Planner Agent for a solo business owner.\n"
        "- Your input is a summary of business performance and marketing insights.\n"
        "- Your job is to convert this into a short, prioritized task list.\n"
        "- Group tasks by priority: HIGH (this week), MEDIUM (this month), LOW (later).\n"
        "- Use the task_store_tool to save the tasks so they can be reused later.\n"
        "- Return a clear explanation plus a bullet list of tasks."
    ),
    tools=[task_store_tool],
)


# --- 4.3 Notes Agent (lightweight memory helper) ---

notes_agent = LlmAgent(
    name="notes_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a Notes Agent that maintains structured notes about the business.\n"
        "- You receive short descriptions of the user, their preferences and goals.\n"
        "- Your job is to turn them into compact notes that can be stored in session memory.\n"
        "- Focus on stable information: industry, main channels, currencies, main KPIs and long-term goals.\n"
        "- Output should be a short JSON-like summary plus one sentence in plain English."
    ),
    tools=[],
)


# --- 4.3.5 BizOps Orchestrator Agent ---

# Wrap specialist agents as tools
analytics_tool = AgentTool(agent=analytics_agent)
planner_tool = AgentTool(agent=planner_agent)
notes_tool = AgentTool(agent=notes_agent)

bizops_orchestrator_agent = LlmAgent(
    name="bizops_orchestrator_agent",
    model="gemini-2.5-pro",  # more capable model for coordination and reasoning
    instruction=(
        "You are the main SoloBiz Ops Agent for a small business owner.\n"
        "You are the only agent the user talks to directly.\n\n"
        "Responsibilities:\n"
        "1) Understand the user's question about their business performance, marketing or planning.\n"
        "2) When data analysis is needed, call the analytics_agent (as a tool) with a clear instruction.\n"
        "3) When a plan or task list is needed, call the planner_agent with the analytics summary.\n"
        "4) Optionally, update long-term notes by calling the notes_agent.\n"
        "5) Return a final answer that combines:\n"
        "   - key KPIs or insights,\n"
        "   - a short explanation in friendly language,\n"
        "   - a small prioritized list of concrete next actions.\n\n"
        "General rules:\n"
        "- Always prefer using tools/agents over hallucinating numbers.\n"
        "- If something is unclear (e.g. time range), ask the user a short clarifying question.\n"
        "- Keep answers focused on what the user can do next."
    ),
    tools=[analytics_tool, planner_tool, notes_tool],
)



# ===== 5. Multi-Agent Orchestration =====

import asyncio

import nest_asyncio
nest_asyncio.apply()

from google.adk.runners import InMemoryRunner
from google.genai import types  # part of google-genai, used by ADK runners


# --- 5.2.1 Create runners for each specialized agent ---

analytics_runner = InMemoryRunner(
    agent=analytics_agent,
    app_name="solobiz_analytics_app",
)

planner_runner = InMemoryRunner(
    agent=planner_agent,
    app_name="solobiz_planner_app",
)

notes_runner = InMemoryRunner(
    agent=notes_agent,
    app_name="solobiz_notes_app",
)


# --- 5.2.2 Small helper to create a session for a given runner ---

def create_session(runner: InMemoryRunner, user_id: str = "demo_user"):
    """Create a new session for a given runner."""
    session = asyncio.run(
        runner.session_service.create_session(
            app_name=runner.app_name,
            user_id=user_id,
        )
    )
    return session


# --- 5.2.3 Helper to run an agent once and collect its text output ---

def run_agent_once(
    runner: InMemoryRunner,
    session_id: str,
    user_message: str,
    user_id: str = "demo_user",
) -> str:
    """
    Send a single user message to the agent and collect the text response.

    Returns:
        The concatenated text of all events emitted by the agent.
    """
    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_message)],
    )

    full_response = ""

    for event in runner.run(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
    ):
        # Collect only text parts from the agent
        if event.content and event.content.parts:
            part = event.content.parts[0]
            if getattr(part, "text", None):
                full_response += part.text

    return full_response.strip()


# --- 5.2.4 High-level orchestration function ---

def run_solobiz_workflow(
    user_question: str,
    period: str = "last_30_days",
    user_id: str = "demo_user",
) -> Dict[str, Any]:
    """
    Orchestrate a simple multi-agent workflow:

    1. Use Analytics Agent to analyze the data and produce a KPI-based summary.
    2. Use Planner Agent to convert the summary into a prioritized task list.
    3. Use Notes Agent to maintain compact notes about goals & focus.
    4. Return all intermediate outputs plus current task store state.

    Args:
        user_question: The natural language question from the business owner.
        period: Time window for analysis (e.g. 'last_30_days', 'last_7_days', 'all').
        user_id: Logical user id, reused across runs if desired.

    Returns:
        A dictionary with:
            - analytics_summary: str
            - plan_text: str
            - notes_summary: str
            - task_store_state: dict (output of task_store_tool(action="list"))
    """
    # Optional: clear old tasks for a fresh run in the notebook demo
    task_store_tool(action="clear")

    # ---- Step 1: Analytics Agent ----
    analytics_session = create_session(analytics_runner, user_id=user_id)

    analytics_prompt = (
        "You are the Analytics Agent in a SoloBiz Ops multi-agent system.\n"
        "The user asked the following question about their business:\n"
        f"\"{user_question}\"\n\n"
        f"Please analyze the sales, leads and campaigns data, focusing on period='{period}'. "
        "Use your tools (sales, leads, campaigns) to compute KPIs and trends, "
        "and then provide a concise summary with at least 3 concrete insights.\n"
    )

    analytics_summary = run_agent_once(
        analytics_runner,
        session_id=analytics_session.id,
        user_message=analytics_prompt,
        user_id=user_id,
    )

    # ---- Step 2: Planner Agent ----
    planner_session = create_session(planner_runner, user_id=user_id)

    planner_prompt = (
        "You are the Planner Agent in a SoloBiz Ops multi-agent system.\n"
        "You receive the following analytics summary about the business:\n\n"
        f"{analytics_summary}\n\n"
        "Based on this, create a short, prioritized action plan for the business owner.\n"
        "- Group actions into HIGH (this week), MEDIUM (this month), LOW (later).\n"
        "- Use the task_store_tool to add tasks for each action.\n"
        "- Return a clear explanation plus a bullet list of tasks.\n"
    )

    plan_text = run_agent_once(
        planner_runner,
        session_id=planner_session.id,
        user_message=planner_prompt,
        user_id=user_id,
    )

    # ---- Step 3: Notes Agent (optional memory helper) ----
    notes_session = create_session(notes_runner, user_id=user_id)

    notes_prompt = (
        "You are the Notes Agent.\n"
        "Summarize the long-term business context and focus areas based on:\n"
        f"- User question: {user_question}\n"
        f"- Analytics summary: {analytics_summary}\n"
        f"- Action plan: {plan_text}\n\n"
        "Return:\n"
        "1) A compact JSON-like summary with keys such as 'industry', 'focus_channels', "
        "'main_kpis', 'short_term_focus', 'long_term_goal'.\n"
        "2) One short English sentence for the business owner.\n"
    )

    notes_summary = run_agent_once(
        notes_runner,
        session_id=notes_session.id,
        user_message=notes_prompt,
        user_id=user_id,
    )

    # ---- Step 4: Read back task store state ----
    task_store_state = task_store_tool(action="list")

    return {
        "analytics_summary": analytics_summary,
        "plan_text": plan_text,
        "notes_summary": notes_summary,
        "task_store_state": task_store_state,
    }



# ===== 5.3 Demo: Run the multi-agent workflow =====

demo_question = "Review my last month and tell me what I should focus on next week."

result = run_solobiz_workflow(
    user_question=demo_question,
    period="last_7_days",
    user_id="demo_user_1",
)

print("=== User question ===")
print(demo_question)
print("\n=== Step 1: Analytics summary ===\n")
print(result["analytics_summary"])

print("\n=== Step 2: Planner action plan ===\n")
print(result["plan_text"])

print("\n=== Step 3: Notes Agent summary (for memory) ===\n")
print(result["notes_summary"])

print("\n=== Task store (current tasks) ===\n")
print(result["task_store_state"])



# ===== 6.1 Simple in-notebook memory store =====


import json
import re


# Very small, framework-agnostic "memory" for the SoloBiz Ops Agent.
BIZ_MEMORY: Dict[str, Any] = {
    "profile": {
        "industry": None,
        "focus_channels": [],
        "main_kpis": [],
    },
    "short_term_focus": None,
    "long_term_goal": None,
    "notes_history": [],   # keep raw notes from Notes Agent
}


def update_memory_from_notes(notes_text: str) -> None:
    """
    Parse the JSON block from the Notes Agent output
    and update BIZ_MEMORY with real values.
    """
    # Always keep raw text history
    BIZ_MEMORY["notes_history"].append(notes_text)

    match = re.search(r"```json(.*?)```", notes_text, re.DOTALL | re.IGNORECASE)
    if not match:
        return 
    json_block = match.group(1).strip()

    try:
        data = json.loads(json_block)
    except Exception:
        return

    profile = BIZ_MEMORY.get("profile", {})

    if "industry" in data:
        profile["industry"] = data["industry"]

    if "focus_channels" in data and isinstance(data["focus_channels"], list):
        profile["focus_channels"] = data["focus_channels"]

    if "main_kpis" in data and isinstance(data["main_kpis"], list):
        profile["main_kpis"] = data["main_kpis"]

    BIZ_MEMORY["profile"] = profile

    if "short_term_focus" in data:
        BIZ_MEMORY["short_term_focus"] = data["short_term_focus"]

    if "long_term_goal" in data:
        BIZ_MEMORY["long_term_goal"] = data["long_term_goal"]


def format_memory_for_prompt() -> str:
    profile = BIZ_MEMORY.get("profile", {})
    focus_channels = profile.get("focus_channels") or []
    main_kpis = profile.get("main_kpis") or []

    parts = []
    if profile.get("industry"):
        parts.append(f"Industry: {profile['industry']}")
    if focus_channels:
        parts.append(f"Focus channels: {focus_channels}")
    if main_kpis:
        parts.append(f"Main KPIs: {main_kpis}")
    if BIZ_MEMORY.get("short_term_focus"):
        parts.append(f"Short-term focus: {BIZ_MEMORY['short_term_focus']}")
    if BIZ_MEMORY.get("long_term_goal"):
        parts.append(f"Long-term goal: {BIZ_MEMORY['long_term_goal']}")

    if not parts:
        return "No long-term memory yet for this business."
    return " | ".join(parts)




# ===== 6.2 Orchestrator with simple memory integration =====

def run_solobiz_workflow_with_memory(
    user_question: str,
    period: str = "last_30_days",
    user_id: str = "demo_user",
) -> Dict[str, Any]:
    """
    Same multi-agent pipeline as before, but now:
    - includes BIZ_MEMORY in prompts,
    - updates BIZ_MEMORY from the Notes Agent output.
    """
    # Optional: keep existing tasks instead of always clearing,
    # to simulate continuity of planning across runs.
    # task_store_tool(action="clear")  # comment out if you want persistent tasks

    memory_snapshot = format_memory_for_prompt()

    # ---- Step 1: Analytics Agent ----
    analytics_session = create_session(analytics_runner, user_id=user_id)

    analytics_prompt = (
        "You are the Analytics Agent in a SoloBiz Ops multi-agent system.\n"
        f"Current long-term memory about this business: {memory_snapshot}\n\n"
        "The user asked the following question about their business:\n"
        f"\"{user_question}\"\n\n"
        f"Please analyze the sales, leads and campaigns data, focusing on period='{period}'. "
        "Use your tools (sales, leads, campaigns) to compute KPIs and trends, "
        "and then provide a concise summary with at least 3 concrete insights.\n"
    )

    analytics_summary = run_agent_once(
        analytics_runner,
        session_id=analytics_session.id,
        user_message=analytics_prompt,
        user_id=user_id,
    )

    # ---- Step 2: Planner Agent ----
    planner_session = create_session(planner_runner, user_id=user_id)

    planner_prompt = (
        "You are the Planner Agent in a SoloBiz Ops multi-agent system.\n"
        f"Current long-term memory about this business: {memory_snapshot}\n\n"
        "You receive the following analytics summary about the business:\n\n"
        f"{analytics_summary}\n\n"
        "Based on this, create a short, prioritized action plan for the business owner.\n"
        "- Group actions into HIGH (this week), MEDIUM (this month), LOW (later).\n"
        "- Use the task_store_tool to add tasks for each action.\n"
        "- Return a clear explanation plus a bullet list of tasks.\n"
    )

    plan_text = run_agent_once(
        planner_runner,
        session_id=planner_session.id,
        user_message=planner_prompt,
        user_id=user_id,
    )

    # ---- Step 3: Notes Agent ----
    notes_session = create_session(notes_runner, user_id=user_id)

    notes_prompt = (
        "You are the Notes Agent.\n"
        "Summarize the long-term business context and focus areas based on:\n"
        f"- User question: {user_question}\n"
        f"- Analytics summary: {analytics_summary}\n"
        f"- Action plan: {plan_text}\n\n"
        "Return:\n"
        "1) A compact JSON-like summary with keys such as 'industry', 'focus_channels', "
        "'main_kpis', 'short_term_focus', 'long_term_goal'.\n"
        "2) One short English sentence for the business owner.\n"
    )

    notes_summary = run_agent_once(
        notes_runner,
        session_id=notes_session.id,
        user_message=notes_prompt,
        user_id=user_id,
    )

    # Update our simple memory store from the Notes Agent output.
    update_memory_from_notes(notes_summary)

    # ---- Step 4: Read back task store state ----
    task_store_state = task_store_tool(action="list")

    return {
        "analytics_summary": analytics_summary,
        "plan_text": plan_text,
        "notes_summary": notes_summary,
        "task_store_state": task_store_state,
        "memory_snapshot_after": BIZ_MEMORY.copy(),
    }



# ===== 6.3 Demo: run twice and observe memory reuse =====

q1 = "Review my last month and tell me what I should focus on next week."
result_1 = run_solobiz_workflow_with_memory(
    user_question=q1,
    period="last_30_days",
    user_id="demo_user_1",
)

print("=== Run 1 â€“ question ===")
print(q1)
print("\n=== Run 1 â€“ Notes summary ===\n")
print(result_1["notes_summary"])
print("\n=== Run 1 â€“ Memory snapshot ===\n")
print(result_1["memory_snapshot_after"])

q2 = "Given what we discussed before, suggest a concrete email + organic plan for next week."
result_2 = run_solobiz_workflow_with_memory(
    user_question=q2,
    period="last_7_days",
    user_id="demo_user_1",
)

print("\n\n=== Run 2 â€“ question ===")
print(q2)
print("\n=== Run 2 â€“ Analytics summary ===\n")
print(result_2["analytics_summary"])
print("\n=== Run 2 â€“ Planner action plan ===\n")
print(result_2["plan_text"])
print("\n=== Run 2 â€“ Memory snapshot ===\n")
print(result_2["memory_snapshot_after"])



# Configure root logging once (for the whole notebook).
import time
from typing import  List

logger = logging.getLogger("solobiz")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

# Simple in-notebook trace store.
RUN_TRACES: List[Dict[str, Any]] = []


def run_solobiz_workflow_with_observability(
    user_question: str,
    period: str = "last_30_days",
    user_id: str = "demo_user",
) -> Dict[str, Any]:
    """
    Multi-agent pipeline with:
    - BIZ_MEMORY integration (from Section 6),
    - logging at each step,
    - a lightweight trace record stored in RUN_TRACES.
    """
    logger.info("Starting SoloBiz workflow | user_id=%s | period=%s", user_id, period)
    logger.info("User question: %s", user_question)

    # Take a snapshot of memory BEFORE the run (from Section 6 helpers).
    memory_before = format_memory_for_prompt()
    logger.info("Memory before run: %s", memory_before)

    trace: Dict[str, Any] = {
        "user_id": user_id,
        "question": user_question,
        "period": period,
        "memory_before": memory_before,
        "analytics_duration_s": None,
        "planner_duration_s": None,
        "notes_duration_s": None,
        "num_tasks": None,
        "error": None,
    }

    t0 = time.perf_counter()

    try:
        # ---- Step 1: Analytics Agent ----
        t_analytics_start = time.perf_counter()
        analytics_session = create_session(analytics_runner, user_id=user_id)

        analytics_prompt = (
            "You are the Analytics Agent in a SoloBiz Ops multi-agent system.\n"
            f"Current long-term memory about this business: {memory_before}\n\n"
            "The user asked the following question about their business:\n"
            f"\"{user_question}\"\n\n"
            f"Please analyze the sales, leads and campaigns data, focusing on period='{period}'. "
            "Use your tools (sales, leads, campaigns) to compute KPIs and trends, "
            "and then provide a concise summary with at least 3 concrete insights.\n"
        )

        analytics_summary = run_agent_once(
            analytics_runner,
            session_id=analytics_session.id,
            user_message=analytics_prompt,
            user_id=user_id,
        )
        t_analytics_end = time.perf_counter()

        trace["analytics_duration_s"] = round(t_analytics_end - t_analytics_start, 3)
        logger.info("Analytics step finished in %.3f s", trace["analytics_duration_s"])

        # ---- Step 2: Planner Agent ----
        t_planner_start = time.perf_counter()
        planner_session = create_session(planner_runner, user_id=user_id)

        planner_prompt = (
            "You are the Planner Agent in a SoloBiz Ops multi-agent system.\n"
            f"Current long-term memory about this business: {memory_before}\n\n"
            "You receive the following analytics summary about the business:\n\n"
            f"{analytics_summary}\n\n"
            "Based on this, create a short, prioritized action plan for the business owner.\n"
            "- Group actions into HIGH (this week), MEDIUM (this month), LOW (later).\n"
            "- Use the task_store_tool to add tasks for each action.\n"
            "- Return a clear explanation plus a bullet list of tasks.\n"
        )

        plan_text = run_agent_once(
            planner_runner,
            session_id=planner_session.id,
            user_message=planner_prompt,
            user_id=user_id,
        )
        t_planner_end = time.perf_counter()

        trace["planner_duration_s"] = round(t_planner_end - t_planner_start, 3)
        logger.info("Planner step finished in %.3f s", trace["planner_duration_s"])

        # ---- Step 3: Notes Agent ----
        t_notes_start = time.perf_counter()
        notes_session = create_session(notes_runner, user_id=user_id)

        notes_prompt = (
            "You are the Notes Agent.\n"
            "Summarize the long-term business context and focus areas based on:\n"
            f"- User question: {user_question}\n"
            f"- Analytics summary: {analytics_summary}\n"
            f"- Action plan: {plan_text}\n\n"
            "Return:\n"
            "1) A compact JSON-like summary with keys such as 'industry', 'focus_channels', "
            "'main_kpis', 'short_term_focus', 'long_term_goal'.\n"
            "2) One short English sentence for the business owner.\n"
        )

        notes_summary = run_agent_once(
            notes_runner,
            session_id=notes_session.id,
            user_message=notes_prompt,
            user_id=user_id,
        )
        t_notes_end = time.perf_counter()

        trace["notes_duration_s"] = round(t_notes_end - t_notes_start, 3)
        logger.info("Notes step finished in %.3f s", trace["notes_duration_s"])

        # Update our simple memory store from the Notes Agent output (Section 6 helper).
        update_memory_from_notes(notes_summary)

        # ---- Step 4: Read back task store state ----
        task_store_state = task_store_tool(action="list")
        num_tasks = len(task_store_state.get("tasks", []))
        trace["num_tasks"] = num_tasks

        logger.info("Workflow finished successfully | tasks_created=%d", num_tasks)

    except Exception as e:
        # Log the error, store it in trace, and re-raise (so we also see the stack trace).
        logger.exception("Error during SoloBiz workflow: %s", str(e))
        trace["error"] = str(e)
        # Re-raise to make debugging easier in the notebook.
        raise

    finally:
        total_time = time.perf_counter() - t0
        trace["total_duration_s"] = round(total_time, 3)
        logger.info("Total workflow time: %.3f s", trace["total_duration_s"])

        # Store the final memory snapshot after the run.
        trace["memory_after"] = BIZ_MEMORY.copy()

        # Append trace to global list for later analysis.
        RUN_TRACES.append(trace)

    # Return main artifacts for the user.
    return {
        "analytics_summary": analytics_summary,
        "plan_text": plan_text,
        "notes_summary": notes_summary,
        "task_store_state": task_store_state,
        "memory_snapshot_after": BIZ_MEMORY.copy(),
        "trace": trace,
    }



def show_last_traces(n: int = 5) -> None:
    """
    Print the last n traces in a compact form.
    You can also convert RUN_TRACES to a DataFrame if you prefer.
    """
    if not RUN_TRACES:
        print("No traces recorded yet.")
        return

    subset = RUN_TRACES[-n:]
    for i, t in enumerate(subset, start=1):
        print(f"\n=== Trace {len(RUN_TRACES) - len(subset) + i} ===")
        print(f"user_id: {t['user_id']}")
        print(f"question: {t['question']}")
        print(f"period: {t['period']}")
        print(f"total_duration_s: {t.get('total_duration_s')}")
        print(f"analytics_duration_s: {t.get('analytics_duration_s')}")
        print(f"planner_duration_s: {t.get('planner_duration_s')}")
        print(f"notes_duration_s: {t.get('notes_duration_s')}")
        print(f"num_tasks: {t.get('num_tasks')}")
        print(f"error: {t.get('error')}")



demo_question = "Review my last month and tell me what I should focus on next week."

result = run_solobiz_workflow_with_observability(
    user_question=demo_question,
    period="last_7_days",
    user_id="demo_user_obs",
)

print("=== Final answer â€“ Planner plan ===\n")
print(result["plan_text"])

print("\n=== Trace for this run ===")
print(result["trace"])

print("\n=== Last traces summary ===")
show_last_traces(n=3)



# ===== 8. Simple Evaluation â€“ rule-based planner scoring =====

def evaluate_plan_rule_based(plan_text: str) -> Dict[str, Any]:
    """
    Very simple, heuristic evaluation of the Planner's output.
    This is NOT a replacement for human or LLM judging, but
    it demonstrates an automatic quality check inside the notebook.

    Criteria (0â€“5 total):
    - +1 if all three priority labels HIGH / MEDIUM / LOW appear.
    - +1 if there are at least 3 bullet points (lines starting with "- ").
    - +1 if text length is at least 400 characters (non-trivial answer).
    - +1 if there is at least one explicit time hint (e.g. 'this week', 'today', 'this month').
    - +1 if there is at least one business metric mention (e.g. 'revenue', 'conversion', 'leads').
    """
    score = 0
    reasons: List[str] = []

    text_lower = plan_text.lower()

    # 1) Priority structure
    has_high = "high" in text_lower
    has_medium = "medium" in text_lower
    has_low = "low" in text_lower
    if has_high and has_medium and has_low:
        score += 1
        reasons.append("Contains HIGH / MEDIUM / LOW priority groups.")
    else:
        reasons.append("Missing one or more priority groups (HIGH/MEDIUM/LOW).")

    # 2) Number of bullets
    bullets = re.findall(r"^-\\s+", plan_text, flags=re.MULTILINE)
    if len(bullets) >= 3:
        score += 1
        reasons.append(f"Contains {len(bullets)} bullet points (>= 3).")
    else:
        reasons.append(f"Only {len(bullets)} bullet points (< 3).")

    # 3) Length
    if len(plan_text) >= 400:
        score += 1
        reasons.append("Plan is reasonably detailed (>= 400 characters).")
    else:
        reasons.append("Plan is very short (< 400 characters).")

    # 4) Time hints
    time_hints = ["today", "tomorrow", "this week", "next week", "this month"]
    if any(h in text_lower for h in time_hints):
        score += 1
        reasons.append("Includes explicit time-related guidance (e.g. 'this week').")
    else:
        reasons.append("No explicit time-related guidance detected.")

    # 5) Business metric hints
    metric_hints = ["revenue", "conversion", "leads", "cost per lead", "ctr", "click-through"]
    if any(m in text_lower for m in metric_hints):
        score += 1
        reasons.append("Mentions at least one business metric (revenue, leads, conversion, etc.).")
    else:
        reasons.append("No clear business metric mentions found.")

    return {
        "score": score,
        "max_score": 5,
        "reasons": reasons,
    }



# A tiny "evaluation set" â€“ just a few realistic questions
EVAL_QUESTIONS = [
    "Look at my last month and tell me what 3 things I should focus on next week.",
    "Based on recent sales and leads, how can I increase conversion from lead to paying client?",
    "What should I do this month to stabilize my revenue and reduce marketing waste?",
]



def run_simple_evaluation(
    questions: List[str],
    period: str = "last_30_days",
    user_id_prefix: str = "eval_user_",
) -> List[Dict[str, Any]]:
    """
    Run the full SoloBiz workflow for a few questions and
    evaluate only the Planner output using a rule-based rubric.

    Returns a list of dicts with:
    - question
    - score
    - reasons
    - short length and meta info for quick inspection
    """
    results: List[Dict[str, Any]] = []

    for idx, q in enumerate(questions):
        user_id = f"{user_id_prefix}{idx+1}"
        print(f"\n=== Running evaluation case {idx+1} for user_id={user_id} ===")
        print(f"Question: {q}\n")

        workflow_result = run_solobiz_workflow_with_observability(
            user_question=q,
            period=period,
            user_id=user_id,
        )

        plan_text = workflow_result["plan_text"]
        eval_result = evaluate_plan_rule_based(plan_text)

        print(f"Score: {eval_result['score']} / {eval_result['max_score']}")
        for r in eval_result["reasons"]:
            print(f"- {r}")

        results.append(
            {
                "question": q,
                "user_id": user_id,
                "score": eval_result["score"],
                "max_score": eval_result["max_score"],
                "reasons": eval_result["reasons"],
                "plan_length": len(plan_text),
            }
        )

    return results



eval_results = run_simple_evaluation(EVAL_QUESTIONS, period="last_30_days")

print("\n=== Summary table ===")
for r in eval_results:
    print(
        f"Q: {r['question'][:60]}... | "
        f"score={r['score']}/{r['max_score']} | "
        f"plan_length={r['plan_length']}"
    )



# Runner and simple demo
orchestrator_runner = InMemoryRunner(
    agent=bizops_orchestrator_agent,
    app_name="solobiz_orchestrator_app",
)

session = create_session(orchestrator_runner, user_id="kaggle_demo_user")

demo_question = (
    "Please review my recent sales and tell me what I should focus on "
    "next week to improve revenue."
)

final_answer = run_agent_once(
    orchestrator_runner,
    session_id=session.id,
    user_message=demo_question,
    user_id="kaggle_demo_user",
)

print("=== ğŸ’¬ Final SoloBiz Ops answer (Path A) ===\n")
print(final_answer)


# Orchestrator agent used only as a final formatter (no tools needed here)
finalizer_agent = LlmAgent(
    name="bizops_finalizer_agent",
    model="gemini-2.5-pro",
    instruction=(
        "You receive internal results from specialist agents and must speak "
        "directly to the business owner.\n\n"
        "Use the analytics summary, the action plan and the current notes "
        "to produce:\n"
        "- 2â€“3 sentences summarizing the situation,\n"
        "- a short bullet list with the most important next actions.\n"
        "Do not mention internal agent names or technical details."
    ),
    tools=[],  # no tools in this path
)

finalizer_runner = InMemoryRunner(
    agent=finalizer_agent,
    app_name="solobiz_finalizer_app",
)


def run_solobiz_workflow_and_finalize(
    user_question: str,
    period: str = "last_30_days",
    user_id: str = "demo_user",
):
    """
    Path B:
    1) Run the explicit multi-agent workflow (Analytics + Planner + Notes).
    2) Ask the finalizer agent to turn internal results into a final answer.
    """
    # 1) Multi-agent workflow with observability (already defined in previous sections)
    workflow_result = run_solobiz_workflow_with_observability(
        user_question=user_question,
        period=period,
        user_id=user_id,
    )

    analytics_summary = workflow_result["analytics_summary"]
    plan_text = workflow_result["plan_text"]
    notes_summary = workflow_result["notes_summary"]

    # 2) Finalizer agent: produce a user-facing answer
    session = create_session(finalizer_runner, user_id=user_id)

    finalizer_prompt = (
        "You are preparing a summary for a small business owner.\n\n"
        f"User question: {user_question}\n\n"
        f"Analytics summary:\n{analytics_summary}\n\n"
        f"Action plan:\n{plan_text}\n\n"
        f"Notes summary:\n{notes_summary}\n\n"
        "Write a concise answer:\n"
        "- 2â€“3 sentences overview,\n"
        "- a bullet list of next actions."
    )

    final_answer = run_agent_once(
        finalizer_runner,
        session_id=session.id,
        user_message=finalizer_prompt,
        user_id=user_id,
    )

    return {
        "final_answer": final_answer,
        "analytics_summary": analytics_summary,
        "plan_text": plan_text,
        "notes_summary": notes_summary,
        "trace": workflow_result["trace"],
    }


# Small demo
demo_result_b = run_solobiz_workflow_and_finalize(
    user_question="Review my last month and suggest what to focus on next week.",
    period="last_30_days",
    user_id="kaggle_demo_user",
)

print("=== ğŸ’¬ Final SoloBiz Ops answer (Path B) ===\n")
print(demo_result_b["final_answer"])

