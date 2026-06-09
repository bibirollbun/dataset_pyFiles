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
    print("✅ Gemini API key setup complete.")
except Exception as e:
    print(
        f"🔑 Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )



# Install ADK
# pip install google-adk --quiet

# Check if ADK is already installed
try:
    import google.adk
    print("google-adk already installed.")
except ImportError:
    print("Installing google-adk...")
    !pip install google-adk --quiet
    print("google-adk installed.")

# Imports
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime
from google.adk.agents.llm_agent import Agent
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
from google.adk.plugins.logging_plugin import (
    LoggingPlugin,
)

# Display settings
plt.style.use('default')
pd.set_option('display.max_rows', 20)

print("Environment ready.")


business_data = {
    "month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
    "revenue": [12000, 15000, 17000, 16000, 19000, 21000],
    "costs":   [8000,  9000,  9500,  9200, 10500, 11500],
}

df_business = pd.DataFrame(business_data)
df_business["profit"] = df_business["revenue"] - df_business["costs"]

display(df_business)


retry_config = types.HttpRetryOptions(
    attempts=5,              # Maximum retry attempts
    exp_base=7,              # Delay multiplier (exponential backoff)
    initial_delay=1,         # Seconds
    http_status_codes=[429, 500, 503, 504],  # Errors to retry
)


APP_NAME = "enterprise_helper_demo"
USER_ID = "demo-user-1"
SESSION_ID = "session-1"   # we'll just reuse one session for now
MODEL_NAME = Gemini(
    model="gemini-2.5-flash",   # or "gemini-2.5-flash"
    retry_options=retry_config,      # this is the key part
)   



from collections import defaultdict
from collections import Counter
from datetime import datetime

# -------------------------------
# Long-term memory (simple, global)
# -------------------------------

# Stores conversation turns across sessions
conversation_memory: list[dict] = []

# Stores tool calls + high-level result info
tool_memory: list[dict] = []

# -------------------------------------------------
# Global variable to store last health evaluation
# -------------------------------------------------
agent_health: dict | None = None

def remember_conversation(session_name: str, user_text: str, agent_text: str) -> None:
    """
    Store a single user → agent exchange in long-term memory.
    Only simple strings & timestamps so it's safe to serialize.
    """
    conversation_memory.append({
        "timestamp": datetime.utcnow().isoformat(),
        "session": session_name,
        "user": user_text,
        "agent": agent_text,
    })


def remember_tool_call(
    tool_name: str,
    status: str,
    extra_info: dict | None = None,
) -> None:
    """
    Store a summary of a tool invocation in long-term memory.
    """
    tool_memory.append({
        "timestamp": datetime.utcnow().isoformat(),
        "tool": tool_name,
        "status": status,
        "info": extra_info or {},
    })


def summarize_memory() -> dict:
    """
    Tool: Return a compact snapshot of recent memory so the agent
    can remind the user what happened in earlier interactions.

    This is our 'long-term memory' tool.
    """
    # To keep things small, just send the last few entries
    recent_conversation = conversation_memory[-10:]
    recent_tools = tool_memory[-10:]

    return {
        "status": "success",
        "recent_conversation": recent_conversation,
        "recent_tool_calls": recent_tools,
    }

print("✅ Long-term memory helpers defined.")



def evaluate_agent_health() -> dict:
    """
    Simple evaluation / health snapshot based on long-term tool memory.

    Returns:
      - total tool calls
      - successes vs errors
      - per-tool call counts
    """
    global agent_health
    if not tool_memory:
        agent_health = {
            "status": "success",
            "message": "No tool calls logged yet.",
            "metrics": {},
        }
        return agent_health

    total_calls = len(tool_memory)
    successes = sum(1 for t in tool_memory if t.get("status") == "success")
    errors = total_calls - successes

    tool_names = [t.get("tool", "unknown") for t in tool_memory]
    per_tool_counts = dict(Counter(tool_names))

    agent_health = {
        "status": "success",
        "metrics": {
            "total_tool_calls": total_calls,
            "successful_calls": successes,
            "error_calls": errors,
            "per_tool_counts": per_tool_counts,
        },
    }
    return agent_health
    
 

def plot_agent_health_dashboard() -> dict:
    """
    Create a visualization dashboard summarizing the agent’s performance
    using the latest evaluate_agent_health() data.

    It displays:
    - Total tool calls
    - Successful vs error calls
    - Per-tool usage bar chart
    - Success rate percentage

    Dashboard is drawn in the notebook.
    """
    global agent_health  # You stored the evaluate_agent_health() result here.

    try:
        if agent_health is None or "metrics" not in agent_health:
            result = {
                "status": "error",
                "message": "No agent health metrics found. Run evaluate_agent_health() first."
            }
            remember_tool_call("plot_agent_health_dashboard", "error")
            return result

        metrics = agent_health["metrics"]

        total_calls = metrics["total_tool_calls"]
        success_calls = metrics["successful_calls"]
        error_calls = metrics["error_calls"]
        per_tool = metrics["per_tool_counts"]

        success_rate = (success_calls / total_calls * 100) if total_calls > 0 else 0

        # ---- Plot Dashboard ----
        plt.figure(figsize=(10, 6))

        # 1) Bar chart of tool usage
        plt.subplot(2, 1, 1)
        plt.bar(per_tool.keys(), per_tool.values(), color='skyblue')
        plt.title("Tool Usage Counts")
        plt.ylabel("Calls")
        plt.xticks(rotation=45)

        # 2) Success vs error + success rate
        plt.subplot(2, 1, 2)
        plt.bar(["Success", "Errors"], [success_calls, error_calls],
                color=["green", "red"])
        plt.title(f"Overall Agent Performance (Success Rate: {success_rate:.1f}%)")
        plt.ylabel("Number of Calls")

        plt.tight_layout()
        plt.show()

        result = {
            "status": "success",
            "message": "Agent health dashboard plotted."
        }

        remember_tool_call(
            "plot_agent_health_dashboard",
            "success",
            {"success_rate": success_rate}
        )
        return result

    except Exception as e:
        print(f"[plot_agent_health_dashboard] Error: {e}")
        result = {
            "status": "error",
            "message": str(e)
        }
        remember_tool_call("plot_agent_health_dashboard", "error")
        return result





# This will always hold the *currently active* dataset
current_df: pd.DataFrame | None = df_business.copy()




# Very simple demo tool (we'll replace/extend this later)
def draft_summary(user_request: str) -> dict:
    """
    Simple helper tool: returns the raw user request as a 'task'
    that the agent can refine into a nicer answer.
    For now it's just a placeholder so that our agent has at least one tool.
    """
    return {
        "status": "success",
        "task": user_request,
    }

# ============================================
# Tool 0: load_data_csv  (LOAD DATA TOOL)
# ============================================

def load_data_csv(file_path: str) -> dict:
    """
    Load a CSV file from the given file_path into the *current_df* variable.

    The agent should call this when the user asks to load or switch datasets.
    The tool returns a short summary of the loaded data (rows, columns, head).

    Args:
        file_path: Path to a CSV file visible in this notebook environment.
                   Example: '/kaggle/input/your-dataset/file.csv'
    Returns a short summary: shape + first few column names.               
    """
    global current_df

    try:
        df = pd.read_csv(file_path)

        # Basic sanity checks
        if df.empty:
            result = {
                "status": "error",
                "message": f"Loaded file '{file_path}' is empty.",
            }
            # log tool call
            remember_tool_call(
                tool_name="load_data_csv",
                status=result["status"],
                extra_info={"file_path": file_path},
            )
            return result

        current_df = df  # update the active dataset the agent works with

        result = {
            "status": "success",
            "file_path": file_path,
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "columns": list(df.columns[:20]),  # show up to 20 column names
        }

        remember_tool_call(
            tool_name="load_data_csv",
            status=result["status"],
            extra_info={
                "file_path": file_path,
                "n_rows": result["n_rows"],
                "n_cols": result["n_cols"],
            },
        )
        return result

    except Exception as e:
        print(f"[load_data_csv] Error: {e}")
        result = {
            "status": "error",
            "message": f"Could not load '{file_path}': {e}",
        }
        remember_tool_call(
            tool_name="load_data_csv",
            status=result["status"],
            extra_info={"file_path": file_path},
        )
        return result
    
# ============================================
# ============================================
# Tool 1: get_basic_kpis
# ============================================

def get_basic_kpis() -> dict:
    """
    Returns a few basic KPIs from the *currently loaded* business dataset.

    The agent should call this tool whenever the user asks for
    an overview of recent business performance.

    The dataset is read from the global 'current_df', which can be changed
    by the 'load_data_csv' tool.
    """
    global current_df

    if current_df is None:
        result = {
            "status": "error",
            "message": "No dataset is loaded yet. Please call load_data_csv first.",
        }
        remember_tool_call("get_basic_kpis", result["status"], {})
        return result

    df = current_df

    # Case-insensitive lookup for column names
    cols = {c.lower(): c for c in df.columns}
    if "revenue" not in cols or "profit" not in cols:
        result = {
            "status": "error",
            "message": (
                "Dataset must have 'revenue' and 'profit' columns "
                "(case-insensitive) for KPI calculation."
            ),
        }
        remember_tool_call("get_basic_kpis", result["status"], {})
        return result

    rev_col = cols["revenue"]
    prof_col = cols["profit"]

    try:
        total_revenue = float(df[rev_col].sum())
        total_profit = float(df[prof_col].sum())
        avg_margin_pct = float((df[prof_col].sum() / df[rev_col].sum()) * 100.0)

        latest_row = df.iloc[-1]
        latest_month = str(latest_row.get("month", "latest_period"))
        latest_revenue = float(latest_row[rev_col])
        latest_profit = float(latest_row[prof_col])

        result = {
            "status": "success",
            "kpis": {
                "total_revenue": total_revenue,
                "total_profit": total_profit,
                "avg_margin_pct": avg_margin_pct,
                "latest_month": latest_month,
                "latest_revenue": latest_revenue,
                "latest_profit": latest_profit,
            },
        }

        remember_tool_call(
            tool_name="get_basic_kpis",
            status=result["status"],
            extra_info={
                "latest_month": latest_month,
                "latest_revenue": latest_revenue,
                "latest_profit": latest_profit,
            },
        )
        return result

    except Exception as e:
        print(f"[get_basic_kpis] Error: {e}")
        result = {
            "status": "error",
            "message": str(e),
        }
        remember_tool_call("get_basic_kpis", result["status"], {})
        return result
# ============================================
# Tool 2: visualize_revenue_profit
# ============================================

def plot_revenue_profit_over_time() -> dict:
    """
    Plot revenue and profit over time from the *currently loaded* dataset.

    The agent should call this when the user asks for charts/plots/visualizations
    of business performance.
    """
    global current_df

    if current_df is None:
        result = {
            "status": "error",
            "message": "No dataset is loaded yet. Please call load_data_csv first.",
        }
        remember_tool_call(
            tool_name="plot_revenue_profit_over_time",
            status=result["status"],
            extra_info={"reason": "no dataset loaded"}
        )
        return result

    df = current_df.copy()

    # Case-insensitive column lookup
    cols = {c.lower(): c for c in df.columns}
    if "revenue" not in cols or "profit" not in cols:
        result = {
            "status": "error",
            "message": (
                "To plot revenue vs profit, the dataset must have "
                "'revenue' and 'profit' columns (any case)."
            ),
        }
        remember_tool_call(
            tool_name="plot_revenue_profit_over_time",
            status=result["status"],
            extra_info={"reason": "missing revenue/profit columns"}
        )
        return result

    rev_col = cols["revenue"]
    prof_col = cols["profit"]
    month_col = cols.get("month")  # may be None

    # Ensure numeric
    df[rev_col] = pd.to_numeric(df[rev_col], errors="coerce")
    df[prof_col] = pd.to_numeric(df[prof_col], errors="coerce")

    # Group by month if available
    if month_col:
        df[month_col] = pd.to_numeric(df[month_col], errors="coerce")
        grouped = (
            df.groupby(month_col)[[rev_col, prof_col]]
            .sum()
            .reset_index()
            .sort_values(month_col)
        )
        x = grouped[month_col]
        x_label = "Month"
    else:
        grouped = df[[rev_col, prof_col]].reset_index(drop=True)
        grouped["period"] = grouped.index + 1
        x = grouped["period"]
        x_label = "Period"

    # Create the plot
    plt.figure(figsize=(8, 4))
    plt.plot(x, grouped[rev_col], label="Revenue")
    plt.plot(x, grouped[prof_col], label="Profit")
    plt.xlabel(x_label)
    plt.ylabel("Amount")
    plt.title("Revenue & Profit Over Time")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Tool call memory log
    remember_tool_call(
        tool_name="plot_revenue_profit_over_time",
        status="success",
        extra_info={
            "x_axis": x_label,
            "n_points": len(x),
        }
    )

    return {
        "status": "success",
        "message": "Revenue vs profit chart generated in the notebook output.",
    }
# ============================================
# Tool 3: get_monthly_kpi
# ============================================

def get_monthly_kpi(month: int) -> dict:
    """
    Return KPIs for a specific month from the *currently loaded* dataset.

    Args:
        month: Month number as an integer (e.g. 4 for Month 4) that matches
               the values in the 'month' column of the dataset.
    """
    global current_df

    # If no data is loaded
    if current_df is None:
        result = {
            "status": "error",
            "message": "No dataset is loaded yet. Please call load_data_csv first.",
        }
        remember_tool_call("get_monthly_kpi", result["status"], {"month": month})
        return result

    df = current_df

    # Case-insensitive column lookup
    cols = {c.lower(): c for c in df.columns}
    if "revenue" not in cols or "profit" not in cols or "month" not in cols:
        result = {
            "status": "error",
            "message": (
                "Dataset must contain 'month', 'revenue' and 'profit' columns "
                "(case-insensitive) to compute monthly KPIs."
            ),
        }
        remember_tool_call("get_monthly_kpi", result["status"], {"month": month})
        return result

    rev_col = cols["revenue"]
    prof_col = cols["profit"]
    month_col = cols["month"]

    try:
        # Ensure numeric
        df[rev_col] = pd.to_numeric(df[rev_col], errors="coerce")
        df[prof_col] = pd.to_numeric(df[prof_col], errors="coerce")
        df[month_col] = pd.to_numeric(df[month_col], errors="coerce")

        month_df = df[df[month_col] == month]

        if month_df.empty:
            result = {
                "status": "error",
                "message": f"No rows found for month {month}.",
            }
            remember_tool_call("get_monthly_kpi", result["status"], {"month": month})
            return result

        total_revenue = float(month_df[rev_col].sum())
        total_profit = float(month_df[prof_col].sum())
        avg_margin_pct = float((month_df[prof_col].sum() / month_df[rev_col].sum()) * 100.0)

        result = {
            "status": "success",
            "month": int(month),
            "kpis": {
                "revenue": total_revenue,
                "profit": total_profit,
                "profit_margin_pct": avg_margin_pct,
            },
        }

        remember_tool_call(
            tool_name="get_monthly_kpi",
            status=result["status"],
            extra_info={
                "month": int(month),
                "revenue": total_revenue,
                "profit": total_profit,
                "profit_margin_pct": avg_margin_pct,
            },
        )
        return result

    except Exception as e:
        print(f"[get_monthly_kpi] Error: {e}")
        result = {
            "status": "error",
            "message": str(e),
        }
        remember_tool_call("get_monthly_kpi", result["status"], {"month": month})
        return result

# ============================================
# Tool 4: analyze_orders_customers
# ============================================

def analyze_orders_customers() -> dict:
    """
    Returns high-level KPIs about *orders* and *customers* from the
    currently loaded dataset.

    The agent should call this tool when the user asks about:
      - orders
      - customers
      - demand / traffic
      - average order value

    Data is read from the global 'current_df', which is set by load_data_csv().
    """
    global current_df

    # No dataset loaded
    if current_df is None:
        result = {
            "status": "error",
            "message": "No dataset is loaded yet. Please call load_data_csv first.",
        }
        remember_tool_call("analyze_orders_customers", result["status"], {})
        return result

    df = current_df

    # Case-insensitive column lookup
    cols = {c.lower(): c for c in df.columns}
    if "orders" not in cols or "customers" not in cols:
        result = {
            "status": "error",
            "message": (
                "Dataset must have 'orders' and 'customers' columns "
                "(case-insensitive) to analyze orders and customers."
            ),
        }
        remember_tool_call("analyze_orders_customers", result["status"], {})
        return result

    orders_col = cols["orders"]
    customers_col = cols["customers"]

    # Make sure columns are numeric
    df[orders_col] = pd.to_numeric(df[orders_col], errors="coerce")
    df[customers_col] = pd.to_numeric(df[customers_col], errors="coerce")

    # ---- Core metrics (cast to plain Python types, not numpy) ----
    total_orders = int(df[orders_col].sum())
    total_customers = int(df[customers_col].sum())
    avg_orders_per_day = float(df[orders_col].mean())
    avg_customers_per_day = float(df[customers_col].mean())

    # Try to get Average Order Value (AOV)
    avg_order_value = None
    if "avg_order_value" in cols:  # if dataset already has this column
        aov_col = cols["avg_order_value"]
        df[aov_col] = pd.to_numeric(df[aov_col], errors="coerce")
        avg_order_value = float(df[aov_col].mean())
    elif "revenue" in cols:       # otherwise compute from revenue / orders
        rev_col = cols["revenue"]
        df[rev_col] = pd.to_numeric(df[rev_col], errors="coerce")
        if total_orders > 0:
            avg_order_value = float(df[rev_col].sum() / total_orders)

    result_payload = {
        "total_orders": total_orders,
        "total_customers": total_customers,
        "avg_orders_per_day": avg_orders_per_day,
        "avg_customers_per_day": avg_customers_per_day,
    }
    if avg_order_value is not None:
        result_payload["avg_order_value"] = avg_order_value

    result = {
        "status": "success",
        "orders_customers": result_payload,
    }

    # Long-term memory logging
    remember_tool_call(
        tool_name="analyze_orders_customers",
        status=result["status"],
        extra_info={
            "total_orders": total_orders,
            "total_customers": total_customers,
            "avg_order_value": avg_order_value,
        },
    )

    return result




root_agent = LlmAgent(
    model=MODEL_NAME,
    name="enterprise_helper_agent",
    description="Business insights agent.",
    instruction=(
        "You are a business insights assistant.\n\n"

        "============================\n"
        "   DATA HANDLING (TOOLS)\n"
        "============================\n"
        "1) When the user asks to *load*, *switch*, or *use* a dataset, "
        "always call the 'load_data_csv' tool.\n\n"

        "2) When the user asks about revenue, profit, growth, performance, "
        "business KPIs, or summary of financials, "
        "always call 'get_basic_kpis' first. After receiving the tool output, "
        "explain it clearly in natural language.\n\n"

        "2b) When the user asks for KPIs for a **specific month except Month 5 which is latest month** "
        "(for example, 'Month 4' or 'the KPIs for month X'), "
        "call 'get_monthly_kpi' with the appropriate month number. "
        "After that, you may compare that month with the latest month "
        "using information from 'get_basic_kpis'.\n\n"

        "3) When the user asks about *orders*, *customers*, or *average order value* "
        "(for example: demand, traffic, number of customers, order volume), "
        "call 'analyze_orders_customers' first and then explain the metrics "
        "clearly in plain English.\n\n"

        "4) When the user asks for visualizations, charts, trends, or "
        "'revenue vs profit over time', call 'plot_revenue_profit_over_time'. "
        "After the chart is created, describe the insight clearly.\n\n"

        "5) Use 'draft_summary' ONLY to rewrite or refine text after a tool has returned.\n\n"

        "6) When the user asks for metrics about your own performance, logs, or an evaluation dashboard "
        "or says things like 'how healthy is the agent', FIRST call the 'evaluate_agent_health' tool "
        "to get a summary of tool usage. THEN call the 'plot_agent_health_dashboard' tool to visualize "
        "those metrics.\n\n"

        "=====================================\n"
        "   LONG-TERM MEMORY (YOUR NEW FEATURE)\n"
        "=====================================\n"
        "You have access to long-term memory stored inside this notebook. "
        "This memory persists across all sessions and stores:\n"
        "- Previous user questions\n"
        "- Your final answers\n"
        "- All tool calls and their results (success or error)\n\n"

        "You may call the 'summarize_memory' tool whenever:\n"
        "- The user asks 'What did we do earlier?'\n"
        "- The user asks 'What dataset did we load last time?'\n"
        "- The user asks 'Remind me what KPIs you gave earlier'\n"
        "- The user refers to past behaviour, previous months, earlier sessions, "
        "or prior responses.\n\n"

        "When using memory:\n"
        "- FIRST call 'summarize_memory'\n"
        "- THEN read the returned details\n"
        "- THEN produce a clear explanation based on it\n\n"

        "=====================================\n"
        "   ERROR HANDLING\n"
        "=====================================\n"
        "If any tool returns an error, do NOT invent results. "
        "Explain the error and suggest the correct usage.\n"
    ),
    tools=[load_data_csv, 
           get_basic_kpis, 
           get_monthly_kpi, 
           plot_revenue_profit_over_time, 
           analyze_orders_customers, 
           summarize_memory, 
           evaluate_agent_health, 
           plot_agent_health_dashboard, 
           draft_summary
          ],
    
)

print("Agent ready.")



# 1) Session service
session_service = InMemorySessionService()

# 2) Logging plugin (built-in observability)
logging_plugin = LoggingPlugin()

# 3) Runner: now with logging
runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service,
    plugins=[
        LoggingPlugin()
    ],   # 👈 this is the new bit
)

print("Runner + sessions + logging ready.")





import asyncio

# Helper to run a conversational session with our agent
async def run_session(
    runner_instance: Runner,
    user_queries: list[str] | str,
    session_name: str = "session-1",
):
    print(f"\n### Session: {session_name}")

    app_name = runner_instance.app_name

    # Create or get the session (so memory can persist)
    try:
        session = await session_service.create_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
    except Exception:
        session = await session_service.get_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )

    # Normalize to list
    if isinstance(user_queries, str):
        user_queries = [user_queries]

    for query_text in user_queries:
        print(f"\nUser  > {query_text}")

        # Store user query later in memory
        content = types.Content(role="user", parts=[types.Part(text=query_text)])

        last_agent_text = None

        async for event in runner_instance.run_async(
            user_id=USER_ID, session_id=session.id, new_message=content
        ):
            # Only print the final response text
            if event.is_final_response() and event.content and event.content.parts:
                text = event.content.parts[0].text
                if text:
                    last_agent_text = text
                    print(f"{MODEL_NAME} > {text}")

        # After the turn is complete, save it into long-term memory
        if last_agent_text is not None:
            remember_conversation(
                session_name=session_name,
                user_text=query_text,
                agent_text=last_agent_text,
            )

print("✅ Helper function with memory logging defined.")



import time

await run_session(
    runner,
    "Load the demo CSV at '/kaggle/input/synthetic-business-dataset/synthetic_business_data_500_lowercase_profit.csv' and then give me "
    "a quick overview of our business performance and key KPIs."
)
time.sleep(7) 


await run_session(
    runner,
    "Load the demo CSV at '/kaggle/input/synthetic-business-dataset/synthetic_business_data_500_lowercase_profit.csv' and then give me "
    "and generate a chart of revenue vs profit."
)
time.sleep(7) 


await run_session(
    runner,
    [
        "Give me a quick KPI overview.",
        "Remind me what you told me earlier.",
    ]
)
time.sleep(7) 


await run_session(
    runner,
    "I'm starting a new session now. Based on your long-term memory, "
    "remind me which CSV dataset we loaded earlier and what KPIs or charts "
    "you generated for me before.",
    session_name="session-2",
)
time.sleep(7) 


await run_session(
    runner,
    "Using what you remember from earlier, tell me how our latest month's performance "
    "compares to the overall average, and mention any key insights from past sessions.",
    session_name="session-2",
)
time.sleep(7) 


await run_session(
    runner,
    "Using the dataset we loaded earlier, tell me the revenue, profit, and margin for Month 1." 
    "and compare it to the latest month we discussed in our previous sessions." 
    "Also mention anything relevant from earlier sessions.",
    session_name="session-3"
)
time.sleep(7) 


await run_session(
    runner,
    "Using the dataset we loaded earlier, tell me the revenue, profit, and margin for Month 2." 
    "and compare it to the latest month we discussed in our previous sessions." 
    "Also tell me the margin for the latest month."
    "Also mention anything relevant from earlier sessions.",
    session_name="session-3"
)
time.sleep(7) 


await run_session(
    runner,
    "Load the demo CSV at "
    "'/kaggle/input/synthetic-business-dataset/synthetic_business_data_500_lowercase_profit.csv' "
    "and then give me a summary of our orders and customers, including total orders, "
    "total customers, and the average order value.",
    session_name="session-3"
)
time.sleep(7)


await run_session(
    runner,
    "Using what you remember from earlier, remind me of the key orders and customer "
    "metrics you found, and tell me whether our demand looks stable or growing.",
    session_name="session-4"
)
time.sleep(7)


await run_session(
    runner,
    "Show me your internal metrics: how many tool calls did you make and how many failed?",
    session_name="session-5"
)
time.sleep(7)

