# Imports and Gemini Initialization

import logging
logging.getLogger("google_genai.types").setLevel(logging.ERROR)

import os
import uuid
from datetime import timedelta
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from kaggle_secrets import UserSecretsClient

from google import genai
from google.genai import types

# ADK imports
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

# Load API key from Kaggle Secrets
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY is missing in Kaggle Secrets.")

os.environ["GOOGLE_API_KEY"] = api_key

# Initialize Gemini Client
client = genai.Client()
print("Gemini client initialized successfully.")



# Load Dataset and Aggregate Daily Sales

# Load training data
train_path = "/kaggle/input/store-sales-time-series-forecasting/train.csv"
df = pd.read_csv(train_path)

df["date"] = pd.to_datetime(df["date"])

# Aggregate total sales per day across all stores and items
daily = (
    df.groupby("date", as_index=False)["sales"]
      .sum()
      .sort_values("date")
      .reset_index(drop=True)
)

print("Rows in raw training data:", len(df))
print("Rows in daily aggregated data:", len(daily))

daily.tail()



# Utility Functions for Summary, Forecasting, and Evaluation

def get_recent_history(days: int) -> pd.DataFrame:
    """Return the last `days` days of aggregated daily sales."""
    days = max(1, int(days))
    return daily.tail(days).copy()


def get_sales_summary(days: int):
    """Return summary statistics and raw records for the last `days` days."""
    hist = get_recent_history(days)

    summary = {
        "days": int(days),
        "start_date": hist["date"].iloc[0].strftime("%Y-%m-%d"),
        "end_date": hist["date"].iloc[-1].strftime("%Y-%m-%d"),
        "total_sales": float(hist["sales"].sum()),
        "avg_sales": float(hist["sales"].mean()),
        "min_sales": float(hist["sales"].min()),
        "max_sales": float(hist["sales"].max()),
    }

    return {
        "summary": summary,
        "records": hist.to_dict(orient="records"),
    }


def forecast_future(history_df: pd.DataFrame, horizon: int = 14):
    """Train a simple Linear Regression model and forecast the next `horizon` days."""
    df_hist = history_df.sort_values("date").reset_index(drop=True).copy()
    df_hist["t"] = np.arange(len(df_hist))

    X = df_hist[["t"]].values
    y = df_hist["sales"].values

    model = LinearRegression()
    model.fit(X, y)

    last_t = df_hist["t"].iloc[-1]
    last_date = df_hist["date"].iloc[-1]

    forecasts = []
    for i in range(1, horizon + 1):
        t_future = last_t + i
        pred = model.predict([[t_future]])[0]

        forecasts.append({
            "date": (last_date + timedelta(days=i)).strftime("%Y-%m-%d"),
            "forecast": float(round(pred, 2)),
        })

    return forecasts


def evaluate_forecast(history_df: pd.DataFrame, forecast: list[dict]):
    """Evaluate forecast accuracy against a naive last-value baseline."""
    df_hist = history_df.sort_values("date").reset_index(drop=True).copy()
    last_val = float(df_hist["sales"].iloc[-1])

    preds = np.array([f["forecast"] for f in forecast], dtype=float)
    naive = np.full_like(preds, last_val)

    mae = float(np.mean(np.abs(preds - naive)))

    return {
        "naive_last_value": last_val,
        "mae_vs_naive": mae,
    }



# Define ADK Tools, Agents, and Sequential Workflow

# --- Tool wrapper functions (JSON-safe signatures) --- #

def tool_get_sales_summary(days: int = 30):
    """Tool: return recent sales summary for the last `days` days."""
    return get_sales_summary(days)


def tool_run_forecast(history_days: int = 90, horizon: int = 14):
    """Tool: train forecasting model and evaluate against naive baseline."""
    history_df = get_recent_history(history_days)
    forecast = forecast_future(history_df, horizon=horizon)
    evaluation = evaluate_forecast(history_df, forecast)

    return {
        "history_days": int(history_days),
        "horizon": int(horizon),
        "forecast": forecast,
        "evaluation": evaluation,
    }


# --- Agent: Sales Summary (Last 30 Days) --- #

analysis_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="analysis_agent",
    description="Generates a structured 30-day sales summary.",
    instruction=(
        "You are a sales analysis agent.\n"
        "Produce ONLY the following Markdown section using values from the sales-summary tool:\n\n"
        "## Sales Summary (Last 30 Days)\n\n"
        "Over the period **<START_DATE> – <END_DATE>**, total sales were **$<TOTAL_SALES>**, "
        "with an average of **$<AVG_SALES> per day**.\n\n"
        "- **Trend:** Describe the overall direction and key peaks.\n"
        "- **Seasonality:** Explain weekday vs weekend patterns.\n"
        "- **Key Outliers:**\n"
        "  - **Highest:** <DATE> at **$<MAX_VALUE>**\n"
        "  - **Lowest:** <DATE> at **$<MIN_VALUE>**\n\n"
        "Finish with one concise sentence summarizing overall performance.\n\n"
        "Rules:\n"
        "- Do not mention tools or forecasting.\n"
        "- Do not ask questions.\n"
        "- Do not repeat instructions.\n"
        "- Do not add extra headings.\n"
        "- Replace all placeholders with real values.\n"
    ),
    tools=[tool_get_sales_summary],
)


# --- Agent: 14-Day Forecast + Model Quality --- #

forecast_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="forecast_agent",
    description="Generates a 14-day forecast and evaluates model accuracy.",
    instruction=(
        "You are a sales forecasting specialist.\n"
        "Using the forecast tool, produce ONLY the following Markdown structure:\n\n"
        "## Forecast (Next 14 Days)\n\n"
        "Provide a short sentence describing the next 14 days "
        "(e.g., 'slightly declining but stable sales from <START_DATE> to <END_DATE>').\n\n"
        "| Date       | Forecast     |\n"
        "| :--------- | :----------- |\n"
        "| 2017-08-16 | $<VALUE_1>   |\n"
        "| 2017-08-17 | $<VALUE_2>   |\n"
        "| ...        | ...          |\n"
        "| 2017-08-29 | $<VALUE_14>  |\n\n"
        "Follow with 1–2 sentences summarizing the overall pattern.\n\n"
        "## Model Quality\n\n"
        "- **Model MAE:** $<MODEL_MAE>\n"
        "- **Naive MAE:** $<NAIVE_MAE>\n\n"
        "Add a short statement on whether the model outperforms the naive baseline.\n\n"
        "Rules:\n"
        "- Never say you cannot forecast.\n"
        "- Do not ask questions.\n"
        "- Do not add extra headings.\n"
        "- Replace all placeholders with real values.\n"
    ),
    tools=[tool_run_forecast],
)


# --- Sequential workflow: Summary → Forecast --- #

sales_forecasting_workflow = SequentialAgent(
    name="sales_forecasting_workflow",
    description="Runs 30-day summary followed by 14-day forecast.",
    sub_agents=[analysis_agent, forecast_agent],
)

print("Agents and workflow created.")



# Create Session Service and Runner

APP_NAME = "store_sales_enterprise_agent"

# In-memory session storage for this notebook
session_service = InMemorySessionService()

# Runner connects the workflow, sessions, and Gemini model
runner = Runner(
    agent=sales_forecasting_workflow,
    app_name=APP_NAME,
    session_service=session_service,
)

print("Runner initialized.")



# Function to Execute the Sales Agent

async def run_sales_agent(user_query: str) -> str:
    """Run the workflow end-to-end and return the final text output."""
    
    user_id = "kaggle_user"
    session_id = f"session_{uuid.uuid4().hex[:8]}"

    # Create a fresh session for each run
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
    )

    content = types.Content(
        role="user",
        parts=[types.Part(text=user_query)],
    )

    final_chunks = []

    # Execute workflow asynchronously and collect only LLM text outputs
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    final_chunks.append(part.text)

    return "".join(final_chunks).strip()



# Run the Sales Analysis + Forecast Workflow

import asyncio

user_query = "Help me understand recent sales and forecast the next two weeks."

async def main():
    result = await run_sales_agent(user_query)
    print(result)

await main()





