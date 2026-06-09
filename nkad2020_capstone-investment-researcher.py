!pip install -q google-adk google-generativeai yfinance




import asyncio
import os
import pandas as pd
import yfinance as yf

from kaggle_secrets import UserSecretsClient
import google.generativeai as genai

import os

user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("GOOGLE_API_KEY")

os.environ["GOOGLE_API_KEY"] = api_key
os.environ["GOOGLE_GENAI_API_KEY"] = api_key


import asyncio
import yfinance as yf
import pandas as pd
from typing import List, Dict, Any

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

MODEL_ID = "gemini-2.5-flash-lite"

APP_NAME = "kaggle-finance-mvp"
USER_ID = "kaggle-user"




def fetch_price_history(
    ticker: str,
    period: str = "1mo",
    interval: str = "1d",
):
    data = yf.download(ticker, period=period, interval=interval, progress=False)

    if data.empty:
        return []

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ["_".join(str(x) for x in col) for col in data.columns]

    data = data.tail(5).reset_index()
    data["Date"] = data["Date"].astype(str)
    return data.to_dict(orient="records")



import yfinance as yf
import math

def json_safe(val):
    """Convert numpy + NaN + Pandas to JSON-safe primitives."""
    if val is None:
        return None
    if isinstance(val, (int, float, str, bool)):
        # convert NaN → None
        if isinstance(val, float) and math.isnan(val):
            return None
        return val
    try:
        # numpy types → python
        if math.isnan(float(val)):
            return None
        return float(val)
    except:
        return None


def clean_info(info: dict):
    """Return a JSON-safe dict with dangerous fields removed."""
    safe = {}
    for k, v in info.items():
        val = json_safe(v)
        safe[k] = val
    return safe


import yfinance as yf
from typing import List, Dict, Any


def fetch_stocks_by_factor(
    style: str = "growth",
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    Simple yfinance-based factor screener for a small curated universe.

    Args:
        style: one of {"growth", "value", "momentum"} (case-insensitive)
        limit: number of tickers to return (best N by score)

    Returns:
        List of dicts:
        [
          {
            "ticker": "NVDA",
            "score": 45.2,         # style-specific score
            "ret_12m": 45.2,       # % 12-month return
            "ret_3m": 10.4,        # % 3-month return
          },
          ...
        ]
    """
    style = style.lower()

    # --- 1. Curated universes per style (MVP-friendly) -----------------
    # For a real system you’d swap this with an API-based universe.
    growth_universe = [
        "NVDA", "AVGO", "META", "AMZN", "TSLA",
        "ADBE", "CRM", "NOW", "AMD", "CDNS",
    ]

    value_universe = [
        "BRK-B", "JNJ", "PG", "PEP", "KO",
        "XOM", "CVX", "UNH", "MRK", "ABBV",
    ]

    momentum_universe = [
        "NVDA", "META", "AMZN", "TSLA", "AVGO",
        "ADBE", "LRCX", "PANW", "ANET", "COST",
    ]

    if style == "growth":
        universe = growth_universe
    elif style == "value":
        universe = value_universe
    elif style == "momentum":
        universe = momentum_universe
    else:
        # Fallback: use growth universe if style is unknown
        universe = growth_universe

    results: List[Dict[str, Any]] = []

    # --- 2. Score each ticker individually -----------------------------
    for ticker in universe:
        try:
            hist = yf.Ticker(ticker).history(period="1y", interval="1d")
            if hist.empty or len(hist) < 60:
                continue

            close = hist["Close"]
            ret_12m = float((close.iloc[-1] / close.iloc[0] - 1) * 100.0)

            # 3-month momentum as rough short-term factor
            if len(close) > 60:
                close_3m = close.iloc[-60:]
                ret_3m = float((close_3m.iloc[-1] / close_3m.iloc[0] - 1) * 100.0)
            else:
                ret_3m = ret_12m

            # --- 3. Style-specific scoring ----------------------------
            if style == "growth":
                # Use 12m return as a crude growth proxy
                score = ret_12m
            elif style == "momentum":
                # Use 3m return as momentum
                score = ret_3m
            else:  # "value"
                # Favor names that *haven’t* run too much (lower 12m return)
                # Negate so "cheaper" (lower return) => higher score when sorted asc
                score = -ret_12m

            results.append(
                {
                    "ticker": ticker,
                    "score": round(score, 2),
                    "ret_12m": round(ret_12m, 2),
                    "ret_3m": round(ret_3m, 2),
                }
            )
        except Exception:
            # On any yfinance hiccup, just skip this ticker
            continue

    if not results:
        return []

    # --- 4. Sort & slice -----------------------------------------------
    if style in {"growth", "momentum"}:
        # higher is better
        results.sort(key=lambda r: r["score"], reverse=True)
    else:
        # for value, we used negative return as score, so sort ascending
        results.sort(key=lambda r: r["score"])

    return results[:limit]



research_agent = LlmAgent(
    name="research_agent",
    model=MODEL_ID,
    description="Collects market context, screener results, and clarifies objective.",
    instruction=(
        "You receive: user question, price history, and optional screener data.\n"
        "If the user asks for stock lists or ETF lists, you MUST call the screener tool.\n"
        "If ticker-specific: use price_history.\n"
        "When the user asks for growth/value/momentum stocks, call fetch_stocks_by_factor(style=...)\n"
        "and list the tickers with their ret_12m/ret_3m as justification.\n"
        "Do NOT guess tickers.\n"
        "Output strictly:\n"
        "RESEARCH SUMMARY:\n- ...\n\n"
        "KEY RISKS:\n- ...\n"
    ),
    tools=[fetch_price_history, fetch_stocks_by_factor],
)


planner_agent = LlmAgent(
    name="planner_agent",
    model=MODEL_ID,
    description="Structured investment plan.",
    instruction=(
        "You receive RESEARCH SUMMARY and KEY RISKS.\n"
        "Output strictly as:\n\n"
        "PLAN:\n"
        "- Thesis: ...\n"
        "- Time horizon: ...\n"
        "- Entry framing: ...\n"
        "- Exit framing: ...\n"
        "- Risk factors:\n"
        "  - ...\n"
        "  - ...\n"
    ),
)

evaluator_agent = LlmAgent(
    name="evaluator_agent",
    model=MODEL_ID,
    description="Plan evaluator.",
    instruction=(
        "Restate research, restate plan, then score it.\n\n"
        "RESEARCH SUMMARY (CONDENSED):\n"
        "- ...\n\n"
        "PLAN (RESTATED):\n"
        "- Thesis: ...\n"
        "- Time horizon: ...\n"
        "- Entry framing: ...\n"
        "- Exit framing: ...\n"
        "- Risk factors:\n"
        "  - ...\n\n"
        "PLAN REVIEW:\n"
        "- Clarity: x/10\n"
        "- Realism: x/10\n"
        "- Risk awareness: x/10\n\n"
        "VERDICT: BUY | WATCH | AVOID"
    ),
)

root_agent = SequentialAgent(
    name="pipeline",
    description="Research → Planning → Evaluation",
    sub_agents=[research_agent, planner_agent, evaluator_agent],
)



async def run_once(question: str) -> str:
    runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)

    # Create a NEW session each time → Kaggle-friendly
    session = await runner.session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
    )

    content = types.Content(role="user", parts=[types.Part(text=question)])

    final_text = ""

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session.id,
        new_message=content
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_text = event.content.parts[0].text

    return final_text



from IPython.display import Markdown, display

async def ask(question: str):
    """Run one query through the pipeline and render result as markdown."""
    print(f"[User] {question}")

    answer = await run_once(question)

    md = f"""### User  
{question}

---

### Assistant  

{answer}
"""
    display(Markdown(md))




await ask("What's the outlook for AMD for the next 12-18 months?")





