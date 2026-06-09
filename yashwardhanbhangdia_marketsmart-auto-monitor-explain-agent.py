pip install google-adk pandas yfinance requests python-dotenv


#Imports
import os
import asyncio
import logging
from datetime import datetime, timedelta
import traceback

import pandas as pd
import yfinance as yf
import requests

# ADK imports
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.loop_agent import LoopAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.tools import google_search
from google.adk.tools import FunctionTool

from google.genai import types

from google.adk.models.google_llm import Gemini


#Setting up all the environment Variables or Fetching env config
import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


try:
    AV_API_KEY = UserSecretsClient().get_secret("AV_API_KEY")
    os.environ["AV_API_KEY"] = AV_API_KEY
    print("âœ… Alpha Vantage API key setup complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'AV_API_KEY' to your Kaggle secrets. Details: {e}"
    )


# -------- config (env) ----------
WATCHLIST = os.getenv("WATCHLIST", "TSLA,MSFT,SMX").split(",")
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "60"))
WINDOW_MINUTES = int(os.getenv("WINDOW_MINUTES", "1"))
PCT_THRESHOLD = float(os.getenv("PCT_THRESHOLD", "0.1"))  # percent
# NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
MODEL_NAME = os.getenv("ADK_MODEL", "gemini-2.5-flash-lite")  # or whichever model name you have access to
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
AV_API_KEY = os.getenv("AV_API_KEY")

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("adk-market-agent")


# --------- Utility functions (used as ADK tools) ----------
async def fetch_price(ticker: str, window_minutes: int):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": ticker,
        "interval": f"{window_minutes}min",
        "apikey": AV_API_KEY,
        "outputsize": "compact"
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    series = data.get(f"Time Series ({window_minutes}min)")
    if not series:
        raise RuntimeError(f"Alpha Vantage error: {data}")

    records = []
    for ts, values in list(series.items())[:window_minutes]:
        records.append({
            "timestamp": ts,
            "price": float(values["4. close"])
        })

    history = list(reversed(records))
    return {"ticker":ticker, "current_price": history[-1]["price"], "old_price": history[0]["price"]}


# await fetch_price("AAPL", 1)


# def compute_indicators(ticker: str, time_window: int):
#     """
#     ADK tool: compute SMA and RSI and basic vol spike; expects history_records as list-of-dicts
#     """
#     history_records = fetch_intraday_history(ticker, window_minutes)
#     prices = [r["price"] for r in history_records]
#     try:
#         if not history_records:
#             return {"status": "error", "error": "no_history", "ticker": ticker}
#         df = pd.DataFrame(history_records).set_index("Datetime") if "Datetime" in history_records[0] else pd.DataFrame(history_records)
#         # normalize column names to 'Close' and 'Volume' if needed
#         if "Close" not in df.columns and "close" in df.columns:
#             df = df.rename(columns={"close": "Close", "volume": "Volume"})
#         # compute indicators
#         df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    #     df = df.dropna(subset=["Close"])
    #     if df.empty:
    #         return {"status": "error", "error": "no_valid_close", "ticker": ticker}
    #     # SMA
    #     df["sma_20"] = df["Close"].rolling(window=20, min_periods=1).mean()
    #     df["sma_50"] = df["Close"].rolling(window=50, min_periods=1).mean()
    #     # RSI (14)
    #     delta = df["Close"].diff()
    #     gain = delta.clip(lower=0).rolling(window=14, min_periods=1).mean()
    #     loss = -delta.clip(upper=0).rolling(window=14, min_periods=1).mean().replace(0, 1e-8)
    #     rs = gain / loss
    #     df["rsi_14"] = 100 - (100 / (1 + rs))
    #     # volume spike heuristic
    #     if "Volume" in df.columns:
    #         vol_med = df["Volume"].median()
    #         df["vol_spike"] = df["Volume"].apply(lambda v: (v / vol_med) if vol_med > 0 else 1.0)
    #     else:
    #         df["vol_spike"] = 1.0
    #     latest = df.iloc[-1].to_dict()
    #     indicators = {
    #         "sma_20": float(latest.get("sma_20")) if latest.get("sma_20") is not pd.NA else None,
    #         "sma_50": float(latest.get("sma_50")) if latest.get("sma_50") is not pd.NA else None,
    #         "rsi_14": float(latest.get("rsi_14")) if latest.get("rsi_14") is not pd.NA else None,
    #         "vol_spike": float(latest.get("vol_spike")) if latest.get("vol_spike") is not pd.NA else None
    #     }
    #     return {"status": "success", "ticker": ticker, "indicators": indicators}
    # except Exception as e:
    #     logger.exception("compute_indicators_tool error for %s", ticker)
    #     return {"status": "error", "error": str(e), "ticker": ticker}



# CUSTOM TOOL DEFINITIONS. 
fetch_price_tool = FunctionTool(func=fetch_price)
# compute_indicators_tool = FunctionTool(func=compute_indicators)


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)



# --------- LLM Agent (reasoner) ----------
# The LLM agent has tools attached (above) and is responsible for creating a concise explanation.
reasoner_instruction = """
You are a concise market explainer agent. When invoked, you will be given:
 - a ticker (e.g., AAPL)
 - a percent change (%)
 - a time window (minutes)
You have tools available: google_search tool and fetch_price_tool.
Use fetch_price_tool to fetch the real time price related details for the mentioned ticker. 
Use google_search tool to retrieve the most recent real-time news related to the ticker.
Check if the price has moved by the percent change provided within the time winodw specific and then only create a detailed 2-4 sentence explanation linking news and indicators to the price move. 
End the explanation with a confidence score between 0 and 1.
Be explicit about which evidence you used (indicator or news source).

Here is the output structure:
Ticker: <name of the ticker mentioned>
Current Price: <current_price>
Price <time window> minutes ago: <old_price>
<if price change is more than the percent specified>
Reason for price change: <Reason you identified using latest news>
Confidence: <confidence>
Sources: <source>
<else>
No major price change update in the last <time window> minutes. 

Follow the output structue.

"""

# Create LlmAgent instance. ADK will route tool calls the agent makes to the Python tool functions we provide.
root_reasoner = LlmAgent(
    model=Gemini(model="gemini-2.0-flash", retry_options=retry_config),
    name="market_reasoner",
    description="Explains abnormal price moves using tools and indicators.",
    instruction=reasoner_instruction,
    # ADK expects tools as a list of callables. The framework will expose these to the LLM.
    tools=[google_search]
    # Custom tools can be passed here to perform more sophisticated mathemical operations in order to gain more insights from the market data.
)

# --------- Build a workflow:
# We'll use a SequentialAgent that calls the reasoner (which uses the tools) and returns explanation.
# The LoopAgent will repeatedly invoke the sequential pipeline for each ticker.
sequential_monitor = SequentialAgent(
    name="monitor_sequence",
    sub_agents=[root_reasoner]
)

# A helper: a small custom wrapper / "driver" agent that iterates through a watchlist and triggers the
# sequential flow per ticker. ADK loop agents normally iterate their sub_agents â€” here we keep it
# simple by creating a LoopAgent that will perform iterations and allow us to use ADK session state.
loop_agent = LoopAgent(
    name="market_loop",
    sub_agents=[sequential_monitor],
    max_iterations=1  # run until stopped by runner / external interrupt (None means indefinite)
)



# ---------- Runner & Session ----------
async def run_agent_watchers():
    """
    Instantiate session services and start the ADK runner.
    The ADK runner runs the loop agent and provides a dev UI if `adk web` is used.
    """

    # In-memory session & artifact services (demo / prototype)
    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()

    # create a session for this app (app_name/user/session can be any stable strings)
    session = await session_service.create_session(app_name="market_monitor_app", user_id="user1", session_id="default")

    # Create the Runner bound to our loop_agent, session service, and artifact service
    runner = Runner(agent=loop_agent, app_name="market_monitor_app", session_service=session_service, artifact_service=artifact_service)

    # The ADK runner can be run synchronously or with asyncio depending on version. Use asyncio.run to be safe.
    logger.info("Starting ADK Runner for market_loop. Watchlist=%s", WATCHLIST)

    # For each ticker we'll call the runner with a prompt that instructs the reasoner which ticker to process.
    # This is a minimal driver loop: we schedule a run for each ticker with a prompt payload.
    async def driver():
        try:
            while True:
                for ticker in WATCHLIST:
                    # Build a user prompt describing the job for the LLM agent (ADK will invoke LlmAgent)
                    prompt = f"Monitor ticker {ticker.strip().upper()}. If and only if the price moved by {PCT_THRESHOLD*100}% or more over the last {WINDOW_MINUTES} minutes, then explain the move using tools. Return a short explanation + confidence. Use tools rather than hallucinating."
                    query = types.Content(role="user", parts=[types.Part(text=prompt)])
                    # Run the agent with an invocation context that contains the prompt; the runner will start
                    # the loop_agent which will call the sequential sub-agent (our LlmAgent).
                    # Depending on ADK version, use runner.run(prompt=...) or runner.run_async
                    async for event in runner.run_async(user_id="user1",session_id="default",new_message=query):
                        # print("EVENT:", event)                   
                        if hasattr(event, "content") and event.content:
                            text = "".join(
                                p.text for p in event.content.parts
                                    if hasattr(p, "text") and p.text
                            )
                        print(f"Checking for {ticker}: {text}", "\n")
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            print("Driver cancelled, shutting down.")
        except Exception as e:
            print("Driver encountered exception")
            print("\nDRIVER CRASHED WITH ERROR:\n", repr(e))
            import traceback
            traceback.print_exc()

    # run loop
    try:
        await driver()
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, exiting.")
    except Exception:
        print("Runner failed")




await run_agent_watchers()

