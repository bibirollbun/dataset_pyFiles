pip install -q google-adk[a2a]


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


pip install feedparser fpdf yfinance


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


# 1. IMPORTS, LOGGING, METRICS, A2A MESSAGE TYPE

import os
import sys
import json
import time
import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from concurrent.futures import ThreadPoolExecutor

import requests
import feedparser
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF

import google.generativeai as genai
from google.api_core import retry
from kaggle_secrets import UserSecretsClient

warnings.filterwarnings("ignore", category=RuntimeWarning, module="pandas")

VERBOSE = False  # flip to True if you want to see all INFO logs

log_level = logging.INFO if VERBOSE else logging.WARNING
logging.basicConfig(
    level=log_level,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("livestock_agents")
logger.setLevel(log_level)


# 2. AUTHENTICATION & MODEL INITIALIZATION

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
except Exception:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY not found. Set it via Kaggle Secrets or environment.")
    model = None
    retry_config = None
else:
    genai.configure(api_key=GOOGLE_API_KEY)
    retry_config = retry.Retry(
        initial=1.0,
        maximum=10.0,
        multiplier=2.0,
        deadline=90.0,
    )
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        logger.info("Model initialized: gemini-2.5-flash-lite")
    except Exception as e:
        logger.warning(f"gemini-2.5-flash-lite failed: {e}. Falling back to gemini-1.5-flash.")
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            logger.info("Model initialized: gemini-1.5-flash")
        except Exception as e2:
            logger.error(f"Critical error initializing model: {e2}")
            model = None


# 3. TOOL DEFINITIONS (COMMODITIES, WEATHER, NEWS, COST MODEL)

try:
    tz_cache_dir = os.path.join(os.getcwd(), ".yfinance_tz_cache")
    os.makedirs(tz_cache_dir, exist_ok=True)
    yf.set_tz_cache_location(tz_cache_dir)
except Exception as e:
    logger.warning(f"Could not set yfinance tz cache dir: {e}")

def get_region_weather_coords(region: str):
    r = (region or "United States").lower()
    if "canada" in r:
        return 51.0, -113.0, "Southern Alberta feedlot belt (Canada)"
    elif "brazil" in r:
        return -13.0, -56.0, "Central Brazil cattle region"
    else:
        return 41.25, -96.0, "Central Nebraska â€“ proxy for US feedlot region"

def get_commodities_data(period="1mo", commodity_type="cattle"):
    """
    Dynamic price fetch:
    - cattle -> Live Cattle (LE=F) + Corn (ZC=F)
    - hog    -> Lean Hogs (HE=F)  + Corn (ZC=F)
    """
    if commodity_type == "hog":
        tickers = ['HE=F', 'ZC=F']
        logger.info("Tool: fetching Lean Hogs + Corn prices from Yahoo Finance")
    else:
        tickers = ['LE=F', 'ZC=F']
        logger.info("Tool: fetching Live Cattle + Corn prices from Yahoo Finance")

    try:
        data = yf.download(tickers, period=period, progress=False, auto_adjust=False)
        if data.empty:
            return "Prices_Error: No data found."

        try:
            df = data.xs('Close', level=0, axis=1).reset_index()
        except Exception:
            df = data['Close'].reset_index()

        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        return df.tail(10).to_string(index=False)
    except Exception as e:
        logger.error(f"Error fetching commodities: {e}")
        return f"Prices_Error: {e}"

WEATHER_TOOL_SCHEMA = {
    "name": "get_weather_forecast",
    "description": "Get 1â€“7 day livestock-relevant weather forecast for a region.",
    "parameters": {
        "type": "object",
        "properties": {
            "region": {
                "type": "string",
                "description": "Human-readable region name, e.g. 'United States', 'Canada', 'Brazil'.",
            },
            "days": {
                "type": "integer",
                "description": "Number of days to forecast (1â€“7).",
                "minimum": 1,
                "maximum": 7,
                "default": 3,
            },
        },
        "required": ["region"],
    },
}

def get_real_weather(region="United States", days=3):
    """HTTP tool called according to WEATHER_TOOL_SCHEMA."""
    days = max(1, min(7, int(days)))
    lat, lon, label = get_region_weather_coords(region)
    logger.info(f"Tool: fetching weather for {label}, {days} days")
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&daily=temperature_2m_max,precipitation_sum&timezone=auto"
        )
        res = requests.get(url).json()['daily']
        report = f"Forecast ({label}):\n"
        for i in range(days):
            c = res['temperature_2m_max'][i]
            f_val = c * 9/5 + 32
            rain = res['precipitation_sum'][i]
            report += f"- Day {i+1}: High {f_val:.1f}Â°F ({c:.1f}Â°C), Precip {rain:.1f}mm\n"
        return report
    except Exception as e:
        logger.error(f"Weather API error: {e}")
        return f"Weather_Error: {e}"

def call_weather_tool_from_schema(arguments: dict):
    """
    Example of 'OpenAPI-style' structured call.
    In a full MCP setup, the LLM would emit these arguments as JSON.
    """
    region = arguments.get("region", "United States")
    days = arguments.get("days", 3)
    return get_real_weather(region=region, days=days)

def get_google_news_rss(commodity_type="cattle", region="United States"):
    if commodity_type == "hog":
        base_query = "hog market"
    else:
        base_query = "cattle market"

    r = region.lower()
    q_region = ""
    if "canada" in r:
        q_region = " canada"
    elif "brazil" in r:
        q_region = " brazil"
    elif "united states" in r or "usa" in r or "america" in r:
        q_region = " usa"

    query = (base_query + q_region).strip()
    logger.info(f"Tool: fetching Google News RSS for '{query}'")

    try:
        rss_url = (
            "https://news.google.com/rss/search?"
            f"q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
        )
        feed = feedparser.parse(rss_url)
        if not feed.entries:
            return "News_Error: No news found."
        return "\n".join([f"- {e.title}" for e in feed.entries[:3]])
    except Exception as e:
        logger.error(f"News feed error: {e}")
        return f"News_Error: {e}"

def calculate_cost_model(price, corn_price, commodity_type="cattle"):
    """
    Livestock/Corn ratio:
    - Hogs: Hog/Corn ratio thresholds ~14, 19
    - Cattle: Steer/Corn thresholds ~20, 25
    """
    logger.info(f"Tool: running {commodity_type.title()}/Corn ratio model")
    try:
        corn_dollars_per_bushel = float(corn_price) / 100.0
        if corn_dollars_per_bushel <= 0:
            return "Margins_Error: Invalid corn price."

        ratio = float(price) / corn_dollars_per_bushel

        if commodity_type == "hog":
            if ratio < 14:
                comment = "Ratio is low â€“ margins compressed (risk of liquidation)."
            elif ratio < 19:
                comment = "Ratio is neutral â€“ margins are okay but not exciting."
            else:
                comment = "Ratio is >19 â€“ margins look attractive (supportive for expansion)."
            label = "Hog/Corn Ratio"
        else:
            if ratio < 20:
                comment = "Ratio is low â€“ margins compressed."
            elif ratio < 25:
                comment = "Ratio is moderate â€“ margins are okay."
            else:
                comment = "Ratio is high â€“ margins look very attractive."
            label = "Cattle/Corn (Steer/Corn) Ratio"

        return f"{label}: {ratio:.1f}. {comment}"
    except Exception as e:
        logger.error(f"Cost model error: {e}")
        return f"Margins_Error: {e}"


# 3. TOOL DEFINITIONS (COMMODITIES, WEATHER, NEWS, COST MODEL)
def get_region_weather_coords(region: str):
    r = (region or "United States").lower()
    if "canada" in r:
        return 51.0, -113.0, "Southern Alberta feedlot belt (Canada)"
    elif "brazil" in r:
        return -13.0, -56.0, "Central Brazil cattle region"
    else:
        return 41.25, -96.0, "Central Nebraska â€“ proxy for US feedlot region"

def get_commodities_data(period="1mo", commodity_type="cattle"):
    """
    Dynamic price fetch:
    - cattle -> Live Cattle (LE=F) + Corn (ZC=F)
    - hog    -> Lean Hogs (HE=F)  + Corn (ZC=F)
    """
    if commodity_type == "hog":
        tickers = ['HE=F', 'ZC=F']
        logger.info("Tool: fetching Lean Hogs + Corn prices from Yahoo Finance")
    else:
        tickers = ['LE=F', 'ZC=F']
        logger.info("Tool: fetching Live Cattle + Corn prices from Yahoo Finance")

    try:
        data = yf.download(tickers, period=period, progress=False, auto_adjust=False)
        if data.empty:
            return "Prices_Error: No data found."

        try:
            df = data.xs('Close', level=0, axis=1).reset_index()
        except Exception:
            df = data['Close'].reset_index()

        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        return df.tail(10).to_string(index=False)
    except Exception as e:
        logger.error(f"Error fetching commodities: {e}")
        return f"Prices_Error: {e}"


WEATHER_TOOL_SCHEMA = {
    "name": "get_weather_forecast",
    "description": "Get 1â€“7 day livestock-relevant weather forecast for a region.",
    "parameters": {
        "type": "object",
        "properties": {
            "region": {
                "type": "string",
                "description": "Human-readable region name, e.g. 'United States', 'Canada', 'Brazil'.",
            },
            "days": {
                "type": "integer",
                "description": "Number of days to forecast (1â€“7).",
                "minimum": 1,
                "maximum": 7,
                "default": 3,
            },
        },
        "required": ["region"],
    },
}

def get_real_weather(region="United States", days=3):
    """HTTP tool called according to WEATHER_TOOL_SCHEMA."""
    days = max(1, min(7, int(days)))
    lat, lon, label = get_region_weather_coords(region)
    logger.info(f"Tool: fetching weather for {label}, {days} days")
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&daily=temperature_2m_max,precipitation_sum&timezone=auto"
        )
        res = requests.get(url).json()['daily']
        report = f"Forecast ({label}):\n"
        for i in range(days):
            c = res['temperature_2m_max'][i]
            f_val = c * 9/5 + 32
            rain = res['precipitation_sum'][i]
            report += f"- Day {i+1}: High {f_val:.1f}Â°F ({c:.1f}Â°C), Precip {rain:.1f}mm\n"
        return report
    except Exception as e:
        logger.error(f"Weather API error: {e}")
        return f"Weather_Error: {e}"

def call_weather_tool_from_schema(arguments: dict):
    """
    Example of 'OpenAPI-style' structured call.
    In a full MCP setup, the LLM would emit these arguments as JSON.
    """
    region = arguments.get("region", "United States")
    days = arguments.get("days", 3)
    return get_real_weather(region=region, days=days)

def get_google_news_rss(commodity_type="cattle", region="United States"):
    if commodity_type == "hog":
        base_query = "hog market"
    else:
        base_query = "cattle market"

    r = region.lower()
    q_region = ""
    if "canada" in r:
        q_region = " canada"
    elif "brazil" in r:
        q_region = " brazil"
    elif "united states" in r or "usa" in r or "america" in r:
        q_region = " usa"

    query = (base_query + q_region).strip()
    logger.info(f"Tool: fetching Google News RSS for '{query}'")

    try:
        rss_url = (
            "https://news.google.com/rss/search?"
            f"q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
        )
        feed = feedparser.parse(rss_url)
        if not feed.entries:
            return "News_Error: No news found."
        return "\n".join([f"- {e.title}" for e in feed.entries[:3]])
    except Exception as e:
        logger.error(f"News feed error: {e}")
        return f"News_Error: {e}"

def calculate_cost_model(price, corn_price, commodity_type="cattle"):
    """
    Livestock/Corn ratio:
    - Hogs: Hog/Corn ratio thresholds ~14, 19
    - Cattle: Steer/Corn thresholds ~20, 25
    """
    logger.info(f"Tool: running {commodity_type.title()}/Corn ratio model")
    try:
        corn_dollars_per_bushel = float(corn_price) / 100.0
        if corn_dollars_per_bushel <= 0:
            return "Margins_Error: Invalid corn price."

        ratio = float(price) / corn_dollars_per_bushel

        if commodity_type == "hog":
            if ratio < 14:
                comment = "Ratio is low â€“ margins compressed (risk of liquidation)."
            elif ratio < 19:
                comment = "Ratio is neutral â€“ margins are okay but not exciting."
            else:
                comment = "Ratio is >19 â€“ margins look attractive (supportive for expansion)."
            label = "Hog/Corn Ratio"
        else:
            if ratio < 20:
                comment = "Ratio is low â€“ margins compressed."
            elif ratio < 25:
                comment = "Ratio is moderate â€“ margins are okay."
            else:
                comment = "Ratio is high â€“ margins look very attractive."
            label = "Cattle/Corn (Steer/Corn) Ratio"

        return f"{label}: {ratio:.1f}. {comment}"
    except Exception as e:
        logger.error(f"Cost model error: {e}")
        return f"Margins_Error: {e}"


# 4. LONG-RUNNING BACKTEST TOOL + MEMORY BANK

BACKTEST_STATE_FILE = "backtest_state.json"
MEMORY_FILE = "memory_bank.json"

def load_json_file(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default

def save_json_file(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def run_backtest(job_id: str, commodity_type="cattle", period="6mo", chunk_size=5, resume=False):
    """
    Simulated long-running operation:
    - downloads historical prices
    - processes chunk_size rows at a time
    - saves progress to BACKTEST_STATE_FILE keyed by job_id
    """
    logger.info(f"Backtest: job_id={job_id}, commodity={commodity_type}, resume={resume}")

    state = load_json_file(BACKTEST_STATE_FILE, {})
    job_state = state.get(job_id, {"index": 0})

    main_ticker = "HE=F" if commodity_type == "hog" else "LE=F"
    tickers = [main_ticker, "ZC=F"]

    data = yf.download(tickers, period=period, progress=False, auto_adjust=False)
    if data.empty:
        return f"Backtest_Error: No data for {tickers}"

    try:
        df = data.xs("Close", level=0, axis=1).reset_index()
    except Exception:
        df = data["Close"].reset_index()

    start_idx = job_state.get("index", 0)
    end_idx = min(start_idx + chunk_size, len(df))

    if start_idx >= len(df):
        return f"Backtest job {job_id} already completed. Total steps: {len(df)}"

    logger.info(f"Backtest job {job_id}: processing rows {start_idx} to {end_idx-1}")
    time.sleep(0.3)  # simulate heavy work

    job_state["index"] = end_idx
    state[job_id] = job_state
    save_json_file(BACKTEST_STATE_FILE, state)

    done = end_idx >= len(df)
    return (
        f"Backtest job {job_id} processed rows {start_idx}â€“{end_idx-1} "
        f"of {len(df)}. done={done}"
    )

def append_memory_entry(timestamp: str, commodity: str, region: str, verdict: str, confidence: float):
    memory = load_json_file(MEMORY_FILE, [])
    memory.append(
        {
            "timestamp": timestamp,
            "commodity": commodity,
            "region": region,
            "verdict": verdict,
            "confidence": confidence,
        }
    )
    memory = memory[-100:]  # keep last 100
    save_json_file(MEMORY_FILE, memory)

def get_recent_memory(commodity: str, region: str, limit: int = 5):
    memory = load_json_file(MEMORY_FILE, [])
    filtered = [
        m for m in memory
        if m["commodity"] == commodity and m["region"] == region
    ]
    return filtered[-limit:]

def format_memory_for_session(commodity: str, region: str):
    mem = get_recent_memory(commodity, region, limit=5)
    if not mem:
        return "Memory: no prior runs for this commodity/region."
    lines = [f"Memory: last {len(mem)} runs for {commodity} in {region}:"]
    for m in mem:
        lines.append(
            f"- {m['timestamp']}: verdict={m['verdict']}, confidence={m['confidence']:.2f}"
        )
    return "\n".join(lines)


# 5. CHART / EXPORT TOOLS + STATS HELPERS

def generate_analytics_charts(data_string, filename="market_analytics.png"):
    logger.info(f"Tool: generating dual-axis chart {filename}")
    try:
        df = pd.read_csv(StringIO(data_string), sep=r"\s+")
        df['Date'] = pd.to_datetime(df['Date'])

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Livestock (Main Contract)')

        for col in ['LE=F', 'HE=F']:
            if col in df.columns:
                ax1.plot(df['Date'], df[col], linewidth=2, label=col)
                break

        ax2 = ax1.twinx()
        ax2.set_ylabel('Feed Corn (ZC=F)')
        if 'ZC=F' in df.columns:
            ax2.plot(df['Date'], df['ZC=F'], linestyle='--', linewidth=2, label='ZC=F')

        plt.title("Price Trend: Livestock vs Corn")
        fig.tight_layout()
        plt.savefig(filename)
        plt.close()
        return "Chart created."
    except Exception as e:
        logger.error(f"Chart error: {e}")
        return f"Chart_Error: {e}"

def export_to_excel(data_string, filename="livestock_prices.xlsx"):
    logger.info(f"Tool: exporting Excel {filename}")
    try:
        df = pd.read_csv(StringIO(data_string), sep=r"\s+")
        df.to_excel(filename, index=False)
        return "Excel created."
    except Exception as e:
        logger.error(f"Excel error: {e}")
        return f"Excel_Error: {e}"

def export_to_pdf(report_text, chart_path, filename="livestock_report.pdf"):
    logger.info(f"Tool: exporting PDF {filename}")
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Livestock Market Intelligence", ln=True, align='C')
        pdf.ln(5)

        if os.path.exists(chart_path):
            pdf.image(chart_path, x=10, y=30, w=190)
            pdf.ln(100)

        pdf.set_font("Arial", size=11)
        safe_text = report_text.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 7, safe_text)
        pdf.output(filename)
        return "PDF created."
    except Exception as e:
        logger.error(f"PDF error: {e}")
        return f"PDF_Error: {e}"

def extract_price_df_from_table(table_str):
    try:
        df = pd.read_csv(StringIO(table_str), sep=r"\s+")
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        return df
    except Exception:
        return None

def compute_price_stats(table_str, main_ticker="LE=F"):
    df = extract_price_df_from_table(table_str)
    if df is None:
        return {"error": "Could not parse price table."}

    stats = {}
    for col in [main_ticker, 'ZC=F']:
        if col not in df.columns:
            continue
        series = df[col]
        series = series.replace([float("inf"), float("-inf")], pd.NA).dropna()

        last = series.iloc[-1]
        first = series.iloc[0]
        pct_change = (last - first) / first * 100
        col_stats = {
            "first": round(float(first), 2),
            "last": round(float(last), 2),
            "pct_change": round(float(pct_change), 2),
            "max": round(float(series.max()), 2),
            "min": round(float(series.min()), 2),
            "std": round(float(series.std()), 4) if len(series) > 1 else 0.0,
        }
        if len(series) >= 5:
            col_stats["sma_5"] = round(float(series.tail(5).mean()), 2)
        stats[col] = col_stats

    return stats

def get_latest_prices_from_history(session, main_ticker="LE=F"):
    for msg in reversed(session.history):
        if msg.get("name") == "tool_output" and "Tool Output (Prices):" in msg["content"]:
            text = msg["content"]
            idx = text.find("Date")
            if idx == -1:
                return None, None, None
            table_str = text[idx:]
            df = extract_price_df_from_table(table_str)
            if df is None:
                return None, None, None
            if main_ticker not in df.columns or 'ZC=F' not in df.columns:
                return None, None, None
            last_row = df.iloc[-1]
            try:
                date_str = last_row['Date'].strftime('%Y-%m-%d')
                livestock_price = float(last_row[main_ticker])
                corn_price = float(last_row['ZC=F'])
                return date_str, livestock_price, corn_price
            except Exception:
                return None, None, None
    return None, None, None


# 6. AGENT DEFINITIONS (WITH MEMORY & MCP )
class CommoditiesAgent:
    def __init__(self):
        self.name = "commodities_agent"
        self.system_prompt = (
            "You are the CommoditiesAgent. "
            "Use ONLY the 'Tool Output (Prices)' and 'Tool Output (Stats)' messages "
            "to describe price ranges and trends. "
            "Identify whether the data is for Live Cattle (LE=F) or Lean Hogs (HE=F). "
            "Do not invent prices."
        )

    def execute_task(self, commodity_type="cattle"):
        data = get_commodities_data(commodity_type=commodity_type)
        return f"Tool Output (Prices):\n{data}"

class WeatherAgent:
    def __init__(self):
        self.name = "weather_agent"
        self.system_prompt = (
            "You are the WeatherAgent. "
            "Use ONLY the latest 'Tool Output (Forecast)' message. "
            "Explain how the weather affects livestock performance, stress, and feed use. "
            "Temperatures are given in Â°F (with Â°C in parentheses)."
        )

    def execute_task(self, region="United States", days=3):
        data = call_weather_tool_from_schema({"region": region, "days": days})
        return f"Tool Output (Forecast):\n{data}"

class NewsAgent:
    def __init__(self):
        self.name = "news_agent"
        self.system_prompt = (
            "You are the NewsAgent. "
            "Use ONLY the latest 'Tool Output (News)' message. "
            "Summarize supply-side and demand-side sentiment, particularly exports, "
            "using emojis for sentiment (ğŸ˜ƒ bullish, ğŸ˜� mixed, ğŸ˜Ÿ bearish). "
            "If the news output contains 'News_Error', say no reliable recent news is available."
        )

    def execute_task(self, commodity_type="cattle", region="United States"):
        data = get_google_news_rss(commodity_type=commodity_type, region=region)
        return f"Tool Output (News):\n{data}"

class CostModelAgent:
    def __init__(self):
        self.name = "cost_model"
        self.system_prompt = (
            "You are the CostModelAgent. "
            "Use ONLY the latest 'Tool Output (Margins)' message. "
            "Explain whether producer margins look compressed, neutral, or very attractive. "
            "Do NOT invent breakeven math."
        )

    def execute_task(self, livestock_price=185, corn_price=430, commodity_type="cattle"):
        data = calculate_cost_model(livestock_price, corn_price, commodity_type=commodity_type)
        return f"Tool Output (Margins):\n{data}"

class BacktestAgent:
    def __init__(self):
        self.name = "backtest_agent"
        self.system_prompt = (
            "You are the BacktestAgent. "
            "You simply report the status messages from 'Tool Output (Backtest)' "
            "and do not add any extra interpretation."
        )

    def execute_task(self, job_id: str, commodity_type="cattle", resume=False):
        msg = run_backtest(job_id=job_id, commodity_type=commodity_type, resume=resume)
        return f"Tool Output (Backtest): {msg}"

class AnalystAgent:
    def __init__(self, prefs: dict):
        self.name = "analyst"
        self.prefs = prefs
        self.system_prompt = self._build_prompt()

    def _build_prompt(self) -> str:
        mode = self.prefs["mode"]
        perspective = self.prefs["perspective"]
        focus = self.prefs["focus"]
        commodity_label = self.prefs["commodity_label"]
        main_ticker = self.prefs["main_ticker"]
        region = self.prefs["region"]

        return f"""You are the Senior Strategist. Write a {commodity_label} ({main_ticker}) market commentary
for {region} producers and market participants.

USER_MODE = {mode}.
- If USER_MODE is 'brief', keep the answer under 250 words.
- If USER_MODE is 'detailed', you may use 400â€“800 words.

USER_PERSPECTIVE = {perspective}.
- If 'producer', emphasize producer margins and hedging decisions.
- If 'trader', emphasize price levels and risk/reward.
- Otherwise, keep a balanced general perspective.

USER_FOCUS = {focus}.
- If 'weather', put extra weight on weather impacts.
- If 'exports', put extra weight on export/demand headlines.

DATA RULES:
- For trend language (uptrend/downtrend/flat/volatile), use ONLY the latest
  'Tool Output (Stats)' JSON.
- Use cost model messages for Livestock/Corn ratio.
- Ignore any *_Error messages.
- Use 'Tool Output (Memory)' to compare today's outlook with previous runs.

REPORT STRUCTURE:
1. **TL;DR** with 'Overall: BULLISH/BEARISH/NEUTRAL' and 1â€“2 main reasons.
2. **Price Action** ğŸ“ˆ/ğŸ“‰ with a Markdown table (main contract + Corn).
3. **Input Costs & Livestock/Corn Ratio**.
4. **Weather Impact**.
5. **Sentiment & News Flow** ğŸ˜ƒ/ğŸ˜�/ğŸ˜Ÿ with bullet points.
6. **Scorecard & Final Verdict** with scores -2..+2 and explanation.
7. **Producer Playbook: Watch Levels & Scenarios** with support/resistance and bull/bear cases.

At the end:
- Output a line: SUMMARY_JSON: {{"verdict":"BULLISH","confidence":0.9}}
- Then a line: [SCORE_VERIFIED]: 95%
"""

    def execute_task(self):
        return None  # pure LLM

class SupervisorAgent:
    def __init__(self):
        self.name = "supervisor"
        self.system_prompt = """You are a Robotic Workflow Manager.

Phase 1 â€“ Orchestration:
- Call these agents (in any order, but at least once):
  1) Commodities_Agent
  2) Weather_Agent
  3) News_Agent
  4) Cost_Model
  5) Analyst
- Optionally call Backtest_Agent if the user mentions 'backtest'.

Phase 2 â€“ Review & Challenge:
- After the Analyst has produced a forecast with SUMMARY_JSON, review it against:
  - Livestock/Corn ratio (Cost_Model),
  - Price Stats,
  - News sentiment.
- If verdict is wildly inconsistent with tools (e.g. strongly BEARISH with very attractive margins),
  request 1 revision:
  "Analyst, revise your final prediction for consistency with the tools."
- After at most 1 revision, output "[PROCESS_COMPLETE]".

Your responses MUST be exactly one of:
- "Commodities_Agent, check prices."
- "Weather_Agent, check forecast."
- "News_Agent, check headlines."
- "Cost_Model, calculate margins."
- "Backtest_Agent, run backtest."
- "Analyst, provide final prediction."
- "Analyst, revise your final prediction for consistency with the tools."
- "[PROCESS_COMPLETE]"
"""

    def determine_next_step(self, reply, agents_called):
        r = reply.lower()

        if "[process_complete]" in r:
            return "COMPLETE"

        mapping = {
            "commodities_agent": "commodities_agent",
            "weather_agent": "weather_agent",
            "news_agent": "news_agent",
            "cost_model": "cost_model",
            "backtest_agent": "backtest_agent",
        }

        for key, name in mapping.items():
            if key in r and name not in agents_called:
                return name

        if "analyst, provide final prediction" in r:
            return "analyst"
        if "analyst, revise your final prediction" in r:
            return "analyst"

        return None


# 7. MASTER EXECUTION SYSTEM (PARALLEL TOOLS, MEMORY, METRICS)
LAST_ANALYST_REPORT = None
LAST_SUMMARY_JSON = None
LAST_ARTIFACTS = None
LAST_SESSION_HISTORY = None  # new: store full conversation

class AgentSession:
    def __init__(self):
        self.history = []

    def add_message(self, role, content, name=None):
        self.history.append({"role": role, "content": content, "name": name})
        logger.debug(f"Session add: role={role}, name={name}, len={len(self.history)}")

def generate_llm_reply(messages, system_instruction):
    global model, retry_config
    if not model:
        return "Error: Model not initialized."

    if 'retry_config' not in globals() or retry_config is None:
        retry_config = retry.Retry(
            initial=1.0,
            maximum=10.0,
            multiplier=2.0,
            deadline=90.0,
        )

    full_prompt = f"System Instruction: {system_instruction}\n\n"
    for msg in messages:
        role = msg['role'].title()
        content = msg['content']
        name = msg.get('name', '')
        if name:
            full_prompt += f"{role} ({name}): {content}\n"
        else:
            full_prompt += f"{role}: {content}\n"
    full_prompt += "\nAssistant:"

    try:
        resp = model.generate_content(
            full_prompt,
            request_options={'retry': retry_config},
        )
        return resp.text
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return f"LLM_Error: {e}"

def interpret_user_query(query: str) -> dict:
    q = query.lower()

    mode = "detailed"
    if any(x in q for x in ["brief", "short", "summary"]):
        mode = "brief"

    perspective = "general"
    if "producer" in q or "feedlot" in q or "farmer" in q:
        perspective = "producer"
    elif "trader" in q or "speculator" in q:
        perspective = "trader"

    focus = "general"
    if "weather" in q:
        focus = "weather"
    elif "export" in q or "exports" in q:
        focus = "exports"

    if any(x in q for x in ["hog", "hogs", "lean hog", "pork"]):
        commodity_type = "hog"
        commodity_label = "Lean Hogs"
        main_ticker = "HE=F"
    else:
        commodity_type = "cattle"
        commodity_label = "Live Cattle"
        main_ticker = "LE=F"

    if "canada" in q:
        region = "Canada"
    elif "brazil" in q:
        region = "Brazil"
    elif "europe" in q or "eu " in q:
        region = "Europe"
    elif "united states" in q or " us " in q or "america" in q:
        region = "United States"
    else:
        region = "Global (defaulting to North American futures as reference)"

    wants_backtest = "backtest" in q

    return {
        "mode": mode,
        "perspective": perspective,
        "focus": focus,
        "commodity_type": commodity_type,
        "commodity_label": commodity_label,
        "main_ticker": main_ticker,
        "region": region,
        "wants_backtest": wants_backtest,
    }

def run_master_system(user_query: str):
    global LAST_ANALYST_REPORT, LAST_SUMMARY_JSON, LAST_ARTIFACTS, LAST_SESSION_HISTORY, METRICS

    if not model:
        logger.error("Model not initialized; aborting run.")
        return

    prefs = interpret_user_query(user_query)
    commodity_type = prefs["commodity_type"]
    main_ticker = prefs["main_ticker"]
    region = prefs["region"]

    logger.info(f"Run start: commodity={commodity_type}, region={region}")

    supervisor = SupervisorAgent()
    agent_registry = {
        "commodities_agent": CommoditiesAgent(),
        "weather_agent": WeatherAgent(),
        "news_agent": NewsAgent(),
        "cost_model": CostModelAgent(),
        "backtest_agent": BacktestAgent(),
        "analyst": AnalystAgent(prefs),
    }

    session = AgentSession()
    session.add_message("user", user_query)

    # Inject memory bank
    memory_text = format_memory_for_session(commodity_type, region)
    session.add_message("user", memory_text, name="tool_output")

    current_agent_name = "supervisor"
    agents_called = set()
    agent_times = {}
    parallel_env_done = False

    for _ in range(40):
        if current_agent_name == "supervisor":
            t0 = time.time()
            reply = generate_llm_reply(session.history, supervisor.system_prompt)
            agent_times.setdefault("supervisor", []).append(time.time() - t0)

            session.add_message("assistant", reply, name="supervisor")
            log_a2a("supervisor", "all", "command", reply)

            next_step = supervisor.determine_next_step(reply, agents_called)
            if next_step == "COMPLETE":
                break
            elif next_step:
                current_agent_name = next_step
            else:
                continue

        else:
            # Parallel Weather + News tools
            if current_agent_name in ("weather_agent", "news_agent") and not parallel_env_done:
                parallel_env_done = True
                logger.info("Running Weather + News tools in parallel")

                def weather_tool():
                    agent = agent_registry["weather_agent"]
                    return "weather_agent", agent.execute_task(region=region)

                def news_tool():
                    agent = agent_registry["news_agent"]
                    return "news_agent", agent.execute_task(
                        commodity_type=commodity_type,
                        region=region,
                    )

                with ThreadPoolExecutor(max_workers=2) as ex:
                    futures = [ex.submit(weather_tool), ex.submit(news_tool)]
                    for fut in futures:
                        name, tool_output = fut.result()
                        session.add_message("user", tool_output, name="tool_output")
                        log_a2a("supervisor", name, "tool_result", tool_output)

                for name in ("weather_agent", "news_agent"):
                    agent_obj = agent_registry[name]
                    t0 = time.time()
                    reply = generate_llm_reply(session.history, agent_obj.system_prompt)
                    agent_times.setdefault(name, []).append(time.time() - t0)

                    session.add_message("assistant", reply, name=name)
                    log_a2a(name, "supervisor", "analysis", reply)
                    agents_called.add(name)

                current_agent_name = "supervisor"
                continue

            agent_obj = agent_registry.get(current_agent_name)
            if not agent_obj:
                current_agent_name = "supervisor"
                continue
          
            tool_output = None
            if hasattr(agent_obj, "execute_task"):
                if current_agent_name == "commodities_agent":
                    tool_output = agent_obj.execute_task(commodity_type=commodity_type)

                elif current_agent_name == "cost_model":
                    _, lp, cp = get_latest_prices_from_history(session, main_ticker=main_ticker)
                    if lp is not None and cp is not None:
                        tool_output = agent_obj.execute_task(
                            livestock_price=lp,
                            corn_price=cp,
                            commodity_type=commodity_type,
                        )
                    else:
                        tool_output = agent_obj.execute_task(commodity_type=commodity_type)

                elif current_agent_name == "backtest_agent" and prefs["wants_backtest"]:
                    tool_output = agent_obj.execute_task(
                        job_id="demo_job",
                        commodity_type=commodity_type,
                        resume=True,
                    )

                elif current_agent_name not in ("weather_agent", "news_agent"):
                    tool_output = agent_obj.execute_task()

                if tool_output:
                    session.add_message("user", tool_output, name="tool_output")
                    log_a2a("supervisor", current_agent_name, "tool_result", tool_output)

                    if current_agent_name == "commodities_agent" and "Tool Output (Prices):" in tool_output:
                        idx = tool_output.find("Date")
                        if idx != -1:
                            table_str = tool_output[idx:]
                            stats = compute_price_stats(table_str, main_ticker=main_ticker)
                            stats_json = json.dumps(stats)
                            stats_msg = (
                                f"Tool Output (Stats): {stats_json}\n"
                                "Use this JSON as the authoritative source for price trends."
                            )
                            session.add_message("user", stats_msg, name="tool_output")


            t0 = time.time()
            reply = generate_llm_reply(session.history, agent_obj.system_prompt)
            agent_times.setdefault(current_agent_name, []).append(time.time() - t0)
            session.add_message("assistant", reply, name=current_agent_name)
            log_a2a(current_agent_name, "supervisor", "analysis", reply)

            agents_called.add(current_agent_name)
            current_agent_name = "supervisor"

    # Ensure Analyst report exists
    has_summary = any(
        msg.get("name") == "analyst" and "SUMMARY_JSON:" in msg["content"]
        for msg in session.history
    )
    if not has_summary:
        logger.warning("No Analyst SUMMARY_JSON found; forcing final Analyst pass.")
        analyst_obj = agent_registry["analyst"]
        t0 = time.time()
        reply = generate_llm_reply(session.history, analyst_obj.system_prompt)
        agent_times.setdefault("analyst", []).append(time.time() - t0)
        session.add_message("assistant", reply, name="analyst")

 
    logger.info("Finalizing report and artifacts.")
    csv_data = ""
    final_report = "No report."
    summary_json = None

    for msg in reversed(session.history):
        if msg.get("name") == "analyst" and "SUMMARY_JSON:" in msg["content"]:
            final_report = msg["content"]
            for line in final_report.splitlines():
                if line.strip().startswith("SUMMARY_JSON:"):
                    js = line.split("SUMMARY_JSON:", 1)[1].strip()
                    try:
                        summary_json = json.loads(js)
                    except Exception:
                        summary_json = None
            break

    if final_report == "No report.":
        for msg in reversed(session.history):
            if msg.get("name") == "analyst":
                final_report = msg["content"]
                break

    for msg in reversed(session.history):
        if msg.get("name") == "tool_output" and "Date" in msg["content"]:
            idx = msg["content"].find("Date")
            csv_data = msg["content"][idx:]
            break

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    chart_file = f"market_analytics_{ts}.png"
    excel_file = f"livestock_prices_{ts}.xlsx"
    pdf_file = f"livestock_report_{ts}.pdf"

    if csv_data:
        generate_analytics_charts(csv_data, filename=chart_file)
        export_to_excel(csv_data, filename=excel_file)
    export_to_pdf(final_report, chart_file, filename=pdf_file)

    LAST_ANALYST_REPORT = final_report
    LAST_SUMMARY_JSON = summary_json
    LAST_ARTIFACTS = {
        "chart": chart_file,
        "excel": excel_file,
        "pdf": pdf_file,
    }


    METRICS["run_count"] += 1
    verdict = summary_json.get("verdict") if summary_json else None
    if verdict:
        METRICS["verdict_counts"][verdict] = METRICS["verdict_counts"].get(verdict, 0) + 1

    for agent, times in agent_times.items():
        avg = sum(times) / len(times)
        old_avg = METRICS["agent_durations"].get(agent)
        if old_avg is None:
            METRICS["agent_durations"][agent] = avg
        else:
            METRICS["agent_durations"][agent] = (old_avg + avg) / 2

   
    if summary_json:
        ts_iso = datetime.now().isoformat(timespec="seconds")
        append_memory_entry(
            ts_iso,
            commodity=prefs["commodity_type"],
            region=prefs["region"],
            verdict=summary_json["verdict"],
            confidence=float(summary_json["confidence"]),
        )


    LAST_SESSION_HISTORY = session.history

    logger.info(f"Run complete. Verdict={verdict}, metrics={METRICS}")

def display_last_report():
    if LAST_ANALYST_REPORT is None:
        print("No analyst report stored yet.")
        return

    print("\n================ FINAL ANALYST REPORT ================\n")
    print(LAST_ANALYST_REPORT)
    print("\n=====================================================\n")

    if LAST_SUMMARY_JSON:
        print("Parsed SUMMARY_JSON:", LAST_SUMMARY_JSON)

    if LAST_ARTIFACTS:
        print("\nArtifacts created:")
        for k, v in LAST_ARTIFACTS.items():
            print(f" - {k}: {LAST_ARTIFACTS[k]}")

    print("\nMetrics:", METRICS)


def display_agent_conversation(max_chars_per_msg: int = 400):
    """
    Pretty-print the full conversation between user, supervisor, tools, and agents
    from the last run of run_master_system().
    """
    if LAST_SESSION_HISTORY is None:
        print("No session history yet. Run run_master_system(...) first.")
        return

    print("=============== AGENT CONVERSATION (LAST RUN) ===============\n")
    for i, msg in enumerate(LAST_SESSION_HISTORY, 1):
        role = msg.get("role", "").upper()
        name = msg.get("name")
        label = role if not name else f"{role} ({name})"

        content = msg.get("content", "")
        if "Tool Output (Stats):" in content and len(content) > 120:
            content = "Tool Output (Stats): {...} (truncated for display)"
        elif len(content) > max_chars_per_msg:
            content = content[:max_chars_per_msg] + "..."

        print(f"{i:02d}. {label}")
        print(content)
        print("-" * 70)

def display_a2a_log(max_chars_per_msg: int = 120):
    """
    Show the Agent-to-Agent (A2A) messages captured in A2A_LOG.
    """
    if not A2A_LOG:
        print("No A2A messages logged yet. Run run_master_system(...) first.")
        return

    print("=============== A2A MESSAGE LOG (ALL RUNS) ===============\n")
    for i, m in enumerate(A2A_LOG, 1):
        snippet = m.content
        if len(snippet) > max_chars_per_msg:
            snippet = snippet[:max_chars_per_msg] + "..."
        print(f"{i:02d}. {m.sender} -> {m.receiver} [{m.msg_type}]")
        print(snippet)
        print("-" * 70)



# 8. AGENT EVALUATION HARNESS

TEST_CASES = [
    {
        "name": "Cattle US detailed producer exports",
        "prompt": "Give me a detailed market outlook for Live Cattle in the United States "
                  "from a producer perspective, with focus on exports.",
        "expected_verdict": {"BULLISH", "BEARISH", "NEUTRAL"},  # smoke test
    },
    {
        "name": "Hogs Canada brief producer exports",
        "prompt": "Give me a brief market outlook for Lean Hogs in Canada from a producer "
                  "perspective, with focus on exports.",
        "expected_verdict": {"NEUTRAL", "BULLISH", "BEARISH"},
    },
]

def run_evaluation():
    results = []
    for case in TEST_CASES:
        logger.info(f"Eval: {case['name']}")
        run_master_system(case["prompt"])
        verdict = (LAST_SUMMARY_JSON or {}).get("verdict", "UNKNOWN")
        passed = verdict in case["expected_verdict"]
        results.append((case["name"], verdict, passed))
        print(f"[{case['name']}] verdict={verdict} -> {'PASS' if passed else 'FAIL'}")

    passed_count = sum(1 for _, _, p in results if p)
    print(f"\nEvaluation summary: {passed_count}/{len(results)} tests passed.")


# 9. SIMPLE GRADIO UI (NOTEBOOK "DEPLOYMENT")

import gradio as gr

def gradio_run(prompt: str):
    run_master_system(prompt)
    if LAST_ANALYST_REPORT:
        lines = LAST_ANALYST_REPORT.splitlines()
        return "\n".join(lines[:40])
    else:
        return "No report produced."

demo = gr.Interface(
    fn=gradio_run,
    inputs=gr.Textbox(lines=3, label="Enter market question"),
    outputs=gr.Textbox(lines=20, label="Analyst Report (TL;DR + sections)"),
    title="Livestock Multi-Agent Market Intelligence",
    description=(
        "Multi-agent system (Supervisor, Commodities, Weather, News, Cost Model, Analyst) "
        "with tools, memory bank, and structured outputs."
    ),
)


# 10. EXAMPLE TEST RUNS

if __name__ == "__main__":
    # Test 1: Cattle â€“ US
    run_master_system(
        "Give me a detailed market outlook for Live Cattle in the United States "
        "from a producer perspective, with focus on exports."
    )
    display_last_report()


if __name__ == "__main__":
    run_master_system(
        "Give me a brief market outlook for Lean Hogs in Canada from a producer perspective, "
        "with focus on exports."
    )
    display_last_report()

