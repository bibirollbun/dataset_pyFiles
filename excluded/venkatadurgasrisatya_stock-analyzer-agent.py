# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here are several helpful packages to load

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


!pip install yfinance pandas numpy matplotlib


import uuid
import time
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
from kaggle_secrets import UserSecretsClient
import google.generativeai as genai

# Load the secret from Kaggle
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Kaggle secret loaded successfully.")
except Exception as e:
    raise RuntimeError(
        f"â�Œ Gemini API key not found. Please add GOOGLE_API_KEY in Kaggle Secrets. Details: {e}"
    )

# âœ… THIS IS THE CRITICAL LINE YOU WERE MISSING:
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

print("ğŸ¤– Gemini AI is now properly configured.")



# =========================================================
# 0. Observability: logging + basic metrics
# =========================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("stock_multi_agent")

METRICS: Dict[str, List[float]] = {}

def record_metric(name: str, value: float):
    METRICS.setdefault(name, []).append(value)

print("âœ… Observability Setup is completed")


# =========================================================
# 1. A2A Protocol Message
# =========================================================
@dataclass
class A2AMessage:
    session_id: str
    task_id: str
    agent_id: str
    payload: Dict[str, Any]
    trace_id: str
    parent_id: Optional[str] = None



# =========================================================
# 2. Sessions & Memory (simple in-memory Memory Bank)
# =========================================================

MEMORY_BANK = []

def memory_store(event: Dict[str, Any]):
    """
    Stores events into a simple long-term memory structure.
    Replace with DB / Redis in real deployments.
    """
    MEMORY_BANK.append(event)
    
class InMemorySessionService:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(self) -> str:
        sid = f"SES-{uuid.uuid4()}"
        self.sessions[sid] = {"history": [], "checkpoints": {}}
        return sid

    def append_history(self, session_id: str, event: Dict[str, Any]):
        self.sessions.setdefault(session_id, {}).setdefault("history", []).append(event)

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        return self.sessions.get(session_id, {}).get("history", [])

    def save_checkpoint(self, session_id: str, key: str, value: Any):
        self.sessions.setdefault(session_id, {}).setdefault("checkpoints", {})[key] = value

    def get_checkpoint(self, session_id: str, key: str, default=None):
        return self.sessions.get(session_id, {}).get("checkpoints", {}).get(key, default)

SESSION_SERVICE = InMemorySessionService()

print("âœ… Sessions & Memory (simple in-memory Memory Bank) Created")


# =========================================================
# 3. Context engineering (context compaction)
# =========================================================
def compact_context(history: List[Dict[str, Any]], max_events: int = 10) -> str:
    if not history:
        return "No previous context."
    recent = history[-max_events:]
    lines = []
    for h in recent:
        lines.append(f"[{h['agent_id']}] {h['event']}")
    return "\n".join(lines)


# =========================================================
# 4. Tools (MCP-style): MarketData, CodeExecution
# =========================================================
class MarketDataTool:
    """Custom tool that wraps yfinance for daily OHLC / close data."""
    def fetch_daily(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        t0 = time.time()
        df = yf.download(
            symbol,
            start=start,
            end=end,
            interval="1d",
            progress=False,
            auto_adjust=False,
        )
        if df.empty:
            raise ValueError(f"No data for {symbol}")

        if "Close" in df.columns:
            df = df.rename(columns={"Close": "close"})
        elif "Adj Close" in df.columns:
            df = df.rename(columns={"Adj Close": "close"})

        df = df[["close"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.sort_index()
        df.index.name = "date"
        record_metric("tool_market_latency", time.time() - t0)
        return df

MARKET_TOOL = MarketDataTool()

class CodeExecutionTool:
    """Toy built-in tool to demonstrate tool usage."""
    def run(self, code: str) -> str:
        loc = {}
        try:
            exec(code, {}, loc)
            return f"OK, defined: {list(loc.keys())}"
        except Exception as e:
            return f"ERROR: {e}"

CODE_TOOL = CodeExecutionTool()

print("âœ… Required Tools are initialized")


# =========================================================
# 5. Base Agent
# =========================================================
class Agent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    def handle(self, msg: A2AMessage) -> Dict[str, Any]:
        raise NotImplementedError

    def _log_event(self, msg: A2AMessage, event: str, payload: Dict[str, Any]):
        logger.info(f"[{self.agent_id}] {event}")
        SESSION_SERVICE.append_history(msg.session_id, {
            "ts": datetime.utcnow().isoformat(),
            "agent_id": self.agent_id,
            "event": event,
        })
        memory_store({
            "session_id": msg.session_id,
            "task_id": msg.task_id,
            "agent_id": self.agent_id,
            "payload": payload,
            "ts": datetime.utcnow().isoformat(),
        })


# =========================================================
# 6. Concrete Agents
# =========================================================

# 6.1 Data Agent (tool-based)
class DataAgent(Agent):
    def __init__(self):
        super().__init__("data_agent")

    def handle(self, msg: A2AMessage) -> Dict[str, Any]:
        symbol = msg.payload["symbol"]
        start = msg.payload["start"]
        end = msg.payload["end"]

        df = MARKET_TOOL.fetch_daily(symbol, start, end)
        table = df.reset_index()
        # index name is 'date', so we already get 'date' column
        table["date"] = table["date"].astype(str)
        out = table.to_dict(orient="list")

        self._log_event(msg, f"Fetched data for {symbol}", {"symbol": symbol})
        return {"symbol": symbol, "data": out}

# 6.2 Indicator Agent (sequential after data)
class IndicatorAgent(Agent):
    def __init__(self):
        super().__init__("indicator_agent")

    @staticmethod
    def sma(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window).mean()

    @staticmethod
    def rsi(series: pd.Series, window: int = 14) -> pd.Series:
        series = series.astype(float)
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ma_up = up.rolling(window).mean()
        ma_down = down.rolling(window).mean()
        rs = ma_up / (ma_down + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        rsi_1d = np.ravel(np.asarray(rsi))
        return pd.Series(rsi_1d, index=series.index)

    def handle(self, msg: A2AMessage) -> Dict[str, Any]:
        symbol = msg.payload["symbol"]
        data = msg.payload["data"]
        df = pd.DataFrame(data).set_index("date")
        df.index = pd.to_datetime(df.index)

        df["sma20"] = self.sma(df["close"], 20)
        df["sma50"] = self.sma(df["close"], 50)
        df["sma200"] = self.sma(df["close"], 200)
        df["rsi14"] = self.rsi(df["close"], 14)

        self._log_event(msg, f"Computed indicators for {symbol}", {"symbol": symbol})
        return {"symbol": symbol, "data_with_indicators": df.reset_index().to_dict(orient="list")}

# 6.3 LLM-powered Summary Agent (single LLM agent)
class SummaryAgent(Agent):
    def __init__(self, model_name="gemini-2.5-flash"):
        super().__init__("summary_llm_agent")
        self.model = genai.GenerativeModel(model_name)

    def _to_scalar(self, x):
        import numpy as np
        import pandas as pd
        if isinstance(x, pd.Series):
            return float(x.iloc[-1]) if len(x) else np.nan
        try:
            return float(x)
        except Exception:
            arr = np.asarray(x).ravel()
            return float(arr[-1]) if arr.size else np.nan

    def handle(self, msg: A2AMessage) -> Dict[str, Any]:
        symbol = msg.payload["symbol"]
        data = msg.payload["data_with_indicators"]

        df = pd.DataFrame(data).set_index("date")
        df.index = pd.to_datetime(df.index)
        latest = df.iloc[-1]

        close = self._to_scalar(latest.get("close"))
        sma20 = self._to_scalar(latest.get("sma20"))
        sma50 = self._to_scalar(latest.get("sma50"))
        sma200 = self._to_scalar(latest.get("sma200"))
        rsi = self._to_scalar(latest.get("rsi14"))

        history = SESSION_SERVICE.get_history(msg.session_id)
        context = compact_context(history)

        prompt = f"""
You are an AI Stock Analysis Agent.

Stock: {symbol}
Last Close: {close:.2f}
SMA20: {sma20:.2f}
SMA50: {sma50:.2f}
SMA200: {sma200:.2f}
RSI14: {rsi:.2f}

Session Context:
{context}

TASK:
Explain the current stock condition in simple language.
Do NOT give buy/sell advice.
Mention trend, momentum, and risk level.
Keep it under 4 sentences.
"""

        try:
            response = self.model.generate_content(prompt)
            summary = response.text.strip()
        except Exception as e:
            summary = f"Gemini failed to respond: {e}"

        self._log_event(msg, f"Generated Gemini summary for {symbol}", {"symbol": symbol})
        return {"symbol": symbol, "summary": summary}

    
    
# 6.4 Simple Evaluation Agent
class EvaluationAgent(Agent):
    """Agent evaluation: basic check on data sufficiency/indicator validity."""
    def __init__(self):
        super().__init__("evaluation_agent")

    def handle(self, msg: A2AMessage) -> Dict[str, Any]:
        symbol = msg.payload["symbol"]
        data = msg.payload["data_with_indicators"]
        df = pd.DataFrame(data)

        # Simple quality metric: number of non-NaN rows for SMA200 + RSI
        valid_rows = df["sma200"].notna() & df["rsi14"].notna()
        coverage = valid_rows.sum() / len(df) if len(df) > 0 else 0.0

        score = float(coverage)  # 0..1
        self._log_event(msg, f"Evaluated analysis for {symbol}", {"score": score})

        return {"symbol": symbol, "analysis_quality": score}



# =========================================================
# 7. Orchestrator (sequential pipeline + pause/resume hooks)
# =========================================================
class StockAnalysisOrchestrator:
    def __init__(self):
        self.trace_id = str(uuid.uuid4())
        self.data_agent = DataAgent()
        self.indicator_agent = IndicatorAgent()
        self.summary_agent = SummaryAgent()
        self.eval_agent = EvaluationAgent()

    def run_once(
        self,
        symbol: str,
        days_back: int = 180,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Sequential pipeline: Data â†’ Indicators â†’ Summary â†’ Evaluation."""
        session_id = session_id or SESSION_SERVICE.create_session()
        task_id = f"TASK-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"

        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=days_back)

        # 1) Data
        msg = A2AMessage(
            session_id=session_id,
            task_id=task_id,
            agent_id="client",
            payload={"symbol": symbol, "start": start_date.isoformat(), "end": end_date.isoformat()},
            trace_id=self.trace_id,
        )
        data_out = self.data_agent.handle(msg)

        # 2) Indicators
        ind_msg = A2AMessage(
            session_id=session_id,
            task_id=task_id,
            agent_id=self.data_agent.agent_id,
            payload=data_out,
            trace_id=self.trace_id,
        )
        ind_out = self.indicator_agent.handle(ind_msg)

        # 3) Summary (LLM agent)
        sum_msg = A2AMessage(
            session_id=session_id,
            task_id=task_id,
            agent_id=self.indicator_agent.agent_id,
            payload=ind_out,
            trace_id=self.trace_id,
        )
        summary_out = self.summary_agent.handle(sum_msg)

        # 4) Evaluation agent
        eval_msg = A2AMessage(
            session_id=session_id,
            task_id=task_id,
            agent_id=self.summary_agent.agent_id,
            payload=ind_out,
            trace_id=self.trace_id,
        )
        eval_out = self.eval_agent.handle(eval_msg)

        # "Long-running" / pause-resume hook: store checkpoint
        SESSION_SERVICE.save_checkpoint(session_id, "last_analysis", {
            "symbol": symbol,
            "task_id": task_id,
            "end_date": end_date.isoformat(),
        })

        result = {
            "session_id": session_id,
            "task_id": task_id,
            "symbol": symbol,
            "data_with_indicators": ind_out["data_with_indicators"],
            "summary": summary_out["summary"],
            "analysis_quality": eval_out["analysis_quality"],
        }
        return result


# =========================================================
# 8. Simple top-level "Simple Stock Analysis Agent" API
# =========================================================
class SimpleStockAnalysisAgent:
    """
    Deployable facade: hides multi-agent complexity behind a simple .analyze(...) call.
    """
    def __init__(self):
        self.orchestrator = StockAnalysisOrchestrator()

    def analyze(self, symbol: str, days_back: int = 180, plot: bool = True):
        res = self.orchestrator.run_once(symbol=symbol, days_back=days_back)
        df = pd.DataFrame(res["data_with_indicators"])
        df = df.set_index("date")
        df.index = pd.to_datetime(df.index)

        print("=== SUMMARY ===")
        print(res["summary"])
        print(f"\nAnalysis quality score: {res['analysis_quality']:.2f}")
        print("\n(Metrics & memory are kept in METRICS and MEMORY_BANK variables.)")

        if plot:
            plt.figure(figsize=(12, 5))
            plt.plot(df.index, df["close"], label="Close")
            if "sma20" in df: plt.plot(df.index, df["sma20"], label="SMA20")
            if "sma50" in df: plt.plot(df.index, df["sma50"], label="SMA50")
            if "sma200" in df: plt.plot(df.index, df["sma200"], label="SMA200")
            plt.title(f"{symbol} - Price with SMAs")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            if "rsi14" in df:
                plt.figure(figsize=(12, 3))
                plt.plot(df.index, df["rsi14"], label="RSI14")
                plt.axhline(70, linestyle="--")
                plt.axhline(30, linestyle="--")
                plt.title(f"{symbol} - RSI14")
                plt.xlabel("Date")
                plt.ylabel("RSI")
                plt.grid(True)
                plt.tight_layout()
                plt.show()

        return res


agent = SimpleStockAnalysisAgent()

# US example
agent.analyze("AAPL", days_back=180)

# NSE example (daily)
# agent.analyze("RELIANCE.NS", days_back=365)



agent.analyze("RELIANCE.NS", days_back=365)

