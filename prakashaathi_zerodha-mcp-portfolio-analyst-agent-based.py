# Install dependencies

# Standard imports
import os
import logging
import asyncio
import json
from datetime import datetime
from kaggle_secrets import UserSecretsClient

# Data and scraping
import pandas as pd
import requests
from bs4 import BeautifulSoup

# ADK / LLM imports
from google.genai import types
from google.adk.agents import LlmAgent, Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner, InMemoryRunner
from google.adk.sessions import InMemorySessionService

# ADK tools
from google.adk.tools.google_search_tool import google_search
from google.adk.plugins.logging_plugin import LoggingPlugin

# MCP toolset imports
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

# Set up Python logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("capstone")
logger.setLevel(logging.INFO)

print("Imports complete.")



# On Kaggle: Add a secret named GOOGLE_API_KEY in Add-ons -> Secrets
GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    logger.warning("⚠️ GOOGLE_API_KEY not set. LLM calls will fail until you add the secret.")
else:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    logger.info("✅ GOOGLE_API_KEY configured.")

# Zerodha MCP toggle
MCP_AVAILABLE = True  # Set to True if you want real Zerodha login

if MCP_AVAILABLE:
    logger.info(
        " MCP_AVAILABLE=True -> Using **REAL Zerodha MCP login**.\n"
        "   -> You will be prompted to authenticate via Kite.\n"
        "   -> Portfolio holdings & LTP will be fetched from your real account."
    )
else:
    logger.info(
        "  MCP_AVAILABLE=False -> Running in **Demo Mode (Mock Zerodha Server)**.\n"
        "   -> No login required.\n"
        "   -> Holdings, LTP, and tool responses are simulated so notebook runs end-to-end."
    )


from google.adk.plugins.base_plugin import BasePlugin
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest

class CountInvocationPlugin(BasePlugin):

    def __init__(self):
        super().__init__(name="count_invocation")
        self.agent_count = 0
        self.tool_count = 0
        self.llm_request_count = 0

    async def before_agent_callback(self, **kwargs):
        agent = kwargs.get("agent")
        ctx = kwargs.get("callback_context")

        self.agent_count += 1
        logger.info(
            "[plugin] before_agent agent_count=%d agent=%s invocation=%s trace=%s",
            self.agent_count,
            getattr(agent, "name", None),
            getattr(ctx, "invocation_id", None),
            getattr(ctx, "trace_id", None),
        )

    async def before_tool_callback(self, **kwargs):
        tool = kwargs.get("tool")
        ctx = kwargs.get("callback_context")
        tool_args = kwargs.get("tool_args")
        tool_kwargs = kwargs.get("tool_kwargs")
        tool_context = kwargs.get("tool_context") 

        self.tool_count += 1
        logger.info(
            "[plugin] before_tool tool_count=%d tool=%s args=%s kwargs=%s invocation=%s tool_context=%s",
            self.tool_count,
            getattr(tool, "name", None),
            tool_args,
            tool_kwargs,
            getattr(ctx, "invocation_id", None),
            tool_context,
        )

    async def before_model_callback(self, **kwargs):
        ctx = kwargs.get("callback_context")
        llm_request = kwargs.get("llm_request")

        self.llm_request_count += 1

        prompt_text = ""
        try:
            if hasattr(llm_request, "messages"):
                for msg in llm_request.messages:
                    if hasattr(msg, "content"):
                        for part in msg.content:
                            if hasattr(part, "text") and part.text:
                                prompt_text += part.text
        except Exception:
            pass

        prompt_len = len(prompt_text)

        logger.info(
            "[plugin] before_model llm_request_count=%d prompt_len=%d invocation=%s",
            self.llm_request_count,
            prompt_len,
            getattr(ctx, "invocation_id", None),
        )



import nest_asyncio
nest_asyncio.apply()

session_service = InMemorySessionService()

plugins = [LoggingPlugin(), CountInvocationPlugin()]

def make_runner(agent, app_name="capstone_app"):
    runner = Runner(
        agent=agent,
        session_service=session_service,
        app_name=app_name,
        plugins=plugins
    )
    logger.info("Runner created for agent: %s", agent.name)
    return runner

USER_ID = "userA"
SESSION_ID = "sessA"

async def create_session_async():
    await session_service.create_session(
        app_name="capstone_app",
        user_id=USER_ID,
        session_id=SESSION_ID
    )

await create_session_async()

logger.info("Session created: user=%s session=%s", USER_ID, SESSION_ID)


from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset

ZERODHA_MCP_URL = "https://mcp.kite.trade/mcp"
ZERODHA_TOOLS = ["login", "get_holdings", "get_ltp"]

if MCP_AVAILABLE:

    # -------------------------
    # REAL MCP TOOLSET
    # -------------------------
    zerodha_toolset = McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="npx",
                args=["-y", "mcp-remote", ZERODHA_MCP_URL],
            ),
            timeout=300,
        ),
        tool_filter=ZERODHA_TOOLS,
    )
    logger.info("Configured REAL Zerodha MCP toolset.")

else:

    # -------------------------
    # STUB MCP TOOLSET
    # -------------------------
    class StubMcpTool(BaseTool):
        def __init__(self, name, fn):
            super().__init__(
                name=name,
                description=f"Stub MCP tool: {name}",
            )
            self._fn = fn

        async def run(self, *args, **kwargs):
            return self._fn(kwargs)

    class StubMcpToolset(BaseToolset):
        def __init__(self):
            super().__init__()

            # ---- Stubbed responses ----
            def stub_login(args):
                return {"content": [{"text": "stub: logged in successfully"}]}

            def stub_get_holdings(args):
                holdings = [
                    {
                        "tradingsymbol": "AXISBANK-EQ",
                        "exchange": "NSE",
                        "quantity": 10,
                        "average_price": 720.0,
                        "instrument_token": 12345,
                    },
                    {
                        "tradingsymbol": "TATAMOTORS-EQ",
                        "exchange": "NSE",
                        "quantity": 5,
                        "average_price": 480.0,
                        "instrument_token": 54321,
                    },
                ]
                return {"content": [{"text": json.dumps(holdings)}]}

            def stub_get_ltp(args):
                token = args.get("instrument_token", 0)
                return 1000.0 + (token % 300)

            # Register stub tools
            self._tools = {
                "login": StubMcpTool("login", stub_login),
                "get_holdings": StubMcpTool("get_holdings", stub_get_holdings),
                "get_ltp": StubMcpTool("get_ltp", stub_get_ltp),
            }

        async def get_tools(self, readonly_context=None):
            return list(self._tools.values())

        def __iter__(self):
            return iter(self._tools.values())

    zerodha_toolset = StubMcpToolset()
    logger.info("MCP not available — using STUB MCP toolset.")


import re

mcp_agent = LlmAgent(
    name="zerodha_agent",
    model=Gemini(model="gemini-2.5-flash"),
    instruction="You MUST call MCP tools. Return JSON only. No explanation.",
    tools=[zerodha_toolset],
)

mcp_runner = make_runner(mcp_agent, app_name="zerodha_mcp_app")

async def create_mcp_session():
    await session_service.create_session(
        app_name="zerodha_mcp_app",
        user_id=USER_ID,
        session_id=SESSION_ID
    )

await create_mcp_session()
logger.info("MCP session created for app=zerodha_mcp_app session=%s", SESSION_ID)


async def call_mcp_tool(tool_name: str, args: dict):
    logger.info("Calling MCP tool: %s args=%s", tool_name, args)

    content = types.Content(
        role="user",
        parts=[types.Part(text=f"Call {tool_name} with {json.dumps(args)}")]
    )

    result = None

    async for event in mcp_runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=content
    ):
        if hasattr(event, "tool_response") and event.tool_response:
            result = event.tool_response.content

        if hasattr(event, "content") and event.content:
            for part in event.content.parts:
                if part.function_response:
                    result = part.function_response.response

    return result


# Clean login URL extractor
def extract_zerodha_login_url(login_response):
    if not login_response:
        return None

    try:
        text = login_response["content"][0]["text"]
    except:
        return None

    url_regex = r"https:\/\/kite\.zerodha\.com\/connect\/login\?api_key=.*?(?=\s|$|\))"
    m = re.search(url_regex, text)
    return m.group(0) if m else None


# Run MCP login
if MCP_AVAILABLE:
    print("Logging in to Zerodha MCP…")
    login_response = await call_mcp_tool("login", {})
    
    clean_url = extract_zerodha_login_url(login_response)

    print("\n================ ZERODHA LOGIN REQUIRED ================\n")
    print("Click or copy-paste this URL into browser:\n")
    
    print(clean_url or "No login URL found in response")
    
    print("\nAfter logging in, come back to the notebook.")
    print("========================================================\n")
else:
    print("Using stub MCP; login skipped.")




if MCP_AVAILABLE:
    # real MCP tool call
    raw_holdings_resp = asyncio.run(call_mcp_tool("get_holdings", {}))

    if isinstance(raw_holdings_resp, dict) and "content" in raw_holdings_resp:
        try:
            holdings_list = json.loads(raw_holdings_resp["content"][0]["text"])
        except Exception:
            holdings_list = []
    else:
        holdings_list = []

    logger.info("Holdings parsed (MCP): %d", len(holdings_list))

else:
    # STUB / MOCK MODE — NO LLM, NO TOOLS
    logger.info("MCP not available — using MOCK HOLDINGS (offline mode).")

    holdings_list = [
        {
            "tradingsymbol": "AXISBANK-EQ",
            "exchange": "NSE",
            "quantity": 10,
            "average_price": 720.0,
            "instrument_token": 12345,
        },
        {
            "tradingsymbol": "TATAMOTORS-EQ",
            "exchange": "NSE",
            "quantity": 5,
            "average_price": 480.0,
            "instrument_token": 54321,
        },
        {
            "tradingsymbol": "INFY-EQ",
            "exchange": "NSE",
            "quantity": 3,
            "average_price": 1450.0,
            "instrument_token": 67890,
        },
    ]

logger.info("Final holdings (used by pipeline): %d", len(holdings_list))
holdings_list[:5]


from datetime import datetime
import math

def clean_symbol(stock):
    ts = stock["tradingsymbol"].replace("-EQ", "").replace("-BE", "")
    if stock.get("exchange", "") == "NSE":
        return ts + ".NS"
    return ts + ".BO"

def fetch_yahoo_ohlc(symbol, period="6mo", interval="1d", timeout=5):
    """
    Fetch OHLC from Yahoo. Returns a pandas DataFrame or None.
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={period}&interval={interval}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        js = r.json().get("chart", {}).get("result")
        if not js:
            return None
        js = js[0]
        ts = js.get("timestamp", [])
        q = js.get("indicators", {}).get("quote", [{}])[0]
        df = pd.DataFrame({
            "timestamp": [datetime.fromtimestamp(t) for t in ts],
            "open": q.get("open"),
            "high": q.get("high"),
            "low": q.get("low"),
            "close": q.get("close"),
            "volume": q.get("volume")
        })
        return df
    except Exception as e:
        logger.warning("fetch_yahoo_ohlc failed for %s: %s", symbol, e)
        return None

def compute_indicators(df):
    if df is None or df.empty:
        return None
    df = df.copy()
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["sma200"] = df["close"].rolling(200).mean()
    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss.replace(0, float("nan")))
    df["rsi14"] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["signal"] = df["macd"].ewm(span=9).mean()
    return df

def fetch_fundamentals(symbol):
    """
    Fetch simple fundamentals from Screener.in (best-effort).
    Returns dictionary like {'pe': '12', 'pb': '1.2', 'roe': '20%'}
    """
    try:
        name = symbol.split(".")[0]
        url = f"https://www.screener.in/company/{name}/"
        html = requests.get(url, timeout=5).text
        soup = BeautifulSoup(html, "html.parser")
        fundamentals = {
            li.select_one("span.name").text.strip(): li.select_one("span.value").text.strip()
            for li in soup.select("li.flex") if li.select_one("span.name") and li.select_one("span.value")
        }
        return {
            "pe": fundamentals.get("P/E"),
            "pb": fundamentals.get("P/B"),
            "roe": fundamentals.get("ROE"),
            "roce": fundamentals.get("ROCE"),
        }
    except Exception as e:
        logger.info("fetch_fundamentals failed for %s: %s", symbol, e)
        return {}



async def get_live_price(token):
    ex = stock.get("exchange", "NSE")
    ts = stock.get("tradingsymbol")
    
    resp = await call_mcp_tool("get_ltp", {
        "instruments": [f"{ex}:{ts}"]
    })
    # If stub returns a number directly, return it
    if isinstance(resp, dict) and "content" in resp:
        try:
            data = json.loads(resp["content"][0]["text"])
            return data.get("last_price")  # MCP returns dict
        except Exception:
            logger.warning("Unexpected LTP response shape: %s", str(resp)[:200])
            return stock.get("last_price")



def analyze_stock(stock):
    logger.info("Analyzing stock: %s", stock.get("tradingsymbol"))

    symbol = clean_symbol(stock)

    # --- OHLC + Technical ---
    df = fetch_yahoo_ohlc(symbol)
    df_ind = compute_indicators(df)

    # --- Fundamentals ---
    fundamentals = fetch_fundamentals(symbol)

    # --- Live Price ---
    if MCP_AVAILABLE:
        # Real MCP call
        try:
            live_price = asyncio.run(get_live_price(stock.get("instrument_token")))
        except Exception as e:
            logger.warning("Live price fetch failed: %s", e)
            live_price = None
    else:
        # Deterministic MOCK price (so nothing breaks)
        token = stock.get("instrument_token") or 0
        live_price = 1000.0 + (token % 500)   # stable mock price

    if df_ind is not None and not df_ind.empty:
        tech = {
            "sma20": float(df_ind["sma20"].iloc[-1]) if not pd.isna(df_ind["sma20"].iloc[-1]) else None,
            "sma50": float(df_ind["sma50"].iloc[-1]) if not pd.isna(df_ind["sma50"].iloc[-1]) else None,
            "sma200": float(df_ind["sma200"].iloc[-1]) if not pd.isna(df_ind["sma200"].iloc[-1]) else None,
            "rsi14": float(df_ind["rsi14"].iloc[-1]) if not pd.isna(df_ind["rsi14"].iloc[-1]) else None,
            "macd": float(df_ind["macd"].iloc[-1]) if not pd.isna(df_ind["macd"].iloc[-1]) else None,
            "signal": float(df_ind["signal"].iloc[-1]) if not pd.isna(df_ind["signal"].iloc[-1]) else None,
        }
    else:
        tech = None

    # --- P&L Calculation ---
    q = stock.get("quantity", 0)
    avgp = stock.get("average_price", 0.0)
    unrealized = None
    if live_price is not None and q:
        try:
            unrealized = (live_price - avgp) * q
        except Exception:
            unrealized = None

    return {
        "symbol": stock.get("tradingsymbol"),
        "clean_symbol": symbol,
        "exchange": stock.get("exchange"),
        "quantity": q,
        "average_price": avgp,
        "live_price": live_price,
        "technical": tech,
        "fundamentals": fundamentals,
        "unrealized_pnl": unrealized,
    }


N = 4  # adjust as needed for demo to save tokens in repeted test
final_report = []
for s in holdings_list[:N]:
    try:
        r = analyze_stock(s)
        final_report.append(r)
    except Exception as e:
        logger.exception("Error analyzing stock %s: %s", s.get("tradingsymbol"), e)

logger.info("Analysis complete for %d stocks", len(final_report))
final_report



REPORT_USER = "report_user"
REPORT_SESSION = "report_sess_2"

report_agent = Agent(
    name="portfolio_report_agent",
    model=Gemini(model="gemini-2.5-flash"),
    instruction=(
        "You are an Indian equities portfolio analyst. "
        "Use the google_search tool for latest news when asked. "
        "Output must strictly follow the requested structure and be decisive."
    ),
    tools=[google_search],
)

report_runner = make_runner(report_agent)

async def setup_report_session():
    await session_service.create_session(app_name="portfolio_report_app", user_id=REPORT_USER, session_id=REPORT_SESSION)

asyncio.run(setup_report_session())
logger.info("Report session ready: %s / %s", REPORT_USER, REPORT_SESSION)



PORTFOLIO_PROMPT = """
You are “Portfolio Analyst Agent”, specializing in Indian equities.
Output MUST be decisive, simple, direct, and written like a trading coach.

SECTION 1 — OVERALL ACTION ITEMS (STRONG DECISIONS ONLY)
For all the stocks combined, give 8–12 BULLET POINTS with:
- direct verdicts (e.g., “AXISBANK HOLD, overbought, wait for cool-off”)
- stop-loss levels if a stock is risky
- upside targets or entry levels if it’s a buy
- warnings (e.g., “TATAMOTORS showing breakdown risk below ₹X”)
- call out overvalued / overheated stocks
- call out deep corrections with possible bounce zones
- use internet + technical + fundamental signals to justify each point

SECTION 2 — STOCK-WISE ANALYSIS (VERY STRUCTURED)
For EACH STOCK, give:
1) Clear Verdict (first line)
2) Snapshot: quantity, avg buy price, live price, unrealized P/L, valuation quick check
3) Latest News (use Google Search)
4) Technical Levels
5) Fundamentals (very short)
6) Action Plan (BUY RANGE, STOP-LOSS, TARGET)

SECTION 3 — CONSOLIDATED RANKING TABLE
Stock | Verdict | Buy Zone | Stop-loss | Target | Reason

DATA (DO NOT IGNORE):
{data}
"""



import nest_asyncio
nest_asyncio.apply() 

REPORT_APP_NAME = "portfolio_report_app"

async def ensure_report_session():
    try:
        await session_service.create_session(
            app_name=REPORT_APP_NAME,
            user_id=REPORT_USER,
            session_id=REPORT_SESSION
        )
        logger.info("Report session created: app=%s user=%s session=%s",
                    REPORT_APP_NAME, REPORT_USER, REPORT_SESSION)
    except Exception as e:
        logger.info("Session may already exist: %s", e)

await ensure_report_session()


report_runner = make_runner(report_agent, app_name=REPORT_APP_NAME)
logger.info("Report runner attached to app '%s'", REPORT_APP_NAME)


async def run_report(prompt_text: str):
    text = ""
    content = types.Content(
        role="user",
        parts=[types.Part(text=prompt_text)]
    )

    async for event in report_runner.run_async(
        user_id=REPORT_USER,
        session_id=REPORT_SESSION,
        new_message=content
    ):
        if hasattr(event, "content") and event.content:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    text += part.text
    return text


# --- Build prompt ---
prompt = PORTFOLIO_PROMPT.format(
    data=json.dumps(final_report, indent=2)
)

logger.info("Running report agent with payload length: %d", len(prompt))

# --- Run agent ---
report_output = await run_report(prompt)

logger.info("Report generation complete. Output length: %d", len(report_output))


# --- Display truncated output ---
print("\n\n===== REPORT OUTPUT (truncated) =====\n")
print(report_output[:8000])
print("\n\n===== END =====\n")

# save for submission to see generated file
with open("report_output.txt", "w", encoding="utf-8") as f:
    f.write(report_output)
logger.info("Saved report_output.txt")


# Cell 15 — Simple consolidated table synthesized from final_report

rows = []

for r in final_report:
    # --- Extract live price  ---
    live = r.get("live_price")
    live = float(live) if isinstance(live, (int, float)) else None

    # --- Extract PE  ---
    pe = None
    try:
        pe_str = r["fundamentals"].get("pe")
        if pe_str:
            pe = float(pe_str.replace(",", "").split()[0])
    except Exception:
        pe = None

    # --- Valuation label ---
    if pe is None:
        val_label = "data unavailable"
    elif pe < 12:
        val_label = "cheap"
    elif pe < 25:
        val_label = "fair"
    else:
        val_label = "expensive"

    # --- Defaults ---
    verdict = "HOLD"
    buy_zone = "-"
    stop_loss = "-"
    target = "-"

    tech = r.get("technical")
    rsi = tech.get("rsi14") if tech else None

    if rsi is not None and live is not None:

        if rsi < 30:
            verdict = "BUY"
            buy_zone = f"below {live * 0.97:.2f}"
            stop_loss = f"{live * 0.92:.2f}"
            target = f"{live * 1.12:.2f}"

        elif rsi > 70:
            verdict = "HOLD"
            stop_loss = f"{live * 0.94:.2f}"
            target = f"{live * 1.05:.2f}"

    else:
        # Not enough data to compute technical actions
        verdict = "NEUTRAL"
        buy_zone = "insufficient data"
        stop_loss = "insufficient data"
        target = "insufficient data"

    rows.append({
        "Stock": r["symbol"],
        "Verdict": verdict,
        "Buy Zone": buy_zone,
        "Stop-loss": stop_loss,
        "Target": target,
        "Reason": val_label,
    })

df_summary = pd.DataFrame(rows)
df_summary



with open("final_report.json", "w") as f:
    json.dump(final_report, f, default=str, indent=2)
logger.info("Saved final_report.json")


