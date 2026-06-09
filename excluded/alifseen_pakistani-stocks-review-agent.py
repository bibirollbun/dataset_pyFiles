!pip install tradingview_ta


!pip install google-adk


import os

from google.adk.agents import Agent, LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types
from google.adk.plugins.logging_plugin import (LoggingPlugin)
from tradingview_ta import TA_Handler, Interval


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


retry_config=types.HttpRetryOptions(
    attempts=5,
    exp_base=7,  
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)


def get_technicals_indicators(ticker: str) -> dict:
    """Looks up the important technical indicators for a given company stock.
    This tool provides technical indicators on 1 Day, 1 Week, and 1 Month time horizon
    based on the name of the stock provided by the user.

    Args:
        method: The ticker of the stock. e.g. 'OGDC' or 'PSO'

    Returns:
        Dictionary with horizon and technical indicator name.
        1 Week RSI: {"1W-RSI": 54}
        1 Day RSI: {"1D-RSI": 35}
    """
    # This holds the values of the indicators
    technicals = {}

    # Instantiate the module 3 times, 1 for each time horizon.
    TICKER_1D = TA_Handler(
        symbol=f"{ticker}",
        exchange="PSX",
        screener="pakistan",
        interval=Interval.INTERVAL_1_DAY
    )

    TICKER_1W = TA_Handler(
        symbol=f"{ticker}",
        exchange="PSX",
        screener="pakistan",
        interval=Interval.INTERVAL_1_WEEK
    )

    TICKER_1M = TA_Handler(
        symbol=f"{ticker}",
        exchange="PSX",
        screener="pakistan",
        interval=Interval.INTERVAL_1_MONTH
    )

    # This fetches all the technical indicators for the three time horizons
    tradingview_technicals_1d = TICKER_1D.get_indicators()
    tradingview_technicals_1w = TICKER_1W.get_indicators()
    tradingview_technicals_1m = TICKER_1M.get_indicators()


    # This keeps only the relevant indicators in the dictionary
    for k, v in tradingview_technicals_1d.items():
        if k not in ['RSI', 'Stoch.K', 'MACD.macd', 'CCI20', 'W.R', 'EMA10', 'EMA50', 'EMA200',
                     'Pivot.M.Classic.S1', 'Pivot.M.Classic.R1']:
            continue
        else:
            technicals[f"1D-{k}"] = v

    for k, v in tradingview_technicals_1w.items():
        if k not in ['RSI', 'Stoch.K', 'MACD.macd', 'CCI20', 'W.R', 'EMA10', 'EMA50', 'EMA200',
                     'Pivot.M.Classic.S1', 'Pivot.M.Classic.R1']:
            continue
        else:
            technicals[f"1W-{k}"] = v

    for k, v in tradingview_technicals_1m.items():
        if k not in ['RSI', 'Stoch.K', 'MACD.macd', 'CCI20', 'W.R', 'EMA10', 'EMA50', 'EMA200',
                     'Pivot.M.Classic.S1', 'Pivot.M.Classic.R1']:
            continue
        else:
            technicals[f"1M-{k}"] = v

    return technicals


news_agent = LlmAgent(
    name = 'News_agent',
    model = Gemini(
        model='gemini-2.5-flash-lite',
        retry_options=retry_config
    ),
    instruction='''You are a specialized news agent for Pakistan Stock Market. Your only job is to use 
    the google_search tool and find the 1-2 pakistani financial news related to relevant stock in the past 7 days. ALWAYS present
    your findings with citations so user can verify it. 
    ''',
    tools=[google_search],
    output_key='news_findings',
)


technical_analysis_agent = LlmAgent(
    name = 'Technical_agent',
    model = Gemini(
        model='gemini-2.5-flash-lite',
        retry_options=retry_config
    ),
    instruction='''You are a Stock technical analysis agent. Your objective is to use the get_technical_analysis
    function and provide an analsysis for the relevant stock on 1 day, 1 Week and 1 Month time horizon using the indicators provided. 
    Do not provide the list of indicators, instead only return a paragraph or two of your analysis of them.
    ''',
    tools=[get_technicals_indicators],
    output_key='technical_analysis'
)


root_agent = LlmAgent(
    name = 'Orchestator',
    model = 'gemini-2.5-flash',
    instruction = """
    You are an Stock Analyst. Your objective is to answer the user's query by orchestrating a workflow for the relevant stock:
    1. First, you MUST call the news_agent tool to find the relevant news for the stock provided by the user.
    2. Then, you MUST call the technical_analysis_agent to get an analysis summary for the stock provided by the user.
    3. Finally, combine the two into a final executive summary of the stock based on news and the technical summary.

    Rules:
    - First give news under "New Summary" heading. Then give technical analysis under "Technical Analysis Summary" heading.
    - Remove any irrelevant news that is not linked to the pakistani company or companies stock or is older than a month.
    """,
    tools=[AgentTool(news_agent), AgentTool(technical_analysis_agent)]
)


session_service = InMemorySessionService()

app_name = 'stock_analyst'
user_id = 'user001'
session_id = 'session001'

runner = Runner(agent=root_agent, app_name=app_name, session_service=session_service, plugins=[LoggingPlugin()])


async def run(runner, userid, sessionid, message):
    try:
        session = await session_service.create_session(app_name=runner.app_name, user_id=userid, session_id=sessionid)
    except:
        session = await session_service.get_session(app_name=runner.app_name, user_id=userid, session_id=sessionid)
    
    print("User > ", message)

    query = types.Content(role='user', parts=[types.Part(text=message)])

    async for event in runner.run_async(user_id=user_id, session_id=session.id, new_message=query):
        if event.content and event.content.parts:
            if (
                event.content.parts[0].text != "None"
                and event.content.parts[0].text
            ):
                print("Agent > ", event.content.parts[0].text)


message = 'Tell me about OGDC'

await run(runner, user_id, session_id, message)

