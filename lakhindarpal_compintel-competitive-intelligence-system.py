# Install the Google Agent Development Kit
!pip install -q google-adk[a2a] nest_asyncio


import os
import sys
import nest_asyncio
import logging
from kaggle_secrets import UserSecretsClient

nest_asyncio.apply()

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
except Exception as e:
    print(f"Error loading API Key: {e}")


from google.adk.tools import ToolContext

def calc_growth_rate(current_val: float, previous_val: float) -> dict:
    """
    Calculates year-over-year growth rate percentage.
    
    Args:
        current_val: The revenue/value for the current period.
        previous_val: The revenue/value for the previous period.
        
    Returns:
        Dictionary containing the growth percentage and status.
    """
    if previous_val == 0:
        return {"error": "Previous value cannot be zero"}
    
    growth = ((current_val - previous_val) / previous_val) * 100
    return {
        "status": "success", 
        "growth_percent": round(growth, 2),
        "message": f"Calculated growth: {round(growth, 2)}%"
    }

print("✅ Custom Tools defined.")


from google.adk.agents import Agent, ParallelAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import google_search, FunctionTool
from google.genai import types

MODEL_NAME = "gemini-2.5-flash-lite"

retry_config = types.HttpRetryOptions(
    attempts=5, 
    initial_delay=1,
    http_status_codes=[429, 500, 503]
)

# --- 1. News Researcher (Search Only) ---
news_agent = Agent(
    name="NewsResearcher",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction="""
    Search for the latest 3 major news headlines regarding the target company.
    Focus on product launches, legal issues, or market shifts.
    """,
    tools=[google_search],
    output_key="news_data"
)

# --- 2A. Finance Searcher (Search Only) ---
finance_searcher = Agent(
    name="FinanceSearcher",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction="""
    Find the annual revenue for the last two available years for the target company.
    Output the specific revenue numbers for each year clearly.
    """,
    tools=[google_search],
    output_key="raw_finance_data"
)

# --- 2B. Finance Calculator (Math Tool Only) ---
finance_calculator = Agent(
    name="FinanceCalculator",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction="""
    Read the revenue data from {raw_finance_data}.
    Use the 'calc_growth_rate' tool to calculate the year-over-year growth percentage.
    Report the final growth metric.
    """,
    tools=[FunctionTool(calc_growth_rate)],
    output_key="financial_analysis"
)

# --- 3. Strategic Analyst ---
analyst_agent = Agent(
    name="StrategicAnalyst",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction="""
    You are a Lead Strategic Analyst.
    Read the news from {news_data} and financial analysis from {financial_analysis}.
    Create a SWOT analysis (Strengths, Weaknesses, Opportunities, Threats).
    """,
    output_key="swot_analysis"
)

# --- 4. Report Writer ---
writer_agent = Agent(
    name="ReportWriter",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction="""
    Format the {swot_analysis} into a clean Markdown executive summary.
    """,
)

print("✅ Agents configured")


# --- Architecture Definition ---

# 1. Finance Sub-Pipeline (Sequential)
# Runs Search THEN Calculator to avoid tool mixing error
finance_pipeline = SequentialAgent(
    name="FinancePipeline",
    sub_agents=[finance_searcher, finance_calculator]
)

# 2. Parallel Research Team
# Runs News and Finance Pipeline at the same time
research_team = ParallelAgent(
    name="ResearchTeam",
    sub_agents=[news_agent, finance_pipeline]
)

# 3. Main System
comp_intel_pipeline = SequentialAgent(
    name="CompIntelPipeline",
    sub_agents=[research_team, analyst_agent, writer_agent]
)

print("✅ Architecture built: User -> [News || [FinSearch -> FinCalc]] -> Analyst -> Writer")


import asyncio
from google.adk.runners import InMemoryRunner
from google.adk.plugins.logging_plugin import LoggingPlugin

async def run_analysis(company_name):
    runner = InMemoryRunner(
        agent=comp_intel_pipeline,
        plugins=[LoggingPlugin()] 
    )
    
    query = f"Analyze {company_name}."
    
    try:
        responses = await runner.run_debug(query)
        
        if responses and responses[-1].content and responses[-1].content.parts:
            print(responses[-1].content.parts[0].text)
        else:
            print("No final text generated.")
            
    except ExceptionGroup as eg:
        for exc in eg.exceptions:
            print(f"Parallel Execution Error: {exc}")
    except Exception as e:
        print(f"Error: {e}")


# --- EXECUTE THE SYSTEM ---
if __name__ == "__main__":
    asyncio.run(run_analysis("Tesla Inc."))




