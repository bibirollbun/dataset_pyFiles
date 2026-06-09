pip install google-genai pydantic


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("GOOGLE_API_KEY")



import os
from kaggle_secrets import UserSecretsClient
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import json
from typing import Callable
import time
# --- 1. SETUP AND AUTHENTICATION ---
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GEMINI_API_KEY"] = GOOGLE_API_KEY
    print("Gemini API Key loaded successfully.")
except Exception as e:
    print("FATAL ERROR: GOOGLE_API_KEY not found in Kaggle Secrets.")
    raise

client = genai.Client()


# --- 2. STRUCTURED OUTPUT (Pydantic Schema) ---
class MarketAnalysis(BaseModel):
    """Defines the strict JSON schema for the final analysis report."""
    stock_ticker: str = Field(description="The financial ticker symbol (e.g., 'AAPL' or 'TSLA').")
    sentiment_score: int = Field(description="A score from 1 (Very Negative) to 10 (Very Positive) based on the news.")
    key_factors: list[str] = Field(description="A list of 3-5 critical factors driving the sentiment.")
    investment_summary: str = Field(description="A 1-2 sentence actionable summary for the investor (e.g., 'A strong buy due to ...').")


# --- 3. TOOL USE (Simulated Function) ---
def google_search_tool(query: str) -> str:
    """A tool that simulates searching the web for the latest financial news articles."""
    print(f"\nðŸ“ž TOOL CALLED: Executing Google Search for: '{query}'")
    
    # This simulates receiving real-time data from an external API call.
    if "Tesla" in query:
        return """
        [Snippet 1] Tesla's new cheaper Model 2 factory begins construction in Berlin, promising volume growth.
        [Snippet 2] CEO Elon Musk confirms massive job cuts across engineering and sales, causing short-term stock volatility.
        [Snippet 3] Analysts raise Q3 price target for Tesla, citing improved production efficiency and reduced overhead.
        """
    else:
        return f"Real-time news check for {query}: No recent high-impact financial events found. Sentiment is neutral."

# --- 4. MULTI-AGENT SYSTEM (System Instructions) ---

# Agent 1: The Planner
PLANNER_SYSTEM_INSTRUCTION = """
You are the *Market Planner Agent*. Your sole role is to take a user's broad, natural language request and distill it into a single, highly precise search query optimized for the 'google_search_tool'. Do not analyze; only refine the query.

Your output must be *ONLY* the refined search query string, nothing else.
"""

# Agent 2: The Analyst
ANALYST_SYSTEM_INSTRUCTION = f"""
You are the *Expert Financial Analyst Agent*. Your task is to provide a real-time, objective market analysis.

Your core process must be:
1.  *Tool Call:* Immediately use the 'google_search_tool' to fetch news based on the query.
2.  *Analysis & Reasoning:* Scrutinize the search results. Before finalizing, you must internally confirm that the 'sentiment_score' directly reflects the retrieved 'key_factors'. Adjust the score and justify the change if they contradict.
3.  *Structured Output:* Generate the final analysis report strictly conforming to the JSON schema below.

Strict JSON Output Schema:
{json.dumps(MarketAnalysis.model_json_schema(), indent=2)}
"""


# --- 5. EXECUTION FUNCTION ---
def run_market_agent(client: genai.Client, user_request: str):
    """Orchestrates the Planner and Analyst Agents to generate the final report."""
    print(f"--- USER REQUEST ---\n'{user_request}'\n")
    
    # STEP 1: Planner Agent Refines the Query
    print("--- 1. PLANNER AGENT: Strategizing the Search Query ---")
    planner_response = client.models.generate_content(
        model='gemini-2.5-flash', 
        contents=user_request,
        config=types.GenerateContentConfig(system_instruction=PLANNER_SYSTEM_INSTRUCTION)
    )
    refined_query = planner_response.text.strip()
    print(f"Refined Query: {refined_query}")
    time.sleep(3)
    # --- STEP 2A: ANALYST AGENT: Execute Tool Call (Data Retrieval) ---
    # This call focuses ONLY on Tool Use (no structured output)
    print("\n--- 2A. ANALYST AGENT: Executing Tool Call (Fetching Data) ---")
    
    tool_use_response = client.models.generate_content(
        model='gemini-2.5-flash', 
        # Prompt guides the agent to use the tool based on the query.
        contents=f"Use the google_search_tool to find real-time financial news for this query: {refined_query}", 
        config=types.GenerateContentConfig(
            system_instruction=ANALYST_SYSTEM_INSTRUCTION, 
            tools=[google_search_tool], # Tool is included here
        )
    )
    
    # Get the raw search result text from the tool call
    search_results = tool_use_response.text 
    print(f"Tool Results Received: \n{search_results[:100]}...")
    
    # --- STEP 2B: ANALYST AGENT: Structured Analysis (JSON Generation) ---
    # This call focuses ONLY on Structured Output (no tool)
    print("\n--- 2B. ANALYST AGENT: Structured Analysis (JSON Generation) ---")
    
    final_analysis_response = client.models.generate_content(
        model='gemini-2.5-flash', 
        # Pass the search results to the model to generate the JSON report.
        contents=f"Based on the following news snippets: '{search_results}', analyze the market and generate the final report in the strict JSON format.", 
        config=types.GenerateContentConfig(
            system_instruction=ANALYST_SYSTEM_INSTRUCTION, 
            response_mime_type="application/json", # Structured output is used here
            response_schema=MarketAnalysis
        )
    )

    final_analysis = final_analysis_response.text
    
    print("\nâœ… FINAL STRUCTURED ANALYSIS (JSON OUTPUT):")
    try:
        # Load and print the JSON in a human-readable format
        print(json.dumps(json.loads(final_analysis), indent=4))
    except json.JSONDecodeError:
        print("ERROR: Failed to decode JSON output. Raw output:")
        print(final_analysis)


import pandas as pd

# Run the Agent
run_market_agent(client, "Tell me what the latest news suggests about investing in Tesla stock today.")

# This file is not used for grading, but is required by the submission interface.
dummy_submission = pd.DataFrame({'id': [1], 'analysis': ['completed']})
dummy_submission.to_csv('submission.csv', index=False)

print("\nSuccessfully created required 'submission.csv' file.")

