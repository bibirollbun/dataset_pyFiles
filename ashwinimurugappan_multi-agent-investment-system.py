import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


import asyncio
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types

print("âœ… ADK components imported successfully.")


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


# --- 2. Agent Definition ---


market_analyst = Agent(
    name="Nifty_50_Market_Analyst", 
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="""A specialized financial agent that uses Google Search to provide real-time analysis 
    of the Nifty 50 stock market index.""",
    instruction="""You are a professional financial analyst. 
    Your task is to provide a concise, data-driven summary of the current market sentiment 
    and key factors affecting the Nifty 50 index. 
    Always use the Google Search tool for the latest data.""",
    tools=[google_search],
    output_key="final_recommendation"
)
print(f"âœ… Agent '{market_analyst.name}' defined successfully.")


async def run_analysis(stock_ticker: str):
    """Initializes the runner and executes the single-agent analysis for one stock."""
    print(f"\n--- Starting Real-Time Market Analysis for {stock_ticker} ---")
    
    # 1. Initialize Runner (Execution Environment)
    runner = InMemoryRunner(agent=market_analyst)
    
    # 2. Define the search prompt directly
    prompt_string = f"Today's closing summary and key movement reason for Nifty stock {stock_ticker}"
    
    # 3. Execute the agent
    try:
        # runner.run_debug() returns a list of events.
        result_events = await runner.run_debug(
            prompt_string
        )
        
        # Get the final event object, which is the last item in the list of steps.
        final_result_obj = result_events[-1]
            
        # 4. Extract and print the final output
        # FINAL FIX: Accessing the result through the nested 'actions.state_delta', 
        # which reliably holds the agent's output dictionary when using run_debug.
        final_recommendation = final_result_obj.actions.state_delta.get("final_recommendation")
        
        # Check if we successfully extracted the text
        if final_recommendation:
            print(f"\n====================== {stock_ticker} ANALYSIS ======================")
            print(final_recommendation)
            print("==============================================================")
        else:
            # If extraction fails, print the final object for further debugging
            print(f"â�Œ No final recommendation found for {stock_ticker}.")
            # Note: We skip printing the massive object to keep the log clean, 
            # but this is where it would be printed if needed for diagnosis.
        
    except Exception as e:
        # Note: If this block is hit, it means something failed *before* the final event  was returned
        print(f"â�Œ An error occurred during the analysis of {stock_ticker}: {e}")


# --- 4. Main Parallel Execution Function ---

# Define a representative subset of Nifty 50 stocks to analyze
STOCKS_TO_ANALYZE = ["RELIANCE", "HDFCBANK", "INFY", "TCS", "ICICIBANK"] 

async def main_parallel_analysis():
    """Runs all stock analyses concurrently using asyncio.gather()."""
    print(f"ğŸš€ Starting PARALLEL real-time analysis for {len(STOCKS_TO_ANALYZE)} key Nifty 50 stocks...")
    
    # Create a list of all the asynchronous tasks (coroutines)
    tasks = [run_analysis(stock) for stock in STOCKS_TO_ANALYZE]
    
    # Run all tasks concurrently
    await asyncio.gather(*tasks)
    
    print("\nâœ… All parallel analyses complete!")

# --- 5. Execution in Notebook Environment ---

# This top-level 'await' command executes the asynchronous function directly
# in the notebook's event loop, avoiding the RuntimeError.
await main_parallel_analysis()




