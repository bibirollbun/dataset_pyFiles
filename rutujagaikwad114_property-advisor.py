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


# Install Google ADK 
!pip install google-adk --quiet --no-deps
!pip install google-genai --quiet
!pip install nest_asyncio --quiet



import os
import logging
import asyncio

from google.genai import types as genai_types
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import google_search, FunctionTool, load_memory


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("propmate")

MODEL_ID = "gemini-2.5-flash"
APP_NAME = "propmate_app"
USER_ID = "kaggle_user"


#cell 4- IntakeAgent
intake_agent = Agent(
    model=MODEL_ID,
    name="IntakeAgent",
    instruction="""
Extract structured property preferences from the user's message.

Return:

STRUCTURED_CRITERIA:
{
 "city": "...",
 "max_price": ...,
 "min_bedrooms": ...,
 "property_type": "...",
 "must_haves": "..."
}

Be concise.
"""
)



#CELL 5 â€” SearchAgent 
search_agent = Agent(
    model=MODEL_ID,
    name="SearchAgent",
    tools=[google_search],
    instruction="""
Use STRUCTURED_CRITERIA to build a query:

"<min_bedrooms> bedroom <property_type or 'home'> for sale in <city> under <max_price> CAD"

Call google_search(query=<your string>).

After tool returns:
- Extract top 3 URLs/snippets
- No hallucinations
"""
)


#CELL 6 â€” ExtractionAgent
extraction_agent = Agent(
    model=MODEL_ID,
    name="ExtractionAgent",
    instruction="""
Convert google_search results into CLEAN_LISTINGS:

[
 {"title": "...", "url": "...", "snippet": "...", "price_estimate": null}
]

Do not invent details; only use what you see.
Limit to 3â€“5 listings.
"""
)


#CELL 7 â€” Advisor Agents

# Lifestyle advisor
advisor_pros = Agent(
    model=MODEL_ID,
    name="AdvisorPros",
    instruction="""
Give lifestyle-focused pros and cons for the top CLEAN_LISTINGS.
Use only the available listing data. No hallucinations.
"""
)

# Budget advisor
advisor_budget = Agent(
    model=MODEL_ID,
    name="AdvisorBudget",
    instruction="""
Evaluate whether the listings fit the user's budget from STRUCTURED_CRITERIA.
If prices are missing, state that clearly and explain caution.
"""
)

# Investment advisor
investment_advisor = Agent(
    model=MODEL_ID,
    name="InvestmentAdvisorAgent",
    tools=[google_search],
    instruction="""
You evaluate the INVESTMENT potential of the property search.

STEPS:
1. Read STRUCTURED_CRITERIA (city, budget, bedrooms, property_type).
2. Perform 1â€“2 google_search tool calls using queries like:
   - "<property_type> investment potential in <city>"
   - "real estate trends <city> 2024 2025"
   - "rental demand <city> vacancy rate"
   - "is <property_type> in <city> a good investment"
3. Extract signals:
   - price trend (up/flat/down)
   - rental demand
   - vacancy levels
   - neighborhood growth indicators
   - buyer vs seller market
4. Output a JSON block:

INVESTMENT_ANALYSIS:
{
  "investment_score": <1â€“10>,
  "summary": "...",
  "pros": ["...", "..."],
  "risks": ["...", "..."],
  "recommendations": ["...", "...", "..."]
}

RULES:
- Use only information visible in google_search results.
- If evidence is weak, say so clearly.
- Absolutely NO hallucinated data.
"""
)

# Parallel advisor
advisor_parallel = ParallelAgent(
    name="AdvisorParallel",
    sub_agents=[
        advisor_pros,
        advisor_budget,
        investment_advisor,   
    ],
)



#CELL 8 â€” Mortgage Tool +Summary Agent
def mortgage_calc(price: int, rate: float = 5.0, years: int = 25):
    monthly_rate = (rate/100)/12
    n = years * 12
    m = price * (monthly_rate * (1 + monthly_rate)**n) / ((1 + monthly_rate)**n - 1)
    return {"monthly_payment": round(m, 2)}

mortgage_tool = FunctionTool(mortgage_calc)

summary_agent = Agent(
    model=MODEL_ID,
    name="SummaryAgent",
    tools=[mortgage_tool],
    instruction="""
You are the FINAL summarizing agent. You MUST always produce a plain natural-language output.
Never leave a function_call as your last message.

Use the following inputs if available:
- STRUCTURED_CRITERIA
- CLEAN_LISTINGS
- AdvisorProsAgent output
- AdvisorBudgetAgent output
- InvestmentAdvisorAgent output

RULES:
1. If CLEAN_LISTINGS is empty or unclear:
   - Say that live listings were not available from Google Search.
   - DO NOT invent prices, specs, or listings.
   - Still answer the user's underlying question using investment advisor output.

2. Incorporate investment insight:
   - State whether a semi-detached in Mississauga around the user's budget
     appears to be a good investment based on signals found by InvestmentAdvisorAgent.
   - Summarize risks + opportunities.
   - Provide a simple investment verdict (Good / Fair / Risky).

3. ONLY call the mortgage_calc tool if a price_estimate exists.
   If not, explain that a mortgage estimate cannot be computed without a real price.

4. Provide 3â€“5 next steps like:
   - check actual sold data
   - speak to agent
   - compare neighborhoods
   - verify rent demand

5. Must end with a friendly, human-like closing.

Avoid hallucinations.
"""
)



#Build Multi-Agent System
prop_agent = SequentialAgent(
    name="PropMateRoot",
    sub_agents=[
        intake_agent,
        search_agent,
        extraction_agent,
        advisor_parallel,
        summary_agent,
    ],
)


#Runner + Memory +RunConfig
session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()

runner = Runner(
    agent=prop_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service,
)

run_config = RunConfig(max_llm_calls=12)

print("Runner ready âœ”")



#CELL 11 â€” run_query Helper / run_property_query()
async def run_property_query(msg: str, session_id="prop_session"):

    # Ensure session exists
    try:
        await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=session_id,
        )
    except Exception:
        pass

    message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=msg)]
    )

    final = ""

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=message,
        run_config=run_config,
    ):
        if event.is_final_response():
            final = event.content.parts[0].text

    session_obj = await session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id
    )

    await memory_service.add_session_to_memory(session_obj)

    return final




import os

# Detect Kaggle automated "Run & Save All"
KAGGLE_AUTO = False

# Case 1: Papermill (most reliable)
if os.environ.get("PAPERMILL_INPUT_PATH") or os.environ.get("PAPERMILL_OUTPUT_PATH"):
    KAGGLE_AUTO = True

# Case 2: Kaggle Batch mode (Save & Run All ALWAYS sets this)
elif os.environ.get("KAGGLE_KERNEL_RUN_TYPE") == "Batch":
    KAGGLE_AUTO = True

# Final behavior
if KAGGLE_AUTO:
    user_query = "Find a semi-detached home in Mississauga under 900k and advise investment potential."
    print("Auto-mode detected. Using default query:")
    print(user_query)
else:
    user_query = input("Describe the property you want:\n")

# Run pipeline
result = await run_property_query(user_query)

print("\nâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Property Advisor Response â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€\n")
print(result)



import pandas as pd
import os

# Automatic fallback output (only used when Kaggle does Run & Save All)
auto_message = "Notebook ran successfully. User input not available during version save."

# If result exists, save actual result.
if "result" in globals():
    output_text = result
    query_text = user_query
else:
    output_text = auto_message
    query_text = "N/A (no input during version save)"

df = pd.DataFrame({
    "user_query": [query_text],
    "advisor_output": [output_text]
})

save_path = "/kaggle/working/submission.csv"
df.to_csv(save_path, index=False)

print("CSV saved to:", save_path)
print("Files in working dir:", os.listdir('/kaggle/working/'))



