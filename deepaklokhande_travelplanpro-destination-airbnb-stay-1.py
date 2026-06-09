import os
import asyncio
from typing import Any, Dict, List, Optional

from kaggle_secrets import UserSecretsClient

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.google_search_tool import google_search
from google.adk.runners import InMemoryRunner
from google.genai import types


print("Step 1: Loading authentication...")

# ============================================================
# 1. Authentication (Kaggle -> env var)
# ============================================================
GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
print("Authentication loaded.")


# ============================================================
# 2. Retry configuration
# ============================================================
print("Step 2: Configuring retry options...")
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)


# ============================================================
# 3. Python tools (ADK safe)
# ============================================================
print("Step 3: Creating Python tools...")


def filter_airbnbs(
    listings: List[Dict[str, Any]],
    max_price_per_night: float = 999_999.0,
    min_rating: float = 4.0,
    max_results: int = 10,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Filter Airbnb-style listings by price, rating, and limit result count.

    Args:
        listings: List of listing dicts. Each listing should contain
                  "price_per_night" or "price", and optionally "rating".
        max_price_per_night: Maximum allowed price per night (float).
        min_rating: Minimum rating required (float).
        max_results: Max number of listings to return (int).

    Returns:
        A dict with a single key:
          {
            "filtered_listings": [ ...listings... ]
          }
    """
    filtered: List[Dict[str, Any]] = []

    for listing in listings:
        if not isinstance(listing, dict):
            continue

        try:
            price = float(
                listing.get("price_per_night", listing.get("price", 0.0))
            )
            rating = float(listing.get("rating", 0.0))
        except Exception:
            # Skip malformed listings
            continue

        if price <= max_price_per_night and rating >= min_rating:
            filtered.append(listing)

    return {"filtered_listings": filtered[:max_results]}


def validate_dates(
    check_in: str,
    check_out: str,
    travelers: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Validate check-in/check-out dates (as strings) and optional traveler count.

    Args:
        check_in: Check-in date as a string (e.g., "2025-12-24").
        check_out: Check-out date as a string (e.g., "2025-12-28").
        travelers: Optional number of travelers (int).

    Returns:
        JSON-serializable dict with validation status and echo of inputs.
    """
    if not check_in or not check_out:
        return {
            "status": "error",
            "message": "Both check_in and check_out must be provided.",
        }

    return {
        "status": "ok",
        "check_in": check_in,
        "check_out": check_out,
        "travelers": travelers,
        "message": "Dates validated.",
    }


# ============================================================
# 4. Sub agents
# ============================================================
print("Step 4: Creating sub-agents...")

airbnb_search_agent = LlmAgent(
    name="airbnb_search_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="Searches Airbnb-style stays using web search.",
    instruction="""
Use google_search to find Airbnb-style listings for the given destination and dates.
Return a structured JSON list of listings, where each listing includes:
- name/title
- location
- price_per_night (number, in USD if possible)
- rating (0–5)
- url
- description (short)
""",
    tools=[google_search],
)

weather_agent = LlmAgent(
    name="weather_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="Gets weather information.",
    instruction="""
Use google_search to get weather forecasts for the specific destination and trip dates.
Return concise structured JSON, for example:
{
  "destination": "...",
  "daily_forecast": [
    {"date": "...", "summary": "...", "high": 00, "low": 00}
  ]
}
""",
    tools=[google_search],
)

calendar_itinerary_agent = LlmAgent(
    name="calendar_itinerary_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="Validates dates and generates itinerary.",
    instruction="""
First, call the validate_dates tool with:
- check_in (string)
- check_out (string)
- travelers (integer if available)

If validate_dates.status == "ok", then build a short day-by-day itinerary that fits within the provided dates.
Return a structured JSON itinerary plus a friendly natural-language summary.
""",
    tools=[validate_dates],
)


# ============================================================
# 5. Root multi-agent planner
# ============================================================
print("Step 5: Creating root travel planner agent...")

travel_planner_agent = LlmAgent(
    name="travel_planner_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="Multi-agent travel planner.",
    instruction="""
You are a multi-agent travel planner.

High-level steps:
1. Extract from the user message:
   - destination
   - check-in and check-out dates
   - number of people (travelers)
   - budget range
   - preferences (e.g. beach view)

2. Call airbnb_search_agent to get a list of possible stays.

3. Call filter_airbnbs with the following arguments:
   - listings: the list of results from airbnb_search_agent
   - max_price_per_night: a numeric value inferred from the user's budget
   - min_rating: typically 4.0
   - max_results: around 10

4. Call weather_agent to get weather info for the destination and dates.

5. Call calendar_itinerary_agent with:
   - check_in: the parsed check-in date string
   - check_out: the parsed check-out date string
   - travelers: the number of people traveling

6. Combine:
   - the best filtered stay options (especially beach view if requested),
   - the weather summary,
   - and the generated itinerary,
   into a final nicely formatted travel plan.

The final answer should be:
- Easy to read (use headings and bullet points where helpful).
- Include 2–5 recommended stays with key details.
- Include a short weather overview.
- Include a simple day-by-day itinerary.
""",
    tools=[
        AgentTool(agent=airbnb_search_agent),
        AgentTool(agent=weather_agent),
        AgentTool(agent=calendar_itinerary_agent),
        filter_airbnbs,
    ],
)


# ============================================================
# 6. Runner
# ============================================================
print("Step 6: Initializing runner...")
runner = InMemoryRunner(agent=travel_planner_agent)
print("Runner initialized.")


# ============================================================
# 7. Output extraction helper
# ============================================================
def extract_output(result: Any) -> str:
    if hasattr(result, "final_output") and result.final_output:
        return result.final_output
    if hasattr(result, "output_text") and result.output_text:
        return result.output_text
    if hasattr(result, "result") and result.result:
        return result.result
    return str(result)


# ============================================================
# 8. CLI entry point
# ============================================================
async def plan_trip_cli() -> None:
    print("Step 7: Travel planner started.")
    print("Enter your trip details below.")
    text = input("Describe your trip: ")

    print("Step 8: Processing your request...")

    result = await runner.run_debug(text, quiet=True)
    final = extract_output(result)

    print("\n----- TRAVEL PLAN -----\n")
    print(final)
    print("\n------------------------\n")


# Automatically run CLI after setup finishes (Kaggle supports top-level await)
await plan_trip_cli()

# Example input when prompted:
# Plan a Galveston trip for 3 people from Dec 24–28, 2025, budget 3000–5000 USD, prefer beach view.


