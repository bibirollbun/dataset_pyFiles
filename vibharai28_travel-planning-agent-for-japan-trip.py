# CELL: Install ADK
!pip install --upgrade google-adk --quiet
print("pip install finished. Now verify installation in the next cell.")



# CELL: Verify google-adk is installed and show candidate modules
import pkgutil, sys, subprocess
print("Python:", sys.version.splitlines()[0])

# Show package info
try:
    import google.adk as adk
    print("Imported google.adk:", adk)
except Exception as e:
    print("Import google.adk failed:", repr(e))

# Show google.adk top-level submodules present on disk
mods = [m.name for m in pkgutil.iter_modules() if 'adk' in m.name or 'google' in m.name][:200]
print("Some installed modules containing 'adk' or 'google' (partial):", mods)

# Show pip metadata for google-adk
!pip show google-adk || true



# CELL: Confirm GoogleApiToolset path
try:
    from google.adk.toolsets.google_api import GoogleApiToolset
    print("âœ… google.adk.toolsets.google_api.GoogleApiToolset is available.")
    gt = GoogleApiToolset()
    print("GoogleApiToolset instance created. Methods:", [m for m in dir(gt) if 'search' in m or 'web' in m][:20])
except Exception as e:
    print("GoogleApiToolset import failed:", repr(e))
    print("If this fails, paste the output of the verification cell here and I'll pick the right import path.")



import pkgutil
import google.adk as adk
print([m.name for m in pkgutil.iter_modules(adk.__path__)])



import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


# === NEXT STEP: Initialize Gemini ===
import google.generativeai as genai
import os

if "GOOGLE_API_KEY" not in os.environ:
    raise RuntimeError("Please set GOOGLE_API_KEY in environment before running.")

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# A simple helper to call Gemini in any environment
def call_llm(prompt, model="gemini-2.0-flash"):
    response = genai.GenerativeModel(model).generate_content(prompt)
    return response.text



# === Research Agent ===
def research_agent(destination, dates, prefs):
    prompt = f"""
    Research the following trip:

    Destination: {destination}
    Start Date: {dates['start']}
    End Date: {dates['end']}
    Interests: {prefs.get("interests", [])}

    Provide:

    1. Top 5 attractions relevant to interests
    2. Weather during dates
    3. Local transportation options
    4. Cultural & safety tips
    5. Cost hints

    Write in clear sections.
    """

    return call_llm(prompt)



# === Itinerary Agent ===
def itinerary_agent(research_output, nights, prefs):
    prompt = f"""
    Create a {nights}-day itinerary based on this research:

    {research_output}

    Travel style: {prefs.get('travel_style','balanced')}

    Include:
    - Morning / afternoon / evening blocks
    - Travel times
    - Backup options for weather
    - Rest hours

    Format as Day 1, Day 2, ...
    """

    return call_llm(prompt)



# === Budget Agent ===
def budget_agent(itinerary_output, nights, budget_limit):
    prompt = f"""
    Create an itemized travel budget.

    Budget limit: {budget_limit}
    Nights: {nights}

    Itinerary:
    {itinerary_output}

    Include:
    - Flights
    - Accommodation (per night + total)
    - Food (daily)
    - Activities
    - Transport
    - Emergency buffer (10%)

    If total exceeds budget, propose reductions.
    """

    return call_llm(prompt)



# === Orchestrator ===
def plan_trip(user_id, trip):
    destination = trip["destination"]
    dates = {"start": trip["start_date"], "end": trip["end_date"]}
    prefs = trip.get("preferences", {})
    nights = trip.get("nights", 3)
    budget_limit = trip.get("budget", 2500)

    print("Step 1: Researching destination...")
    research = research_agent(destination, dates, prefs)

    print("Step 2: Creating itinerary...")
    itinerary = itinerary_agent(research, nights, prefs)

    print("Step 3: Calculating budget...")
    budget = budget_agent(itinerary, nights, budget_limit)

    final = {
        "user_id": user_id,
        "destination": destination,
        "dates": dates,
        "research": research,
        "itinerary": itinerary,
        "budget": budget
    }

    print("âœ” Trip planning complete!")
    return final



example = {
    "destination": "Tokyo, Japan",
    "start_date": "2025-09-01",
    "end_date": "2025-09-07",
    "nights": 6,
    "budget": 3000,
    "preferences": {
        "interests": ["food", "culture", "technology"],
        "travel_style": "balanced"
    }
}

plan = plan_trip("demo_user", example)

print("\n=== FINAL TRIP PLAN ===\n")
print(plan["research"])
print("\n---\n")
print(plan["itinerary"])
print("\n---\n")
print(plan["budget"])


