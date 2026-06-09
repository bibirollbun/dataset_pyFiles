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


!pip install -q -U google-genai


from google import genai
from kaggle_secrets import UserSecretsClient

# Load secret key from Kaggle Add-ons > Secrets
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)
print("Gemini client initialized!")


from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from google import genai
from kaggle_secrets import UserSecretsClient


# ===============================
# 0. GEMINI CLIENT (Kaggle secret)
# ===============================

user_secrets = UserSecretsClient()
GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)


# ===============================
# 1. Data structures
# ===============================

@dataclass
class TripPreferences:
    origin: Optional[str] = None
    destination: Optional[str] = None
    start_date: Optional[str] = None  # "YYYY-MM-DD"
    end_date: Optional[str] = None    # "YYYY-MM-DD"
    budget_inr: Optional[int] = None
    interests: Optional[List[str]] = None
    traveler_type: Optional[str] = None   # "solo", "couple", "family", "friends"
    num_travelers: int = 1
    trip_mode: str = "balanced"  # NEW: "chill", "explore", "budget", "balanced"

@dataclass
class DayPlan:
    date: str
    morning: str
    afternoon: str
    evening: str

@dataclass
class TripPlan:
    summary: str
    days: List[DayPlan]
    hotel_suggestion: str
    estimated_total_cost_inr: int
    cost_breakdown: Dict[str, int]


# ===============================
# 2. Tool functions (rule-based)
# ===============================

def parse_dates(start_date: str, end_date: str) -> List[str]:
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    days = []
    current = start
    while current <= end:
        days.append(str(current))
        current += timedelta(days=1)
    return days


def suggest_activities(destination: str, interests: List[str]) -> Dict[str, List[str]]:
    base_activities = {
        "history": [
            f"Visit the main historical fort in {destination}",
            f"Explore the old town / heritage walk in {destination}",
            f"Spend time in the city museum of {destination}",
        ],
        "food": [
            f"Street food tour in {destination}",
            f"Try a famous local restaurant in {destination}",
            f"Visit a popular cafe in {destination}",
        ],
        "nature": [
            f"Morning walk in a park / lake in {destination}",
            f"Short hike to a viewpoint near {destination}",
            f"Sunset at a scenic spot in {destination}",
        ],
        "beach": [
            f"Relax at the main beach in {destination}",
            f"Try basic water sports at {destination} beach",
            f"Walk on the beach at sunset in {destination}",
        ],
        "nightlife": [
            f"Explore pubs / bars in {destination}",
            f"Attend a live music venue in {destination}",
            f"Visit a rooftop lounge in {destination}",
        ],
        "shopping": [
            f"Visit a local market / bazaar in {destination}",
            f"Explore a popular mall in {destination}",
            f"Buy local souvenirs in {destination}",
        ],
    }

    chosen: Dict[str, List[str]] = {}
    for interest in interests:
        key = interest.lower().strip()
        if key in base_activities:
            chosen[key] = base_activities[key]

    if not chosen:
        chosen["general"] = [
            f"Walking tour in central {destination}",
            f"Try a popular local restaurant",
            f"Evening at a famous landmark in {destination}",
        ]
    return chosen


def estimate_budget(
    num_days: int,
    origin: str,
    destination: str,
    budget_inr: Optional[int],
    num_travelers: int = 1,
) -> Dict[str, Any]:
    # per person per day
    per_day_food = 800
    per_day_local_travel = 400
    per_day_activities = 600
    hotel_per_room_per_day = 1500

    rooms_needed = max(1, (num_travelers + 1) // 2)

    food_total = per_day_food * num_days * num_travelers
    local_travel_total = per_day_local_travel * num_days * num_travelers
    activities_total = per_day_activities * num_days * num_travelers
    hotel_total = hotel_per_room_per_day * num_days * rooms_needed

    if origin.lower() == destination.lower():
        flight_per_person = 0
    else:
        flight_per_person = 6000  # dummy
    flight = flight_per_person * num_travelers

    total = food_total + local_travel_total + activities_total + hotel_total + flight

    return {
        "flight_inr": flight,
        "hotel_inr": hotel_total,
        "food_inr": food_total,
        "local_travel_inr": local_travel_total,
        "activities_inr": activities_total,
        "estimated_total_inr": total,
        "fits_user_budget": None if budget_inr is None else total <= budget_inr,
    }


def suggest_hotel(destination: str, budget_inr: int, traveler_type: str) -> str:
    if budget_inr < 15000:
        tier = "budget-friendly 2-3 star stay"
    elif budget_inr < 40000:
        tier = "comfortable 3-4 star hotel"
    else:
        tier = "premium 4-5 star hotel or boutique stay"

    if traveler_type in ("solo", "friends"):
        extra = "You can also consider hostels to save money and meet other travellers."
    elif traveler_type == "family":
        extra = "Look for family rooms and kid-friendly facilities like pool or play area."
    else:
        extra = "Pick something with good privacy and ambience."

    return (
        f"For {destination}, a {tier} near the city center would be ideal. "
        f"Look for properties with good reviews (4.2+ rating) and free breakfast. {extra}"
    )


# ========= NEW FEATURE 2: Trip mode logic =========

def apply_trip_mode(
    base_morning: str,
    base_afternoon: str,
    base_evening: str,
    destination: str,
    mode: str,
) -> (str, str, str):
    """Adjust day structure based on trip mode."""
    mode = (mode or "balanced").lower()

    if mode == "chill":
        morning = f"Slow morning, free time near your stay in {destination}"
        afternoon = base_afternoon
        evening = f"Relaxed evening walk / cafe time in {destination}"
    elif mode == "budget":
        morning = f"Free sightseeing walk in {destination} (no ticketed spots)"
        afternoon = base_afternoon + " (pick low-cost or free options)"
        evening = f"Visit a popular but budget-friendly street food area in {destination}"
    elif mode == "explore":
        # keep as much activity as possible
        morning = base_morning
        afternoon = base_afternoon
        evening = base_evening
    else:  # balanced / default
        morning = base_morning
        afternoon = base_afternoon
        evening = base_evening

    return morning, afternoon, evening


def build_itinerary(
    prefs: TripPreferences,
    activities_db: Dict[str, List[str]]
) -> TripPlan:
    date_list = parse_dates(prefs.start_date, prefs.end_date)
    interests = list(activities_db.keys()) or ["general"]

    days: List[DayPlan] = []
    interest_index = 0

    for d in date_list:
        interest_key = interests[interest_index % len(interests)]
        interest_index += 1

        acts = activities_db.get(interest_key, [])
        base_morning = acts[0] if len(acts) > 0 else f"Free time to explore {prefs.destination}"
        base_afternoon = acts[1] if len(acts) > 1 else f"Cafe hopping in {prefs.destination}"
        base_evening = acts[2] if len(acts) > 2 else f"Relax at a viewpoint in {prefs.destination}"

        morning, afternoon, evening = apply_trip_mode(
            base_morning,
            base_afternoon,
            base_evening,
            prefs.destination,
            prefs.trip_mode,
        )

        days.append(
            DayPlan(
                date=d,
                morning=morning,
                afternoon=afternoon,
                evening=evening,
            )
        )

    budget_info = estimate_budget(
        num_days=len(date_list),
        origin=prefs.origin,
        destination=prefs.destination,
        budget_inr=prefs.budget_inr,
        num_travelers=prefs.num_travelers,
    )

    summary_lines = [
        f"Trip from {prefs.origin} to {prefs.destination}",
        f"Travellers: {prefs.num_travelers} ({prefs.traveler_type})",
        f"Mode: {prefs.trip_mode}",
        f"Dates: {prefs.start_date} to {prefs.end_date} ({len(date_list)} days)",
        f"Rough estimated total cost: ~â‚¹{budget_info['estimated_total_inr']}",
    ]
    if prefs.budget_inr is not None:
        if budget_info["fits_user_budget"]:
            summary_lines.append("âœ… This seems to fit within your budget.")
        else:
            summary_lines.append("âš ï¸� This may exceed your budget. Consider reducing trip length or hotel cost.")

    summary = "\n".join(summary_lines)

    hotel_suggestion = suggest_hotel(
        prefs.destination,
        prefs.budget_inr or budget_info['estimated_total_inr'],
        prefs.traveler_type or "solo",
    )

    return TripPlan(
        summary=summary,
        days=days,
        hotel_suggestion=hotel_suggestion,
        estimated_total_cost_inr=budget_info["estimated_total_inr"],
        cost_breakdown={
            "flight_inr": budget_info["flight_inr"],
            "hotel_inr": budget_info["hotel_inr"],
            "food_inr": budget_info["food_inr"],
            "local_travel_inr": budget_info["local_travel_inr"],
            "activities_inr": budget_info["activities_inr"],
        },
    )


def replan_day(trip_plan: TripPlan, day_index: int, new_interest: str, destination: str):
    """Re-plan a single day based on a new interest."""
    activities_db = suggest_activities(destination, [new_interest])
    acts = list(activities_db.values())[0]

    if 0 <= day_index < len(trip_plan.days):
        day = trip_plan.days[day_index]
        if len(acts) > 0:
            day.morning = acts[0]
        if len(acts) > 1:
            day.afternoon = acts[1]
        if len(acts) > 2:
            day.evening = acts[2]


# ========= NEW FEATURE 3: Season advice =========

def season_advice(destination: str, start_date: str) -> str:
    """Very simple rule-based season/weather advice."""
    try:
        dt = datetime.strptime(start_date, "%Y-%m-%d")
        month = dt.month
    except Exception:
        return "Season info unavailable (invalid date)."

    dest_lower = destination.lower()

    if "goa" in dest_lower:
        if 6 <= month <= 8:
            return "Note: This is monsoon season in Goa. Expect heavy rains and limited water sports."
        elif 11 <= month <= 2:
            return "Note: This is peak season in Goa. Great weather but higher prices and crowd."
        else:
            return "Goa has mixed weather this time. Some rain possible, but generally okay for travel."
    if "manali" in dest_lower:
        if month in (12, 1, 2):
            return "Note: Very cold in Manali this time. Expect snow and carry heavy winter clothing."
        elif 7 <= month <= 9:
            return "Note: You may face some rain/landslides in Himachal in monsoon season."
        else:
            return "Pleasant weather in Manali during this time. Good for sightseeing."
    # generic fallback
    if month in (4,5,6):
        return f"Note: It can be quite hot in many parts of India around this time. Stay hydrated in {destination}."
    elif month in (7,8,9):
        return f"Note: Monsoon season in many regions. Check local weather forecasts for {destination}."
    elif month in (10,11,12,1,2,3):
        return f"Note: Generally a good time to travel to {destination}, but always check local conditions before you go."
    return "Season advice not available."


# ===============================
# 3. Agent orchestration
# ===============================

class TravelPlannerAgent:
    """Agent that collects user preferences and uses tools to build a trip plan."""

    def __init__(self):
        self.state = TripPreferences()

    def collect_preferences(
        self,
        origin: str,
        destination: str,
        start_date: str,
        end_date: str,
        budget_inr: Optional[int],
        interests: List[str],
        traveler_type: str = "solo",
        num_travelers: int = 1,
        trip_mode: str = "balanced",
    ):
        self.state.origin = origin.strip()
        self.state.destination = destination.strip()
        self.state.start_date = start_date
        self.state.end_date = end_date
        self.state.budget_inr = budget_inr
        self.state.interests = interests
        self.state.traveler_type = traveler_type.lower().strip()
        self.state.num_travelers = max(1, num_travelers)
        self.state.trip_mode = trip_mode.lower().strip() or "balanced"

    def plan_trip(self) -> TripPlan:
        missing = []
        for field_name, value in asdict(self.state).items():
            if value is None:
                missing.append(field_name)

        if missing:
            raise ValueError(f"Missing fields: {missing}. Please collect all preferences first.")

        activities_db = suggest_activities(self.state.destination, self.state.interests)
        trip_plan = build_itinerary(self.state, activities_db)
        return trip_plan


# ===============================
# 4. Pretty printing
# ===============================

def pretty_print_trip(trip_plan: TripPlan):
    print("\n========== TRIP SUMMARY ==========")
    print(trip_plan.summary)
    print("\n========== HOTEL SUGGESTION ==========")
    print(trip_plan.hotel_suggestion)
    print("\n========== COST BREAKDOWN (INR) ==========")
    for k, v in trip_plan.cost_breakdown.items():
        print(f"{k}: â‚¹{v}")


# ========= NEW FEATURE 1: Budget Optimizer Tool =========

def optimize_budget(prefs: TripPreferences, trip_plan: TripPlan) -> Dict[str, Any]:
    """
    Suggests ways to move closer to user budget.
    Does NOT change the plan, only returns scenarios.
    """
    num_days = len(trip_plan.days)
    base = estimate_budget(
        num_days=num_days,
        origin=prefs.origin,
        destination=prefs.destination,
        budget_inr=prefs.budget_inr,
        num_travelers=prefs.num_travelers,
    )

    user_budget = prefs.budget_inr
    if user_budget is None:
        return {
            "has_budget": False,
            "message": "No user budget provided, so no optimization needed.",
            "scenarios": [],
        }

    base_total = base["estimated_total_inr"]
    scenarios = []

    # Scenario 1: Cheaper hotel (30% cut on hotel cost)
    cheaper_hotel_total = base_total - int(base["hotel_inr"] * 0.3)
    scenarios.append({
        "label": "Cheaper stay",
        "estimated_total": cheaper_hotel_total,
        "description": "Use budget hotels/hostels instead of mid-range stays (approx. 30% cheaper on stay)."
    })

    # Scenario 2: One day shorter
    if num_days > 2:
        reduced_total = int(base_total * (num_days - 1) / num_days)
        scenarios.append({
            "label": "Shorter trip by 1 day",
            "estimated_total": reduced_total,
            "description": "Reduce the trip by one day (drop the last day) to save on all daily costs."
        })

    # Scenario 3: Lower activities spend (40% cut)
    cheaper_activities_total = base_total - int(base["activities_inr"] * 0.4)
    scenarios.append({
        "label": "Fewer paid activities",
        "estimated_total": cheaper_activities_total,
        "description": "Replace some paid activities with free sightseeing and walks."
    })

    # Mark which scenarios fit the budget
    for sc in scenarios:
        sc["fits_budget"] = (sc["estimated_total"] <= user_budget)

    return {
        "has_budget": True,
        "user_budget": user_budget,
        "base_total": base_total,
        "scenarios": scenarios,
    }


# ===============================
# 5. Gemini helpers & features
# ===============================

def generate_travel_guide_with_gemini(trip_plan: TripPlan) -> str:
    """Simple Gemini guide based on summary + cost."""
    summary_text = trip_plan.summary.replace("\n", " | ")
    total_cost = trip_plan.estimated_total_cost_inr
    cost_text = ", ".join(f"{k}: â‚¹{v}" for k, v in trip_plan.cost_breakdown.items())

    prompt = f"""
You are a concise travel assistant.

Trip summary: {summary_text}
Total estimated cost: â‚¹{total_cost}
Cost breakdown: {cost_text}

Write a short 80â€“120 word travel advice:
- 2 lines: overall opinion about the trip (value, vibe)
- 3 bullet tips: saving budget, booking, packing, safety
Keep it simple and to the point.
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=prompt
    )

    text = getattr(response, "text", None)
    if not text:
        return "âš ï¸� Gemini did not return text. Raw response:\n" + repr(response)

    return text.strip()


def chat_about_trip_with_gemini(
    trip_plan: TripPlan,
    user_message: str,
    history: List[Dict[str, str]]
) -> str:
    """Simple chat based on summary and total cost."""
    summary_text = trip_plan.summary.split("\n")[0]
    total_cost = trip_plan.estimated_total_cost_inr

    prompt = f"""
You are a travel assistant.

Trip summary: {summary_text}
Total estimated cost: â‚¹{total_cost}

User question: {user_message}

Rules:
- Answer in less than 80 words.
- Only talk about this trip (budget, days, activities, etc.)
- Be direct and practical.
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=prompt
    )

    text = getattr(response, "text", None)
    if not text:
        return "âš ï¸� Gemini couldn't answer. Raw response:\n" + repr(response)

    return text.strip()


# ========= NEW FEATURE 4: What-if Gemini Analyzer =========

def gemini_what_if_extend_trip(trip_plan: TripPlan, extra_days: int, extra_budget: int) -> str:
    """
    Ask Gemini: if we add extra_days and extra_budget,
    how would you improve/enhance the trip (high-level)?
    """
    summary_text = trip_plan.summary.replace("\n", " | ")
    total_cost = trip_plan.estimated_total_cost_inr

    prompt = f"""
You are a travel planning expert.

Current trip:
- {summary_text}
- Current total cost: â‚¹{total_cost}

What-if scenario:
- User can add {extra_days} more day(s).
- User can add extra budget of â‚¹{extra_budget}.

In 4â€“6 bullet points, suggest how to enhance the trip:
- New experiences to add
- Upgrades (stay/activities)
- Any changes in pacing

Keep it short and practical.
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=prompt
    )
    text = getattr(response, "text", None)
    if not text:
        return "âš ï¸� Gemini couldn't generate what-if analysis."

    return text.strip()


# ===============================
# 6. NON-INTERACTIVE DEMO (no input)
# ===============================

def run_auto_demo():
    """
    Auto demo for Kaggle:
    - No input() calls
    - Uses fixed example preferences
    - Shows: trip summary, season advice,
      budget optimization, Gemini guide, chat, and what-if analysis.
    """

    print("ğŸŒ� Welcome to AI Trip Planner Agent")
    print("------------------------------------------------\n")

    # ğŸ‘‰ Yahin par values change karke tum alag-alag trips test kar sakte ho
    agent = TravelPlannerAgent()
    agent.collect_preferences(
        origin="Patna",
        destination="Goa",
        start_date="2025-07-10",
        end_date="2025-07-15",
        budget_inr=40000,
        interests=["beach", "food", "nightlife"],
        traveler_type="friends",
        num_travelers=2,
        trip_mode="chill",   # try "chill", "explore", "budget", "balanced"
    )

    # 1) Build trip plan
    trip = agent.plan_trip()
    pretty_print_trip(trip)

    # 2) Season advice
    print("\n========== SEASON ADVICE ==========\n")
    print(season_advice(agent.state.destination, agent.state.start_date))

    # 3) Budget optimization suggestions
    print("\n========== BUDGET OPTIMIZER ==========\n")
    opt = optimize_budget(agent.state, trip)
    if not opt["has_budget"]:
        print(opt["message"])
    else:
        print(f"User budget: â‚¹{opt['user_budget']}")
        print(f"Baseline estimate: â‚¹{opt['base_total']}")
        print("\nScenarios:")
        for sc in opt["scenarios"]:
            status = "âœ… fits budget" if sc["fits_budget"] else "âš  above budget"
            print(f"- {sc['label']}: ~â‚¹{sc['estimated_total']} ({status})")
            print(f"  {sc['description']}")

    # 4) Gemini travel guide (auto)
    print("\n========== AI TRAVEL GUIDE (Gemini) ==========\n")
    guide = generate_travel_guide_with_gemini(trip)
    print(guide)

    # 5) Auto Q&A example with Gemini
    print("\n========== SAMPLE GEMINI Q&A ==========\n")
    sample_question = "How can I make this trip more budget friendly without losing the main experiences?"
    print("User:", sample_question)
    answer = chat_about_trip_with_gemini(trip, sample_question, history=[])
    print("\nAgent:", answer)

    # 6) What-if analysis
    print("\n========== WHAT-IF: EXTEND TRIP ==========\n")
    what_if_text = gemini_what_if_extend_trip(trip_plan=trip, extra_days=2, extra_budget=15000)
    print(what_if_text)


# ===============================
# 7. Entry point
# ===============================

if __name__ == "__main__":
    # Always run auto demo (no input, Kaggle-safe)
    run_auto_demo()


