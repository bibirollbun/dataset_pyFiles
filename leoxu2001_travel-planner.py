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

user_secrets = UserSecretsClient()
GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

@dataclass
class TripPreferences:
    origin: Optional[str] = None
    destination: Optional[str] = None
    start_date: Optional[str] = None  # "YYYY-MM-DD"
    end_date: Optional[str] = None    # "YYYY-MM-DD"
    interests: Optional[List[str]] = None

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
        "city": [
            f"Explore pubs / bars in {destination}",
            f"Attend a live music venue in {destination}",
            f"Visit a rooftop lounge in {destination}",
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

    summary_lines = [
        f"Trip from {prefs.origin} to {prefs.destination}",
        f"Dates: {prefs.start_date} to {prefs.end_date} ({len(date_list)} days)",
        
    summary = "\n".join(summary_lines)

    return TripPlan(
        summary=summary,
        days=days
    )

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
    ):
        self.state.origin = origin.strip()
        self.state.destination = destination.strip()
        self.state.start_date = start_date
        self.state.end_date = end_date

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

def pretty_print_trip(trip_plan: TripPlan):
    print("\n========== TRIP SUMMARY ==========")
    print(trip_plan.summary)

def generate_travel_guide_with_gemini(trip_plan: TripPlan) -> str:
    """A travel guide"""
    summary_text = trip_plan.summary.replace("\n", " | ")

    prompt = f"""
You are a concise travel assistant.

Trip summary: {summary_text}
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=prompt
    )

    text = getattr(response, "text", None)
    if not text:
        return "Response is not text. Raw response:\n" + repr(response)

    return text.strip()


def chat_about_trip_with_gemini(
    trip_plan: TripPlan,
    user_message: str,
    history: List[Dict[str, str]]
) -> str:
    """Chatting about the trip"""
    summary_text = trip_plan.summary.split("\n")[0]

    prompt = f"""
You are a travel assistant.

Trip summary: {summary_text}

User question: {user_message}
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=prompt
    )

    text = getattr(response, "text", None)
    if not text:
        return "Response is not text. Raw response:\n" + repr(response)

    return text.strip()

def run_demo():
    #Demo Run

    print("Travel Planner")
    print("----------------------------------------------\n")

    
    agent = TravelPlannerAgent()
    agent.collect_preferences(
        origin="New York",
        destination="Los Angeles",
        start_date="2026-07-10",
        end_date="2026-07-15",
        interests=["nature", "beach", "city"]
    )
    trip = agent.plan_trip()
    pretty_print_trip(trip)
    
    print("\n========== TRAVEL GUIDE ==========\n")
    guide = generate_travel_guide_with_gemini(trip)
    print(guide)

if __name__ == "__main__":
    run_demo()

