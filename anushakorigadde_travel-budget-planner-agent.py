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


!pip install pandas pillow fpdf openpyxl --quiet



import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, List
import pandas as pd
import math, random, datetime
from pathlib import Path



# Simple realistic-looking defaults (per traveler unless noted)
DEST_TIER = {
    "lowcost": {"flight_base": 200, "hotel_per_night": 40, "food_per_day": 15, "local_transport_per_day": 5},
    "moderate": {"flight_base": 500, "hotel_per_night": 100, "food_per_day": 30, "local_transport_per_day": 15},
    "expensive": {"flight_base": 1200, "hotel_per_night": 220, "food_per_day": 60, "local_transport_per_day": 40}
}

ACTIVITY_ESTIMATE_PER_DAY = {"low": 10, "moderate": 40, "high": 120}
TAX_INSURANCE_PERCENT = 0.08   # approx
TIP_PERCENT = 0.10



@dataclass
class TripInfo:
    destination: str
    days: int
    travel_style: str  # 'lowcost', 'moderate', 'expensive'
    num_travelers: int = 1
    depart_date: str = None

SESSION: Dict[str, Any] = {}
def save_session(key, value): SESSION[key] = value
def load_session(key, default=None): return SESSION.get(key, default)



def collect_trip_info():
    info = TripInfo(
        destination="Lisbon, Portugal",
        days=6,
        travel_style="moderate",
        num_travelers=2,
        depart_date="2026-03-15"
    )
    assert info.days > 0, "days must be > 0"
    assert info.travel_style in DEST_TIER, f"travel_style must be one of {list(DEST_TIER.keys())}"
    save_session("trip_info", info)
    return info

trip = collect_trip_info()
trip



def estimate_flight_and_hotel(trip: TripInfo):
    tier = DEST_TIER[trip.travel_style]
    flight_variation = 0.9 + random.uniform(-0.15, 0.15)
    flight_cost = tier["flight_base"] * flight_variation * trip.num_travelers

    rooms = math.ceil(trip.num_travelers / 2)
    hotel_cost = tier["hotel_per_night"] * trip.days * rooms

    return {"flight_cost": round(flight_cost, 2), "hotel_cost": round(hotel_cost, 2)}

estimate_flight_and_hotel(trip)



def estimate_daily_expenses(trip: TripInfo, activity_level="moderate"):
    tier = DEST_TIER[trip.travel_style]
    food = tier["food_per_day"] * trip.days * trip.num_travelers
    transport = tier["local_transport_per_day"] * trip.days * trip.num_travelers
    activities = ACTIVITY_ESTIMATE_PER_DAY[activity_level] * trip.days * trip.num_travelers

    return {
        "food_cost": round(food,2),
        "transport_cost": round(transport,2),
        "activities_cost": round(activities,2)
    }

estimate_daily_expenses(trip, activity_level="moderate")



def suggest_savings(trip: TripInfo, breakdown: Dict[str, float]):
    suggestions = []
    if trip.travel_style == "expensive":
        suggestions.append("Consider shifting to 'moderate' style for ~20-40% savings.")

    suggestions.append("Check flights on Tue/Wed â€” often cheaper mid-week.")

    avg_hotel_per_night = breakdown["hotel_cost"] / trip.days
    if avg_hotel_per_night > 150:
        suggestions.append("Consider 3-star hotels or apartment rentals to reduce nightly rate.")

    suggestions.append("Look for free walking tours and city passes to cut activity costs.")
    return suggestions



def aggregate_budget(trip: TripInfo, activity_level="moderate"):
    fh = estimate_flight_and_hotel(trip)
    daily = estimate_daily_expenses(trip, activity_level)
    subtotal = fh["flight_cost"] + fh["hotel_cost"] + daily["food_cost"] + daily["transport_cost"] + daily["activities_cost"]

    fees = subtotal * TAX_INSURANCE_PERCENT
    tips = daily["food_cost"] * TIP_PERCENT
    total = subtotal + fees + tips

    breakdown = {
        **fh, **daily,
        "subtotal": round(subtotal,2),
        "taxes_insurance_est": round(fees,2),
        "tips_est": round(tips,2),
        "total_estimated_cost": round(total,2)
    }
    suggestions = suggest_savings(trip, breakdown)
    return breakdown, suggestions

breakdown, suggestions = aggregate_budget(trip)

def pretty_report(trip: TripInfo, breakdown: Dict[str,float], suggestions: List[str]):
    print("=== Travel Budget Report ===")
    print(f"Destination: {trip.destination}")
    print(f"Dates / Days: {trip.depart_date} / {trip.days} days")
    print(f"Travelers: {trip.num_travelers}")
    print("\nBreakdown:")
    for k,v in breakdown.items():
        print(f"  {k}: ${v}")
    print("\nSuggestions:")
    for s in suggestions:
        print(" -", s)

def export_to_csv(trip, breakdown, suggestions, filename="budget_report.csv"):
    df = pd.DataFrame([breakdown])
    for k,v in asdict(trip).items(): df[k] = v
    df["suggestions"] = "; ".join(suggestions)
    df.to_csv(filename, index=False)
    print("Saved:", filename)

pretty_report(trip, breakdown, suggestions)
export_to_csv(trip, breakdown, suggestions)


