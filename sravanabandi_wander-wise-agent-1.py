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


"""
wander_wise_agents.py

Self-contained multi-agent style system for travel outputs.
Author: Sravana Bandi (example)
Run: python wander_wise_agents.py
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import random
import math
import datetime

# ----------------------
# Data models
# ----------------------
@dataclass
class TripRequest:
    user_name: str
    destination: str
    start_date: datetime.date
    nights: int
    travelers: int
    budget: float  # total budget in INR (for example)
    preferences: Dict[str, Any]  # e.g., {"food": "veg", "comfort": "mid", "avoid": ["heights"]}

# ----------------------
# Utility helpers
# ----------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# ----------------------
# Agents
# ----------------------

class BudgetPlannerAgent:
    """
    Computes a sample budget breakdown for the trip.
    Strategy: split budget across categories using heuristics based on nights and travelers.
    """
    def plan_budget(self, req: TripRequest) -> Dict[str, float]:
        base = req.budget
        nights = req.nights
        travelers = req.travelers

        # heuristics
        accommodation_share = 0.36
        food_share = 0.18
        local_transport_share = 0.08
        activities_share = 0.25
        shopping_share = 0.10
        # tweak by comfort preference
        comfort = req.preferences.get("comfort", "mid")
        if comfort == "luxury":
            accommodation_share += 0.12
            activities_share -= 0.05
        elif comfort == "budget":
            accommodation_share -= 0.12
            activities_share += 0.03

        # calculate
        breakdown = {
            "Accommodation": round(base * accommodation_share),
            "Food": round(base * food_share),
            "Local Transport": round(base * local_transport_share),
            "Activities / Tickets": round(base * activities_share),
            "Shopping / Extra": round(base * shopping_share)
        }

        # Small correction so sums to total
        allocated = sum(breakdown.values())
        diff = round(base) - allocated
        if diff != 0:
            breakdown["Accommodation"] += diff  # adjust accommodation to match total

        # per-person & per-night estimates
        breakdown_meta = {
            "total_budget": base,
            "per_person_per_night_estimate": round(base / max(1, travelers * nights), 2),
            "breakdown": breakdown
        }
        return breakdown_meta


class WeatherHelperAgent:
    """
    Returns mock weather forecasts + packing suggestions.
    In a production system, hook this up to a weather API (OpenWeatherMap, etc).
    """

    def get_forecast_and_packing(self, req: TripRequest) -> List[Dict[str, str]]:
        # We'll produce one entry per day
        out = []
        for d in range(req.nights):
            date = req.start_date + datetime.timedelta(days=d)
            # mock conditions
            temp_c = random.randint(18, 36)  # sample
            cond = random.choice(["Sunny", "Partly Cloudy", "Rainy", "Windy"])
            suggestions = []
            if "Rainy" in cond or temp_c < 20:
                suggestions.append("Carry umbrella / light rain jacket")
            if temp_c >= 30:
                suggestions.append("Sunscreen, hat, water bottle")
            if temp_c < 18:
                suggestions.append("Light sweater or jacket")
            out.append({
                "date": date.isoformat(),
                "temp_c": f"{temp_c}°C",
                "condition": cond,
                "packing_suggestions": "; ".join(suggestions) if suggestions else "Standard light clothing"
            })
        return out


class LocalFoodAgent:
    """
    Returns must-try foods and recommended local eateries (mocked).
    In production this could query Yelp/Google Places.
    """

    _local_food_db = {
        "Jaipur": {
            "must_try": ["Dal Bati Churma", "Ghevar", "Pyaaz Kachori"],
            "spots": [
                {"name": "LMB", "type": "Restaurant", "why": "Iconic sweets & Rajasthani thali"},
                {"name": "Rawat Mishtan Bhandar", "type": "Street/Shop", "why": "Known for pyaaz kachori"}
            ]
        },
        "Goa": {
            "must_try": ["Prawn Curry", "Bebinca", "Goan Sausages"],
            "spots": [{"name": "Fisherman's Wharf", "type": "Restaurant", "why": "Seafood location"}]
        }
    }

    def get_food_recommendations(self, destination: str, pref: Dict[str, Any]) -> Dict[str, Any]:
        entry = self._local_food_db.get(destination, None)
        if not entry:
            # fallback generic
            generic = {
                "must_try": ["Local street snack", "Regional curry/dish", "Local dessert"],
                "spots": [{"name": "Top Rated Local Spot", "type": "Restaurant", "why": "Popular with locals"}]
            }
            return generic
        # filter/sort by preference - simplistic
        return entry


class HiddenGemsAgent:
    """
    Returns lesser-known spots / viewpoints / local experiences.
    """

    _hidden_db = {
        "Jaipur": [
            {"name": "Panna Meena ka Kund", "notes": "Stepwell with photogenic symmetry"},
            {"name": "Jawahar Circle Garden", "notes": "Local park, good for sunrise/sunset"},
            {"name": "Nahargarh Sunset Point", "notes": "Great sunset view above the city"}
        ],
        "Goa": [
            {"name": "Arambol Viewpoint", "notes": "Quiet beach and cliff viewpoint"}
        ]
    }

    def get_hidden_spots(self, destination: str) -> List[Dict[str, str]]:
        return self._hidden_db.get(destination, [{"name": "Local hidden gem", "notes": "Ask locals!"}])


class SafetyAdvisorAgent:
    """
    Provides a safety overview and emergency information.
    In production, this could rely on local crime stats and official advisories.
    """

    _general_advice = {
        "India": {
            "police": "100",
            "ambulance": "108",
            "tips": [
                "Use official/taxi apps for rides",
                "Avoid isolated areas at night",
                "Keep photocopy of passport/ID"
            ]
        }
    }

    def compute_safety_score(self, destination: str) -> Dict[str, Any]:
        # Mocked safety score: 0-100
        base = 70
        # small randomness to simulate place-specific nuance
        score = clamp(base + random.randint(-10, 10), 40, 95)
        category = "Safe" if score >= 75 else ("Moderate" if score >= 55 else "Caution")
        return {
            "destination": destination,
            "safety_score": score,
            "category": category,
            "advice": self._general_advice.get("India")
        }

    def nearest_emergency(self, hotel_location: str = None) -> Dict[str, str]:
        # Mock nearest hospital
        return {
            "nearest_hospital": "Manipal Hospital (approx 2.1 km)",
            "police_number": "100",
            "ambulance_number": "108"
        }


class PriceComparisonAgent:
    """
    Mocks price comparisons for hotels/flights and suggests cheaper options.
    Real version would query multiple APIs and return real-time price differences.
    """

    def compare_hotels(self, destination: str, nights: int, travelers: int) -> List[Dict[str, Any]]:
        # Mock a few options with prices
        # price per night per room
        sample_options = [
            {"name": f"{destination} Comfort Hotel", "price_per_night": 3500, "platform": "Booking"},
            {"name": f"{destination} Heritage Stay", "price_per_night": 4500, "platform": "Agoda"},
            {"name": f"{destination} Budget Inn", "price_per_night": 1800, "platform": "MakeMyTrip"}
        ]
        # total cost estimate for 1 room (assuming 1 room accommodates travelers/group)
        for opt in sample_options:
            opt["total_cost_for_stay"] = opt["price_per_night"] * nights
        # sort by total cost
        sample_options.sort(key=lambda x: x["total_cost_for_stay"])
        return sample_options

    def compare_flights(self, origin: str, destination: str, date: datetime.date) -> List[Dict[str, Any]]:
        # mock flight options
        options = [
            {"airline": "AirFast", "price": 4500, "stops": 0, "departure": "07:00"},
            {"airline": "FlyHigh", "price": 3800, "stops": 1, "departure": "13:20"},
            {"airline": "BudgetAir", "price": 3200, "stops": 2, "departure": "23:30"},
        ]
        options.sort(key=lambda o: o["price"])
        return options


# ----------------------
# Orchestrator (coordinates agents)
# ----------------------
class OrchestratorAgent:
    def __init__(self):
        self.budget = BudgetPlannerAgent()
        self.weather = WeatherHelperAgent()
        self.food = LocalFoodAgent()
        self.hidden = HiddenGemsAgent()
        self.safety = SafetyAdvisorAgent()
        self.price = PriceComparisonAgent()

    def build_full_report(self, req: TripRequest) -> Dict[str, Any]:
        report = {}
        report["request_summary"] = asdict(req)
        report["budget"] = self.budget.plan_budget(req)
        report["forecast"] = self.weather.get_forecast_and_packing(req)
        report["food"] = self.food.get_food_recommendations(req.destination, req.preferences)
        report["hidden_gems"] = self.hidden.get_hidden_spots(req.destination)
        report["safety"] = self.safety.compute_safety_score(req.destination)
        report["emergency_contacts"] = self.safety.nearest_emergency()
        report["hotel_comparisons"] = self.price.compare_hotels(req.destination, req.nights, req.travelers)
        report["flight_comparisons"] = self.price.compare_flights("OriginCity", req.destination, req.start_date)
        report["itinerary_tips"] = self._generate_itinerary_tips(req.destination)
        return report

    def _generate_itinerary_tips(self, destination: str) -> List[str]:
        # Simple rules of thumb for best time slots for common attractions
        tips = [
            f"For popular landmarks in {destination}, visit early morning (7-10 AM) to avoid crowds.",
            "Aim to group nearby attractions on the same day to reduce travel time.",
            "Check local event calendars — weekends may be crowded or closed depending on the site."
        ]
        # add a couple destination-specific tips
        if destination.lower() == "jaipur":
            tips.append("Amber Fort: Visit at opening time to avoid heat & crowds. Sunset points at Nahargarh are recommended.")
        return tips

    def print_user_friendly(self, report: Dict[str, Any]):
        # Nicely formatted console output for tourists
        print("\n===== Wander Wise — Trip Report =====\n")
        req = report["request_summary"]
        print(f"Traveler: {req['user_name']}")
        print(f"Destination: {req['destination']}")
        print(f"Dates: {req['start_date']}  — Nights: {req['nights']}")
        print(f"Travelers: {req['travelers']}  | Budget: ₹{req['budget']}\n")

        print("---- Budget Breakdown ----")
        b = report["budget"]
        for k, v in b["breakdown"].items():
            print(f"  {k}: ₹{v}")
        print(f"  Per person / per night estimate: ₹{b['per_person_per_night_estimate']}\n")

        print("---- Weather & Packing Suggestions (by day) ----")
        for day in report["forecast"]:
            print(f"  {day['date']}: {day['temp_c']} — {day['condition']}")
            print(f"    Packing: {day['packing_suggestions']}")
        print()

        print("---- Local Food Recommendations ----")
        print("  Must try:", ", ".join(report["food"]["must_try"]))
        for spot in report["food"]["spots"]:
            print(f"  - {spot['name']} ({spot['type']}): {spot['why']}")
        print()

        print("---- Hidden Gems ----")
        for gem in report["hidden_gems"]:
            print(f"  - {gem['name']}: {gem.get('notes','')}")
        print()

        print("---- Safety Overview ----")
        s = report["safety"]
        print(f"  Safety Score: {s['safety_score']} ({s['category']})")
        print("  Tips:")
        for t in s["advice"]["tips"]:
            print(f"   - {t}")
        print(f"  Emergency: Police {report['emergency_contacts']['police_number']}, Ambulance {report['emergency_contacts']['ambulance_number']}")
        print()

        print("---- Price Comparisons (Hotels) ----")
        for opt in report["hotel_comparisons"]:
            print(f"  {opt['name']} on {opt['platform']}: ₹{opt['price_per_night']}/night — total ₹{opt['total_cost_for_stay']}")
        cheapest = report["hotel_comparisons"][0]
        print(f"  Suggested Save: Book '{cheapest['name']}' to save money.\n")

        print("---- Price Comparisons (Flights) ----")
        for f in report["flight_comparisons"]:
            print(f"  {f['airline']}: ₹{f['price']} — Stops: {f['stops']} — Departs: {f['departure']}")
        print()

        print("---- Itinerary Tips ----")
        for tip in report["itinerary_tips"]:
            print(f"  - {tip}")
        print("\n======================================\n")



   
       # ----------------------
# Example usage / main
# ----------------------

# Example trip request
trip = TripRequest(
    user_name="Asha",
    destination="Jaipur",
    start_date=datetime.date.today() + datetime.timedelta(days=7),
    nights=3,
    travelers=2,
    budget=25000.0,
    preferences={"food": "veg", "comfort": "mid", "avoid": []}
)

orchestrator = OrchestratorAgent()
report = orchestrator.build_full_report(trip)
orchestrator.print_user_friendly(report)

# === CODE TO CREATE OUTPUT FILE ===
import json

def serialize_report(obj):
    # Function to handle converting date objects to strings for JSON
    if isinstance(obj, datetime.date):
        return obj.isoformat()
    # ... rest of the serialize_report function ... (keep it)
# ...

final_json_report = serialize_report(report)
output_filename = 'wanderwise_trip_report.json'
with open(output_filename, 'w') as f:
    json.dump(final_json_report, f, indent=4)

print(f"\nSuccessfully saved the final report to: {output_filename}")
# === END ADDED CODE ===
    
    
            
   

