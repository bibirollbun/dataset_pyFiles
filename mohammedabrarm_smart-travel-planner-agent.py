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


from dataclasses import dataclass
from typing import List, Dict, Any


# -----------------------------
# Data classes (simple structs)
# -----------------------------
@dataclass
class FlightOption:
    carrier: str
    price: float
    duration_hours: float
    description: str


@dataclass
class StayOption:
    name: str
    price_per_night: float
    nights: int
    description: str


# -----------------------------
# Tools (mocked for now)
# -----------------------------
def estimate_total_trip_cost(
    flight: FlightOption,
    stay: StayOption,
    daily_expense_estimate: float,
    days: int,
) -> float:
    """Simple cost calculator."""
    lodging_cost = stay.price_per_night * stay.nights
    daily_cost = daily_expense_estimate * days
    return flight.price + lodging_cost + daily_cost


def mock_flight_search(origin: str, destination: str, days: int) -> List[FlightOption]:
    """Fake flight search to keep things simple for now."""
    return [
        FlightOption(
            carrier="Example Air",
            price=680.0,
            duration_hours=11.5,
            description=f"{origin} → {destination} (standard)",
        ),
        FlightOption(
            carrier="Budget Air",
            price=550.0,
            duration_hours=14.0,
            description=f"{origin} → {destination} (budget)",
        ),
    ]


def mock_stay_search(destination: str, nights: int, preference: str) -> List[StayOption]:
    """Fake hotel/hostel search."""
    return [
        StayOption(
            name=f"{destination} Central Hotel",
            price_per_night=84.0,
            nights=nights,
            description="Mid-range, central location.",
        ),
        StayOption(
            name=f"{destination} Budget Capsule",
            price_per_night=45.0,
            nights=nights,
            description="Budget capsule hotel.",
        ),
    ]


# -----------------------------
# Memory (very simple)
# -----------------------------
class SimpleMemory:
    """Very small in-memory store just for this run."""

    def __init__(self):
        self.store = {}

    def set(self, key, value):
        self.store[key] = value

    def get(self, key, default=None):
        return self.store.get(key, default)


# -----------------------------
# Agents
# -----------------------------
class FlightSearchAgent:
    def run(self, origin: str, destination: str, days: int) -> List[FlightOption]:
        flights = mock_flight_search(origin, destination, days)
        return flights


class StaySearchAgent:
    def run(self, destination: str, nights: int, preference: str) -> List[StayOption]:
        stays = mock_stay_search(destination, nights, preference)
        return stays


class BudgetOptimizationAgent:
    def choose_best_combo(
        self,
        flights: List[FlightOption],
        stays: List[StayOption],
        budget: float,
        days: int,
        daily_expense_estimate: float,
        max_iterations: int = 5,
    ):
        """
        Simple "loop": check combinations, try to find one within budget.
        Returns (flight, stay, total_cost).
        """
        best_combo = None
        best_cost = float("inf")

        for _ in range(max_iterations):
            for f in flights:
                for s in stays:
                    total_cost = estimate_total_trip_cost(
                        flight=f,
                        stay=s,
                        daily_expense_estimate=daily_expense_estimate,
                        days=days,
                    )
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_combo = (f, s)

            if best_combo is not None and best_cost <= budget:
                break

        if best_combo is None:
            return None, None, float("inf")

        return best_combo[0], best_combo[1], best_cost


class ItineraryAgent:
    def generate_itinerary(
        self,
        destination: str,
        days: int,
        travel_style: str,
        budget_summary: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Placeholder itinerary generator.
        Later we can replace this with a call to Gemini.
        """
        itinerary = []
        for day in range(1, days + 1):
            itinerary.append(
                {
                    "day": day,
                    "title": f"Day {day} in {destination}",
                    "activities": [
                        f"Morning: Explore a neighborhood based on '{travel_style}'.",
                        "Afternoon: Visit a major landmark.",
                        "Evening: Try a local restaurant.",
                    ],
                }
            )
        return itinerary


class TravelCoordinatorAgent:
    def __init__(self):
        self.flight_agent = FlightSearchAgent()
        self.stay_agent = StaySearchAgent()
        self.budget_agent = BudgetOptimizationAgent()
        self.itinerary_agent = ItineraryAgent()
        self.memory = SimpleMemory()

    def plan_trip(self, user_request: Dict[str, Any]) -> Dict[str, Any]:
        origin = user_request["origin"]
        destination = user_request["destination"]
        days = user_request["days"]
        budget = user_request["budget"]
        travel_style = user_request.get("travel_style", "mixed")
        daily_expense_estimate = user_request.get("daily_expense_estimate", 60.0)

        # Save some preferences
        self.memory.set("travel_style", travel_style)
        self.memory.set("budget", budget)

        # "Parallel" work (in this simple version: sequential calls)
        flights = self.flight_agent.run(origin, destination, days)
        stays = self.stay_agent.run(destination, days, travel_style)

        # Budget optimization loop
        best_flight, best_stay, total_cost = self.budget_agent.choose_best_combo(
            flights=flights,
            stays=stays,
            budget=budget,
            days=days,
            daily_expense_estimate=daily_expense_estimate,
        )

        budget_summary = {
            "total_cost": total_cost,
            "budget": budget,
            "under_budget": total_cost <= budget,
        }

        itinerary = self.itinerary_agent.generate_itinerary(
            destination=destination,
            days=days,
            travel_style=travel_style,
            budget_summary=budget_summary,
        )

        return {
            "request": user_request,
            "flight": best_flight,
            "stay": best_stay,
            "budget_summary": budget_summary,
            "itinerary": itinerary,
        }


# -----------------------------
# Demo run
# -----------------------------
def demo():
    coordinator = TravelCoordinatorAgent()

    user_request = {
        "origin": "SFO",
        "destination": "Tokyo",
        "days": 5,
        "budget": 1500.0,
        "travel_style": "food and culture",
        "daily_expense_estimate": 60.0,
    }

    result = coordinator.plan_trip(user_request)

    print("=== Trip Summary ===")
    print(f"From: {result['request']['origin']} → {result['request']['destination']}")
    print(f"Days: {result['request']['days']}")
    print(f"Total Cost: {result['budget_summary']['total_cost']:.2f}")
    print(f"Within Budget: {result['budget_summary']['under_budget']}")

    print("\n=== Selected Flight ===")
    f = result["flight"]
    print(f"{f.carrier} - ${f.price} - {f.duration_hours}h")
    print(f"Details: {f.description}")

    print("\n=== Selected Stay ===")
    s = result["stay"]
    print(f"{s.name} - ${s.price_per_night}/night x {s.nights} nights")
    print(f"Details: {s.description}")

    print("\n=== Itinerary ===")
    for day in result["itinerary"]:
        print(f"\nDay {day['day']}: {day['title']}")
        for act in day["activities"]:
            print(f"- {act}")


# Run demo
demo()


