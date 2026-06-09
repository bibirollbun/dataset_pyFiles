import numpy as np
import pandas as pd
import os
import json
import time
import random
from typing import Dict, Any, List


def mock_weather_api(location: str):
    conditions = ["Sunny", "Cloudy", "Rainy", "Windy", "Snowy"]
    return {
        "location": location,
        "temp_c": random.randint(15, 30),
        "condition": random.choice(conditions)
    }

def mock_places_api(query: str):
    sample_data = {
        "delhi": ["India Gate", "Qutub Minar", "Lotus Temple"],
        "paris": ["Eiffel Tower", "Louvre Museum", "Notre Dame"],
        "tokyo": ["Shibuya Crossing", "Senso-ji Temple", "Tokyo Skytree"]
    }
    return sample_data.get(query.lower(), ["No results found"])


def research_agent(user_query: str):
    print("Research Agent Running...")
    places = mock_places_api(user_query)
    return {
        "query": user_query,
        "recommendations": places
    }

def weather_agent(city: str):
    print("Weather Agent Running...")
    return mock_weather_api(city)

def itinerary_agent(city: str, days: int):
    print("Itinerary Agent Running...")
    research = research_agent(city)
    weather = weather_agent(city)

    plan = []
    for i in range(days):
        place = research["recommendations"][i % len(research["recommendations"])]
        plan.append({
            "day": i+1,
            "place": place,
            "weather": weather["condition"]
        })

    return {
        "city": city,
        "weather": weather,
        "itinerary": plan
    }


def orchestrator(user_input: Dict[str, Any]):
    city = user_input["city"]
    days = user_input["days"]

    print("Orchestrating Travel Plan...\n")
    result = itinerary_agent(city, days)

    print("\nFinal Trip Plan Ready!\n")
    return result


user_request = {
    "city": "Tokyo",
    "days": 3
}

trip = orchestrator(user_request)
trip


df = pd.DataFrame(trip["itinerary"])
df


def long_running_task():
    print("Generating bookings...\n")
    for i in range(5):
        print(f"Step {i+1}/5 completed")
        time.sleep(0.5)
    return "Long-running job finished!"

long_running_task()


session_memory = []

def remember(x):
    session_memory.append(x)

def recall():
    return session_memory

remember("User likes vegetarian food")
remember("User prefers museums")

recall()


with open("trip_plan.json", "w") as f:
    json.dump(trip, f, indent=4)

"trip_plan.json saved successfully"


def test(city, days):
    print(f"\nTrip to: {city}, {days} days")
    try:
        result = orchestrator({"city": city, "days": days})
        print(pd.DataFrame(result["itinerary"]))
    except Exception as e:
        print("Error:", e)

test("Delhi", 2)
test("Paris", 5)
test("known City", 2)




