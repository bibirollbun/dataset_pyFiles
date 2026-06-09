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


# imports used across the notebook
import requests
import json
import math
import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from geopy.geocoders import Nominatim
import pandas as pd
import matplotlib.pyplot as plt



# CONFIG - put your real keys here if you have them; otherwise mock_mode=True will use sample data.
CONFIG = {
    "OPENTRIPMAP_KEY": "YOUR_OPENTRIPMAP_KEY",
    "OPENWEATHER_KEY": "YOUR_OPENWEATHER_KEY",
    "mock_mode": True  # Set False when you add real API keys
}



geolocator = Nominatim(user_agent="ai_travel_planner_notebook")

def geocode_place(place: str):
    """Return latitude and longitude for place name. Works with internet; will raise if not found."""
    loc = geolocator.geocode(place)
    if not loc:
        raise ValueError(f"Could not geocode {place}")
    return {"lat": loc.latitude, "lon": loc.longitude, "name": loc.address}

def haversine(a_lat, a_lon, b_lat, b_lon):
    R = 6371  # km
    phi1, phi2 = math.radians(a_lat), math.radians(b_lat)
    dphi = math.radians(b_lat - a_lat)
    dlambda = math.radians(b_lon - a_lon)
    x = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(x))

def pretty_print_itinerary(itinerary: Dict[str, List[Dict[str,Any]]]):
    for day, spots in itinerary.items():
        print(f"\n{day}:")
        for i, s in enumerate(spots, 1):
            name = s.get("name", "Unknown")
            cat = s.get("kinds", "")
            dur = s.get("duration_hours", "?")
            print(f"  {i}. {name} ({cat}) — est {dur} hr")



class DiscoveryAgent:
    def __init__(self, api_key: str = None, mock_mode: bool = True):
        self.api_key = api_key
        self.mock = mock_mode

    def get_pois(self, lat: float, lon: float, radius: int = 8000, kinds: str = "historic,architecture,interesting_places,foods", limit: int = 50):
        """
        Returns a list of POIs. If mock_mode True, returns canned sample POIs.
        Each POI: {'name','lat','lon','dist','kinds','xid','duration_hours'}
        """
        if self.mock:
            # canned sample POIs (for Istanbul-style demo)
            sample = [
                {"name":"Hagia Sophia", "lat":41.0086, "lon":28.9802, "dist":0.4, "kinds":"historic,architecture", "xid":"poi_1", "duration_hours":1.5},
                {"name":"Topkapi Palace", "lat":41.0115, "lon":28.9833, "dist":0.6, "kinds":"historic,architecture", "xid":"poi_2", "duration_hours":2.0},
                {"name":"Grand Bazaar", "lat":41.0105, "lon":28.9680, "dist":0.9, "kinds":"shopping,cultural", "xid":"poi_3", "duration_hours":1.5},
                {"name":"Sultanahmet Square", "lat":41.0056, "lon":28.9768, "dist":0.2, "kinds":"park,cultural", "xid":"poi_4", "duration_hours":0.75},
                {"name":"Spice Bazaar", "lat":41.0165, "lon":28.9714, "dist":1.0, "kinds":"food,market", "xid":"poi_5", "duration_hours":1.0},
                {"name":"Galata Tower", "lat":41.0256, "lon":28.9744, "dist":2.0, "kinds":"historic,viewpoint", "xid":"poi_6", "duration_hours":1.0},
                {"name":"Istiklal Street", "lat":41.0369, "lon":28.9849, "dist":3.0, "kinds":"shopping,food,nightlife", "xid":"poi_7", "duration_hours":2.0},
            ]
            return sample[:limit]
        # Real API call to OpenTripMap (if not mock)
        url = "https://api.opentripmap.com/0.1/en/places/radius"
        params = {
            "apikey": self.api_key,
            "radius": radius,
            "lon": lon,
            "lat": lat,
            "kinds": kinds,
            "limit": limit,
            "format": "json"
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        raw = resp.json()
        pois = []
        for p in raw:
            pois.append({
                "name": p.get("name","Unknown"),
                "lat": p["point"]["lat"],
                "lon": p["point"]["lon"],
                "dist": p.get("dist"),
                "kinds": p.get("kinds",""),
                "xid": p.get("xid"),
                "duration_hours": 1.25  # default guess; better enrichment possible
            })
        return pois



class WeatherAgent:
    def __init__(self, api_key: str = None, mock_mode: bool = True):
        self.api_key = api_key
        self.mock = mock_mode

    def get_daily_forecast(self, lat: float, lon: float, days: int = 7):
        """Return list of daily forecasts (date, temp_day, weather_main)"""
        if self.mock:
            today = datetime.date.today()
            sample = []
            temps = [20, 21, 19, 18, 22, 24, 23]
            weathers = ["Clear", "Clouds", "Rain", "Clear", "Clear", "Clouds", "Rain"]
            for i in range(min(days, len(temps))):
                sample.append({"date": str(today + datetime.timedelta(days=i)), "temp_day": temps[i], "weather": weathers[i]})
            return sample
        # Real API: OpenWeather OneCall
        url = "https://api.openweathermap.org/data/2.5/onecall"
        params = {"lat": lat, "lon": lon, "exclude": "minutely,hourly,alerts", "units": "metric", "appid": self.api_key}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        out = []
        for d in data.get("daily", [])[:days]:
            out.append({
                "date": datetime.date.fromtimestamp(d["dt"]).isoformat(),
                "temp_day": d["temp"]["day"],
                "weather": d["weather"][0]["main"]
            })
        return out



class ItineraryAgent:
    def __init__(self, travel_pace: str = "normal"):
        """
        travel_pace: 'relaxed' (3-4 hrs/day), 'normal' (6-8 hrs/day), 'packed' (9-12 hrs/day)
        """
        self.pace = travel_pace

    def hours_per_day(self) -> float:
        return {"relaxed":4.0, "normal":7.0, "packed":10.0}.get(self.pace, 7.0)

    def create_itinerary(self, pois: List[Dict[str,Any]], days: int = 3, start_lat: Optional[float]=None, start_lon: Optional[float]=None):
        """
        Simple greedy packing: sort by proximity to start (or center), then fill each day until hours budget reached.
        """
        if not pois:
            return {}
        # Default center: average of POIs if start not provided
        if start_lat is None or start_lon is None:
            start_lat = sum(p["lat"] for p in pois) / len(pois)
            start_lon = sum(p["lon"] for p in pois) / len(pois)
        # compute distance from center
        for p in pois:
            p["dist_center"] = haversine(start_lat, start_lon, p["lat"], p["lon"])
            # ensure a duration value exists
            p.setdefault("duration_hours", 1.25)
        pois_sorted = sorted(pois, key=lambda x: x["dist_center"])
        hours_budget = self.hours_per_day()
        itinerary = {f"Day {i+1}": [] for i in range(days)}
        day = 0
        remaining = hours_budget
        for p in pois_sorted:
            dur = p["duration_hours"]
            if dur <= remaining:
                itinerary[f"Day {day+1}"].append(p)
                remaining -= dur
            else:
                day += 1
                if day >= days:
                    # put remainder into last day (overflow)
                    itinerary[f"Day {days}"].append(p)
                else:
                    remaining = hours_budget - dur
                    itinerary[f"Day {day+1}"].append(p)
        return itinerary



class BudgetAgent:
    def __init__(self, currency: str = "INR"):
        self.currency = currency

    def estimate(self, destination: str, days: int = 3, travelers: int = 2, level: str = "mid"):
        """
        Return a budget dict. Heuristics; replace with real APIs for more accuracy.
        level: 'budget','mid','lux'
        """
        level_mult = {'budget': 0.7, 'mid': 1.0, 'lux': 1.8}.get(level, 1.0)
        # base numbers in INR -- you can adjust per city for realism
        accommodation_per_night = 3000 * level_mult
        food_per_person_day = 800 * level_mult
        local_transport_per_person_day = 300
        attractions_per_person_day = 500
        total_accom = accommodation_per_night * days
        total_food = food_per_person_day * days * travelers
        total_transport = local_transport_per_person_day * days * travelers
        total_attractions = attractions_per_person_day * days * travelers
        subtotal = (total_accom * travelers) + total_food + total_transport + total_attractions
        contingency = int(subtotal * 0.08)
        total = int(subtotal + contingency)
        return {
            "currency": self.currency,
            "destination": destination,
            "days": days,
            "travelers": travelers,
            "level": level,
            "breakdown": {
                "accommodation": int(total_accom * travelers),
                "food": int(total_food),
                "local_transport": int(total_transport),
                "attractions": int(total_attractions),
                "contingency": contingency
            },
            "total_estimate": total
        }



class AssistantAgent:
    def __init__(self):
        pass

    def packing_list(self, forecast: List[Dict[str,Any]]):
        temps = [d["temp_day"] for d in forecast]
        avg = sum(temps)/len(temps)
        items = ["passport/ID", "phone & charger", "medications", "credit card/cash"]
        if avg >= 25:
            items += ["light shirts", "sunscreen", "hat"]
        elif avg < 10:
            items += ["warm jacket", "gloves"]
        else:
            items += ["light jacket", "umbrella"]
        # if any rainy day
        if any("Rain" in d["weather"] for d in forecast):
            items += ["compact umbrella", "waterproof shoes"]
        return items

    def summarize(self, destination: str, itinerary: Dict[str,Any], budget: Dict[str,Any], forecast: List[Dict[str,Any]]):
        days = budget.get("days", "?")
        travelers = budget.get("travelers", "?")
        total = budget.get("total_estimate", "?")
        summary = f"Trip to {destination} for {travelers} traveler(s), {days} days. Estimated total cost: {total} {budget.get('currency')}. "
        summary += "Below is a suggested itinerary and notes. "
        # short day summary
        for day, spots in itinerary.items():
            names = [s["name"] for s in spots]
            summary += f"{day}: " + (", ".join(names) if names else "No suggestions") + ". "
        summary += "Weather highlights: "
        summary += ", ".join([f'{d["date"]}: {d["weather"]} ({d["temp_day"]}°C)' for d in forecast[:min(3, len(forecast))]])
        packing = self.packing_list(forecast)
        summary += f" Packing suggestions: {', '.join(packing)}."
        return summary



@dataclass
class Orchestrator:
    discovery: DiscoveryAgent
    weather: WeatherAgent
    itinerary_agent: ItineraryAgent
    budget_agent: BudgetAgent
    assistant: AssistantAgent

    def plan_trip(self, destination: str, start_date: str, days: int = 3, travelers: int = 2, interests: List[str] = None, budget_level: str = "mid"):
        """
        Returns dict with geodata, pois, forecast, itinerary, budget, and human summary.
        start_date: 'YYYY-MM-DD' string. (Used for forecasting window length)
        """
        result = {"destination_input": destination, "start_date": start_date, "days": days, "travelers": travelers}
        # 1. Geocode
        geo = geocode_place(destination)
        result["geo"] = geo
        # 2. Discovery
        pois = self.discovery.get_pois(geo["lat"], geo["lon"], radius=8000, kinds=",".join(interests) if interests else None)
        result["pois_raw"] = pois
        # 3. Weather
        forecast = self.weather.get_daily_forecast(geo["lat"], geo["lon"], days=days)
        result["forecast"] = forecast
        # 4. Itinerary
        itinerary = self.itinerary_agent.create_itinerary(pois, days=days, start_lat=geo["lat"], start_lon=geo["lon"])
        result["itinerary"] = itinerary
        # 5. Budget
        budget = self.budget_agent.estimate(destination, days=days, travelers=travelers, level=budget_level)
        result["budget"] = budget
        # 6. Assistant summary
        summary = self.assistant.summarize(destination, itinerary, budget, forecast)
        result["summary"] = summary
        return result



# instantiate agents according to CONFIG
discovery = DiscoveryAgent(api_key=CONFIG["OPENTRIPMAP_KEY"], mock_mode=CONFIG["mock_mode"])
weather = WeatherAgent(api_key=CONFIG["OPENWEATHER_KEY"], mock_mode=CONFIG["mock_mode"])
itinerary_agent = ItineraryAgent(travel_pace="normal")
budget_agent = BudgetAgent(currency="INR")
assistant_agent = AssistantAgent()

orch = Orchestrator(discovery, weather, itinerary_agent, budget_agent, assistant_agent)



demo = orch.plan_trip(destination="Istanbul, Turkey", start_date=str(datetime.date.today()+datetime.timedelta(days=30)), days=4, travelers=2, interests=["historic","food"], budget_level="mid")
print("SUMMARY:\n", demo["summary"])
pretty_print_itinerary(demo["itinerary"])
print("\nBUDGET:\n", json.dumps(demo["budget"], indent=2))
print("\nFORECAST (first days):", demo["forecast"][:4])



def plot_budget(budget_dict):
    bd = budget_dict["breakdown"]
    names = list(bd.keys())
    vals = [bd[k] for k in names]
    plt.figure(figsize=(6,4))
    plt.bar(names, vals)
    plt.title("Budget breakdown")
    plt.ylabel(budget_dict.get("currency","INR"))
    plt.show()

plot_budget(demo["budget"])



def export_plan(result: Dict[str,Any], filename: str = "trip_plan.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return filename

exported = export_plan(demo, "demo_trip_plan.json")
print("Exported to:", exported)


