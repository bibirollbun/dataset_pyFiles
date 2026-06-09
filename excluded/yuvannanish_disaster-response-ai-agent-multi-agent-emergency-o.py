import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import requests  # for OpenAPI tools (if you use real APIs)



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("disaster-agent")



class InMemorySessionService:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def get_session(self, user_id: str) -> Dict[str, Any]:
        if user_id not in self.sessions:
            self.sessions[user_id] = {
                "last_city": None,
                "last_disaster_type": None,
                "last_safe_point": None,
            }
        return self.sessions[user_id]

    def update_session(self, user_id: str, **kwargs):
        session = self.get_session(user_id)
        session.update(kwargs)



session_service = InMemorySessionService()



@dataclass
class Metrics:
    alerts_processed: int = 0
    scenarios_run: int = 0
    false_alerts_flagged: int = 0
    avg_response_time_sec: float = 0.0
    history: List[Dict[str, Any]] = field(default_factory=list)

    def log_scenario(self, scenario_name: str, response_time: float, alerts_count: int):
        self.scenarios_run += 1
        self.alerts_processed += alerts_count
        # simple running average
        if self.scenarios_run == 1:
            self.avg_response_time_sec = response_time
        else:
            self.avg_response_time_sec = (
                (self.avg_response_time_sec * (self.scenarios_run - 1)) + response_time
            ) / self.scenarios_run
        self.history.append({
            "scenario": scenario_name,
            "response_time": response_time,
            "alerts_count": alerts_count
        })

metrics = Metrics()



!pip install newsapi-python



from newsapi import NewsApiClient

# Add your API key here
NEWS_API_KEY = "38249c2acf264751a89d2cea1ffc9219"   # <-- replace with your real key
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

def news_search_tool(location: str, disaster_type: str) -> List[Dict[str, Any]]:
    """
    Real NewsAPI integration.
    Fetches the latest news articles related to the disaster and location.
    """
    query = f"{location} {disaster_type}"

    try:
        articles = newsapi.get_everything(
            q=query,
            language='en',
            sort_by='publishedAt',
            page_size=5
        )

        results = []
        for a in articles.get("articles", []):
            results.append({
                "title": a["title"],
                "source": a["source"]["name"],
                "summary": a["description"],
                "url": a["url"]
            })

        if not results:
            print("[news_search_tool] No real news articles found.")
        else:
            print(f"[news_search_tool] Found {len(results)} real news articles.")

        return results

    except Exception as e:
        print(f"[news_search_tool] Error fetching real news: {e}")
        return []



def geo_safe_route_tool(location: str, disaster_type: str) -> Dict[str, Any]:
    """
    Simulated OpenAPI tool for maps/geo.
    Returns safe and hazard zones along with a suggested route.
    """
    logger.info(f"[Tool] Computing safe route for {disaster_type} in {location}")
    return {
        "safe_areas": [
            {"name": "City Hall Relief Center", "description": "Higher ground, official camp"},
            {"name": "Central School Shelter", "description": "Designated shelter point"}
        ],
        "hazard_zones": [
            {"name": "Riverfront Area", "description": "High flood risk"},
        ],
        "recommended_route": "Avoid Riverfront Area, move via Main Road to City Hall Relief Center."
    }



def nearby_resources_tool(location: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Simulated OpenAPI for nearby hospitals and shelters.
    """
    logger.info(f"[Tool] Fetching resources near {location}")
    return {
        "hospitals": [
            {"name": "General Hospital", "distance_km": 2.1, "emergency_available": True},
            {"name": "City Medical Center", "distance_km": 3.5, "emergency_available": False},
        ],
        "shelters": [
            {"name": "Community Hall Shelter", "distance_km": 1.2, "capacity": "High"},
            {"name": "School Auditorium Shelter", "distance_km": 2.8, "capacity": "Medium"},
        ],
    }



class NewsMonitorAgent:
    def __init__(self):
        pass

    def run(self, location: str, disaster_type: str) -> List[Dict[str, Any]]:
        news = news_search_tool(location, disaster_type)
        logger.info(f"[NewsMonitorAgent] Found {len(news)} news items")
        return news



    def run_polling(self, location: str, disaster_type: str, iterations: int = 3, delay_sec: int = 5):
        """
        Simulates a long-running monitor with pause/resume.
        """
        all_alerts = []
        for i in range(iterations):
            logger.info(f"[NewsMonitorAgent] Poll iteration {i+1}")
            alerts = self.run(location, disaster_type)
            all_alerts.extend(alerts)
            time.sleep(delay_sec)
        return all_alerts



class GeoNavigationAgent:
    def __init__(self):
        pass

    def run(self, location: str, disaster_type: str) -> Dict[str, Any]:
        geo_info = geo_safe_route_tool(location, disaster_type)
        logger.info(f"[GeoNavigationAgent] Safe areas: {len(geo_info['safe_areas'])}, "
                    f"Hazard zones: {len(geo_info['hazard_zones'])}")
        return geo_info



class ResourcesAgent:
    def __init__(self, session_service: InMemorySessionService):
        self.session_service = session_service

    def run(self, user_id: str, location: str) -> Dict[str, Any]:
        session = self.session_service.get_session(user_id)
        session["last_city"] = location

        resources = nearby_resources_tool(location)
        # Update "last safe point" as first shelter for example
        if resources["shelters"]:
            session["last_safe_point"] = resources["shelters"][0]["name"]

        logger.info(f"[ResourcesAgent] Found {len(resources['hospitals'])} hospitals and "
                    f"{len(resources['shelters'])} shelters")
        return resources



def fake_llm_generate(prompt: str) -> str:
    # Placeholder. Replace with a real LLM call in your environment.
    return "SIMULATED LLM RESPONSE:\n" + prompt[:1000]



class SummaryInstructionAgent:
    def __init__(self):
        pass

    def run(self, location: str, disaster_type: str,
            news: List[Dict[str, Any]],
            geo_info: Dict[str, Any],
            resources: Dict[str, Any]) -> str:
        prompt = f"""
You are a disaster response assistant.

User location: {location}
Disaster type: {disaster_type}

Latest alerts:
{news}

Geo information (safe and hazard zones, routes):
{geo_info}

Nearby resources (hospitals, shelters):
{resources}

Task:
1. Briefly summarize the current situation in {location}.
2. Provide a recommended safe route and safe areas.
3. List nearest hospitals and shelters with very short descriptions.
4. Give 5-8 bullet points of safety instructions and basic first-aid tips for this type of disaster.
5. Use clear, calm, human-friendly language.
"""
        response = fake_llm_generate(prompt)
        logger.info("[SummaryInstructionAgent] Generated response for user")
        return response



class DisasterResponseOrchestrator:
    def __init__(self, session_service: InMemorySessionService, metrics: Metrics):
        self.news_agent = NewsMonitorAgent()
        self.geo_agent = GeoNavigationAgent()
        self.resources_agent = ResourcesAgent(session_service)
        self.summary_agent = SummaryInstructionAgent()
        self.metrics = metrics

    def handle_request(self, user_id: str, location: str, disaster_type: str,
                       scenario_name: str = "default_scenario") -> str:
        start = time.time()

        # In real code, you could run news_agent and geo_agent in parallel threads
        news = self.news_agent.run(location, disaster_type)
        geo_info = self.geo_agent.run(location, disaster_type)
        resources = self.resources_agent.run(user_id, location)

        response = self.summary_agent.run(location, disaster_type, news, geo_info, resources)

        end = time.time()
        self.metrics.log_scenario(
            scenario_name=scenario_name,
            response_time=end - start,
            alerts_count=len(news)
        )
        return response



orchestrator = DisasterResponseOrchestrator(session_service, metrics)

user_id = "demo_user_1"
location = "Chennai, India"
disaster_type = "flood"

final_response = orchestrator.handle_request(
    user_id=user_id,
    location=location,
    disaster_type=disaster_type,
    scenario_name="Chennai_flood_demo"
)

print(final_response)



import requests
from typing import List, Dict, Any, Optional, Tuple

# Reusable headers for Nominatim (required by usage policy)
OSM_HEADERS = {
    "User-Agent": "DisasterResponseAgent/1.0 (Kaggle Notebook; contact: example@example.com)"
}


OPEN_CAGE_KEY = "cd0dc66e37b4407e9eab500d2dfa49d0"

def geocode_location_osm(location):
    url = "https://api.opencagedata.com/geocode/v1/json"
    params = {"q": location, "key": OPEN_CAGE_KEY}

    r = requests.get(url, params=params)
    data = r.json()

    lat = data["results"][0]["geometry"]["lat"]
    lon = data["results"][0]["geometry"]["lng"]
    return lat, lon



OVERPASS_URL = "https://overpass-api.de/api/interpreter"

OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.nchc.org.tw/api/interpreter"
]

def _run_overpass_query(query: str):
    """
    Robust Overpass client with automatic retries across multiple servers.
    Prevents 429 and 504 errors from breaking the pipeline.
    """
    for server in OVERPASS_SERVERS:
        try:
            print(f"[Overpass] Trying server: {server}")
            r = requests.post(server, data={'data': query}, timeout=40)
            r.raise_for_status()
            print(f"[Overpass] Success on {server}")
            return r.json()

        except Exception as e:
            print(f"[Overpass] Server failed ({server}): {e}")
            continue

    print("[Overpass] All servers failed.")
    return None


def get_hospitals_overpass(lat: float, lon: float, radius_m: int = 5000) -> List[Dict[str, Any]]:
    """Return nearby hospitals as a list of OSM elements."""
    query = f"""
    [out:json];
    (
      node["amenity"="hospital"](around:{radius_m},{lat},{lon});
      way["amenity"="hospital"](around:{radius_m},{lat},{lon});
      relation["amenity"="hospital"](around:{radius_m},{lat},{lon});
    );
    out center;
    """
    data = _run_overpass_query(query)
    return data.get("elements", []) if data else []


def get_shelters_overpass(lat: float, lon: float, radius_m: int = 5000) -> List[Dict[str, Any]]:
    """Return nearby shelters as a list of OSM elements."""
    query = f"""
    [out:json];
    (
      node["amenity"="shelter"](around:{radius_m},{lat},{lon});
      way["amenity"="shelter"](around:{radius_m},{lat},{lon});
      relation["amenity"="shelter"](around:{radius_m},{lat},{lon});
    );
    out center;
    """
    data = _run_overpass_query(query)
    return data.get("elements", []) if data else []


def get_flood_related_areas(lat: float, lon: float, radius_m: int = 5000) -> List[Dict[str, Any]]:
    """Return OSM features potentially related to flood/hazard/water in the area."""
    query = f"""
    [out:json];
    (
      way["hazard"="flood"](around:{radius_m},{lat},{lon});
      way["flood_prone"="yes"](around:{radius_m},{lat},{lon});
      way["natural"="water"](around:{radius_m},{lat},{lon});
    );
    out center;
    """
    data = _run_overpass_query(query)
    return data.get("elements", []) if data else []


class ResourcesAgent:
    """Agent responsible for discovering nearby hospitals and shelters using Overpass API.

    It also updates session memory with:
    - last_city
    - last_safe_point (first shelter found, if any)
    """
    def __init__(self, session_service):
        self.session_service = session_service

    def run(self, user_id: str, location: str) -> Dict[str, Any]:
        session = self.session_service.get_session(user_id)
        session["last_city"] = location

        geo = geocode_location_osm(location)
        if geo is None:
            print(f"[ResourcesAgent] Could not geocode location: {location}")
            hospitals = []
            shelters = []
        else:
            lat, lon = geo
            hospitals = get_hospitals_overpass(lat, lon)
            shelters = get_shelters_overpass(lat, lon)

        def _summarize_resource(elem):
            tags = elem.get("tags", {})
            name = tags.get("name", "Unnamed")
            amenity = tags.get("amenity", "unknown")
            return {
                "name": name,
                "amenity": amenity,
                "lat": elem.get("lat") or elem.get("center", {}).get("lat"),
                "lon": elem.get("lon") or elem.get("center", {}).get("lon"),
            }

        hospitals_summary = [_summarize_resource(e) for e in hospitals]
        shelters_summary = [_summarize_resource(e) for e in shelters]

        if shelters_summary:
            session["last_safe_point"] = shelters_summary[0]["name"]

        resources = {
            "hospitals": hospitals_summary,
            "shelters": shelters_summary,
        }

        print(f"[ResourcesAgent] Found {len(hospitals_summary)} hospitals and {len(shelters_summary)} shelters")
        return resources


class GeoNavigationAgent:
    """Agent that provides basic geo-context for the disaster using Overpass data."""
    def __init__(self):
        pass

    def run(self, location: str, disaster_type: str) -> Dict[str, Any]:
        geo = geocode_location_osm(location)
        if geo is None:
            print(f"[GeoNavigationAgent] Could not geocode location: {location}")
            return {
                "safe_areas": [],
                "hazard_zones": [],
                "recommended_route": "Unable to compute route due to missing geolocation.",
            }

        lat, lon = geo
        flood_elems = get_flood_related_areas(lat, lon)

        hazard_zones = []
        for e in flood_elems:
            tags = e.get("tags", {})
            name = tags.get("name", "Water / flood-prone area")
            hazard_zones.append({
                "name": name,
                "lat": e.get("lat") or e.get("center", {}).get("lat"),
                "lon": e.get("lon") or e.get("center", {}).get("lon"),
                "description": "Potential flood or water-related hazard zone from OSM data."
            })

        safe_areas = [
            {
                "name": "Higher ground / central civic buildings",
                "description": "Move towards higher elevation and official relief centers where available."
            }
        ]

        recommended_route = (
            "Avoid low-lying areas and regions close to mapped water bodies. "
            "Move towards higher ground and official shelters if indicated in the resources."
        )

        print(f"[GeoNavigationAgent] Hazard zones identified: {len(hazard_zones)}")
        return {
            "safe_areas": safe_areas,
            "hazard_zones": hazard_zones,
            "recommended_route": recommended_route,
        }


def news_search_tool(location: str, disaster_type: str):
    """Placeholder for a real news API integration.

    You can replace this implementation with:
    - NewsAPI.org
    - GNews
    - any other trusted news source.

    Expected return format:
        List[{
            "title": str,
            "source": str,
            "summary": str,
            "url": str
        }]
    """
    print("[news_search_tool] No real news API configured yet. Returning empty list.")
    return []


class NewsMonitorAgent:
    def __init__(self):
        pass

    def run(self, location: str, disaster_type: str):
        news = news_search_tool(location, disaster_type)
        print(f"[NewsMonitorAgent] Retrieved {len(news)} news items")
        return news


def fake_llm_generate(prompt: str) -> str:
    """Placeholder LLM call.

    In a real deployment, this would call an LLM.
    For this notebook, we simply echo the prompt prefix so the pipeline is visible.
    """
    return "LLM RESPONSE (placeholder)\n\n" + prompt[:1200]


class SummaryInstructionAgent:
    def __init__(self):
        pass

    def run(
        self,
        location: str,
        disaster_type: str,
        news: List[Dict[str, Any]],
        geo_info: Dict[str, Any],
        resources: Dict[str, Any],
    ) -> str:
        prompt = f"""
You are a calm, concise disaster response assistant.

User location: {location}
Disaster type: {disaster_type}

Latest alerts (may be empty):
{news}

Geo information (safe areas, hazard zones, general route guidance):
{geo_info}

Nearby resources (hospitals and shelters from OpenStreetMap):
{resources}

Your task:
1. Give a 2–3 sentence situation summary for {location}.
2. List any known hazard zones or flood-prone areas. If none are available, say that explicit hazard zones are not mapped.
3. List the nearest hospitals and shelters in short bullet points (name + one helpful detail).
4. Provide 5–8 clear safety instructions tailored to a {disaster_type}.
5. If information is missing (e.g., no shelters found), tell the user what they should do instead (e.g., contact authorities, move to higher ground).
6. Use simple, reassuring language. Do NOT mention that this came from a model or an API; just speak as an assistant.
"""
        response = fake_llm_generate(prompt)
        print("[SummaryInstructionAgent] Generated guidance text.")
        return response


import time
from dataclasses import dataclass, field

@dataclass
class Metrics:
    alerts_processed: int = 0
    scenarios_run: int = 0
    avg_response_time_sec: float = 0.0
    history: List[Dict[str, Any]] = field(default_factory=list)

    def log_scenario(self, scenario_name: str, response_time: float, alerts_count: int):
        self.scenarios_run += 1
        self.alerts_processed += alerts_count
        if self.scenarios_run == 1:
            self.avg_response_time_sec = response_time
        else:
            self.avg_response_time_sec = (
                (self.avg_response_time_sec * (self.scenarios_run - 1)) + response_time
            ) / self.scenarios_run
        self.history.append(
            {
                "scenario": scenario_name,
                "response_time_sec": response_time,
                "alerts_count": alerts_count,
            }
        )


class InMemorySessionService:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def get_session(self, user_id: str) -> Dict[str, Any]:
        if user_id not in self.sessions:
            self.sessions[user_id] = {
                "last_city": None,
                "last_disaster_type": None,
                "last_safe_point": None,
            }
        return self.sessions[user_id]

    def update_session(self, user_id: str, **kwargs):
        session = self.get_session(user_id)
        session.update(kwargs)


metrics = Metrics()
session_service = InMemorySessionService()


class DisasterResponseOrchestrator:
    def __init__(self, session_service: InMemorySessionService, metrics: Metrics):
        self.news_agent = NewsMonitorAgent()
        self.geo_agent = GeoNavigationAgent()
        self.resources_agent = ResourcesAgent(session_service)
        self.summary_agent = SummaryInstructionAgent()
        self.metrics = metrics

    def handle_request(
        self,
        user_id: str,
        location: str,
        disaster_type: str,
        scenario_name: str = "default_scenario",
    ) -> str:
        start = time.time()

        news = self.news_agent.run(location, disaster_type)
        geo_info = self.geo_agent.run(location, disaster_type)
        resources = self.resources_agent.run(user_id, location)
        response = self.summary_agent.run(location, disaster_type, news, geo_info, resources)

        end = time.time()
        self.metrics.log_scenario(
            scenario_name=scenario_name,
            response_time=end - start,
            alerts_count=len(news),
        )
        return response


# Instantiate orchestrator with real-data-capable agents
orchestrator = DisasterResponseOrchestrator(session_service, metrics)

demo_scenarios = [
    ("user1", "Chennai, India", "flood", "Chennai_flood_demo"),
    ("user2", "Tokyo, Japan", "earthquake", "Tokyo_earthquake_demo"),
    ("user3", "Miami, USA", "hurricane", "Miami_hurricane_demo"),
]

for user_id, loc, dtype, name in demo_scenarios:
    print("\n" + "=" * 80)
    print(f"Scenario: {name} | Location: {loc} | Disaster: {dtype}")
    guidance = orchestrator.handle_request(user_id, loc, dtype, scenario_name=name)
    print(guidance[:800] + "\n...\n")


import pandas as pd

metrics_df = pd.DataFrame(metrics.history)
metrics_df


resources = orchestrator.resources_agent.run("user1", "Chennai, India")



def print_resources(resources):
    print("\n=== Nearby Hospitals ===")
    for h in resources["hospitals"]:
        print(f"- {h['name']}  (lat={h['lat']}, lon={h['lon']})")

    print("\n=== Nearby Shelters ===")
    for s in resources["shelters"]:
        print(f"- {s['name']}  (lat={s['lat']}, lon={s['lon']})")






print_resources(resources)

