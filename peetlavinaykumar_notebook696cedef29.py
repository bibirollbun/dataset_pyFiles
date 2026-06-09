# MEDI-LOCATOR: FULL FIXED Kaggle Notebook (corrected Overpass query syntax)
# - Fix: Overpass QL filters use ["amenity"="..."]
# - Overpass mirrors + retries + backoff retained
# - Reduced radius default to 3000m
# - Distance sorting included
# Paste into your Kaggle notebook and run

# 1) Minimal installs (uncomment if needed)
# Kaggle typically has geopy & requests preinstalled; uncomment if missing:
# !pip install -q geopy

# 2) Imports & basic config
import os
import json
import time
import logging
import math
import requests
from typing import Dict, Any, List, Optional, Tuple
from geopy.geocoders import Nominatim

# Setup logging (Observability)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("medi-locator")

# Notebook configuration
MEMORY_FILE = "medi_memory.json"
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.nchc.org.tw/api/interpreter"
]
GEOCODER_USER_AGENT = "medi_locator_agent_kaggle"
DEFAULT_SEARCH_RADIUS_M = 3000
MAX_OVERPASS_RETRIES = 3

# 3) Utility helpers
def save_json(path: str, data: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def pretty_print_result(res: Dict):
    from pprint import pprint
    pprint(res)

# 4) Geocoding helper using Nominatim
geolocator = Nominatim(user_agent=GEOCODER_USER_AGENT, timeout=10)

def geocode_location(query: str) -> Optional[Tuple[float, float, Dict]]:
    try:
        loc = geolocator.geocode(query)
        if loc:
            return (loc.latitude, loc.longitude, loc.raw)
        else:
            return None
    except Exception as e:
        logger.error("Geocoding failed: %s", e)
        return None

# 5) Memory Agent
class MemoryAgent:
    def __init__(self, filepath: str = MEMORY_FILE):
        self.filepath = filepath
        self.memory = load_json(self.filepath) or {}
        logger.info("Memory loaded: %s keys", len(self.memory.keys()))
    
    def get_user_memory(self, user_id: str) -> Dict:
        return self.memory.get(user_id, {})
    
    def save_user_memory(self, user_id: str, payload: Dict):
        self.memory[user_id] = self.memory.get(user_id, {})
        self.memory[user_id].update(payload)
        save_json(self.filepath, self.memory)
        logger.info("Saved memory for user %s", user_id)
    
    def append_event(self, user_id: str, event: Dict):
        self.memory[user_id] = self.memory.get(user_id, {})
        events = self.memory[user_id].get("events", [])
        events.append({"timestamp": time.time(), **event})
        self.memory[user_id]["events"] = events
        save_json(self.filepath, self.memory)

# 6) Triage Agent
class TriageAgent:
    def __init__(self):
        self.openai_key = os.environ.get("OPENAI_API_KEY", None)
    
    def llm_call(self, prompt: str) -> str:
        if self.openai_key:
            try:
                import openai
                openai.api_key = self.openai_key
                resp = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"user","content":prompt}],
                    temperature=0.2,
                    max_tokens=400
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                logger.warning("LLM call failed: %s", e)
                return "FALLBACK_LLM: error calling LLM."
        else:
            return "FALLBACK_LLM: no API key configured."

    def rule_based_triage(self, symptoms: List[str], vitals: Dict[str,Any]=None) -> Dict[str,Any]:
        s = " ".join(symptoms).lower()
        vitals = vitals or {}
        result = {"level":"low", "reason":[], "actions":[]}
        
        critical_keywords = ["cardiac arrest","not breathing","unconscious","not responsive","chest pain","severe difficulty breathing","severe bleeding"]
        high_keywords = ["severe bleeding","seizure","major burn","loss of consciousness","stroke","sudden weakness","sudden numbness"]
        medium_keywords = ["high fever","vomit","vomiting","dehydration","persistent fever","abdominal pain","moderate pain", "dizziness"]
        
        for kw in critical_keywords:
            if kw in s:
                result["level"] = "critical"
                result["reason"].append(kw)
        if result["level"] != "critical":
            for kw in high_keywords:
                if kw in s:
                    result["level"] = "high"
                    result["reason"].append(kw)
        if result["level"] not in ("critical","high"):
            for kw in medium_keywords:
                if kw in s:
                    result["level"] = "medium"
                    result["reason"].append(kw)
        
        if "bp" in vitals:
            try:
                systolic = int(str(vitals["bp"]).split("/")[0])
                if systolic < 90:
                    result["level"] = "high"
                    result["reason"].append("low systolic BP")
            except Exception:
                pass
        if "temperature_c" in vitals:
            try:
                if float(vitals["temperature_c"]) >= 40:
                    if result["level"] == "low":
                        result["level"] = "medium"
                    result["reason"].append("high fever")
            except Exception:
                pass
        
        if result["level"] == "critical":
            result["actions"] = [
                "CALL EMERGENCY SERVICES IMMEDIATELY (local emergency number).",
                "If trained, start CPR if patient is unresponsive and not breathing.",
                "Place patient in recovery position if breathing and no suspected spinal injury.",
                "Prepare for ambulance: note location, allergies, medications."
            ]
        elif result["level"] == "high":
            result["actions"] = [
                "Get to the nearest emergency department now or call an ambulance.",
                "Control bleeding with direct pressure if present.",
                "Keep patient warm and monitor breathing & consciousness."
            ]
        elif result["level"] == "medium":
            result["actions"] = [
                "Seek urgent care within a few hours.",
                "Provide supportive care: fluids, antipyretics for fever, rest.",
                "If symptoms worsen, escalate to emergency services."
            ]
        else:
            result["actions"] = [
                "Manage symptoms at home and monitor closely for changes.",
                "Follow up with primary care if symptoms persist >48 hours."
            ]
        
        return result

    def triage(self, symptoms: List[str], vitals: Dict[str,Any]=None, use_llm: bool=False) -> Dict:
        if use_llm and self.openai_key:
            prompt = f"Patient reported symptoms: {symptoms}. Vitals: {vitals}.\nClassify emergency level (critical/high/medium/low) and give immediate action steps in JSON."
            llm_resp = self.llm_call(prompt)
            try:
                parsed = json.loads(llm_resp)
                return parsed
            except Exception:
                logger.warning("LLM returned non-JSON or failed; falling back to rule-based triage.")
                return self.rule_based_triage(symptoms, vitals)
        else:
            return self.rule_based_triage(symptoms, vitals)

# 7) Resource Locator Agent (CORRECTED Overpass QL syntax)
class ResourceLocatorAgent:
    def __init__(self, overpass_endpoints: List[str] = OVERPASS_ENDPOINTS):
        self.overpass_endpoints = overpass_endpoints
    
    def _post_with_backoff(self, url: str, data: Dict, timeout: int = 30):
        backoff = 1.0
        for attempt in range(MAX_OVERPASS_RETRIES):
            try:
                resp = requests.post(url, data=data, timeout=timeout)
                resp.raise_for_status()
                return resp
            except requests.exceptions.RequestException as e:
                logger.warning("Request to %s failed (attempt %d/%d): %s", url, attempt+1, MAX_OVERPASS_RETRIES, e)
                time.sleep(backoff)
                backoff *= 2
        return None

    def _overpass_query(self, query: str) -> Dict:
        for url in self.overpass_endpoints:
            try:
                logger.info("Trying Overpass server: %s", url)
                resp = self._post_with_backoff(url, data={"data": query}, timeout=40)
                if resp is None:
                    logger.warning("Attempts failed for %s, trying next endpoint.", url)
                    continue
                try:
                    return resp.json()
                except ValueError:
                    logger.warning("Non-JSON response from %s; skipping.", url)
                    continue
            except Exception as e:
                logger.warning("Overpass endpoint failed (%s): %s", url, e)
                continue
        logger.error("All Overpass servers failed or returned invalid responses.")
        return {}

    def find_amenities(self, lat: float, lon: float, radius_m: int = DEFAULT_SEARCH_RADIUS_M, amenity_tags: List[str] = None) -> List[Dict]:
        if amenity_tags is None:
            amenity_tags = [
                'hospital', 'clinic', 'doctors', 'pharmacy',
                'blood_donation', 'blood_bank',
                'ambulance_station', 'emergency', 'healthcare'
            ]

        # Correctly build tag filters using quoted key/value syntax: ["amenity"="..."]
        node_filters = []
        way_filters = []
        rel_filters = []
        for tag in amenity_tags:
            node_filters.append(f'node(around:{radius_m},{lat},{lon})["amenity"="{tag}"];')
            way_filters.append(f'way(around:{radius_m},{lat},{lon})["amenity"="{tag}"];')
            rel_filters.append(f'relation(around:{radius_m},{lat},{lon})["amenity"="{tag}"];')

        # Also include broader 'healthcare' and 'emergency' keys (which may be used differently)
        extra = f'''
            node(around:{radius_m},{lat},{lon})["healthcare"];
            way(around:{radius_m},{lat},{lon})["healthcare"];
            relation(around:{radius_m},{lat},{lon})["healthcare"];
            node(around:{radius_m},{lat},{lon})["emergency"];
            way(around:{radius_m},{lat},{lon})["emergency"];
            relation(around:{radius_m},{lat},{lon})["emergency"];
        '''

        q = f"""
        [out:json][timeout:25];
        (
          {"".join(node_filters)}
          {"".join(way_filters)}
          {"".join(rel_filters)}
          {extra}
        );
        out center tags;
        """

        data = self._overpass_query(q)
        elements = data.get("elements", []) if isinstance(data, dict) else []
        results = []
        for el in elements:
            if el.get("lat") and el.get("lon"):
                plat, plon = el["lat"], el["lon"]
            else:
                center = el.get("center", {}) or {}
                plat, plon = center.get("lat"), center.get("lon")
            tags = el.get("tags", {}) or {}
            name = tags.get("name") or tags.get("operator") or tags.get("healthcare") or "unknown"
            kind = tags.get("amenity") or tags.get("healthcare") or tags.get("emergency") or "unknown"
            phone = tags.get("phone") or tags.get("contact:phone") or tags.get("contact:telephone")
            opening = tags.get("opening_hours")
            results.append({
                "osmid": el.get("id"),
                "name": name,
                "kind": kind,
                "lat": plat,
                "lon": plon,
                "phone": phone,
                "opening_hours": opening,
                "tags": tags
            })

        # Deduplicate by name+coords
        unique = {}
        for r in results:
            key = (r.get("name"), str(r.get("lat")), str(r.get("lon")))
            if key not in unique:
                unique[key] = r
        return list(unique.values())

    def find_nearest(self, place_query: str, radius_m: int = DEFAULT_SEARCH_RADIUS_M, amenity_tags: List[str]=None) -> Dict:
        gz = geocode_location(place_query)
        if not gz:
            return {"error": f"Could not geocode location '{place_query}'."}
        lat, lon, raw = gz
        amenities = self.find_amenities(lat, lon, radius_m=radius_m, amenity_tags=amenity_tags)

        # compute distances if lat/lon present
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371000
            phi1 = math.radians(lat1)
            phi2 = math.radians(lat2)
            dphi = math.radians(lat2-lat1)
            dlambda = math.radians(lon2-lon1)
            a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
            return 2*R*math.asin(math.sqrt(a))

        enriched = []
        for a in amenities:
            try:
                if a.get("lat") and a.get("lon"):
                    a_dist = haversine(lat, lon, a["lat"], a["lon"])
                else:
                    a_dist = None
            except Exception:
                a_dist = None
            a_copy = dict(a)
            a_copy["distance_m"] = a_dist
            enriched.append(a_copy)

        enriched_sorted = sorted(enriched, key=lambda x: x["distance_m"] if x["distance_m"] is not None else float("inf"))
        return {"query_place": place_query, "lat": lat, "lon": lon, "raw": raw, "amenities": enriched_sorted}

# 8) Coordinator Agent
class CoordinatorAgent:
    def __init__(self, triage_agent: TriageAgent, locator_agent: ResourceLocatorAgent, memory_agent: MemoryAgent):
        self.triage = triage_agent
        self.locator = locator_agent
        self.memory = memory_agent
    
    def handle_request(self, user_id: str, symptoms: List[str], location_str: Optional[str]=None, vitals: Dict[str,Any]=None, use_llm: bool=False):
        logger.info("Coordinator: handling request for user %s", user_id)
        triage_result = self.triage.triage(symptoms, vitals, use_llm=use_llm)
        
        user_mem = self.memory.get_user_memory(user_id)
        if not location_str:
            location_str = user_mem.get("last_known_location") or user_mem.get("city") or None
        
        resources = []
        if triage_result.get("level") in ("critical", "high"):
            if not location_str:
                logger.warning("No location provided and none in memory; cannot locate resources.")
                triage_result["locator_note"] = "No location provided. Please share your city/address for resource lookup."
            else:
                res = self.locator.find_nearest(location_str, radius_m=DEFAULT_SEARCH_RADIUS_M)
                if res.get("error"):
                    triage_result["locator_note"] = res.get("error")
                    resources = []
                else:
                    resources = res.get("amenities", [])
                    triage_result["locator_note"] = f"Found {len(resources)} amenities near {location_str} within {DEFAULT_SEARCH_RADIUS_M/1000:.1f} km."
                    triage_result["resources_summary"] = [{"name": r["name"], "kind": r.get("kind"), "distance_m": r.get("distance_m"), "phone": r.get("phone")} for r in resources[:10]]
        
        self.memory.append_event(user_id, {"symptoms": symptoms, "vitals": vitals or {}, "triage": triage_result, "location": location_str})
        if location_str:
            self.memory.save_user_memory(user_id, {"last_known_location": location_str})
        
        plan = {"triage": triage_result, "resources": resources, "timestamp": time.time()}
        logger.info("Coordinator: completed plan for user %s", user_id)
        return plan

# 9) Observability
class Observability:
    def __init__(self):
        self.metrics = {"requests": 0, "critical_count": 0, "high_count": 0, "medium_count": 0, "low_count": 0}
    
    def record(self, triage_level: str):
        self.metrics["requests"] += 1
        if triage_level == "critical":
            self.metrics["critical_count"] += 1
        elif triage_level == "high":
            self.metrics["high_count"] += 1
        elif triage_level == "medium":
            self.metrics["medium_count"] += 1
        else:
            self.metrics["low_count"] += 1
    
    def report(self):
        return self.metrics

# 10) Instantiate agents
memory_agent = MemoryAgent(MEMORY_FILE)
triage_agent = TriageAgent()
locator_agent = ResourceLocatorAgent(OVERPASS_ENDPOINTS)
coordinator = CoordinatorAgent(triage_agent, locator_agent, memory_agent)
observ = Observability()

# 11) Demo / sample interactive runs
if __name__ == "__main__":
    user_id = "user_vinay"
    symptoms = ["sudden chest pain", "shortness of breath", "sweating"]
    vitals = {"bp":"110/70", "temperature_c": 36.7}
    location_str = "Anantapur, Andhra Pradesh, India"
    
    print("Running demo request (this may take a few seconds if Overpass mirrors are used)...")
    plan = coordinator.handle_request(user_id=user_id, symptoms=symptoms, location_str=location_str, vitals=vitals, use_llm=False)
    observ.record(plan["triage"].get("level", "low"))
    
    print("\n=== TRIAGE RESULT ===")
    pretty_print_result(plan["triage"])
    print("\n=== RESOURCES (top 5) ===")
    for r in plan.get("resources", [])[:5]:
        print("-", r["name"], "|", r.get("kind"), "|", r.get("phone") or "no-phone", f"({r.get('lat')},{r.get('lon')})", "| dist:", r.get("distance_m"))
    
    print("\nMemory snapshot for user:")
    pretty_print_result(memory_agent.get_user_memory(user_id))
    
    user_id2 = "user_demo2"
    symptoms2 = ["mild headache", "low-grade fever"]
    plan2 = coordinator.handle_request(user_id=user_id2, symptoms=symptoms2, location_str=None, vitals={"temperature_c": 37.2})
    observ.record(plan2["triage"].get("level", "low"))
    print("\n=== Example 2 Triage ===")
    pretty_print_result(plan2["triage"])
    
    print("\n=== Observability Metrics ===")
    pretty_print_result(observ.report())

# 12) Interactive helper (use in notebook cells)
def interactive_session(user_id: str):
    print("=== MEDI-LOCATOR INTERACTIVE ===")
    location = input("Enter your city/address (or press Enter to use memory): ").strip() or None
    symptoms_raw = input("Enter symptoms (comma-separated): ").strip()
    symptoms = [s.strip() for s in symptoms_raw.split(",") if s.strip()]
    temp_str = input("Enter temperature in C (optional): ").strip()
    vitals = {}
    if temp_str:
        try:
            vitals["temperature_c"] = float(temp_str)
        except:
            pass
    plan = coordinator.handle_request(user_id=user_id, symptoms=symptoms, location_str=location, vitals=vitals)
    print("\n--- Plan ---")
    pretty_print_result(plan)
    print("\nSaved to memory.")


