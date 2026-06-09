"""
CivicGuard - lightweight prototype core pipeline.
Single-file demo for Kaggle / local runs.
No external APIs required. Swap LLM_STUB with real LLM call easily.
"""

import json
import time
import uuid
import random
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple

# ---------------------------
# Sample Mock Data
# ---------------------------
SAMPLE_SOCIAL_POSTS = [
    {"id": "t1", "source": "tweet", "time": "2025-11-24T10:12:00Z", "geo": {"lat": 37.77, "lon": -122.42},
     "text": "Water rising fast on Elm St near 5th! Cars stuck.", "meta": {"likes": 3}},
    {"id": "t2", "source": "tweet", "time": "2025-11-24T10:13:05Z", "geo": {"lat": 37.7705, "lon": -122.419},
     "text": "Elm St sidewalks flooded, be careful.", "meta": {"likes": 1}},
    {"id": "r1", "source": "reddit", "time": "2025-11-24T10:11:30Z", "geo": {"lat": 37.78, "lon": -122.41},
     "text": "Flooding reported near Riverside Market. Traffic bad.", "meta": {"ups": 5}},
]

SAMPLE_WEATHER = [
    {"id": "w1", "source": "weather_api", "time": "2025-11-24T10:00:00Z", "geo": {"lat": 37.77, "lon": -122.42},
     "text": "Heavy rainfall cell over downtown. Flash flood warning issued.", "meta": {"severity": "high"}}
]

COMMUNITY_ASSETS = [
    {"id": "shelter_1", "name": "Community Hall", "lat": 37.7715, "lon": -122.418, "capacity": 200},
    {"id": "shelter_2", "name": "High School Gym", "lat": 37.765, "lon": -122.425, "capacity": 500},
]

# ---------------------------
# Utilities
# ---------------------------
def now_iso():
    return datetime.now(timezone.utc).isoformat()

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# ---------------------------
# Canonical EventRecord
# ---------------------------
def to_event_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "event_id": str(uuid.uuid4()),
        "source": raw.get("source"),
        "orig_id": raw.get("id"),
        "time": raw.get("time"),
        "geo": raw.get("geo"),
        "text": raw.get("text"),
        "meta": raw.get("meta", {}),
        "ingested_at": now_iso()
    }

# ---------------------------
# Ingestion Agents (parallel)
# ---------------------------
def ingest_social(posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    time.sleep(0.1)
    return [to_event_record(p) for p in posts]

def ingest_weather(weather: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    time.sleep(0.05)
    return [to_event_record(w) for w in weather]

def parallel_ingest(social, weather) -> List[Dict[str, Any]]:
    ingested = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {
            ex.submit(ingest_social, social): 'social',
            ex.submit(ingest_weather, weather): 'weather'
        }
        for fut in as_completed(futures):
            res = fut.result()
            ingested.extend(res)
    return ingested

# ---------------------------
# Validation & Triage Agent
# ---------------------------
def simple_source_trust(source: str) -> float:
    trust = {"weather_api": 0.95, "tweet": 0.5, "reddit": 0.6, "official_rss": 0.9}
    return trust.get(source, 0.4)

def keyword_severity(text: str) -> float:
    text_l = text.lower()
    if any(k in text_l for k in ["flash flood", "flooding", "water entering", "waters rising", "evacuate"]):
        return 0.9
    if any(k in text_l for k in ["flood", "heavy rain", "traffic bad", "blocked"]):
        return 0.6
    return 0.2

def validate_and_score(event: Dict[str, Any]) -> Dict[str, Any]:
    trust = simple_source_trust(event["source"])
    severity = keyword_severity(event["text"])
    age_seconds = (datetime.now(timezone.utc) - datetime.fromisoformat(event["time"].replace("Z","+00:00"))).total_seconds()
    recency = max(0.0, 1 - min(age_seconds / 3600.0, 1))
    evidence_confidence = 0.5 * trust + 0.4 * severity + 0.1 * recency
    event["validation"] = {
        "trust": trust,
        "severity": severity,
        "recency": recency,
        "evidence_confidence": round(evidence_confidence, 3)
    }
    event["priority"] = "high" if evidence_confidence > 0.7 else ("medium" if evidence_confidence > 0.45 else "low")
    return event

# ---------------------------
# Memory & Context (compaction)
# ---------------------------
class MemoryBank:
    def __init__(self, assets: List[Dict[str, Any]]):
        self.assets = assets

    def nearest_shelters(self, lat, lon, k=2):
        scored = []
        for a in self.assets:
            d = haversine_km(lat, lon, a["lat"], a["lon"])
            scored.append((d, a))
        scored.sort(key=lambda x: x[0])
        return [a for _, a in scored[:k]]

    def compact_context(self, location_hint=None) -> str:
        if location_hint:
            s = self.nearest_shelters(location_hint['lat'], location_hint['lon'], k=3)
        else:
            s = self.assets[:3]
        lines = [f"- {p['name']} ({p['capacity']} cap)" for p in s]
        return "Known community assets:\n" + "\n".join(lines)

# ---------------------------
# Resource Registry Tool
# ---------------------------
class ResourceRegistry:
    def __init__(self, assets):
        self.assets = assets

    def query_nearest(self, lat, lon, kind="shelter", max_results=3):
        return MemoryBank(self.assets).nearest_shelters(lat, lon, k=max_results)

# ---------------------------
# Risk Model
# ---------------------------
def risk_score(evidence_confidence, severity_indicator, population_density_factor, historical_incidents):
    score = 0.4 * evidence_confidence + 0.3 * severity_indicator + 0.2 * population_density_factor + 0.1 * historical_incidents
    return round(score, 3)

# ---------------------------
# LLM Stub + Fallback
# ---------------------------
def llm_stub_generate_brief(validated_events, context_compaction):
    top = sorted(validated_events, key=lambda e: e["validation"]["evidence_confidence"], reverse=True)[:3]
    lines = [f"Crisis Brief generated at {now_iso()}:"]
    for e in top:
        lines.append(f"- {e['text']} (confidence {e['validation']['evidence_confidence']})")
    lines.append("\nContext:")
    lines.append(context_compaction)
    return "\n".join(lines), True

def fallback_rule_based_brief(validated_events, context_compaction):
    top = sorted(validated_events, key=lambda e: e["validation"]["evidence_confidence"], reverse=True)[:2]
    brief = f"[Fallback Summary at {now_iso()}]\n"
    brief += "Detected incidents:\n"
    for e in top:
        brief += f"- {e['text']} (confidence {e['validation']['evidence_confidence']})\n"
    return brief + "\n" + context_compaction

# ---------------------------
# Action Recommendation Agent
# ---------------------------
def action_recommendations(primary_event, resource_registry):
    lat = primary_event['geo']['lat']; lon = primary_event['geo']['lon']
    shelters = resource_registry.query_nearest(lat, lon, max_results=2)
    return {
        "nearest_shelters": [{"name": s["name"], "capacity": s["capacity"]} for s in shelters],
        "immediate_actions": [
            "Avoid driving through flooded areas",
            "Move to higher ground if indoors",
            "Call emergency services only for life-threatening issues"
        ]
    }

# ---------------------------
# Evaluation
# ---------------------------
def evaluate_provenance_coverage(validated_events):
    unique_sources = set(e['source'] for e in validated_events)
    return {"coverage": min(1.0, len(unique_sources)/3.0)}

# ---------------------------
# Orchestrator
# ---------------------------
def run_pipeline(social_posts, weather_reports, memory_assets, debug=False):
    ingested = parallel_ingest(social_posts, weather_reports)
    validated = [validate_and_score(e) for e in ingested]
    mem = MemoryBank(memory_assets)
    context = mem.compact_context(validated[0]['geo'])
    top_event = sorted(validated, key=lambda e: e["validation"]["evidence_confidence"], reverse=True)[0]
    risk = risk_score(top_event["validation"]["evidence_confidence"], top_event["validation"]["severity"], 0.6, 0.2)
    brief, ok = llm_stub_generate_brief(validated, context)
    if not ok:
        brief = fallback_rule_based_brief(validated, context)
    actions = action_recommendations(top_event, ResourceRegistry(memory_assets))
    eval_metrics = evaluate_provenance_coverage(validated)
    return {"brief": brief, "risk": risk, "actions": actions, "evaluation": eval_metrics}

# ---------------------------
# Demo
# ---------------------------
result = run_pipeline(SAMPLE_SOCIAL_POSTS, SAMPLE_WEATHER, COMMUNITY_ASSETS, debug=True)
print("=== Crisis Brief ===\n", result["brief"])
print("\n=== Risk Score ===\n", result["risk"])
print("\n=== Actions ===\n", json.dumps(result["actions"], indent=2))
print("\n=== Evaluation ===\n", result["evaluation"])


