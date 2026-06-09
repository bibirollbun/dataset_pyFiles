# Core imports
import math, random, json, os, datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# For datetime convenience
from datetime import datetime, timedelta, time


def haversine(lat1, lon1, lat2, lon2):
    # returns km
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def estimate_travel_time_minutes(lat1, lon1, lat2, lon2, mode='walk'):
    # heuristic speeds: walking 5 km/h, biking 12 km/h, driving 30 km/h (incl overhead)
    km = haversine(lat1, lon1, lat2, lon2)
    speeds = {'walk':5.0, 'bike':12.0, 'drive':30.0, 'transit':20.0}
    spd = speeds.get(mode, 5.0)
    hours = km / spd
    minutes = hours * 60
    # Add small fixed overhead per leg (access/wait)
    overhead = 7 if mode in ('drive','transit') else 5
    return int(minutes + overhead)


# We'll generate a reproducible synthetic POI dataset for Tokyo
random.seed(42)
np.random.seed(42)

TOKYO_CENTER = (35.6895, 139.6917)  # lat, lon

categories = [
    ("Temple/Shrine", 20),
    ("Museum", 25),
    ("Park/Garden", 30),
    ("Viewpoint", 10),
    ("Shopping", 30),
    ("Food/Restaurant", 40),
    ("Cafe", 20),
    ("Market", 15),
    ("Entertainment", 10),
    ("Historic site", 10),
]

def generate_pois(n=200, center=TOKYO_CENTER, radius_km=15):
    pois = []
    for i in range(n):
        cat = random.choices([c for c,_ in categories], weights=[w for _,w in categories])[0]
        # random point within radius (approx)
        ang = random.random() * 2*math.pi
        r = radius_km * math.sqrt(random.random())
        # approx small-display lat lon
        dlat = (r * math.cos(ang)) / 110.574
        dlon = (r * math.sin(ang)) / (111.320 * math.cos(math.radians(center[0])))
        lat = center[0] + dlat
        lon = center[1] + dlon
        # opening hours (simple)
        opens = random.choice([9,10,11,12])
        closes = random.choice([17,18,19,20,21,22])
        if cat in ("Food/Restaurant","Cafe","Market","Entertainment"):
            # later hours
            opens = random.choice([8,10,11,12])
            closes = random.choice([20,21,22,23,24])
        duration = random.choice([30,45,60,90,120])  # minutes typical visit duration
        rating = round(random.uniform(3.0, 4.9),1)
        name = f"{cat} {i+1}"
        pois.append({
            "id": i+1,
            "name": name,
            "category": cat,
            "lat": lat,
            "lon": lon,
            "opens": opens,
            "closes": closes,
            "duration_min": duration,
            "rating": rating
        })
    return pd.DataFrame(pois)

pois_df = generate_pois(200)
pois_df.head()



def load_pois(df=None):
    # accepts DataFrame or path to csv
    if isinstance(df, pd.DataFrame):
        return df.copy()
    else:
        return pd.read_csv(df)

def nearby_pois(center_lat, center_lon, radius_km, df, categories=None):
    df = df.copy()
    df['dist_km'] = df.apply(lambda r: haversine(center_lat, center_lon, r['lat'], r['lon']), axis=1)
    res = df[df['dist_km'] <= radius_km]
    if categories:
        res = res[res['category'].isin(categories)]
    return res.sort_values('dist_km').reset_index(drop=True)

def is_poi_open(poi_row, dt: datetime):
    # poi_row has 'opens' and 'closes' as hour ints (0-24)
    h = dt.hour
    o = int(poi_row['opens'])
    c = int(poi_row['closes']) % 24
    if o <= h < c:
        return True
    # Note: does not handle overnight spans here
    return False



def fit_day_plan(start_lat, start_lon, day_date: datetime.date, start_time: time, end_time: time,
                 candidate_pois: pd.DataFrame, mode='walk', max_stops=8):
    # simple greedy planner: start from start_time, pick nearest open poi that fits time windows
    current_dt = datetime.combine(day_date, start_time)
    end_dt = datetime.combine(day_date, end_time)
    itinerary = []
    cur_lat, cur_lon = start_lat, start_lon
    taken_ids = set()
    attempts = 0
    while current_dt < end_dt and len(itinerary) < max_stops and attempts < 500:
        attempts += 1
        # compute travel times to candidates not taken
        cand = candidate_pois[~candidate_pois['id'].isin(taken_ids)].copy()
        if cand.empty:
            break
        cand['travel_min'] = cand.apply(lambda r: estimate_travel_time_minutes(cur_lat, cur_lon, r['lat'], r['lon'], mode=mode), axis=1)
        # arrival time
        cand['arrive_dt'] = cand['travel_min'].apply(lambda m: current_dt + timedelta(minutes=int(m)))
        # filter those that are open at arrival
        cand['is_open'] = cand['arrive_dt'].apply(lambda dt: is_poi_open(cand.loc[cand.index[0]] if False else cand.iloc[0], dt)) # placeholder fix below
        # NOTE: computing is_open row-wise (correct approach)
        def _is_open_row(row):
            return is_poi_open(row, row['arrive_dt'])
        cand['is_open'] = cand.apply(_is_open_row, axis=1)
        # compute finish time if visited
        cand['finish_dt'] = cand.apply(lambda r: r['arrive_dt'] + timedelta(minutes=int(r['duration_min'])), axis=1)
        # keep those that finish before end_dt
        cand = cand[cand['finish_dt'] <= end_dt]
        if cand.empty:
            break
        # score by a heuristic: rating / (travel_min + 1)
        cand['score'] = cand['rating'] / (cand['travel_min'] + 1)
        cand = cand.sort_values('score', ascending=False)
        chosen = cand.iloc[0]
        itinerary.append({
            "poi_id": int(chosen['id']),
            "name": chosen['name'],
            "category": chosen['category'],
            "lat": float(chosen['lat']),
            "lon": float(chosen['lon']),
            "arrive": chosen['arrive_dt'],
            "finish": chosen['finish_dt'],
            "travel_min": int(chosen['travel_min']),
            "visit_min": int(chosen['duration_min']),
            "rating": float(chosen['rating'])
        })
        # update current time and position
        current_dt = chosen['finish_dt'] + timedelta(minutes=10)  # small break
        cur_lat, cur_lon = chosen['lat'], chosen['lon']
        taken_ids.add(int(chosen['id']))
    return itinerary



def render_gantt(itinerary, title="Itinerary"):
    if not itinerary:
        print("No itinerary to render.")
        return
    rows = []
    for i, it in enumerate(itinerary):
        rows.append({
            "task": f"{i+1}: {it['name']} ({it['category']})",
            "start": it['arrive'],
            "end": it['finish']
        })
    df = pd.DataFrame(rows)
    df['start_num'] = df['start'].map(lambda t: t.timestamp()/60)  # minutes
    df['end_num'] = df['end'].map(lambda t: t.timestamp()/60)
    fig, ax = plt.subplots(figsize=(10, max(2, len(df)*0.6)))
    for i,r in df.iterrows():
        ax.broken_barh([(r['start_num'], r['end_num'] - r['start_num'])], (i-0.4, 0.8))
        ax.text(r['start_num'] + 5, i, r['task'], va='center', ha='left', color='white', fontsize=9)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['task'])
    ax.set_title(title)
    # set x labels as HH:MM
    mins = sorted(set(df['start_num'].tolist() + df['end_num'].tolist()))
    def min_to_str(m):
        dt = datetime.fromtimestamp(m*60)
        return dt.strftime("%H:%M")
    ax.set_xlabel("Time")
    ax.set_xlim(mins[0]-60, mins[-1]+60)
    plt.tight_layout()
    plt.show()



# GEMINI wrapper (example). The notebooks recommend storing API keys in Kaggle Secrets.
# On Kaggle: go to "Settings" -> "Secrets" and add a Secret named "GOOGLE_API_KEY".
# Then access it here via the environment or Kaggle secrets client.

def get_gemini_key_from_env():
    # Kaggle exposes secrets via environment variables in some contexts.
    # If using Kaggle Secrets UI, set a secret 'GEMINI_API_KEY' there.
    key = os.environ.get("GOOGLE_API_KEY")
    if key:
        return key
    # Alternative: Kaggle may use the kaggle_secrets library; show a safe fallback attempt
    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
        v = client.get_secret("GOOGLE_API_KEY")
        return v
    except Exception as e:
        # No key found; in the workshop they instruct not to put API keys in code cells.
        return None

GOOGLE_API_KEY = get_gemini_key_from_env()
print("Gemini key found:", bool(GOOGLE_API_KEY))

def call_gemini_system(prompt: str, temperature=0.0, max_tokens=1024):
    """
    Wrapper: POST to Gemini-like endpoint. Placeholder — DO NOT PUT YOUR KEY HERE.
    In Kaggle environment you can implement the actual HTTP call using requests and the GEMINI_API_KEY.
    Example (pseudocode):
      headers = {"Authorization": f"Bearer {GOOGLE_API_KEY}"}
      resp = requests.post("https://api.example.com/v1/generate", json={...}, headers=headers)
    The workshop notebooks show similar wrapper patterns; we will follow the same call shape.
    """
    if not GOOGLE_API_KEY:
        raise RuntimeError("No GEMINI_API_KEY found in environment. Add it to Kaggle Secrets.")
    # We'll not perform the actual call in this demo cell to keep the notebook runnable offline.
    # When enabled, implement the HTTP call here.
    return {"text": "<<<Gemini response placeholder>>>", "raw": {}}



# POI Ranker & Helper utilities

from typing import Optional
import uuid
import os, json

# Add a reproducible scoring function for ranking POIs based on preferences
def poi_score_vectorized(df: pd.DataFrame, center_lat: float, center_lon: float,
                         preferences: Dict[str, Any], radius_km: float = 15.0):
    """
    Compute scores for POIs using a vectorized heuristic combining:
      - distance penalty
      - category interest match
      - rating
      - category popularity (optional)
    preferences: {"interests": ["Food/Restaurant","Museum"], "max_radius_km": 10, "min_rating": 3.0}
    Returns df with new column 'score'
    """
    df = df.copy()
    df['dist_km'] = df.apply(lambda r: haversine(center_lat, center_lon, r['lat'], r['lon']), axis=1)
    # distance factor: closer -> better
    df['dist_factor'] = 1 / (1 + df['dist_km'])
    # rating factor normalized
    df['rating_factor'] = (df['rating'] - 2.5) / (5.0 - 2.5)  # roughly 0..1
    # interest match
    interests = preferences.get('interests', [])
    if interests:
        df['interest_match'] = df['category'].apply(lambda c: 1.0 if c in interests else 0.0)
    else:
        df['interest_match'] = 0.2  # small baseline
    # category popularity (use frequency in dataset)
    cat_counts = df['category'].value_counts().to_dict()
    df['cat_pop'] = df['category'].apply(lambda c: cat_counts.get(c,1))
    df['cat_pop_norm'] = df['cat_pop'] / (df['cat_pop'].max() + 1e-9)
    # combine into a score
    # weights can be changed by preferences
    w = preferences.get('weights', {'dist':0.4, 'rating':0.3, 'interest':0.2, 'cat':0.1})
    df['score'] = (w['dist'] * df['dist_factor'] +
                   w['rating'] * df['rating_factor'] +
                   w['interest'] * df['interest_match'] +
                   w['cat'] * df['cat_pop_norm'])
    # apply radius and rating filters if provided
    max_r = preferences.get('max_radius_km', radius_km)
    min_r = preferences.get('min_rating', 0.0)
    df = df[(df['dist_km'] <= max_r) & (df['rating'] >= min_r)].copy()
    df = df.sort_values('score', ascending=False).reset_index(drop=True)
    return df

# Quick test: rank near Tokyo center with sample prefs
prefs = {"interests": ["Temple/Shrine", "Food/Restaurant"], "max_radius_km": 10, "min_rating": 3.5}
ranked = poi_score_vectorized(pois_df, TOKYO_CENTER[0], TOKYO_CENTER[1], prefs)
ranked.head(10)



# Simple in-notebook session manager and a file-backed Memory Bank for user prefs

class InMemorySession:
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.created_at = datetime.utcnow().isoformat()
        self.state = {}  # hold ephemeral state like chosen_pois, last_itinerary
        self.trace = []  # observability trace events

    def log(self, event_type: str, payload: Dict[str, Any]):
        ev = {"ts": datetime.utcnow().isoformat(), "type": event_type, "payload": payload}
        self.trace.append(ev)

    def snapshot(self):
        return {"session_id": self.session_id, "created_at": self.created_at, "state": self.state, "trace": self.trace}

class MemoryBank:
    def __init__(self, path="memory_bank.json"):
        self.path = path
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump({}, f)
        self._load()

    def _load(self):
        with open(self.path, "r") as f:
            self.data = json.load(f)

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def get_user(self, user_id: str):
        return self.data.get(user_id, {})

    def set_user(self, user_id: str, info: Dict[str, Any]):
        self.data[user_id] = info
        self.save()

# Initialize a session and memorybank
session = InMemorySession()
mem = MemoryBank()
# Example: save a user preference
user_id = "user_abhilakshya_demo"
mem.set_user(user_id, {"name":"Abhilakshya", "home_country":"India", "preferred_pace":"normal", "veg_pref":False})
print("Memory for user:", mem.get_user(user_id))



# Observability: structured logging of tool calls and decisions
def emit_trace(session: InMemorySession, stage: str, message: str, meta: Dict[str, Any] = None):
    meta = meta or {}
    ev = {"stage": stage, "message": message, "meta": meta}
    session.log("trace", ev)

# Example usage:
emit_trace(session, "ranker", "Ranked 50 POIs based on preferences", {"top_pois": ranked['name'].tolist()[:5]})
session.trace[-1]



def plan_multi_day_itinerary(start_lat: float, start_lon: float,
                             start_date: datetime.date, num_days: int,
                             daily_start_time: time, daily_end_time: time,
                             candidates: pd.DataFrame, preferences: Dict[str, Any],
                             mode='walk', per_day_max=8):
    """
    High-level multi-day planner:
    - Rank candidates once using the POI-Ranker
    - For each day, run a fit_day_plan greedy scheduler starting from either the previous day's end or start location
    - Provide fallback: if a day ends early, consider next best POIs for subsequent days
    Returns: list of day itineraries and a trace of dropped/selected POIs
    """
    results = {"days": [], "selected_pois": [], "dropped": []}
    # rank first
    ranked = poi_score_vectorized(candidates, start_lat, start_lon, preferences, radius_km=preferences.get('max_radius_km',15.0))
    emit_trace(session, "ranker", "Scored candidates", {"num_ranked": len(ranked)})
    remaining = ranked.copy()
    current_start_lat, current_start_lon = start_lat, start_lon
    for d in range(num_days):
        day_date = start_date + timedelta(days=d)
        emit_trace(session, "planner", f"Planning day {d+1} ({day_date})", {"remaining_pois": len(remaining)})
        day_itin = fit_day_plan(current_start_lat, current_start_lon, day_date, daily_start_time, daily_end_time,
                                remaining, mode=mode, max_stops=per_day_max)
        # record selected poi ids
        sel_ids = [it['poi_id'] for it in day_itin]
        results['selected_pois'].extend(sel_ids)
        # remove selected from remaining
        remaining = remaining[~remaining['id'].isin(sel_ids)].reset_index(drop=True)
        # if last visited location exists, set next day's start to last finish location
        if day_itin:
            last = day_itin[-1]
            current_start_lat, current_start_lon = last['lat'], last['lon']
        # append day result
        results['days'].append({"date": str(day_date), "itinerary": day_itin})
    # any leftovers are 'dropped' for the trip (not scheduled)
    results['dropped'] = remaining['id'].tolist()
    emit_trace(session, "planner", "Completed multi-day planning", {"selected_total": len(results['selected_pois']), "dropped": len(results['dropped'])})
    return results

# quick test plan (tiny)
prefs = {"interests": ["Museum","Park/Garden","Food/Restaurant"], "max_radius_km":10, "min_rating":3.5}
demo_plan = plan_multi_day_itinerary(TOKYO_CENTER[0], TOKYO_CENTER[1], datetime.now().date(), 2, time(9,0), time(18,0), pois_df, prefs, mode='walk', per_day_max=6)
# show summary
for d in demo_plan['days']:
    print(d['date'], "=>", len(d['itinerary']), "stops")



def evaluate_itinerary_plan(plan_result):
    """
    Compute:
    - constraint satisfaction: percent of visits within opening hours (we ensure this but still compute)
    - total_travel_minutes and total_visit_minutes
    - travel_overhead_ratio = total_travel / total_visit
    - interest_match_score: proportion of selected POIs matching user's interests (if available)
    """
    total_travel = 0
    total_visit = 0
    matches = 0
    total_selected = 0
    for day in plan_result['days']:
        for it in day['itinerary']:
            total_travel += it.get('travel_min', 0)
            total_visit += it.get('visit_min', 0)
            total_selected += 1
            # simple interest match: compare category with prefs if stored in session
            # we'll rely on last preferences passed in session.state if present
            prefs = session.state.get('last_preferences', {})
            if prefs and 'interests' in prefs:
                if it['category'] in prefs['interests']:
                    matches += 1
    travel_overhead_ratio = total_travel / (total_visit + 1e-9)
    interest_match = matches / (total_selected + 1e-9)
    return {
        "total_selected": total_selected,
        "total_travel_min": total_travel,
        "total_visit_min": total_visit,
        "travel_overhead_ratio": travel_overhead_ratio,
        "interest_match": interest_match
    }

# Evaluate previous demo
session.state['last_preferences'] = prefs
eval_summary = evaluate_itinerary_plan(demo_plan)
eval_summary



# Prompt templates for Gemini-based sub-agents.
PLANNER_SYSTEM_PROMPT = """
You are a strict itinerary planner. You receive structured JSON input specifying candidate POIs, travel times,
and user preferences. You MUST return a JSON object with an "itinerary" array and "explain" field.
You MUST NOT invent new POIs or facts. Use only the information provided. Keep explanation concise (<= 150 words).
"""

REWRITER_SYSTEM_PROMPT = """
You are an assistant that rewrites a structured itinerary into a friendly travel plan for a human user.
Given the itinerary JSON, output a concise natural-language summary per day, and provide packing/contingency tips.
"""

def gemini_plan_from_structured(plan_structured_json: Dict[str, Any], role='planner'):
    """
    Safe wrapper that prepares a prompt payload for Gemini. This function does NOT perform the actual HTTP request
    unless GEMINI_API_KEY is present in environment and the HTTP call is implemented below.
    """
    key = get_gemini_key_from_env()
    system_msg = PLANNER_SYSTEM_PROMPT if role=='planner' else REWRITER_SYSTEM_PROMPT
    prompt = system_msg + "\n\nINPUT_JSON:\n" + json.dumps(plan_structured_json, indent=2, default=str)
    # If API key present, perform an HTTP call (user must implement)
    if not key:
        # Return a placeholder response for offline/demo mode
        return {"ok": False, "reason": "no_key", "summary": "Gemini not enabled in this environment. Add GEMINI_API_KEY in Kaggle Secrets to enable."}
    # else: implement actual HTTP POST to Gemini endpoint (pseudocode)
    # import requests
    # headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    # payload = {"prompt": prompt, "max_tokens": 512}
    # resp = requests.post("https://api.google.com/gemini/v1/generate", json=payload, headers=headers)
    # return resp.json()
    return {"ok": True, "note": "API key found — implement HTTP call in this function to enable live Gemini responses."}

# Example: call with our demo_plan structure (we will pass the JSON containing days & sample metadata)
gemini_demo = gemini_plan_from_structured({"demo_plan": demo_plan}, role='rewriter')
gemini_demo



# Set up three demo scenarios (Tokyo) with different preferences and paces
scenarios = [
    {"name":"1-day Weekend Highlights", "start_date": datetime(2025,12,15).date(), "days":1,
     "start_time": time(9,0), "end_time": time(19,0), "prefs":{"interests":["Viewpoint","Temple/Shrine","Food/Restaurant"], "max_radius_km":8, "min_rating":3.5}, "mode":"walk"},
    {"name":"2-day Relaxed", "start_date": datetime(2025,12,20).date(), "days":2,
     "start_time": time(10,0), "end_time": time(18,0), "prefs":{"interests":["Museum","Park/Garden","Food/Restaurant"], "max_radius_km":12, "min_rating":3.2}, "mode":"transit"},
    {"name":"3-day Packed", "start_date": datetime(2025,12,27).date(), "days":3,
     "start_time": time(8,30), "end_time": time(21,0), "prefs":{"interests":["Shopping","Food/Restaurant","Entertainment"], "max_radius_km":15, "min_rating":3.0}, "mode":"drive"}
]

demo_results = {}
for s in scenarios:
    session = InMemorySession()  # new session per scenario to capture traces
    emit_trace(session, "scenario_start", f"Starting scenario: {s['name']}", {"prefs": s['prefs']})
    session.state['last_preferences'] = s['prefs']
    res = plan_multi_day_itinerary(TOKYO_CENTER[0], TOKYO_CENTER[1],
                                  s['start_date'], s['days'],
                                  s['start_time'], s['end_time'],
                                  pois_df, s['prefs'], mode=s['mode'], per_day_max=6)
    eval_res = evaluate_itinerary_plan(res)
    demo_results[s['name']] = {"plan": res, "eval": eval_res, "session_trace": session.trace}
    # print summary
    print(f"Scenario: {s['name']} -> Selected stops: {eval_res['total_selected']}, Travel min: {eval_res['total_travel_min']}, Interest match: {eval_res['interest_match']:.2f}")

# Render the first day's Gantt for the 1-day scenario
first_scenario = demo_results["1-day Weekend Highlights"]['plan']['days'][0]
render_gantt(first_scenario['itinerary'], title="1-day Weekend Highlights - Day 1")



# Helpers to compute itinerary-level score and validate time windows
def itinerary_total_metrics(itinerary):
    """Return total travel minutes and total visit minutes for a single-day itinerary list"""
    total_travel = sum(it.get('travel_min', 0) for it in itinerary)
    total_visit = sum(it.get('visit_min', 0) for it in itinerary)
    return total_travel, total_visit

def itinerary_score(itinerary, preferences):
    """
    Heuristic itinerary score: weighted combination of:
      - Sum of POI's rating
      - Negative total travel time (less travel is better)
      - Bonus for matching user's interests
    Returns a numeric score (higher is better)
    """
    total_rating = sum(it['rating'] for it in itinerary)
    total_travel, total_visit = itinerary_total_metrics(itinerary)
    interest_bonus = 0
    if preferences and 'interests' in preferences:
        interest_bonus = sum(1 for it in itinerary if it['category'] in preferences['interests'])
    # weights
    score = (2.0 * total_rating) - 0.5 * total_travel + 1.2 * interest_bonus
    return score

def can_insert_poi_into_day(day_date, current_itinerary, candidate_row, start_time, end_time, mode='walk'):
    """
    Check if candidate POI can be appended to the itinerary (naive check).
    We compute travel time from last location (or start location if empty),
    arrival time, and finish time; ensure it doesn't exceed end_time and is open at arrival.
    Returns (ok, arrive_dt, finish_dt, travel_min)
    """
    if current_itinerary:
        last = current_itinerary[-1]
        lat0, lon0 = last['lat'], last['lon']
        current_dt = last['finish'] + timedelta(minutes=10)  # buffer after last
    else:
        # If no items, assume start location is Tokyo center for simplicity here (caller can provide actual)
        lat0, lon0 = TOKYO_CENTER
        current_dt = datetime.combine(day_date, start_time)
    travel_min = estimate_travel_time_minutes(lat0, lon0, candidate_row['lat'], candidate_row['lon'], mode=mode)
    arrive_dt = current_dt + timedelta(minutes=int(travel_min))
    finish_dt = arrive_dt + timedelta(minutes=int(candidate_row['duration_min']))
    if finish_dt > datetime.combine(day_date, end_time):
        return False, None, None, None
    if not is_poi_open(candidate_row, arrive_dt):
        return False, None, None, None
    return True, arrive_dt, finish_dt, travel_min



def iterative_repair_optimize(plan_result: Dict[str, Any], candidates_df: pd.DataFrame,
                              preferences: Dict[str, Any], daily_start_time: time, daily_end_time: time,
                              mode='walk', iterations: int = 200):
    """
    For each day, try to improve the day's itinerary by:
      - considering swapping out one scheduled poi for a promising candidate from remaining pool,
      - or inserting an extra POI if time permits,
      - accept swaps that improve the itinerary heuristic score.
    This is a greedy local-search / first-improvement algorithm (fast and effective for demo).
    Returns new plan_result (deep-copied) and an improvement trace.
    """
    import copy
    new_plan = copy.deepcopy(plan_result)
    # Build a mapping of remaining candidates by id
    remaining = candidates_df[~candidates_df['id'].isin(sum([ [it['poi_id'] for it in d['itinerary']] for d in new_plan['days']], [] ))].reset_index(drop=True)
    improvement_trace = []
    # Precompute ranked candidates to attempt better swaps first
    ranked_candidates = poi_score_vectorized(remaining, TOKYO_CENTER[0], TOKYO_CENTER[1], preferences)
    # iterate days
    for day_idx, day in enumerate(new_plan['days']):
        day_date = datetime.fromisoformat(day['date']).date()
        it = day['itinerary']
        # compute baseline score
        base_score = itinerary_score(it, preferences)
        improved = True
        it_attempts = 0
        while improved and it_attempts < iterations:
            improved = False
            it_attempts += 1
            # try swapping each selected poi with top K ranked candidates
            for sel_index, sel_item in enumerate(it):
                for cand_idx, cand in ranked_candidates.head(50).iterrows():
                    if int(cand['id']) in [x['poi_id'] for x in it]:
                        continue  # skip already present
                    # attempt swap: create a tentative itinerary where selected item replaced by candidate
                    tentative = it.copy()
                    # Check feasibility: compute arrive/finish times for the slot where selected item is placed
                    # We'll approximate by trying to insert the candidate at same position, recomputing times sequentially
                    cur_lat, cur_lon = TOKYO_CENTER if sel_index == 0 and not tentative else (it[0]['lat'], it[0]['lon'])
                    # To recompute times robustly, we'll rebuild the day's itinerary from scratch using the selected set (with replacement),
                    # maintaining order but recomputing travel times and arrival/finish times greedily
                    tentative_ids = [x['poi_id'] for x in it]
                    tentative_ids[sel_index] = int(cand['id'])
                    # rebuild a candidate list in order using ids
                    rebuilt = []
                    cur_dt = datetime.combine(day_date, daily_start_time)
                    cur_lat, cur_lon = TOKYO_CENTER
                    feasible = True
                    for pid in tentative_ids:
                        # find candidate row (either from original pois_df or remaining)
                        row = None
                        if pid in it and any(x['poi_id']==pid for x in it):
                            # original: get row from global pois_df
                            row = pois_df[pois_df['id']==pid].iloc[0] if pid in pois_df['id'].values else None
                        if row is None:
                            # search in candidates (ranked_candidates + remaining)
                            rows = pd.concat([ranked_candidates, remaining], ignore_index=True)
                            rows = rows[rows['id']==pid]
                            if not rows.empty:
                                row = rows.iloc[0]
                        if row is None:
                            feasible = False
                            break
                        travel_min = estimate_travel_time_minutes(cur_lat, cur_lon, row['lat'], row['lon'], mode=mode)
                        arrive_dt = cur_dt + timedelta(minutes=int(travel_min))
                        finish_dt = arrive_dt + timedelta(minutes=int(row['duration_min']))
                        if finish_dt > datetime.combine(day_date, daily_end_time) or not is_poi_open(row, arrive_dt):
                            feasible = False
                            break
                        rebuilt.append({
                            "poi_id": int(row['id']),
                            "name": row['name'],
                            "category": row['category'],
                            "lat": float(row['lat']),
                            "lon": float(row['lon']),
                            "arrive": arrive_dt,
                            "finish": finish_dt,
                            "travel_min": int(travel_min),
                            "visit_min": int(row['duration_min']),
                            "rating": float(row['rating'])
                        })
                        cur_dt = finish_dt + timedelta(minutes=10)
                        cur_lat, cur_lon = row['lat'], row['lon']
                    if not feasible:
                        continue
                    # compute rebuilt score
                    new_score = itinerary_score(rebuilt, preferences)
                    if new_score > base_score + 1e-6:  # accept strict improvement
                        # commit
                        improvement_trace.append({
                            "day": day_idx,
                            "action": "swap",
                            "replaced_poi_id": sel_item['poi_id'],
                            "added_poi_id": int(cand['id']),
                            "old_score": base_score,
                            "new_score": new_score
                        })
                        it[:] = rebuilt  # modify in place
                        base_score = new_score
                        improved = True
                        # update remaining/ranked lists
                        remaining = remaining[~(remaining['id'] == int(cand['id']))].reset_index(drop=True)
                        ranked_candidates = poi_score_vectorized(remaining, TOKYO_CENTER[0], TOKYO_CENTER[1], preferences)
                        break
                if improved:
                    break
            # attempt insertion of an extra POI at the end if time permits
            if not improved:
                # try top ranked candidate for insertion
                for cand_idx, cand in ranked_candidates.head(30).iterrows():
                    ok, arrive_dt, finish_dt, travel_min = can_insert_poi_into_day(day_date, it, cand, daily_start_time, daily_end_time, mode=mode)
                    if ok:
                        # append candidate
                        it.append({
                            "poi_id": int(cand['id']),
                            "name": cand['name'],
                            "category": cand['category'],
                            "lat": float(cand['lat']),
                            "lon": float(cand['lon']),
                            "arrive": arrive_dt,
                            "finish": finish_dt,
                            "travel_min": int(travel_min),
                            "visit_min": int(cand['duration_min']),
                            "rating": float(cand['rating'])
                        })
                        new_score = itinerary_score(it, preferences)
                        improvement_trace.append({
                            "day": day_idx,
                            "action": "insert",
                            "added_poi_id": int(cand['id']),
                            "new_score": new_score
                        })
                        remaining = remaining[~(remaining['id'] == int(cand['id']))].reset_index(drop=True)
                        ranked_candidates = poi_score_vectorized(remaining, TOKYO_CENTER[0], TOKYO_CENTER[1], preferences)
                        improved = True
                        base_score = new_score
                        break
        # end while
    # end day loop
    return new_plan, improvement_trace



# A2A message sender, router, and example flow 
def send_a2a_message(session: InMemorySession, sender: str, recipient: str, msg_type: str, payload: dict):
    """
    Create and log an A2A message into the session trace.
    Returns the message dict.
    """
    msg = {
        "id": str(uuid.uuid4()),
        "ts": datetime.utcnow().isoformat(),
        "from": sender,
        "to": recipient,
        "type": msg_type,
        "payload": payload
    }
    session.log("a2a_message", msg)
    return msg

def process_a2a_message(session: InMemorySession, msg: dict):
    """
    Router that simulates sub-agent processing:
      - POI-Ranker: handles 'rank_request' -> returns top candidates
      - Planner: handles 'plan_request' -> returns a planned itinerary
      - Rewriter: handles 'rewrite_request' -> returns a human-friendly summary (placeholder)
    All responses are sent back as A2A messages and logged to session.trace.
    """
    typ = msg['type']
    sender = msg['from']
    recipient = msg['to']
    payload = msg.get('payload', {})

    # ---------------------------
    # 1) POI-RANKER AGENT
    # ---------------------------
    if recipient == "poi_ranker" and typ == "rank_request":
        prefs = payload.get('preferences', {})
        center = payload.get('center', TOKYO_CENTER)
        # Use the vectorized ranker to get top candidates
        ranked = poi_score_vectorized(pois_df, center[0], center[1], prefs, radius_km=prefs.get('max_radius_km', 15))
        topk = ranked.head(payload.get('top_k', 50)).to_dict(orient='records')
        return send_a2a_message(session, "poi_ranker", sender, "rank_response", {"candidates": topk})

    # ---------------------------
    # 2) PLANNER AGENT
    # ---------------------------
    if recipient == "planner" and typ == "plan_request":
        # Build candidates DataFrame if provided; otherwise fallback to global pois_df
        candidates = pd.DataFrame(payload.get('candidates', []))
        if candidates.empty:
            candidates = pois_df.copy()

        result = plan_multi_day_itinerary(
            payload['start_lat'], payload['start_lon'],
            datetime.fromisoformat(payload['start_date']).date(),
            payload['num_days'],
            time.fromisoformat(payload['daily_start_time']),
            time.fromisoformat(payload['daily_end_time']),
            candidates,
            payload.get('preferences', {}),
            mode=payload.get('mode', 'walk')
        )
        return send_a2a_message(session, "planner", sender, "plan_response", {"plan": result})

    # ---------------------------
    # 3) REWRITER AGENT
    # ---------------------------
    if recipient == "rewriter" and typ == "rewrite_request":
        plan = payload.get('plan', {})
        texts = []
        for d in plan.get('days', []):
            # Safely build a short list of stops for the summary
            stops_list = [f"{it['name']} ({it['category']})" for it in d.get('itinerary', [])[:3]]
            stops = ", ".join(stops_list)
            if len(d.get('itinerary', [])) > 3:
                stops += "..."
            texts.append(f"On {d.get('date')}, you'll visit {len(d.get('itinerary', []))} stops: {stops}")
        summary = " ".join(texts) if texts else "No stops planned."
        return send_a2a_message(session, "rewriter", sender, "rewrite_response", {"summary": summary})

    # ---------------------------
    # DEFAULT: Echo back
    # ---------------------------
    return send_a2a_message(session, recipient, sender, "echo", {"original": payload})


# ---------------------------
# Example A2A Flow
# ---------------------------
session = InMemorySession()  # fresh session to capture traces

# 1) Orchestrator asks Ranker for candidates
rank_request = send_a2a_message(session, "orchestrator", "poi_ranker", "rank_request", {
    "preferences": {"interests": ["Food/Restaurant", "Viewpoint"], "max_radius_km": 8, "min_rating": 3.5},
    "center": TOKYO_CENTER,
    "top_k": 60
})
rank_response = process_a2a_message(session, rank_request)

# 2) Orchestrator forwards those candidates to Planner for a 1-day plan
candidates = rank_response['payload']['candidates']
plan_request = send_a2a_message(session, "orchestrator", "planner", "plan_request", {
    "start_lat": TOKYO_CENTER[0],
    "start_lon": TOKYO_CENTER[1],
    "start_date": datetime(2025, 12, 15).date().isoformat(),
    "num_days": 1,
    "daily_start_time": "09:00",
    "daily_end_time": "19:00",
    "candidates": candidates,
    "preferences": {"interests": ["Food/Restaurant", "Viewpoint"], "max_radius_km": 8, "min_rating": 3.5},
    "mode": "walk"
})
plan_response = process_a2a_message(session, plan_request)

# 3) Orchestrator asks Rewriter to produce a human summary
rewrite_request = send_a2a_message(session, "orchestrator", "rewriter", "rewrite_request", {"plan": plan_response['payload']['plan']})
rewrite_response = process_a2a_message(session, rewrite_request)

# Print last few A2A messages and the rewrite summary
print("Last 4 A2A messages (most recent first):")
for entry in session.trace[-4:]:
    if entry['type'] == 'a2a_message':
        payload = entry['payload']
        print(f"- [{payload['ts']}] {payload['from']} -> {payload['to']} : {payload['type']}")
    else:
        print(entry)

print("\nRewrite summary:")
print(rewrite_response['payload']['summary'])



# Run one of the scenarios again, optimize and compare metrics
scenario = scenarios[0]  # 1-day Weekend Highlights from earlier
session = InMemorySession()
emit_trace(session, "scenario_start", f"Optimizing scenario: {scenario['name']}", {"prefs": scenario['prefs']})
session.state['last_preferences'] = scenario['prefs']

# Generate initial plan using planner (as earlier)
initial_plan = plan_multi_day_itinerary(TOKYO_CENTER[0], TOKYO_CENTER[1],
                                       scenario['start_date'], scenario['days'],
                                       scenario['start_time'], scenario['end_time'],
                                       pois_df, scenario['prefs'], mode=scenario['mode'], per_day_max=6)

# Evaluate before
eval_before = evaluate_itinerary_plan(initial_plan)

# Optimize
optimized_plan, improvements = iterative_repair_optimize(initial_plan, pois_df, scenario['prefs'], scenario['start_time'], scenario['end_time'], mode=scenario['mode'], iterations=200)

# Evaluate after
eval_after = evaluate_itinerary_plan(optimized_plan)

# Print summary of improvements
print("BEFORE: selected:", eval_before['total_selected'], "travel_min:", eval_before['total_travel_min'], "interest_match:", round(eval_before['interest_match'],2))
print("AFTER:  selected:", eval_after['total_selected'], "travel_min:", eval_after['total_travel_min'], "interest_match:", round(eval_after['interest_match'],2))
print("\nImprovements log (top 5):")
for imp in improvements[:5]:
    print(imp)

# Render Gantt of before and after for day 1 (side-by-side)
print("\nGantt BEFORE:")
render_gantt(initial_plan['days'][0]['itinerary'], title="Before Optimization - Day 1")
print("\nGantt AFTER:")
render_gantt(optimized_plan['days'][0]['itinerary'], title="After Optimization - Day 1")

# Add optimization trace to session for observability
emit_trace(session, "optimizer", "Applied iterative repair optimization", {"improvements_count": len(improvements), "sample": improvements[:3]})
session.trace[-1]


