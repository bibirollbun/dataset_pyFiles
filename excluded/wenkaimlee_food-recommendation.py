# 1) Environment setup and version print (English)
# Install required packages quietly; print Python and platform for reproducibility.
%pip install -q kaggle pandas numpy matplotlib seaborn tiktoken
import sys, platform
print({'python': sys.version, 'platform': platform.platform()})


# 2) Load data or create a tiny demo (English)
# Try to read the first CSV under ./data; if none/fails, build a small demo DataFrame
import os
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
import glob
csvs = glob.glob(str(DATA_DIR / "*.csv"))
if csvs:
    try:
        df = pd.read_csv(csvs[0])  # expects columns like: name, latitude, longitude, rating, review_count, cuisine, price
        print("Loaded:", csvs[0], "shape:", df.shape)
    except Exception as e:
        print("Failed to load CSV:", e)
        # Fallback demo rows (lat/lon near Taipei); fields match tool expectations
        df = pd.DataFrame({
            'name':['Demo Sushi','Demo Ramen','Demo Burger'],
            'latitude':[25.033,25.034,25.035],
            'longitude':[121.565,121.566,121.567],
            'rating':[4.5,4.2,3.9],
            'review_count':[120,85,40],
            'cuisine':['Japanese','Japanese','American'],
            'price':['$$','$$','$']
        })
else:
    df = pd.DataFrame({
        'name':['Demo Sushi','Demo Ramen','Demo Burger'],
        'latitude':[25.033,25.034,25.035],
        'longitude':[121.565,121.566,121.567],
        'rating':[4.5,4.2,3.9],
        'review_count':[120,85,40],
        'cuisine':['Japanese','Japanese','American'],
        'price':['$$','$$','$']
    })
print("Data ready, shape:", df.shape)


# 3) User inputs and tools (English)
# Define Haversine distance, a simple in-memory store for preferences, and three tools.
import math
from math import radians, sin, cos, sqrt, atan2
import random

# Haversine distance (km): geographic distance between two lat/lon points on Earth
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

# Minimal memory store for user preferences/state
class MemoryStore:
    def __init__(self):
        self.store = {}
    def put(self, k, v):
        self.store[k] = v
    def get(self, k, d=None):
        return self.store.get(k, d)

memory = MemoryStore()

# Read user inputs from environment or defaults
USER_LAT = float(os.getenv("USER_LAT","25.0330"))
USER_LON = float(os.getenv("USER_LON","121.5654"))
PREF_CUISINE = os.getenv("PREF_CUISINE","Japanese")
PREF_PRICE = os.getenv("PREF_PRICE","$$")
PREF_MAX_KM = float(os.getenv("PREF_MAX_KM","3"))

memory.put("user_location", {"lat": USER_LAT, "lon": USER_LON})
memory.put("user_pref", {"cuisine": PREF_CUISINE, "price": PREF_PRICE, "max_km": PREF_MAX_KM})
print("User prefs:", memory.get("user_pref"))

# Tool: filter restaurants by geo distance and preferences
# Expects df with columns: name, latitude, longitude, rating, review_count, cuisine, price
import pandas as pd

def filter_restaurants(location, cuisine=None, price=None, max_km=5.0):
    lat, lon = location["lat"], location["lon"]
    rows = []
    for _, row in df.iterrows():
        rlat, rlon = row.get("latitude"), row.get("longitude")
        if pd.isna(rlat) or pd.isna(rlon):
            continue
        dist = haversine_km(lat, lon, rlat, rlon)
        if dist > max_km:
            continue
        if cuisine and cuisine.lower() not in str(row.get("cuisine","")).lower():
            continue
        if price and str(row.get("price","")) != price:
            continue
        rows.append({
            "name": row.get("name"),
            "latitude": rlat,
            "longitude": rlon,
            "rating": row.get("rating", 0),
            "review_count": row.get("review_count", 0),
            "cuisine": row.get("cuisine", ""),
            "price": row.get("price", ""),
            "distance_km": round(dist,2),
        })
    return rows

# Tool: rank by a simple weighted score (rating ↑, reviews ↑, distance ↓)
def rank_by_score(candidates, w_rating=0.7, w_reviews=0.2, w_distance=0.1):
    def score(r):
        return w_rating*(r.get("rating") or 0) + \
               w_reviews*min((r.get("review_count") or 0)/1000,1) - \
               w_distance*min((r.get("distance_km") or 0)/10,1)
    return sorted(candidates, key=score, reverse=True)

# Tool: random suggestion for exploration
def random_suggestion(candidates, k=3):
    if not candidates:
        return []
    return random.sample(candidates, k=min(k, len(candidates)))


# 4) Single entry recommendation (English)
# Call simple_pick with a mode and top_k to get a clean table.
# Modes: rating (weighted top), distance (nearest), random (explore).
import pandas as pd

def simple_pick(mode: str = "rating", top_k: int = 5):
    # Read preferences and location from memory
    prefs = memory.get("user_pref", {})
    loc = memory.get("user_location", {})
    # Filter candidates using geo distance and user prefs
    candidates = filter_restaurants(loc, cuisine=prefs.get("cuisine"), price=prefs.get("price"), max_km=prefs.get("max_km"))
    if not candidates:
        return pd.DataFrame([], columns=["name","rating","review_count","price","cuisine","distance_km"])
    # Pick by selected mode
    if mode == "distance":
        picks = sorted(candidates, key=lambda r: r.get("distance_km") or 0)[:top_k]
    elif mode == "random":
        picks = random_suggestion(candidates, k=top_k)
    else:
        picks = rank_by_score(candidates)[:top_k]
    # Return a compact table
    return pd.DataFrame(picks)[["name","rating","review_count","price","cuisine","distance_km"]]

# Example call: override via environment variables
PICK_MODE = os.getenv("PICK_MODE","rating")
TOP_K = int(os.getenv("TOP_K","5"))
result_df = simple_pick(mode=PICK_MODE, top_k=TOP_K)
print("Simple recommendations (Top", TOP_K, ") — mode:", PICK_MODE)
result_df

