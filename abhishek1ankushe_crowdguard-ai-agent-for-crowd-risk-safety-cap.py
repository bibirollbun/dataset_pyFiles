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


# Crowd Risk Safety Agent - Robust geocoding with 403 fallback (Kaggle-friendly)
# Requirements: requests, pandas (optional)
# pip install requests pandas

import requests
import math
import json
import os
from datetime import datetime
from typing import Dict, Any, Tuple

# -------------------------
# Helper: Offline/static lookup (for demo in blocked environments)
# -------------------------
STATIC_LOOKUP = {
    "azad maidan, mumbai, india": {
        "display_name": "Azad Maidan, Mumbai, Maharashtra, India",
        "lat": 18.9750,
        "lon": 72.8225,
        # bounding box: min_lon, min_lat, max_lon, max_lat
        "bbox": (72.8205, 18.9735, 72.8245, 18.9765)
    },
    "wembley stadium, london": {
        "display_name": "Wembley Stadium, London, UK",
        "lat": 51.5560,
        "lon": -0.2796,
        "bbox": (-0.2815, 51.5540, -0.2775, 51.5580)
    },
    "trafalgar square, london": {
        "display_name": "Trafalgar Square, London, UK",
        "lat": 51.5080,
        "lon": -0.1281,
        "bbox": (-0.1295, 51.5070, -0.1265, 51.5090)
    },
}

# Attempt to parse a simple CSV-like local dataset if present
def parse_local_dataset(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Try to parse local dataset file into a mapping of address_lower -> {display_name, lat, lon, bbox}
    This function is forgiving: it expects lines with address,lat,lon or address|lat|lon etc.
    """
    mapping = {}
    if not os.path.exists(path):
        return mapping
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # try common separators
                for sep in [",", "|", "\t", ";"]:
                    if sep in line:
                        parts = [p.strip() for p in line.split(sep)]
                        break
                else:
                    parts = [line]
                # heuristics: find two floats in parts to treat as lat/lon
                floats = []
                for p in parts:
                    try:
                        v = float(p)
                        floats.append(v)
                    except Exception:
                        pass
                if len(floats) >= 2:
                    # assume parts contain address then lat then lon
                    # find indices of floats
                    lat, lon = floats[0], floats[1]
                    # reconstruct address from parts excluding numeric ones
                    text_parts = [p for p in parts if not _is_float(p)]
                    address = text_parts[0] if text_parts else f"{lat},{lon}"
                    mapping[address.lower()] = {
                        "display_name": address,
                        "lat": lat,
                        "lon": lon,
                        # make a small bbox around the point (+/- tiny delta)
                        "bbox": (lon - 0.0006, lat - 0.0006, lon + 0.0006, lat + 0.0006)
                    }
    except Exception:
        # ignore parsing failures, return whatever we collected
        pass
    return mapping

def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False

# Pre-parse local dataset (if available in Kaggle input)
LOCAL_DATASET_PATH = "/kaggle/input/agents-intensive-capstone-project/Hackathon dataset.txt"
LOCAL_MAPPING = parse_local_dataset(LOCAL_DATASET_PATH) if os.path.exists(LOCAL_DATASET_PATH) else {}

# -------------------------
# Utilities / Tools (with robust geocode)
# -------------------------
def get_location_from_address(address: str, limit: int = 1, allow_fallback: bool = True) -> Dict[str, Any]:
    """
    Extended geocoder:
     - Try Nominatim with polite headers
     - On HTTPError 403 or other failures, try local dataset or static lookup fallbacks
    Returns a dict similar to Nominatim result or {} if not found.
    """
    if not address or not address.strip():
        return {}

    addr_clean = address.strip()
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": addr_clean, "format": "jsonv2", "limit": limit}
    # Strong headers: include a real-looking User-Agent and Referer
    headers = {
        "User-Agent": "CrowdRiskAgent/1.0 (demo@example.com)",
        "Referer": "https://www.kaggle.com/"
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=12)
        # If server returns 403 or 429 etc, raise for status to catch below
        resp.raise_for_status()
        results = resp.json()
        if results:
            # ensure we convert to expected shape (display_name, lat, lon, boundingbox)
            r = results[0]
            # Standardize boundingbox: ensure list of 4 floats
            bbox = r.get("boundingbox")
            # Some versions return [south, north, west, east] or [minlat, maxlat, minlon, maxlon]
            # We'll return boundingbox so downstream parser deals with it similarly to earlier code
            return r
        # empty results - fall through to fallback
    except requests.exceptions.HTTPError as he:
        # If forbidden (403) or rate-limited (429), we'll attempt fallback
        status = None
        try:
            status = he.response.status_code
        except Exception:
            status = None
        # Only silently continue to fallback if we allow fallback
        if not allow_fallback:
            raise
    except Exception:
        # network error, timeouts etc -> use fallback
        pass

    # ---------- Fallback logic ----------
    # 1) Check local pre-parsed mapping (from dataset)
    key = addr_clean.lower()
    if key in LOCAL_MAPPING:
        entry = LOCAL_MAPPING[key]
        return {
            "display_name": entry["display_name"],
            "lat": str(entry["lat"]),
            "lon": str(entry["lon"]),
            "boundingbox": [str(entry["bbox"][1]), str(entry["bbox"][3]), str(entry["bbox"][0]), str(entry["bbox"][2])]
        }

    # 2) Try static lookup dictionary
    if key in STATIC_LOOKUP:
        ent = STATIC_LOOKUP[key]
        return {
            "display_name": ent["display_name"],
            "lat": str(ent["lat"]),
            "lon": str(ent["lon"]),
            "boundingbox": [str(ent["bbox"][1]), str(ent["bbox"][3]), str(ent["bbox"][0]), str(ent["bbox"][2])]
        }

    # 3) Try fuzzy match in local mapping/static lookup (contains)
    for mkey, ent in {**LOCAL_MAPPING, **STATIC_LOOKUP}.items():
        if mkey in key or key in mkey:
            # ent may be different shape if from LOCAL_MAPPING
            if isinstance(ent, dict) and "lat" in ent and "lon" in ent:
                return {
                    "display_name": ent.get("display_name", mkey),
                    "lat": str(ent["lat"]),
                    "lon": str(ent["lon"]),
                    "boundingbox": [str(ent["bbox"][1]), str(ent["bbox"][3]), str(ent["bbox"][0]), str(ent["bbox"][2])]
                }
    # Nothing found
    return {}

def bbox_area_m2(bbox: Tuple[float, float, float, float]) -> float:
    """
    Approximate rectangular area in square meters from bbox = (min_lon, min_lat, max_lon, max_lat)
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    avg_lat = (min_lat + max_lat) / 2.0
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * math.cos(math.radians(avg_lat))
    height_m = (max_lat - min_lat) * meters_per_deg_lat
    width_m = (max_lon - min_lon) * meters_per_deg_lon
    area = abs(width_m * height_m)
    return area

def classify_density_risk(density_ppsm: float) -> Tuple[str, int]:
    if density_ppsm < 0.5:
        return "Safe", 20
    elif density_ppsm < 1.5:
        return "Moderate", 60
    else:
        return "High", 95

# -------------------------
# Agent implementations
# -------------------------
def sensor_agent(address: str) -> Dict[str, Any]:
    raw = get_location_from_address(address)
    if not raw:
        return {"error": "address_not_found"}
    # Nominatim returns strings for lat/lon; convert to float
    try:
        lat = float(raw.get("lat"))
        lon = float(raw.get("lon"))
    except Exception:
        return {"error": "invalid_geocode_result"}

    bbox_raw = raw.get("boundingbox") or raw.get("bounding_box") or raw.get("bbox")
    # Normalize bounding box into (min_lon, min_lat, max_lon, max_lat)
    try:
        if bbox_raw and len(bbox_raw) == 4:
            # many Nominatim responses are [south_lat, north_lat, west_lon, east_lon]
            south, north, west, east = map(float, bbox_raw)
            min_lat, max_lat = south, north
            min_lon, max_lon = west, east
        else:
            # fallback tiny box around point
            delta = 0.0005
            min_lon, min_lat, max_lon, max_lat = lon - delta, lat - delta, lon + delta, lat + delta
    except Exception:
        delta = 0.0005
        min_lon, min_lat, max_lon, max_lat = lon - delta, lat - delta, lon + delta, lat + delta

    area_m2 = bbox_area_m2((min_lon, min_lat, max_lon, max_lat))
    display_name = raw.get("display_name") or address
    return {
        "address": display_name,
        "lat": lat,
        "lon": lon,
        "bbox": (min_lon, min_lat, max_lon, max_lat),
        "area_m2": area_m2,
        "raw": raw,
    }

def analyzer_agent(sensor_output: Dict[str, Any], people_count: int) -> Dict[str, Any]:
    if "error" in sensor_output:
        return {"error": "sensor_failure"}
    area = sensor_output.get("area_m2")
    if not area or area <= 0:
        density = float("inf")
    else:
        density = people_count / area
    label, score = classify_density_risk(density)
    return {
        "people_count": people_count,
        "area_m2": area,
        "density_ppl_per_m2": density,
        "risk_label": label,
        "risk_score": score,
    }

def safety_advisor_agent(sensor_output: Dict[str, Any], analysis_output: Dict[str, Any]) -> Dict[str, Any]:
    if "error" in sensor_output or "error" in analysis_output:
        return {"error": "upstream_failure"}
    label = analysis_output.get("risk_label")
    density = analysis_output.get("density_ppl_per_m2")
    people = analysis_output.get("people_count")
    area = analysis_output.get("area_m2")

    recommendations = []
    if label == "Safe":
        recommendations.append("Density is low. Continue regular monitoring.")
        recommendations.append("Ensure clear signage and maintain crowd flow.")
    elif label == "Moderate":
        recommendations.append("Limit additional entrants until density lowers.")
        recommendations.append("Open additional exit/entry corridors to reduce buildup.")
        recommendations.append("Increase staff to guide crowd movement.")
    else:
        recommendations.append("Immediate action required: consider temporary halt to new entrants.")
        recommendations.append("Disperse crowd using controlled routing and public announcements.")
        recommendations.append("Contact local authorities for support and medical standby.")

    report = (
        "Location: {address}\n"
        "Lat/Lon: {lat:.6f}, {lon:.6f}\n"
        "Estimated area: {area:.1f} m^2\n"
        "People: {people}\n"
        "Density: {density:.4f} ppl/m^2\n"
        "Risk: {label} (score {score})\n"
    ).format(
        address=sensor_output.get("address"),
        lat=sensor_output.get("lat"),
        lon=sensor_output.get("lon"),
        area=area if area is not None else 0.0,
        people=people,
        density=density if density is not None else float("inf"),
        label=label,
        score=analysis_output.get("risk_score"),
    )

    return {
        "report_text": report,
        "recommendations": recommendations
    }

# -------------------------
# Orchestrator / Coordinator
# -------------------------
MEMORY_FILE = "crowd_agent_memory.json"

def load_memory() -> Dict[str, Any]:
    try:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"sessions": []}

def save_memory(mem: Dict[str, Any]):
    with open(MEMORY_FILE, "w") as f:
        json.dump(mem, f, indent=2, default=str)

def orchestrator_agent(address: str, people_count: int, session_name: str = None) -> Dict[str, Any]:
    sensor_out = sensor_agent(address)
    analysis_out = analyzer_agent(sensor_out, people_count)
    advice_out = safety_advisor_agent(sensor_out, analysis_out)

    timestamp = datetime.utcnow().isoformat() + "Z"
    session = {
        "session_name": session_name or f"session_{timestamp}",
        "timestamp": timestamp,
        "address_input": address,
        "sensor": sensor_out,
        "analysis": analysis_out,
        "advice": advice_out,
    }

    mem = load_memory()
    mem.setdefault("sessions", [])
    mem["sessions"].append(session)
    save_memory(mem)

    result = {
        "id": session["session_name"],
        "timestamp": timestamp,
        "address": sensor_out.get("address"),
        "lat": sensor_out.get("lat"),
        "lon": sensor_out.get("lon"),
        "area_m2": sensor_out.get("area_m2"),
        "people": analysis_out.get("people_count"),
        "density_ppl_per_m2": analysis_out.get("density_ppl_per_m2"),
        "risk_label": analysis_out.get("risk_label"),
        "risk_score": analysis_out.get("risk_score"),
        "recommendations": advice_out.get("recommendations"),
        "report_text": advice_out.get("report_text"),
    }
    return result

# -------------------------
# Demo: run examples
# -------------------------
if __name__ == "__main__":
    examples = [
        {"address": "Azad Maidan, Mumbai, India", "people": 1500},
        {"address": "Wembley Stadium, London", "people": 60000},
        {"address": "Trafalgar Square, London", "people": 800},
    ]

    results = []
    for e in examples:
        try:
            out = orchestrator_agent(e["address"], e["people"])
            print("-----")
            print("Session:", out["id"])
            print(out["report_text"])
            print("Recommendations:")
            for r in out.get("recommendations", []):
                print(" -", r)
            results.append({
                "session": out["id"],
                "address": out["address"],
                "people": out["people"],
                "area_m2": out["area_m2"],
                "density": out["density_ppl_per_m2"],
                "risk": out["risk_label"],
            })
        except Exception as exc:
            print("Error for", e["address"], ":", exc)

    # Optional: show summary table if pandas available
    try:
        import pandas as pd
        if results:
            df = pd.DataFrame(results)
            print("\nSummary table:")
            print(df.to_string(index=False))
    except Exception:
        pass

    print("\nMemory saved to:", MEMORY_FILE)
    print("Call orchestrator_agent(address, people_count) to test new locations.")



import json
with open("crowd_agent_memory.json") as f:
    mem = json.load(f)
print(len(mem["sessions"]))
print(mem["sessions"][-1])   # show last session



res = orchestrator_agent("Gate of India, Mumbai, India", 2000)
print(res["report_text"])



# Load and display saved sessions
import json
from pathlib import Path
import pandas as pd
from IPython.display import display, JSON, Markdown

MEMORY_FILE = "crowd_agent_memory.json"

def load_sessions(mem_file=MEMORY_FILE):
    p = Path(mem_file)
    if not p.exists():
        print(f"No memory file found at {mem_file}")
        return {"sessions": []}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

mem = load_sessions()
sessions = mem.get("sessions", [])

# 1) Count
display(Markdown(f"**Saved sessions:** {len(sessions)}"))

# 2) Summary table for quick view
rows = []
for s in sessions:
    try:
        rows.append({
            "session_name": s.get("session_name"),
            "timestamp": s.get("timestamp"),
            "address_input": s.get("address_input"),
            "people": s.get("analysis", {}).get("people_count"),
            "area_m2": s.get("analysis", {}).get("area_m2"),
            "density": s.get("analysis", {}).get("density_ppl_per_m2"),
            "risk": s.get("analysis", {}).get("risk_label"),
        })
    except Exception:
        continue

if rows:
    df = pd.DataFrame(rows)
    # sort by timestamp (most recent last)
    try:
        df = df.sort_values("timestamp", ascending=True)
    except Exception:
        pass
    display(df)
else:
    print("No session rows to display.")
    
# 3) Show last session JSON (pretty)
if sessions:
    last = sessions[-1]
    display(Markdown("**Last session (full JSON):**"))
    display(JSON(last))



# Inspect one saved session by index or session_name
def get_session(identifier):
    """
    identifier: int (0-based index) or str session_name
    """
    if not sessions:
        print("No sessions available.")
        return None
    if isinstance(identifier, int):
        if 0 <= identifier < len(sessions):
            return sessions[identifier]
        else:
            print("Index out of range.")
            return None
    elif isinstance(identifier, str):
        for s in sessions:
            if s.get("session_name") == identifier:
                return s
        print("No session found with that name.")
        return None
    else:
        print("Identifier must be int (index) or str (session_name).")
        return None

# Example usage:
# session = get_session(-1)          # last session
# session = get_session(0)           # first session
# session = get_session("session_2025-11-15T06:47:16.998542Z")  # by name

session = get_session(-1)   # change this line as needed
if session:
    import pprint
    pprint.pprint(session)



# Export sessions to a JSON file for submission
EXPORT_FILE = "exported_sessions.json"
with open(EXPORT_FILE, "w", encoding="utf-8") as f:
    json.dump(mem, f, indent=2, ensure_ascii=False)
print("Exported sessions to", EXPORT_FILE)



# Write a minimal agent.py module that imports functions from the notebook if possible,
# otherwise writes a canonical implementation. This avoids duplication when running in the notebook.

module_path = Path("agent.py")
if module_path.exists():
    print("agent.py already exists. Overwriting...")

module_code = '''
# agent.py - Crowd Risk Safety Agent module
# This module exposes orchestrator_agent(address, people_count)

import json
from datetime import datetime

# NOTE: This module expects the heavy functions (sensor_agent, analyzer_agent, safety_advisor_agent)
# to be injected when running in notebook context. If they are not present, this module contains
# fallback implementations (simple versions).
try:
    # If notebook has defined these, import them into this module's globals
    from __main__ import sensor_agent, analyzer_agent, safety_advisor_agent, load_memory, save_memory
except Exception:
    # Fallback minimal implementations (not intended for production)
    def sensor_agent(address):
        return {"address": address, "lat": 0.0, "lon": 0.0, "bbox": (0,0,0,0), "area_m2": 1.0, "raw": {}}
    def analyzer_agent(sensor_output, people_count):
        return {"people_count": people_count, "area_m2": sensor_output.get("area_m2",1.0), "density_ppl_per_m2": people_count/1.0, "risk_label":"Unknown", "risk_score":0}
    def safety_advisor_agent(sensor_output, analysis_output):
        return {"report_text": "Fallback report", "recommendations": []}
    def load_memory():
        return {"sessions": []}
    def save_memory(mem):
        pass

MEMORY_FILE = "crowd_agent_memory.json"

def orchestrator_agent(address: str, people_count: int, session_name: str = None) -> dict:
    sensor_out = sensor_agent(address)
    analysis_out = analyzer_agent(sensor_out, people_count)
    advice_out = safety_advisor_agent(sensor_out, analysis_out)
    timestamp = datetime.utcnow().isoformat() + "Z"
    session = {
        "session_name": session_name or f"session_{timestamp}",
        "timestamp": timestamp,
        "address_input": address,
        "sensor": sensor_out,
        "analysis": analysis_out,
        "advice": advice_out,
    }
    mem = load_memory()
    mem.setdefault("sessions", [])
    mem["sessions"].append(session)
    try:
        save_memory(mem)
    except Exception:
        pass
    return {
        "id": session["session_name"],
        "timestamp": timestamp,
        "address": sensor_out.get("address"),
        "lat": sensor_out.get("lat"),
        "lon": sensor_out.get("lon"),
        "area_m2": sensor_out.get("area_m2"),
        "people": analysis_out.get("people_count"),
        "density_ppl_per_m2": analysis_out.get("density_ppl_per_m2"),
        "risk_label": analysis_out.get("risk_label"),
        "risk_score": analysis_out.get("risk_score"),
        "recommendations": advice_out.get("recommendations"),
        "report_text": advice_out.get("report_text"),
    }
'''

# Save to agent.py
with open(module_path, "w", encoding="utf-8") as f:
    f.write(module_code)
print("Wrote agent.py to", module_path.resolve())
print("You can now `from agent import orchestrator_agent` in future cells or scripts.")



from agent import orchestrator_agent



result = orchestrator_agent("Azad Maidan, Mumbai, India", 1500)
print(result["report_text"])



# Run this once in a Kaggle cell (permissions in Kaggle allow pip)
!pip install shapely pyproj



from shapely.geometry import shape, Polygon, mapping
from shapely.ops import transform
from pyproj import Transformer, CRS
import json
from typing import Any, Dict

# Function: compute polygon area in m^2 given GeoJSON-like geometry (WGS84 lon/lat)
def polygon_area_m2(geojson_geometry: Dict[str, Any]) -> float:
    """
    Takes a GeoJSON geometry (Polygon or MultiPolygon) in WGS84 (lon,lat),
    projects it to an equal-area / metric CRS (EPSG:3857) and returns area in m^2.
    Note: EPSG:3857 is ok for small-to-medium areas; for high accuracy across latitudes
    consider using an equal-area projection like 'EPSG:6933' (NSIDC EASE-Grid) or local UTM.
    """
    geom = shape(geojson_geometry)
    # project from EPSG:4326 -> EPSG:3857 (meters)
    project = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
    geom_m = transform(project, geom)   # now in meters
    return float(geom_m.area)

# Example: simple square polygon around a point (for testing)
# Coordinates are in (lon, lat)
example_poly = {
    "type": "Polygon",
    "coordinates": [[
        [72.8205, 18.9735],
        [72.8245, 18.9735],
        [72.8245, 18.9765],
        [72.8205, 18.9765],
        [72.8205, 18.9735]
    ]]
}

print("Example polygon area (m2):", polygon_area_m2(example_poly))



def sensor_agent(address: str) -> Dict[str, Any]:
    # 1. Try geocoding (with fallback)
    raw = get_location_from_address(address)
    if not raw:
        return {"error": "address_not_found"}

    # Convert lat/lon
    try:
        lat = float(raw.get("lat"))
        lon = float(raw.get("lon"))
    except Exception:
        return {"error": "invalid_geocode_result"}

    # -------------------------------------------
    # 2. AREA CALCULATION (polygon if available)
    # -------------------------------------------
    area_m2 = None

    # If Nominatim / Overpass returns polygon geometry
    if "geojson" in raw and raw["geojson"]:
        try:
            area_m2 = polygon_area_m2(raw["geojson"])
        except Exception:
            area_m2 = None

    # 3. If polygon area not available, fallback to bbox
    if area_m2 is None:
        bbox_raw = raw.get("boundingbox")
        try:
            if bbox_raw and len(bbox_raw) == 4:
                # boundingbox = [south_lat, north_lat, west_lon, east_lon]
                south, north, west, east = map(float, bbox_raw)
                min_lat, max_lat = south, north
                min_lon, max_lon = west, east
            else:
                # fallback tiny box
                delta = 0.0005
                min_lon, min_lat, max_lon, max_lat = lon - delta, lat - delta, lon + delta, lat + delta
        except:
            delta = 0.0005
            min_lon, min_lat, max_lon, max_lat = lon - delta, lat - delta, lon + delta, lat + delta

        area_m2 = bbox_area_m2((min_lon, min_lat, max_lon, max_lat))

    # Final output
    display_name = raw.get("display_name", address)
    return {
        "address": display_name,
        "lat": lat,
        "lon": lon,
        "area_m2": area_m2,
        "bbox": raw.get("boundingbox"),
        "raw": raw,
    }



res = orchestrator_agent("Azad Maidan, Mumbai, India", 1500)
print(res["report_text"])



# Run this once in a notebook cell
!pip install shapely pyproj requests



import requests
import math
import json
import time
from datetime import datetime
from shapely.geometry import shape, Polygon, mapping
from shapely.ops import transform
from pyproj import Transformer
from typing import Dict, Any, Tuple, Optional



def polygon_area_m2(geojson_geometry: Dict[str, Any]) -> float:
    """
    Compute area (m^2) from a GeoJSON geometry in EPSG:4326 (lon,lat).
    Projects to EPSG:3857 (meters) and returns area.
    For best results in large regions consider using local UTM or equal-area CRS.
    """
    geom = shape(geojson_geometry)
    project = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
    geom_m = transform(project, geom)
    return float(geom_m.area)



OVERPASS_URL = "https://overpass-api.de/api/interpreter"

def fetch_overpass_polygon(place_name: str, max_tries: int = 2, pause: float = 1.0) -> Optional[Dict[str, Any]]:
    """
    Try to fetch a polygon geometry (GeoJSON-like) for `place_name` from Overpass.
    Returns a GeoJSON geometry dict (Polygon/MultiPolygon) or None if not found.
    This function will:
      - Query relation[name=place] first, then ways
      - Request 'geom' output to get coordinates
    """
    if not place_name or not place_name.strip():
        return None
    name = place_name.strip()
    # Overpass query: look for relation or way with exact name (case sensitive in Overpass)
    # We'll try a few variants: exact, then case-insensitive via regex.
    queries = [
        f'[out:json][timeout:25];(relation["name"="{name}"];);out geom;',

        f'[out:json][timeout:25];(way["name"="{name}"];);out geom;',

        # regex case-insensitive fallback
        f'[out:json][timeout:25];(relation["name"~"{name}",i];);out geom;',

        f'[out:json][timeout:25];(way["name"~"{name}",i];);out geom;'
    ]

    headers = {
        "User-Agent": "CrowdRiskAgent/1.0 (demo@example.com)",
        "Accept": "application/json",
    }

    for q in queries:
        for attempt in range(max_tries):
            try:
                resp = requests.post(OVERPASS_URL, data=q.encode("utf-8"), headers=headers, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    elements = data.get("elements", [])
                    if not elements:
                        break  # try next query
                    # Prefer relations (they often represent full polygons)
                    # Build GeoJSON geometry from way/relation geometry returned as 'geometry' array
                    # Elements with 'type' 'relation' may have a 'members' with roles; Overpass 'geom' provides 'geometry' for relation too
                    for el in elements:
                        geom = None
                        if "geometry" in el:
                            coords = [(pt["lon"], pt["lat"]) for pt in el["geometry"]]
                            # If element is a closed ring, build Polygon; else try MultiPolygon logic later
                            if len(coords) >= 4 and coords[0] == coords[-1]:
                                geom = {"type": "Polygon", "coordinates": [coords]}
                            else:
                                # As fallback, try to close ring if near-closed
                                if len(coords) >= 3:
                                    coords_closed = coords[:]
                                    if coords_closed[0] != coords_closed[-1]:
                                        coords_closed.append(coords_closed[0])
                                    geom = {"type": "Polygon", "coordinates": [coords_closed]}
                        # If we constructed a geom, return it
                        if geom:
                            return geom
                    # If elements present but couldn't make geom, try next query
                    break
                elif resp.status_code in (429, 504):
                    # rate limited or gateway timeout - backoff and retry
                    time.sleep(pause * (attempt + 1))
                    continue
                else:
                    # other HTTP error -> do not spam, move to next query
                    break
            except requests.exceptions.RequestException:
                time.sleep(pause * (attempt + 1))
                continue
    # Nothing found
    return None



def sensor_agent(address: str) -> Dict[str, Any]:
    """
    Enhanced sensor agent:
     - Try Overpass to fetch polygon geometry (best)
     - If Overpass fails, fall back to Nominatim 'geojson' if present
     - Then fall back to bbox approximation or static lookup
    """
    raw = get_location_from_address(address)  # your existing function; expects Nominatim or fallback dict
    if not raw:
        return {"error": "address_not_found"}

    # parse lat/lon
    try:
        lat = float(raw.get("lat"))
        lon = float(raw.get("lon"))
    except Exception:
        return {"error": "invalid_geocode_result"}

    area_m2 = None

    # First: try Overpass polygon by place name (more reliable footprint)
    try:
        geom = fetch_overpass_polygon(address)
        if geom:
            try:
                area_m2 = polygon_area_m2(geom)
            except Exception:
                area_m2 = None
    except Exception:
        area_m2 = None

    # Second: if Overpass failed, check if Nominatim returned 'geojson' (some endpoints include it)
    if area_m2 is None and "geojson" in raw and raw["geojson"]:
        try:
            area_m2 = polygon_area_m2(raw["geojson"])
        except Exception:
            area_m2 = None

    # Third: fallback to bbox-based area
    if area_m2 is None:
        bbox_raw = raw.get("boundingbox")
        try:
            if bbox_raw and len(bbox_raw) == 4:
                south, north, west, east = map(float, bbox_raw)
                min_lat, max_lat = south, north
                min_lon, max_lon = west, east
            else:
                delta = 0.0005
                min_lon, min_lat, max_lon, max_lat = lon - delta, lat - delta, lon + delta, lat + delta
        except Exception:
            delta = 0.0005
            min_lon, min_lat, max_lon, max_lat = lon - delta, lat - delta, lon + delta, lat + delta

        area_m2 = bbox_area_m2((min_lon, min_lat, max_lon, max_lat))

    display_name = raw.get("display_name", address)
    return {
        "address": display_name,
        "lat": lat,
        "lon": lon,
        "area_m2": area_m2,
        "bbox": raw.get("boundingbox"),
        "raw": raw,
    }



# Test a place that Overpass likely knows
res = orchestrator_agent("Trafalgar Square, London", 1000)
print(res["report_text"])
print("Area used (m2):", res["area_m2"])











