import os
# Set your OpenWeather API key
os.environ["OPENWEATHER_API_KEY"] = "YOUR_API_KEY_HERE"

import os, csv, math, datetime, json, requests
from typing import List, Dict, Any, Optional

# ---------------- Config & data ----------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data") if "__file__" in globals() else "/mnt/data"
SHELTER_CSV = os.path.join(DATA_DIR, "shelters_sample.csv")
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")
os.makedirs(DATA_DIR, exist_ok=True)

# create sample CSV (Option A) if missing
if not os.path.exists(SHELTER_CSV):
    sample = """name,lat,lon,address
Chennai Shelter A,13.0827,80.2707,Chennai City Center, Chennai
Chennai Shelter B,13.0500,80.2500,Anna Nagar Community Hall, Chennai
Coimbatore Shelter A,11.0168,76.9558,Gandhipuram Relief Camp, Coimbatore
Coimbatore Shelter B,11.0200,76.9700,VOC Park Community Hall, Coimbatore
Madurai Shelter A,9.9252,78.1198,Madurai Corporation Shelter, Madurai
Madurai Shelter B,9.9490,78.1210,Anna Nagar Hall, Madurai
Kanchipuram Shelter A,12.8350,79.7000,Kanchi Relief Center, Kanchipuram
Kanchipuram Shelter B,12.8423,79.7155,Ennaikaran Municipal Hall, Kanchipuram
Tirunelveli Shelter A,8.7263,77.7290,Palayamkottai Camp, Tirunelveli
Tirunelveli Shelter B,8.7281,77.6938,Pettai Municipal Shelter, Tirunelveli
Bengaluru Shelter A,12.9716,77.5946,Bengaluru High School, Bengaluru
Bengaluru Shelter B,12.9340,77.6266,Koramangala Shelter, Bengaluru
"""
    with open(SHELTER_CSV, "w", encoding="utf-8") as f:
        f.write(sample)

# ---------------- Helpers ----------------
def now_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

# ---------------- Memory/session ----------------
class InMemorySessionService:
    def __init__(self):
        self.sessions = {}
    def create_session(self, sid):
        self.sessions[sid] = {"history": [], "created_at": now_iso()}; return self.sessions[sid]
    def get_session(self, sid): return self.sessions.get(sid)
    def append_message(self, sid, msg):
        s = self.get_session(sid) or self.create_session(sid); s["history"].append({"timestamp": now_iso(), **msg})

class MemoryBank:
    def __init__(self): self.store = {}
    def save_user_profile(self, uid, profile): self.store[uid] = profile.copy()
    def get_user_profile(self, uid): return self.store.get(uid, {})

session_service = InMemorySessionService()
memory_bank = MemoryBank()

# ---------------- Shelters ----------------
def load_shelters(csv_path=SHELTER_CSV) -> List[Dict[str,Any]]:
    out=[]
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    lat=float(r["lat"]); lon=float(r["lon"])
                except Exception:
                    continue
                addr = (r.get("address") or "")
                parts = [p.strip() for p in addr.split(",") if p.strip()]
                city = parts[-1].lower() if parts else ""
                out.append({"name": r.get("name",""), "lat":lat, "lon":lon, "address":addr, "city":city})
    except FileNotFoundError:
        pass
    return out

def find_nearby_shelters(lat, lon, radius_km=15.0, max_results=5):
    s = load_shelters()
    res=[]
    for item in s:
        d=haversine_km(lat, lon, item["lat"], item["lon"])
        if d<=radius_km:
            it=item.copy(); it["distance_km"]=round(d,2); res.append(it)
    res.sort(key=lambda x: x["distance_km"])
    return res[:max_results]

def find_shelters_by_city(city, lat, lon, max_results=5):
    city_norm=(city or "").strip().lower()
    all_s=load_shelters()
    matched=[it for it in all_s if it.get("city","")==city_norm]
    if not matched:
        return find_nearby_shelters(lat, lon, radius_km=50.0, max_results=max_results)
    res=[]
    for it in matched:
        d=haversine_km(lat, lon, it["lat"], it["lon"])
        i = it.copy(); i["distance_km"]=round(d,2); res.append(i)
    res.sort(key=lambda x: x["distance_km"]); return res[:max_results]

# ---------------- Triage ----------------
IMMEDIATE={"trapped","fire","flooding","drowning","collapsed","injured","bleeding","sinking","explosion"}
WARNING={"warning","evacuate","alert","advisory"}
def triage_text(txt):
    t=(txt or "").lower()
    if any(k in t for k in IMMEDIATE): return {"level":"high","advice":"Immediate danger detected. Call emergency services and, if safe, move to higher ground."}
    if any(k in t for k in WARNING): return {"level":"medium","advice":"Potential risk detected. Prepare to evacuate and follow local authority instructions."}
    return {"level":"low","advice":"No immediate danger detected based on the message. Stay informed."}

# ---------------- Alerts (strict local within 300 km) ----------------
def fetch_openweather_alerts(lat, lon):
    if not OPENWEATHER_API_KEY:
        return [{"source":"OpenWeather","error":"OPENWEATHER_API_KEY not set"}]
    try:
        url=f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude=minutely,hourly&appid={OPENWEATHER_API_KEY}&units=metric"
        r=requests.get(url, timeout=10); r.raise_for_status(); j=r.json()
        alerts=j.get("alerts") or []
        if not alerts: return [{"source":"OpenWeather","info":"no_alerts"}]
        out=[]
        for a in alerts:
            out.append({"source":"OpenWeather","title":a.get("event"),"description":a.get("description"),"sender":a.get("sender_name"),"start":a.get("start"),"end":a.get("end")})
        return out
    except Exception as e:
        return [{"source":"OpenWeather","error":str(e)}]

def fetch_open_meteo_alerts(lat, lon):
    try:
        url=f"https://api.open-meteo.com/v1/gfs?latitude={lat}&longitude={lon}&hourly=temperature_2m&alerts=1&timezone=auto"
        r=requests.get(url, timeout=10); r.raise_for_status(); j=r.json()
        alerts=[]; a=j.get("alerts")
        if isinstance(a, dict):
            nested=a.get("alert") or a.get("alerts") or []
            if isinstance(nested, list): alerts=nested
        elif isinstance(a, list): alerts=a
        out=[]
        for al in alerts: out.append({"source":"OpenMeteo","title":al.get("event") or al.get("title"), "description":al.get("description"), "severity":al.get("severity")})
        if not out: return [{"source":"OpenMeteo","info":"no_alerts"}]
        return out
    except Exception as e:
        return [{"source":"OpenMeteo","error":str(e)}]

def fetch_usgs_nearby(lat, lon, max_km=300):
    try:
        url="https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"
        r=requests.get(url, timeout=10); r.raise_for_status(); j=r.json()
        results=[]
        for feat in j.get("features",[]):
            prop=feat.get("properties",{}); geom=feat.get("geometry",{})
            coords=geom.get("coordinates") or []
            if len(coords)<2: continue
            eq_lon, eq_lat = coords[0], coords[1]
            dist=haversine_km(lat, lon, eq_lat, eq_lon)
            if dist<=max_km:
                mag=prop.get("mag"); place=prop.get("place"); time=prop.get("time")
                results.append({"source":"USGS","title":f"M {mag} - {place}", "magnitude":mag, "place":place, "distance_km":round(dist,1), "time":time})
        if not results: return [{"source":"USGS","info":"no_nearby_earthquakes"}]
        results.sort(key=lambda x: x["distance_km"]); return results[:10]
    except Exception as e:
        return [{"source":"USGS","error":str(e)}]

def fetch_advisories(query, lat=None, lon=None):
    advisories=[]
    if lat is not None and lon is not None:
        ow=fetch_openweather_alerts(lat, lon); advisories.extend(ow)
        # fallback if openweather gave only error/info
        if all((a.get("source")=="OpenWeather" and (a.get("info")=="no_alerts" or a.get("error"))) for a in ow):
            om=fetch_open_meteo_alerts(lat, lon); advisories.extend(om)
        # USGS nearby only (strict local per Option C)
        usgs=fetch_usgs_nearby(lat, lon); advisories.extend(usgs)
    else:
        advisories.append({"source":"OpenWeather","error":"no_coordinates_provided"})
        advisories.extend(fetch_usgs_nearby(0,0))  # will likely return error or no nearby
    return advisories

# ---------------- Weather & reverse geocode ----------------
def get_weather(lat, lon):
    try:
        url=f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        r=requests.get(url, timeout=8); r.raise_for_status(); j=r.json()
        cw=j.get("current_weather",{})
        return {"temperature":cw.get("temperature"), "windspeed":cw.get("windspeed"), "weathercode":cw.get("weathercode")}
    except Exception as e:
        return {"error":str(e)}

def reverse_geocode(lat, lon):
    try:
        url=f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"
        r=requests.get(url, headers={"User-Agent":"safeharbor/1.0"}, timeout=8); r.raise_for_status(); j=r.json()
        addr=j.get("address",{})
        for k in ("city","town","village","county","state"):
            if addr.get(k): return addr.get(k)
        return None
    except Exception:
        return None

# ---------------- Planner & summarizer ----------------
def generate_plan(profile, disaster_type, nearest):
    family_size=int(profile.get("family_size",1))
    mobility=bool(profile.get("mobility_issues",False)); pets=bool(profile.get("pets",False)); meds=profile.get("medications",[])
    plan={"disaster_type":disaster_type,"evacuation_steps":[],"checklist":[]}
    dt=(disaster_type or "").lower()
    if dt in ("flood","cyclone"):
        plan["evacuation_steps"]+=["Move to higher ground away from flood zones.","Turn off gas and electricity if safe to do so.","Avoid driving through flooded roads."]
    elif dt=="earthquake":
        plan["evacuation_steps"]+=["Drop, Cover, and Hold On until shaking stops.","Move to an open area away from buildings and power lines after shaking."]
    else:
        plan["evacuation_steps"]+=["Follow official instructions from authorities."]
    if nearest:
        plan["evacuation_steps"].append(f"Nearest shelter: {nearest.get('name')} ({nearest.get('distance_km')} km) - {nearest.get('address')}")
    plan["checklist"]+=[f"Water - 3 liters per person per day (for {family_size} people)","Non-perishable food - 3 days supply","Flashlight and extra batteries","Battery-powered radio or charged phone with power bank","First-aid kit and essential medications","Important documents in waterproof bag (IDs, insurance)"]
    if meds: plan["checklist"].append("Medications: "+", ".join(meds))
    if mobility: plan["checklist"].append("Mobility aids and assistive devices")
    if pets: plan["checklist"].append("Pet food, carrier, and veterinary records")
    return plan

def summarize_disaster_context_local(q, advisories, weather, locname):
    parts=[f"Query: {q}"]
    if locname: parts.append(f"Location: {locname}")
    if weather and not weather.get("error"): parts.append(f"Current temp: {weather.get('temperature')}°C, wind: {weather.get('windspeed')} m/s")
    if advisories:
        top = ", ".join(sorted(list({a.get("source","") for a in advisories})))
        parts.append(f"Advisories from: {top}.")
    danger=0
    for a in advisories:
        s=(a.get("source") or "").lower()
        if "usgs" in s or "openweather" in s or "openmeteo" in s or a.get("severity"): danger+=1
    if danger>=2: parts.append("Overall risk: elevated. Follow local authority instructions and prepare to evacuate if advised.")
    elif danger==1: parts.append("Overall risk: caution. Prepare emergency kit and monitor official channels.")
    else: parts.append("Overall risk: low based on available feeds.")
    return "\n".join(parts)

# ---------------- Orchestration ----------------
def locate_and_find_shelters(user_text, lat=None, lon=None):
    if lat is None or lon is None: return {"error":"coords_required","shelters":[]}
    txt=(user_text or "").lower()
    known={"chennai","coimbatore","madurai","kanchipuram","tirunelveli","bengaluru"}
    sel=None
    for c in known:
        if c in txt: sel=c; break
    if sel: shelters=find_shelters_by_city(sel, lat, lon)
    else: shelters=find_nearby_shelters(lat, lon, radius_km=50.0)
    return {"lat":lat,"lon":lon,"shelters":shelters,"city_used":sel}

def fetch_weather_and_location(lat, lon): return get_weather(lat, lon), reverse_geocode(lat, lon)

def handle_user_request(session_id, user_id, user_text, lat=None, lon=None, disaster_type="flood"):
    session_service.append_message(session_id, {"from":"user","text":user_text})
    triage=triage_text(user_text)
    advisories=fetch_advisories(user_text, lat=lat, lon=lon)
    loc_info = locate_and_find_shelters(user_text, lat=lat, lon=lon) if (lat is not None and lon is not None) else {"error":"No coords","shelters":[]}
    weather, locname = fetch_weather_and_location(lat, lon)
    user_profile = memory_bank.get_user_profile(user_id) or {}
    nearest = loc_info.get("shelters")[0] if loc_info.get("shelters") else None
    plan = generate_plan(user_profile, disaster_type, nearest)
    summary = summarize_disaster_context_local(user_text, advisories, weather, locname)
    resp = {"timestamp": now_iso(), "triage":triage, "advisories":advisories, "location":loc_info, "weather":weather, "summary":summary, "plan":plan}
    session_service.append_message(session_id, {"from":"assistant","text":resp})
    return resp

# ---------------- CLI (interactive) ----------------
def interactive_run():
    print("SafeHarbor — Interactive Mode\nCommands: /location (change coords) | /exit (quit)\n")
    user_id="cli_user"
    if not memory_bank.get_user_profile(user_id):
        print("Let's create your emergency profile.")
        name=input("Your Name: ") or "User"
        try: family_size=int(input("Family size (number): ") or "1")
        except: family_size=1
        mobility=input("Any mobility issues? (yes/no): ").lower().startswith("y")
        pets=input("Do you have pets? (yes/no): ").lower().startswith("y")
        meds=input("Any medications? (comma separated): ")
        meds_list=[m.strip() for m in meds.split(",")] if meds else []
        memory_bank.save_user_profile(user_id, {"name":name,"family_size":family_size,"mobility_issues":mobility,"pets":pets,"medications":meds_list,"preferred_language":"en"})
        print("Profile saved!\n")
    print("Enter your current location coordinates.")
    while True:
        try:
            lat=float(input("Latitude: ").strip()); lon=float(input("Longitude: ").strip()); break
        except: print("Enter valid numeric coordinates.")
    session_id="interactive_session_1"
    while True:
        user_text=input("\nYou'r query: ").strip()
        if not user_text: continue
        if user_text.lower() in ("/exit","exit","quit"): print("Exiting SafeHarbor. Stay Safe!"); break
        if user_text.lower() in ("/location","/loc"):
            print("Update location:")
            while True:
                try: lat=float(input("New Latitude: ").strip()); lon=float(input("New Longitude: ").strip()); print(f"Location updated to ({lat}, {lon})"); break
                except: print("Enter valid numbers.")
            continue
        disaster_type="flood"
        lo=user_text.lower()
        if "earthquake" in lo or "tremor" in lo: disaster_type="earthquake"
        elif "fire" in lo: disaster_type="fire"
        result=handle_user_request(session_id, user_id, user_text, lat=lat, lon=lon, disaster_type=disaster_type)
        print("\n=== SafeHarbor Response ===")
        print(f"Time: {result['timestamp']}")
        print(f"Triage level: {result['triage']['level']}")
        print(f"Advice: {result['triage']['advice']}\n")
        print("Advisories:")
        for a in result["advisories"][:10]:
            if a.get("error"): print(f"  - {a.get('source')}: ERROR: {a.get('error')}")
            elif a.get("info"): print(f"  - {a.get('source')}: {a.get('info')}")
            else:
                title=a.get("title") or a.get("description") or "(no title)"
                if a.get("source")=="USGS": # nearby eq format
                    place=a.get("place") or ""
                    dist=a.get("distance_km")
                    if dist: print(f"  - {a.get('source')}: {title} ({dist} km away)")
                    else: print(f"  - {a.get('source')}: {title}")
                else:
                    sender=a.get("sender") or a.get("severity") or ""
                    print(f"  - {a.get('source')}: {title} {('- '+sender) if sender else ''}")
        loc_block=result.get("location",{}); used_city=loc_block.get("city_used")
        if used_city: print(f"\nShelters (city detected: {used_city.title()}):")
        else: print("\nShelters (nearest):")
        for s in loc_block.get("shelters",[]): print(f"  - {s.get('name')} ({s.get('distance_km')} km) — {s.get('address')}")
        w=result.get("weather",{})
        if not w.get("error"): print(f"\nWeather: {w.get('temperature')}°C, wind {w.get('windspeed')} m/s")
        else: print(f"\nWeather: ERROR: {w.get('error')}")
        print("\nSuggested Plan / Checklist:")
        for step in result["plan"]["evacuation_steps"]: print(f"  - {step}")
        for item in result["plan"]["checklist"]: print(f"  - {item}")
        print("\n--- End of response ---\n")

if __name__ == "__main__":
    interactive_run()

