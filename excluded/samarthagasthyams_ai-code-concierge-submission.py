import os, re, json, time, uuid, logging, pathlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TypedDict

PROJECT_TITLE = "AI Travel Concierge (Flight Planner & Fare Fetcher)"

# ---- ENV / Config ----
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Provider keys 
AMADEUS_API_KEY    = os.getenv("AMADEUS_API_KEY")
AMADEUS_API_SECRET = os.getenv("AMADEUS_API_SECRET")
TEQUILA_API_KEY    = os.getenv("TEQUILA_API_KEY")  # Kiwi.com


# ---- Logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("travel")
log.info("Loaded %s config", PROJECT_TITLE)



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


metrics = {
    "runs": 0,
    "a2a_messages": 0,
    "tool_calls": 0,
    "gemini_calls": 0,
    "provider_calls": {"amadeus": 0, "tequila": 0, "dummy": 0},
}

def bump(key: str, inc: int = 1):
    metrics[key] = metrics.get(key, 0) + inc



class SessionMessage(TypedDict, total=False):
    role: str
    name: str
    content: str
    timestamp: float

class InMemorySessionService:
    def __init__(self):
        self._sessions: Dict[str, List[SessionMessage]] = {}

    def append(self, session_id: str, msg: SessionMessage):
        arr = self._sessions.get(session_id, [])
        arr.append(msg)
        self._sessions[session_id] = arr

    def get(self, session_id: str) -> List[SessionMessage]:
        return self._sessions.get(session_id, [])

class MemoryBank:
    def __init__(self, path: str = "travel_memory_bank.json"):
        self.path = pathlib.Path(path)
        if not self.path.exists():
            self._save({})
        self.bank = self._load()

    def _load(self) -> Dict[str, List[str]]:
        try:
            return json.loads(self.path.read_text())
        except Exception:
            return {}

    def _save(self, data: Dict[str, List[str]]):
        self.path.write_text(json.dumps(data, indent=2))

    def remember(self, topic: str, fact: str):
        arr = self.bank.get(topic, [])
        arr.append(fact)
        self.bank[topic] = arr
        self._save(self.bank)

    def recall(self, topic: str) -> List[str]:
        return self.bank.get(topic, [])

sessions = InMemorySessionService()
bank      = MemoryBank()



def gemini_client():
    try:
        import google.generativeai as genai
        if GOOGLE_API_KEY and GOOGLE_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
            genai.configure(api_key=GOOGLE_API_KEY)
            return genai.GenerativeModel(GEMINI_MODEL)
    except Exception as e:
        log.warning("Gemini client unavailable: %s", e)
    return None

def gemini_generate(prompt: str) -> str:
    model = gemini_client()
    if not model:
        return "[Gemini unavailable; set GOOGLE_API_KEY] " + prompt[:200]
    bump("gemini_calls")
    try:
        resp = model.generate_content(prompt)
        t = getattr(resp, "text", None)
        if callable(t): return t()
        if isinstance(t, str): return t
        return resp.candidates[0].content.parts[0].text
    except Exception as e:
        return f"[Gemini error: {e}]"

def compact_context(session_id: str, threshold: int = 10) -> Optional[SessionMessage]:
    msgs = sessions.get(session_id)
    if len(msgs) < threshold:
        return None
    joined = "\n".join(
        f"[{datetime.fromtimestamp(m.get('timestamp', 0)/1000.0)}] {m.get('role')}/{m.get('name','')}: {m.get('content')}"
        for m in msgs[-50:]
    )
    summary = gemini_generate(
        "You are the Travel Concierge. Summarize the dialogue into a 150-word brief with traveler prefs, dates, "
        "budgets, cabin choices, and open questions.\n\n" + joined
    )
    return {"role": "system", "name": "context_summary", "content": summary, "timestamp": time.time()*1000}



import requests

def provider_amadeus_search(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Amadeus Flight Offers Search v2.
    Requires AMADEUS_API_KEY / AMADEUS_API_SECRET env vars.
    """
    bump("tool_calls"); metrics["provider_calls"]["amadeus"] += 1
    if not (AMADEUS_API_KEY and AMADEUS_API_SECRET):
        return {"ok": False, "error": "Amadeus credentials missing"}

    try:
        # OAuth2 token
        tok = requests.post(
            "https://api.amadeus.com/v1/security/oauth2/token",
            data={"grant_type":"client_credentials","client_id":AMADEUS_API_KEY,"client_secret":AMADEUS_API_SECRET},
            timeout=15
        ).json()
        access = tok.get("access_token")
        if not access:
            return {"ok": False, "error": "Amadeus token error"}

        q = {
            "originLocationCode": params["origin"],
            "destinationLocationCode": params["destination"],
            "departureDate": params["departDate"],
            **({"returnDate": params["returnDate"]} if params.get("returnDate") else {}),
            "adults": params.get("adults", 1),
            "currencyCode": params.get("currency", "USD"),
            "travelClass": params.get("cabin", "ECONOMY"),
            "max": 15
        }
        r = requests.get(
            "https://api.amadeus.com/v2/shopping/flight-offers",
            headers={"Authorization": f"Bearer {access}"},
            params=q, timeout=20
        )
        r.raise_for_status()
        data = r.json()
        offers = []
        for o in data.get("data", []):
            price = float(o["price"]["total"])
            offers.append({
                "provider":"amadeus",
                "price": price,
                "currency": o["price"]["currency"],
                "oneWay": (params.get("returnDate") is None),
                "meta": {"offerId": o.get("id")}
            })
        return {"ok": True, "offers": offers}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def provider_tequila_search(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Kiwi.com Tequila API (Browse quotes).
    Requires TEQUILA_API_KEY env var.
    """
    bump("tool_calls"); metrics["provider_calls"]["tequila"] += 1
    if not TEQUILA_API_KEY:
        return {"ok": False, "error": "Tequila API key missing"}
    try:
        # Resolve IATA if needed: assume user supplies IATA codes; else you can call locations endpoint
        date_from = params["departDate"]
        date_to   = params.get("returnDate", params["departDate"])
        q = {
            "fly_from": params["origin"],
            "fly_to": params["destination"],
            "date_from": date_from,
            "date_to": date_from,
            "return_from": params.get("returnDate") or "",
            "return_to": params.get("returnDate") or "",
            "adults": params.get("adults", 1),
            "curr": params.get("currency", "USD"),
            "selected_cabins": ({"ECONOMY":"M","PREMIUM_ECONOMY":"W","BUSINESS":"C","FIRST":"F"}.get(params.get("cabin","ECONOMY"),"M")),
            "limit": 15,
            "sort": "price"
        }
        r = requests.get(
            "https://api.tequila.kiwi.com/v2/search",
            headers={"apikey": TEQUILA_API_KEY},
            params=q, timeout=20
        )
        r.raise_for_status()
        data = r.json()
        offers = []
        for it in data.get("data", []):
            price = float(it.get("price", 0.0))
            offers.append({
                "provider":"tequila",
                "price": price,
                "currency": params.get("currency","USD"),
                "oneWay": (params.get("returnDate") is None),
                "meta": {"route": it.get("route", [])[:2], "deep_link": it.get("deep_link")}
            })
        return {"ok": True, "offers": offers}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def provider_backup_search(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Offline fallback. Synthesizes plausible fares so the pipeline works without keys.
    """
    bump("tool_calls"); metrics["provider_calls"]["dummy"] += 1
    base = 120 if params.get("cabin","ECONOMY") == "ECONOMY" else 450
    # Very naive date-based variance
    try:
        y,m,d = map(int, params["departDate"].split("-"))
        weight = (y*13 + m*7 + d) % 70
    except Exception:
        weight = 33
    offers = []
    for i in range(5):
        price = round(base + weight + i*15, 2)
        offers.append({
            "provider":"dummy",
            "price": price,
            "currency": params.get("currency","USD"),
            "oneWay": (params.get("returnDate") is None),
            "meta": {"note":"offline-demo"}
        })
    return {"ok": True, "offers": offers}



def normalize_offers(*results: Dict[str, Any]) -> List[Dict[str, Any]]:
    offers: List[Dict[str, Any]] = []
    for r in results:
        if r and r.get("ok") and isinstance(r.get("offers"), list):
            offers.extend(r["offers"])
    # sort by price asc
    offers.sort(key=lambda x: x.get("price", 1e12))
    # pick top N
    return offers[:10]



@dataclass
class AgentResult:
    ok: bool
    content: str
    data: Any = None
    usage: Dict[str, int] = field(default_factory=dict)
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))

def travel_planner(session_id: str, goal: str) -> Tuple[Dict[str, Any], str]:
    """
    Extracts: origin (IATA), destination (IATA), departDate (YYYY-MM-DD), optional returnDate,
    adults, cabin, currency. Uses Gemini; falls back to regex defaults.
    """
    prompt = (
        "You are a Travel Planner. Parse the developer goal into:\n"
        '{"origin":"IATA","destination":"IATA","departDate":"YYYY-MM-DD","returnDate":"YYYY-MM-DD|null",'
        '"adults":1,"cabin":"ECONOMY|PREMIUM_ECONOMY|BUSINESS|FIRST","currency":"USD"}\n'
        "Return ONLY JSON.\n\nGoal: " + goal
    )
    text = gemini_generate(prompt)
    params: Dict[str, Any]
    try:
        params = json.loads(text)
    except Exception:
        # Fallback: try to pick IATA-like tokens and ISO date
        def pick(regex, default=""):
            m = re.search(regex, goal, re.I)
            return m.group(1).upper() if m else default
        origin      = pick(r"from\s+([A-Z]{3})", "PHX")
        destination = pick(r"to\s+([A-Z]{3})", "JFK")
        date        = pick(r"(\d{4}-\d{2}-\d{2})", datetime.utcnow().strftime("%Y-%m-%d"))
        rtn         = pick(r"return(?:\s*on)?\s*(\d{4}-\d{2}-\d{2})", "")
        params = {
            "origin": origin,
            "destination": destination,
            "departDate": date,
            "returnDate": rtn or None,
            "adults": 1,
            "cabin": "ECONOMY",
            "currency": "USD"
        }
    sessions.append(session_id, {"role":"agent","name":"TravelPlanner","content":json.dumps(params),"timestamp": time.time()*1000})
    # remember user’s common routes
    bank.remember("routes", f"{params['origin']}->{params['destination']}")
    return params, str(uuid.uuid4())

def price_fetcher(session_id: str, plan: Dict[str, Any]) -> AgentResult:
    """
    Calls providers (Amadeus, Tequila, Backup), merges & sorts offers.
    """
    r_dummy   = provider_backup_search(plan)
    r_amadeus = provider_amadeus_search(plan)
    r_tequila = provider_tequila_search(plan)
    merged = normalize_offers(r_amadeus, r_tequila, r_dummy)
    content = "Fetched offers (cheapest first):\n" + json.dumps(merged[:5], indent=2)
    sessions.append(session_id, {"role":"agent","name":"PriceFetcher","content":content,"timestamp": time.time()*1000})
    return AgentResult(ok=bool(merged), content=content, data={"offers": merged})

def itinerary_builder(session_id: str, plan: Dict[str, Any], offers: List[Dict[str, Any]]) -> AgentResult:
    """
    Build a traveler-facing JSON itinerary payload with top 3 options.
    """
    top = offers[:3]
    payload = {
        "plan": plan,
        "options": top,
        "cheapest": (top[0] if top else None),
        "generatedAt": datetime.utcnow().isoformat() + "Z"
    }
    content = "Itinerary JSON:\n" + json.dumps(payload, indent=2)
    sessions.append(session_id, {"role":"agent","name":"ItineraryBuilder","content":content,"timestamp": time.time()*1000})
    return AgentResult(ok=bool(top), content=content, data=payload)

def reviewer(session_id: str, plan: Dict[str, Any], prev: AgentResult) -> AgentResult:
    """
    Sanity checks: at least one option, currency present, price > 0, origin/destination differ.
    If bad, suggests fixes (date flexibility, origin swap, etc.).
    """
    issues = []
    if not prev.data or not prev.data.get("options"):
        issues.append("No offers. Try different dates or nearby airports.")
    if plan.get("origin") == plan.get("destination"):
        issues.append("Origin equals destination. Choose different airports.")
    for opt in prev.data.get("options", [])[:3]:
        if opt.get("price", 0) <= 0: issues.append("Found non-positive price.")
        if not opt.get("currency"): issues.append("Missing currency code.")

    approve = (len(issues) == 0)
    verdict = {"approve": approve, "notes": ("OK" if approve else "; ".join(sorted(set(issues))))}
    content = "Reviewer verdict: " + json.dumps(verdict)
    # One tiny self-heal: if no offers, create a flexible-date hint
    if not approve and "No offers" in verdict["notes"]:
        hint = {
            "tip": "Expand search ±3 days or use alternate airport (e.g., EWR for NYC, OAK for SFO)."
        }
        content += "\nHint: " + json.dumps(hint)
    sessions.append(session_id, {"role":"agent","name":"Reviewer","content":content,"timestamp": time.time()*1000})
    return AgentResult(ok=True, content=content, data=verdict)



def run_travel_concierge(session_id: str, goal: str) -> Dict[str, Any]:
    metrics["runs"] += 1
    sessions.append(session_id, {"role":"user","name":"Traveler","content":goal,"timestamp": time.time()*1000})
    comp = compact_context(session_id)
    if comp: sessions.append(session_id, comp)

    plan, _tid = travel_planner(session_id, goal)
    fetch = price_fetcher(session_id, plan)
    itin  = itinerary_builder(session_id, plan, (fetch.data or {}).get("offers", []))
    rev   = reviewer(session_id, plan, itin)

    return {
        "title": PROJECT_TITLE,
        "plan": plan,
        "final": itin.data,
        "review": rev.data,
        "metrics": metrics.copy(),
        "transcript": sessions.get(session_id),
    }



demo = run_travel_concierge(
    session_id="travel-demo-1",
    goal="Find roundtrip flights from PHX to JFK departing 2026-03-10 and return 2026-03-16 for 1 adult in ECONOMY, currency USD."
)
print("Project:", demo["title"])
print("Plan:", demo["plan"])
print("Cheapest option:", (demo["final"] or {}).get("cheapest"))
print("Review:", demo["review"])
print("Metrics:", demo["metrics"])


