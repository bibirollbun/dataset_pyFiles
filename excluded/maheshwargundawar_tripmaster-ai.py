# TripMaster AI - Kaggle-friendly single-notebook submission
# This notebook is self-contained and uses a simulated LLM (no API keys).
# It demonstrates a multi-agent travel planner: Delhi -> Manali demo.

from IPython.display import Markdown, display
import time, random, json, threading, os
from datetime import datetime
from queue import Queue

# ----------------------------
# Header (display as markdown)
# ----------------------------
header_md = """
# TripMaster AI â€” Smart Travel Planner
**Demo route:** Delhi â†’ Manali

**What this notebook contains:**

- A self-contained **multi-agent architecture** with lightweight simulated LLM calls.  
- Agents: **Intent, Weather, Route, Budget, Itinerary**  
- Tools: Distance estimator, weather simulator, cost estimator, attraction lookup  
- Memory: **Session + Long-term memory** with compaction  
- Orchestrator: parallel agent execution  
- Observability: logs & metrics  
- Demo tests for Delhi â†’ Manali  

> **No API keys required â€” fully Kaggle-friendly**
"""
display(Markdown(header_md))

# ----------------------------
# Utilities: Timestamp & Logging
# ----------------------------
def now_ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

LOGS = []
METRICS = {"requests":0, "agent_calls":0, "tool_calls":0, "mem_reads":0, "mem_writes":0}

def log(agent, status, details=""):
    entry = {"time": now_ts(), "agent": agent, "status": status, "details": details}
    LOGS.append(entry)
    print(f"[{entry['time']}] [{agent}] {status} - {details}")

# ----------------------------
# Memory: Session + Long-term
# ----------------------------
SESSION_MEMORY = []
LONG_TERM = []
MAX_SESSION = 8

def mem_write(role, text):
    SESSION_MEMORY.append({"time": now_ts(), "role": role, "text": text})
    METRICS["mem_writes"] += 1
    if len(SESSION_MEMORY) > MAX_SESSION:
        oldest = SESSION_MEMORY.pop(0)
        LONG_TERM.append(oldest)
        log("Memory", "COMPACT", f"Moved to LT: {oldest['text'][:60]}...")
    log("Memory", "WRITE", f"{role} stored.")

def mem_read(last_n=6):
    METRICS["mem_reads"] += 1
    recent = SESSION_MEMORY[-last_n:]
    lt_summary = " | ".join([m["text"] for m in LONG_TERM[-5:]])
    return {"recent": recent, "long_term_summary": lt_summary}

# ----------------------------
# Fake LLM (lightweight simulation)
# ----------------------------
def fake_llm(prompt, role_hint="assistant"):
    time.sleep(0.15)
    p = prompt.lower()

    if "weather" in p:
        return weather_simulator_response(prompt)
    if "distance" in p or "route" in p:
        return route_simulator_response(prompt)
    if "cost" in p or "budget" in p:
        return cost_simulator_response(prompt)
    if "summarize" in p:
        return simple_summarizer(prompt)

    templates = [
        "Here is a practical travel suggestion and next steps.",
        "I recommend this itinerary refinement.",
        "Below is a concise recommendation based on constraints."
    ]
    return random.choice(templates) + "\n\nContext snippet:\n" + prompt[:300]

# ----------------------------
# Tools (simulated)
# ----------------------------
def route_simulator_response(prompt):
    METRICS["tool_calls"] += 1
    log("Tool/RouteSim", "CALL", prompt[:80])
    if "delhi" in prompt.lower() and "manali" in prompt.lower():
        return "Distance: ~540 km | Time: 12-16 hrs | Suggested stop: Chandigarh"
    return "General route: moderate distance."

def distance_tool(src, dest):
    METRICS["tool_calls"] += 1
    log("Tool/Distance", "CALL", f"{src}->{dest}")
    if src.lower()=="delhi" and dest.lower()=="manali":
        return {"distance_km":540, "typical_hours":13, "notes":"Via NH44"}
    return {"distance_km":120, "typical_hours":3, "notes":"Local route"}

def weather_simulator_response(prompt):
    METRICS["tool_calls"] += 1
    log("Tool/WeatherSim", "CALL", prompt[:80])
    sample = [
        {"day":"Day 1","summary":"Clear, 15Â°C day / 2Â°C night"},
        {"day":"Day 2","summary":"Cloudy, light showers"},
        {"day":"Day 3","summary":"Cold, chance of snow"},
        {"day":"Day 4","summary":"Sunny, pleasant"},
        {"day":"Day 5","summary":"Windy at higher altitudes"}
    ]
    return json.dumps(sample)

def get_weather_for_city(city):
    METRICS["tool_calls"] += 1
    log("Tool/GetWeather", "CALL", city)
    if "manali" in city.lower():
        return [
            ("Day 1", "Clear, 15Â°C"),
            ("Day 2", "Cloudy, 12Â°C"),
            ("Day 3", "Snow chance, 6Â°C"),
            ("Day 4", "Sunny, 14Â°C"),
            ("Day 5", "Windy, 8Â°C")
        ]
    return [("Day 1","Mild")]

def cost_estimator_tool(days, travel_mode="road", tier="mid"):
    METRICS["tool_calls"] += 1
    log("Tool/Cost", "CALL", f"{days} days")
    base = 4000
    stay = days*2000
    food = days*700
    act = 2000
    total = base+stay+food+act
    return {"travel":base,"stay":stay,"food":food,"activities":act,"total_estimate":total}

def attractions_lookup(city):
    METRICS["tool_calls"] += 1
    log("Tool/Attractions", "CALL", city)
    if "manali" in city.lower():
        return [
            {"name":"Hidimba Temple","time":"1 hr"},
            {"name":"Solang Valley","time":"half-day"},
            {"name":"Rohtang Pass","time":"full-day"}
        ]
    return [{"name":"Local Spot","time":"1 hr"}]

def simple_summarizer(text, max_sent=3):
    METRICS["tool_calls"] += 1
    log("Tool/Summarizer", "CALL", text[:80])
    sents = [s.strip() for s in text.replace("\n"," ").split(". ") if s.strip()]
    return ". ".join(sents[:max_sent])

def knowledge_db_lookup(query):
    db = {
        "manali":"Hill station famous for snow and adventure sports.",
        "delhi":"Capital city; main road route to Manali via NH44."
    }
    for k,v in db.items():
        if k in query.lower():
            return v
    return "No match."

# ----------------------------
# Agents
# ----------------------------
METRICS["agent_calls"] = 0

def agent_intent(text):
    METRICS["agent_calls"] += 1
    log("Agent/Intent","START",text[:60])
    t = text.lower()
    src="delhi" if "delhi" in t else "unknown"
    dest="manali" if "manali" in t else "unknown"
    days=3
    for w in t.split():
        if w.isdigit():
            days=int(w)
            break
    budget=None
    nums=[int(s) for s in ''.join(ch if ch.isdigit() or ch==' ' else ' ' for ch in t).split() if s.isdigit()]
    if nums:
        budget=nums[0]

    profile={"source":src,"destination":dest,"days":days,"budget":budget}
    mem_write("intent",str(profile))
    log("Agent/Intent","END",str(profile))
    return profile

def agent_weather(profile):
    METRICS["agent_calls"] += 1
    city=profile.get("destination","manali")
    log("Agent/Weather","START",city)
    w=get_weather_for_city(city)
    mem_write("weather",json.dumps(w))
    log("Agent/Weather","END","ok")
    return w

def agent_route(profile):
    METRICS["agent_calls"] += 1
    src=profile["source"]; dest=profile["destination"]
    log("Agent/Route","START",f"{src}->{dest}")
    r=distance_tool(src,dest)
    r["stop"]="Chandigarh"
    mem_write("route",json.dumps(r))
    log("Agent/Route","END","ok")
    return r

def agent_budget(profile):
    METRICS["agent_calls"] += 1
    log("Agent/Budget","START","estimating")
    b=cost_estimator_tool(profile["days"])
    mem_write("budget",json.dumps(b))
    log("Agent/Budget","END",str(b["total_estimate"]))
    return b

def agent_itinerary(profile, weather, route_info, budget):
    METRICS["agent_calls"] += 1
    log("Agent/Itinerary","START","planning")
    days=profile["days"]
    attractions=attractions_lookup(profile["destination"])

    plan=[]

    plan.append({
        "day":"Day 1",
        "plan":[f"Depart {profile['source']}",f"Travel {route_info['distance_km']} km (~{route_info['typical_hours']} hrs)",
                "Hotel check-in","Evening at Mall Road"]
    })

    idx=0
    for d in range(2,days+1):
        if idx<len(attractions):
            plan.append({"day":f"Day {d}", "plan":[f"Visit {attractions[idx]['name']} ({attractions[idx]['time']})"]})
            idx+=1
        else:
            plan.append({"day":f"Day {d}", "plan":["Local sightseeing / rest"]})

    note=f"Total estimated cost: â‚¹{budget['total_estimate']}"
    mem_write("itinerary",note)
    log("Agent/Itinerary","END","ok")
    return {"itinerary":plan,"note":note}

# ----------------------------
# Orchestrator
# ----------------------------
def orchestrate_trip(text, run_parallel=True, timeout=6):
    log("Orchestrator","START",text[:60])
    METRICS["requests"]+=1

    profile=agent_intent(text)
    q=Queue(); results={}

    def wfn():
        q.put(("weather",agent_weather(profile)))
    def rfn():
        q.put(("route",agent_route(profile)))
    def bfn():
        q.put(("budget",agent_budget(profile)))

    threads=[]
    for fn in [wfn,rfn,bfn]:
        if run_parallel:
            t=threading.Thread(target=fn)
            threads.append(t); t.start()
        else:
            fn()

    start=time.time()
    while len(results)<3 and (time.time()-start)<timeout:
        try:
            k,v=q.get(timeout=0.5)
            results[k]=v
        except:
            pass

    for t in threads: t.join(0.1)

    for k in ["weather","route","budget"]:
        results.setdefault(k,"none")

    itinerary=agent_itinerary(profile, results["weather"], results["route"], results["budget"])

    final_prompt=f"Make a friendly travel summary.\nProfile:{profile}\nWeather:{results['weather']}\nRoute:{results['route']}\nBudget:{results['budget']}\nItinerary:{itinerary}"
    final_text=fake_llm(final_prompt)

    mem_write("final_plan",final_text)
    log("Orchestrator","END","done")

    return {"final":final_text,"profile":profile,"parts":results,
            "itinerary":itinerary,"logs":LOGS[-12:],"metrics":METRICS}

# ----------------------------
# Demo Tests (Delhi â†’ Manali)
# ----------------------------
tests=[
    "Plan a 4 day trip from Delhi to Manali under 15000",
    "Plan 3 day relaxing trip Delhi to Manali",
    "Suggest 5 day sightseeing trip Delhi to Manali"
]

outputs=[]
for t in tests:
    print("\n"+"="*60)
    print("[USER QUERY]",t)
    res=orchestrate_trip(t)
    print("\nFINAL PLAN:\n",res["final"])
    print("\nITINERARY:\n",json.dumps(res["itinerary"],indent=2))
    print("\nPARTS:\n",json.dumps(res["parts"],indent=2))
    print("\nLOGS:")
    for l in res["logs"]:
        print(l)
    print("\nMETRICS:",res["metrics"])
    outputs.append(res)

# Save outputs (Kaggle working directory)
try:
    with open("/kaggle/working/tripmaster_outputs.json","w") as f:
        json.dump(outputs,f,indent=2)
    print("\nSaved demo outputs to /kaggle/working/")
except:
    print("\nRunning outside Kaggle.")

# ----------------------------
# Closing Markdown
# ----------------------------
closing_md = """
---

# ğŸ�� TripMaster AI â€” Notebook Completed

This project shows:
- Multi-agent planning  
- Tools  
- Memory  
- Orchestration  
- Logs + Metrics  
- Delhi â†’ Manali demo  

This notebook is **fully self-contained** and ready for Kaggle submission.

Thank you for reviewing! ğŸ™Œ

---
"""
display(Markdown(closing_md))





