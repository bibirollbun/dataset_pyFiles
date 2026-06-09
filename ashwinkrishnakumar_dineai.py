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


!pip install --quiet openai gradio validators python-dateutil

import os, json, time, uuid, logging
from datetime import datetime, timedelta
from dateutil import parser as dateparser
from typing import Dict, Any, List
import validators
import pprint

pp = pprint.PrettyPrinter(indent=2)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("conciergeflow")



from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("GOOGLE_API_KEY")

MEMORY_FILE = "concierge_memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        return json.load(open(MEMORY_FILE))
    return {"user_profile": {}, "sessions": {}}

def save_memory(mem):
    json.dump(mem, open(MEMORY_FILE,"w"), indent=2)

memory = load_memory()



LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock")  
USE_REAL_TOOLS = False

def llm_call(prompt: str, max_tokens=400) -> str:
    if LLM_PROVIDER=="openai" and os.getenv("OPENAI_API_KEY"):
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=max_tokens
        )
        return resp['choices'][0]['message']['content']
    elif LLM_PROVIDER=="gemini":
        raise NotImplementedError("Plug Gemini/Vertex call here.")
    return mock_llm(prompt)

def mock_llm(prompt: str) -> str:
    if "plan" in prompt.lower() and "book_restaurant" in prompt.lower():
        return "Steps: clarify → search → present_options → confirm → reserve → calendar → notify"
    if "generate options" in prompt.lower():
        return json.dumps([
            {"name":"La Piazza","rating":4.6,"price":"$$","notes":"outdoor seating"},
            {"name":"Trattoria Roma","rating":4.5,"price":"$$","notes":"gluten-free options"}
        ])
    return "I'm a deterministic mock LLM."



def web_search(query: str, limit=3) -> List[Dict]:
    if USE_REAL_TOOLS:
        raise NotImplementedError("Real web search API")
    return [
        {"title":"La Piazza - Italian - outdoor seating","url":"https://example.com/la-piazza"},
        {"title":"Trattoria Roma - gluten-free menu","url":"https://example.com/trattoria"}
    ][:limit]

def find_restaurants(constraints: Dict[str,Any]) -> List[Dict]:
    if USE_REAL_TOOLS:
        raise NotImplementedError("Google Places API integration required.")
    return [
        {"id":"r1","name":"La Piazza","rating":4.6,"price":40,"outdoor":True,"diet":["gluten-free","vegan"],"url":"https://example.com/la-piazza"},
        {"id":"r2","name":"Trattoria Roma","rating":4.5,"price":35,"outdoor":False,"diet":["gluten-free"],"url":"https://example.com/trattoria"}
    ]

def make_reservation(restaurant_id: str, datetime_iso: str, guests: int, contact: Dict[str,str]) -> Dict:
    if USE_REAL_TOOLS:
        raise NotImplementedError("Add real reservation API logic.")
    return {
        "status":"reserved",
        "reservation_id":str(uuid.uuid4()),
        "restaurant_id":restaurant_id,
        "datetime":datetime_iso,
        "guests":guests,
        "contact":contact
    }

def add_calendar_event(summary:str, start_iso:str, end_iso:str, description:str):
    if USE_REAL_TOOLS:
        raise NotImplementedError("Google Calendar integration needed.")
    return {
        "status":"ok",
        "event_id":str(uuid.uuid4()),
        "summary":summary,
        "start":start_iso,
        "end":end_iso,
        "description":description
    }



def parse_user_request(text: str) -> Dict[str,Any]:
    intent = {"task":"unknown", "raw":text, "constraints":{}}
    t = text.lower()

    if "dinner" in t or "restaurant" in t or "book" in t:
        intent["task"]="book_restaurant"
        try:
            dt = dateparser.parse(text, fuzzy=True)
            if dt:
                intent["constraints"]["datetime"]=dt.isoformat()
        except:
            pass
        import re
        m = re.search(r'(\d+)\s*(people|guests|persons)', text)
        if m:
            intent["constraints"]["guests"] = int(m.group(1))
    return intent



def plan_for_intent(intent: Dict[str,Any]) -> List[Dict]:
    if intent["task"] == "book_restaurant":
        return [
            {"step":"clarify","action":"ask_for_missing_info"},
            {"step":"search","action":"find_restaurants"},
            {"step":"present","action":"present_options"},
            {"step":"confirm","action":"confirm_choice"},
            {"step":"reserve","action":"make_reservation"},
            {"step":"calendar","action":"add_calendar"},
            {"step":"notify","action":"send_confirmation"}
        ]
    return [{"step":"unknown","action":"ask_clarify"}]



def execute_plan(plan: List[Dict], intent: Dict[str,Any], user_contact: Dict[str,str]=None) -> Dict:
    trace = []
    session_id = str(uuid.uuid4())
    selected_restaurant = None
    reservation_result = None

    for s in plan:
        action = s["action"]

        if action=="ask_for_missing_info":
            if "datetime" not in intent["constraints"]:
                dt = datetime.utcnow().replace(hour=19, minute=0, second=0, microsecond=0) + timedelta(days=1)
                intent["constraints"]["datetime"] = dt.isoformat()
                trace.append({"action":"fill_datetime","value":intent["constraints"]["datetime"]})
            if "guests" not in intent["constraints"]:
                intent["constraints"]["guests"] = 2
                trace.append({"action":"fill_guests","value":2})

        elif action=="find_restaurants":
            results = find_restaurants(intent["constraints"])
            trace.append({"action":"search_results","results":results})

        elif action=="present_options":
            last = trace[-1]["results"]
            top_two = last[:2] if last else []
            trace.append({"action":"presented_options","options":top_two})

        elif action=="confirm_choice":
            options = trace[-1]["options"]
            selected_restaurant = options[0] if options else None
            trace.append({"action":"auto_confirm","selected":selected_restaurant})

        elif action=="make_reservation":
            if selected_restaurant:
                reservation_result = make_reservation(
                    restaurant_id=selected_restaurant["id"],
                    datetime_iso=intent["constraints"]["datetime"],
                    guests=intent["constraints"]["guests"],
                    contact=user_contact or {}
                )
                trace.append({"action":"reservation","result":reservation_result})
            else:
                trace.append({"action":"error_no_selection"})

        elif action=="add_calendar":
            if reservation_result and selected_restaurant:
                start = intent["constraints"]["datetime"]
                end = (dateparser.parse(start) + timedelta(hours=2)).isoformat()
                event = add_calendar_event(
                    summary=f"Dinner at {selected_restaurant['name']}",
                    start_iso=start,
                    end_iso=end,
                    description=f"Reservation ID: {reservation_result['reservation_id']}"
                )
                trace.append({"action":"calendar_event","event":event})

        elif action=="send_confirmation":
            trace.append({"action":"sent_confirmation","to":user_contact})

        else:
            trace.append({"action":"unrecognized_step","detail":s})

    session = {
        "id":session_id,
        "intent":intent,
        "trace":trace,
        "status":"done",
        "final_choice":selected_restaurant,
        "reservation":reservation_result
    }

    memory["sessions"][session_id] = session
    save_memory(memory)
    return session



def demo_run(user_text:str, contact:Dict=None):
    intent = parse_user_request(user_text)
    plan = plan_for_intent(intent)
    session = execute_plan(plan, intent, user_contact=contact)
    return session

req = "Book a dinner for 4 tomorrow at 7pm at an Italian restaurant with outdoor seating."
output = demo_run(req, contact={"name":"Ashwin","email":"ashwin@example.com"})
pp.pprint(output)


