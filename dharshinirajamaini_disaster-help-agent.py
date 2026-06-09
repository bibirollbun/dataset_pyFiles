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


!pip install --force-reinstall --upgrade tornado==6.4.2 notebook==6.5.7 pandas==2.2.2 requests==2.32.3



# Disaster Help Agent - Colab script 

import os
import time
import random
import json
from datetime import datetime
from geopy.geocoders import Nominatim
from math import radians, sin, cos, sqrt, atan2
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
# ------------------ CONFIG ------------------
# Example (unsafe for shared notebooks): 
# os.environ['GEMINI_API_KEY'] = 'YOUR_KEY'
# os.environ['TWILIO_ACCOUNT_SID'] = 'ACxxxxxxxxxxxxxxxxxxxx'
# os.environ['TWILIO_AUTH_TOKEN'] = 'yyyyyyyyyyyyyyyyyyyyy'
# os.environ['TWILIO_WHATSAPP_FROM'] = 'whatsapp:+14155238886'   # Twilio Sandbox number
# os.environ['TWILIO_WHATSAPP_TO'] = 'whatsapp:+91XXXXXXXXXX'   # Your test WhatsApp number

GEMINI_API_KEY = user_secrets.get_secret('GOOGLE_API_KEY')
TWILIO_ACCOUNT_SID = user_secrets.get_secret('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = user_secrets.get_secret('TWILIO_AUTH_TOKEN')
TWILIO_WHATSAPP_FROM = user_secrets.get_secret('TWILIO_WHATSAPP_FROM')  # e.g. 'whatsapp:+14155238886'
TWILIO_WHATSAPP_TO = user_secrets.get_secret('TWILIO_WHATSAPP_TO')      # e.g. 'whatsapp:+91XXXXXXXXXX'

print('Config summary:')
print(' GEMINI provided:', bool(GEMINI_API_KEY))
print(' Twilio provided:', bool(TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN))
print(' Twilio WhatsApp from/to:', bool(TWILIO_WHATSAPP_FROM), bool(TWILIO_WHATSAPP_TO))

try:
    from twilio.rest import Client as TwilioClient
except Exception:
    TwilioClient = None

# Gemini try-imports (google-genai or google.generativeai)
gemini_client = None
gemini_lib_name = None
if GEMINI_API_KEY:
    try:
        from google import genai
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        gemini_lib_name = 'google-genai.Client'
    except Exception:
        try:
            import google.generativeai as genai2
            genai2.configure(api_key=GEMINI_API_KEY)
            gemini_client = genai2
            gemini_lib_name = 'google.generativeai'
        except Exception:
            gemini_client = None
            gemini_lib_name = None

print('Gemini client:', gemini_lib_name or 'NOT available (simulated LLM fallback)')

# For maps
import folium
from IPython.display import display

# Geolocator
geolocator = Nominatim(user_agent="disaster_help_agent_colab")

# Haversine (km)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

# Geocode helper (fallback-safe)
def geocode_location(query):
    try:
        location = geolocator.geocode(query, timeout=10)
        if location:
            return (location.latitude, location.longitude, location.address)
    except Exception:
        return None
    return None

# Pretty print incident
def pretty_incident(inc):
    print(f"\n--- INCIDENT {inc['id']} ---")
    print('time:', inc['time'])
    print('post:', inc['post'])
    print('is_emergency:', inc['is_emergency'])
    print('location_text:', inc.get('location_text'))
    print('coords:', inc.get('coords'))
    print('assigned_team:', inc.get('assigned_team'))
    print('status:', inc.get('status'))

# ------------------ RESCUE TEAMS (SAMPLE) ------------------
rescue_teams = [
    {"name": "Chennai Fire Dept", "lat": 13.0827, "lon": 80.2707, "available": True, "contact": "+91101111"},
    {"name": "Velachery Rescue Team", "lat": 12.9858, "lon": 80.2206, "available": True, "contact": "+91102222"},
    {"name": "Thane NDRF", "lat": 19.2183, "lon": 72.9781, "available": True, "contact": "+91103333"},
    {"name": "Guwahati Flood Rescue", "lat": 26.1445, "lon": 91.7362, "available": True, "contact": "+91104444"}
]

# In-memory incidents store
INCIDENTS = {}

# Simple ID generator
def new_incident_id():
    return int(time.time()*1000) + random.randint(0,999)

# ------------------ GEMINI WRAPPER & SIMULATIONS ------------------
def call_gemini_json(prompt, timeout_secs=10):
    """
    Attempts to call Gemini with common client patterns.
    Returns parsed JSON dict if finds JSON in response, else returns None.
    """
    if not gemini_client:
        return None
    try:
        if gemini_lib_name == 'google-genai.Client':
            # try generate_text
            try:
                resp = gemini_client.generate_text(model="gemini-proto", input=prompt, max_output_tokens=200)
                text = getattr(resp, 'text', None) or str(resp)
            except Exception:
                resp = gemini_client.generate(model="gemini-proto", prompt=prompt)
                text = str(resp)
        elif gemini_lib_name == 'google.generativeai':
            out = gemini_client.generate_text(model="chat-bison@001", prompt=prompt)
            text = getattr(out, 'text', None) or str(out)
        else:
            return None

        import re
        js = re.search(r"\{[\s\S]*\}", text)
        if js:
            try:
                return json.loads(js.group(0))
            except Exception:
                return {"raw": text}
        return {"raw": text}
    except Exception as e:
        print("Gemini call failed:", e)
        return None

def simulated_llm_response_for_classify(post):
    lower = post.lower()
    keywords = ["help","stuck","flood","earthquake","trapped","urgent","rescue","medical","injured"]
    score = sum(1 for k in keywords if k in lower)
    return {"is_emergency": bool(score>=1), "score": int(score), "reason": "keyword heuristic"}

def simulated_llm_response_for_location(post):
    candidates = ['Velachery','Chennai','Thane','Guwahati','Anna Nagar','Mumbai']
    for c in candidates:
        if c.lower() in post.lower():
            return {"location": c}
    return {"location": "UNKNOWN"}

# ------------------ AGENTS ------------------
sample_posts = [
    "We are stuck in Velachery first floor, water rising please help!",
    "What a beautiful rainy climate",
    "Earthquake damaged our building in Thane, 3 people trapped!",
    "Going to office now",
    "Flood in Guwahati, children stuck in school bus!",
    "My neighbor fell and needs medical help near Anna Nagar",
    "No internet this morning"
]

def social_listening_agent(simulate=True):
    if simulate:
        post = random.choice(sample_posts)
        print('\n[SocialListener] New post:', post)
        return {"post": post, "time": datetime.utcnow().isoformat()}
    else:
        raise NotImplementedError('Wire real streaming source here')

def classification_agent(state):
    post = state['post']
    lower = post.lower()
    keywords = ["help","stuck","flood","earthquake","trapped","urgent","rescue","medical","injured"]
    score = sum(1 for k in keywords if k in lower)
    is_emergency = score >= 1
    severity = score

    if GEMINI_API_KEY and gemini_client:
        prompt = (f"Decide whether the following social media post is an emergency. "
                  f"Return JSON exactly like: {{\"is_emergency\": true/false, \"score\": int, \"reason\": \"short text\"}}.\n\nPost: {post}")
        parsed = call_gemini_json(prompt)
        if parsed and 'is_emergency' in parsed:
            is_emergency = bool(parsed.get('is_emergency'))
            try:
                severity = int(parsed.get('score', severity))
            except Exception:
                pass
        elif not parsed:
            parsed = simulated_llm_response_for_classify(post)
            is_emergency = parsed['is_emergency']
            severity = parsed['score']
    else:
        is_emergency = bool(score >= 1)
        severity = score

    state['is_emergency'] = is_emergency
    state['severity'] = severity
    print('[Classifier] is_emergency=', is_emergency, 'severity=', severity)
    return state

def location_agent(state):
    post = state['post']
    candidates = ['Velachery','Chennai','Thane','Guwahati','Anna Nagar','Mumbai']
    loc_text = None
    for c in candidates:
        if c.lower() in post.lower():
            loc_text = c
            break

    if not loc_text and GEMINI_API_KEY and gemini_client:
        prompt = (f"Extract the most likely location (short place name) from this social post. "
                  f"Return JSON like: {{\"location\": \"PlaceName\"}} or {{\"location\":\"UNKNOWN\"}}.\n\nPost: {post}")
        parsed = call_gemini_json(prompt)
        if parsed and 'location' in parsed:
            loc_text = parsed.get('location')
            if loc_text and loc_text.upper() == 'UNKNOWN':
                loc_text = None
        else:
            parsed = simulated_llm_response_for_location(post)
            if parsed.get('location') and parsed['location'] != 'UNKNOWN':
                loc_text = parsed['location']

    state['location_text'] = loc_text
    if loc_text:
        geo = geocode_location(loc_text)
        if geo:
            state['coords'] = (geo[0], geo[1])
            state['location_address'] = geo[2]
            print('[Locator] Found', loc_text, '->', state['coords'])
        else:
            print('[Locator] Could not geocode', loc_text)
    else:
        print('[Locator] No location text discovered')
    return state

def resource_matching_agent(state):
    if not state.get('coords'):
        state['assigned_team'] = None
        print('[Matcher] No coords to match')
        return state
    lat, lon = state['coords']
    best = None
    best_dist = float('inf')
    for team in rescue_teams:
        if not team.get('available', True):
            continue
        d = haversine(lat, lon, team['lat'], team['lon'])
        if d < best_dist:
            best_dist = d
            best = team
    if best:
        state['assigned_team'] = best['name']
        state['assigned_team_contact'] = best['contact']
        state['assigned_distance_km'] = round(best_dist,2)
        best['available'] = False
        print('[Matcher] Assigned', best['name'], 'distance km', state['assigned_distance_km'])
    else:
        state['assigned_team'] = None
        print('[Matcher] No available team')
    return state

def send_whatsapp_twilio(text):
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TwilioClient and TWILIO_WHATSAPP_FROM and TWILIO_WHATSAPP_TO:
        try:
            client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            msg = client.messages.create(body=text, from_=TWILIO_WHATSAPP_FROM, to=TWILIO_WHATSAPP_TO)
            print('[Notify-WhatsApp] SID', getattr(msg, 'sid', '(no sid)'))
            return True
        except Exception as e:
            print('Twilio send failed ->', e)
            return False
    else:
        print('[Notify-WhatsApp] (simulated):', text)
        return False

def notification_agent(state):
    assigned = state.get('assigned_team')
    if not assigned:
        print('[Notifier] Nothing to notify')
        return state
    text = (f"URGENT SOS — {state['post']}\nLocation: {state.get('location_address', state.get('location_text'))} "
            f"Coords: {state.get('coords')}\nAssigned Team: {assigned} (distance {state.get('assigned_distance_km')} km)")
    w_res = send_whatsapp_twilio(text)
    state['notified'] = True
    state['notification_channels'] = {'whatsapp': w_res}
    return state

def monitor_agent(state):
    state['status'] = 'assigned' if state.get('assigned_team') else 'unassigned'
    if state['status'] == 'assigned':
        print('[Monitor] Waiting for team acknowledgement...')
        time.sleep(0.5)
        state['acknowledged'] = True
        state['status'] = 'in_progress'
        print('[Monitor] Team acknowledged — status now in_progress')
        time.sleep(0.5)
        state['status'] = 'resolved'
        print('[Monitor] Incident resolved (simulated)')
        for team in rescue_teams:
            if team['name'] == state.get('assigned_team'):
                team['available'] = True
    else:
        print('[Monitor] No active assignment')
    return state

# ------------------ ORCHESTRATOR ------------------
def orchestrate_once(simulate=True):
    state = social_listening_agent(simulate=simulate)
    state = classification_agent(state)
    if not state.get('is_emergency'):
        print('[Orchestrator] Not an emergency — aborting pipeline')
        return None
    state = location_agent(state)
    state = resource_matching_agent(state)
    state = notification_agent(state)
    state = monitor_agent(state)

    inc_id = new_incident_id()
    inc = {
        'id': inc_id,
        'time': datetime.utcnow().isoformat(),
        'post': state['post'],
        'is_emergency': state['is_emergency'],
        'location_text': state.get('location_text'),
        'coords': state.get('coords'),
        'assigned_team': state.get('assigned_team'),
        'status': state.get('status')
    }
    INCIDENTS[inc_id] = inc
    pretty_incident(inc)
    return inc
import pandas as pd

df = pd.DataFrame(INCIDENTS)
df.to_csv("incidents.csv", index=False)

print(" incidents.csv saved successfully!")
df.head()


# ------------------ RUN SIMULATION ------------------
for _ in range(4):
    orchestrate_once(simulate=True)
    time.sleep(0.8)

# ------------------ VISUALIZE INCIDENTS ON MAP ------------------
coords = [v['coords'] for v in INCIDENTS.values() if v.get('coords')]
if not coords:
    coords = [(13.0827,80.2707)]
m = folium.Map(location=coords[0], zoom_start=6)
for inc in INCIDENTS.values():
    if inc.get('coords'):
        folium.Marker(location=inc['coords'], popup=f"ID:{inc['id']}\n{inc['post']}").add_to(m)
display(m)

# -------------- NOTES --------------
# Twilio Sandbox: If you use the Twilio sandbox, go to your Twilio Console -> Messaging -> Try it out -> Send WhatsApp messages.
# You'll get a Sandbox number (set TWILIO_WHATSAPP_FROM) and a code to send from your phone (e.g., 'join <code>') to the sandbox number to link your WhatsApp.
# After joining the sandbox, set TWILIO_WHATSAPP_TO to your WhatsApp in 'whatsapp:+<countrycode><number>' format.






