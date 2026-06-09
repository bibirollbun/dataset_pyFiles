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


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


import json, time, uuid
from datetime import datetime
from collections import deque, defaultdict
import sqlite3
import random
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint

SEED = 42
random.seed(SEED)

print("âœ…Libraries imported.")


import sys
print("Python", sys.version)
import pkgutil
for p in ("geopy","sqlalchemy","scikit-learn","flask","requests"):
    print(p, "installed?", pkgutil.find_loader(p) is not None)



# in Kaggle you can run shell commands with !
!pip install geopy sqlalchemy scikit-learn flask requests python-dotenv


def create_message(sender, receiver, kind, payload):
    return {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "sender": sender,
        "receiver": receiver,
        "kind": kind, 
        "payload": payload
    }

def pretty(msg):
    print(json.dumps(msg, indent=2))

print("ğŸ”‘message helper.")



import os
DB_PATH = os.getenv("MEMORY_DB","memory.db")
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
print("Memory Imported.")


# Setup simple SQLite memory bank
conn = sqlite3.connect(':memory:') 
c = conn.cursor()
c.execute('''
CREATE TABLE incidents (
    id TEXT PRIMARY KEY,
    created_at TEXT,
    reporter TEXT,
    text TEXT,
    lat REAL,
    lon REAL,
    severity TEXT,
    triage_ts TEXT
)
''')
conn.commit()

def save_incident(inc):
    c.execute('''
    INSERT INTO incidents (id, created_at, reporter, text, lat, lon, severity, triage_ts)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (inc['id'], inc['created_at'], inc['reporter'], inc['text'], inc['lat'], inc['lon'], inc.get('severity'), inc.get('triage_ts')))
    conn.commit()

def get_incidents(limit=100):
    df = pd.read_sql_query("SELECT * FROM incidents ORDER BY created_at DESC LIMIT ?", conn, params=(limit,))
    return df

print("âœ…Memory Bank Created.")


# Geocode stub (for production replace with geopy or a real API)
def geocode_stub(place_text):
    # fake lat/lon by hashing
    h = abs(hash(place_text)) % 1000
    lat = 20 + (h % 50) * 0.1
    lon = 72 + ((h//50) % 50) * 0.1
    return lat, lon

# Resource DB (in-memory simulation)
resources = [
    {"id":"vol-1","type":"volunteer","lat":20.5,"lon":72.5,"available":True},
    {"id":"shel-1","type":"shelter","lat":20.8,"lon":72.7,"available":True},
    {"id":"med-1","type":"medical_kit","lat":20.6,"lon":72.6,"available":True},
]

def find_nearby_resources(lat, lon, r_km=50):
    # naive distance metric for demo
    def dist(a,b,c,d):
        return ((a-c)**2 + (b-d)**2)**0.5
    res = sorted(resources, key=lambda x: dist(lat, lon, x['lat'], x['lon']))
    return res[:5]

print("ğŸ› ï¸�Tools are created.")


class Session:
    def __init__(self, incident_id):
        self.incident_id = incident_id
        self.history = deque(maxlen=20)  # session messages
        self.created_at = datetime.utcnow().isoformat() + "Z"
    def add(self, msg):
        self.history.append(msg)

class Agent:
    def __init__(self, name):
        self.name = name
        self.log = []
    def send(self, msg):
        # in real system would route message
        self.log.append(msg)
        return msg
    def receive(self, msg):
        self.log.append(msg)
        # override in subclass
print("ğŸ¤–Agent Base Created.")


class ReceiverAgent(Agent):
    def __init__(self, name="receiver"):
        super().__init__(name)
    def receive_report(self, reporter, text, place_text):
        lat, lon = geocode_stub(place_text)
        incident = {
            "id": str(uuid.uuid4()),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "reporter": reporter,
            "text": text,
            "lat": lat,
            "lon": lon
        }
        save_incident(incident)
        msg = create_message(self.name, "triage", "report", incident)
        self.send(msg)
        return msg
print("âœ…Receiver Agent Created.")


def simple_triage_classifier(text):
    text_l = text.lower()
    if any(w in text_l for w in ["dead","fire","trapped","bleeding","collapsed"]):
        return "high"
    if any(w in text_l for w in ["injury","hurt","injured","help"]):
        return "medium"
    return "low"

class TriageAgent(Agent):
    def __init__(self, name="triage"):
        super().__init__(name)
    def process_report(self, msg):
        incident = msg['payload']
        severity = simple_triage_classifier(incident['text'])
        incident['severity'] = severity
        incident['triage_ts'] = datetime.utcnow().isoformat() + "Z"
        # update persistent memory (simple update)
        c.execute('UPDATE incidents SET severity=?, triage_ts=? WHERE id=?', (severity, incident['triage_ts'], incident['id']))
        conn.commit()
        out_msg = create_message(self.name, "coordinator", "triage_result", incident)
        self.send(out_msg)
        return out_msg
print("ğŸ˜�Triage Agent is Created.")


class ResourceAgent(Agent):
    def __init__(self, name="resource"):
        super().__init__(name)
    def match_resources(self, incident):
        lat, lon = incident['lat'], incident['lon']
        nearby = find_nearby_resources(lat, lon)
        # simple allocation: pick first available of each type
        allocation = {}
        for r in nearby:
            if r['available']:
                allocation.setdefault(r['type'], []).append(r['id'])
        payload = {"incident_id": incident['id'], "allocation": allocation}
        msg = create_message(self.name, "coordinator", "resource_allocation", payload)
        self.send(msg)
        return msg
print("ğŸ‘�Resources Agent is created.")


class CoordinatorAgent(Agent):
    def __init__(self, name="coordinator"):
        super().__init__(name)
        self.sessions = {}
    def compact_context(self, session):
        # naive compaction: join last N messages into summary
        texts = [m['payload'].get('text','') if m['kind']=='report' else json.dumps(m['payload']) for m in session.history]
        summary = " | ".join(texts[-5:])
        return summary
    def receive(self, msg):
        inc_id = msg['payload'].get('id') or msg['payload'].get('incident_id')
        if inc_id not in self.sessions:
            self.sessions[inc_id] = Session(inc_id)
        self.sessions[inc_id].add(msg)
        # if triage_result, request resources
        if msg['kind'] == 'triage_result':
            incident = msg['payload']
            # request resources
            resource_req = create_message(self.name, "resource", "resource_request", {"incident": incident})
            self.send(resource_req)
            return resource_req
        if msg['kind'] == 'resource_allocation':
            # finalize briefing
            briefing = {
                "incident_id": msg['payload']['incident_id'],
                "briefing": f"Allocation: {msg['payload']['allocation']}. Context summary: {self.compact_context(self.sessions[inc_id])}"
            }
            final_msg = create_message(self.name, "human", "briefing", briefing)
            self.send(final_msg)
            return final_msg

# Instantiate agents
receiver = ReceiverAgent()
triage = TriageAgent()
resource_agent = ResourceAgent()
coordinator = CoordinatorAgent()

print("âœ…Coordinator Agent Created.")


# Synthetic reports
reports = [
    ("Puneet","There is a large fire and people trapped in the building","Rajkot market"),
    ("Priya","Minor injuries, need first aid","Near lake view"),
    ("Rahul","Fainting and bleeding after accident", "Pune inner road"),
]

messages = []
for r in reports:
    msg = receiver.receive_report(r[0], r[1], r[2])
    messages.append(("receiver->triage", msg))
    tmsg = triage.process_report(msg)
    messages.append(("triage->coordinator", tmsg))
    # coordinator receives triage result and asks resource
    req = coordinator.receive(tmsg)  # this creates resource_request message
    # resource agent handles resource_request:
    res_msg = resource_agent.match_resources(tmsg['payload'])
    messages.append(("resource->coordinator", res_msg))
    final = coordinator.receive(res_msg)
    messages.append(("coordinator->human", final))

# show messages
for tag, m in messages:
    print(f"--- {tag} ---")
    pretty(m)

print("The code is simulated.ğŸ˜€")


# Show current incidents table
df = get_incidents(limit=20)
display(df)

# simple severity counts
print("Severity counts:")
print(df['severity'].value_counts(dropna=False))

# Show coordinator logs
print("\nCoordinator log (last 10):")
for m in coordinator.log[-10:]:
    print(m['kind'], m['timestamp'])

print("âœ…Obeservability clear.")


# small labeled dataset
eval_data = [
    ("someone trapped, building on fire", "high"),
    ("small cut, not severe", "low"),
    ("multiple injured, bleeding", "medium"),
    ("man collapsed, unconscious", "high"),
    ("need food for few people", "low")
]

preds, truths = [], []
for text,label in eval_data:
    preds.append(simple_triage_classifier(text))
    truths.append(label)

eval_df = pd.DataFrame({"text":[t for t,_ in eval_data],"pred":preds,"true":truths})
display(eval_df)
print("Accuracy:", (eval_df['pred']==eval_df['true']).mean())


