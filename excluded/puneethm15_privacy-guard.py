# -------------------------
# SETUP & INSTALLS
# -------------------------
!apt-get update -qq && apt-get install -y -qq tesseract-ocr libtesseract-dev poppler-utils
!pip install -q pytesseract pillow pandas numpy matplotlib python-dateutil fuzzywuzzy[speedup] streamlit

import os, io, json, sqlite3, uuid, datetime, re, time
from PIL import Image
import pytesseract
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fuzzywuzzy import process
from dateutil import parser as dateparser

# -------------------------
# GLOBALS
# -------------------------
DB_PATH = 'memorybank.db'
np.random.seed(42)

# Permission weight map
PERM_WEIGHTS = {
    "contacts": 0.9,
    "sms": 0.9,
    "call_log": 0.9,
    "location": 0.85,
    "camera": 0.7,
    "microphone": 0.7,
    "background_data": 0.6,
    "storage": 0.5,
    "internet": 0.4,
    "notifications": 0.3
}

# Small app reputation cache - can be extended
APP_REPUTATION = {
    "superflashlight": {"tags":["adware"], "note":"Requests odd permissions"},
    "mybank": {"tags":["banking"], "note":"Trusted banking app"},
    "chatly": {"tags":["social","tracking"], "note":"Social app tracking"}
}

# -------------------------
# OBSERVABILITY (logs, traces, metrics)
# -------------------------
LOGS = []
METRICS = {}

def log_event(agent, event_type, details):
    entry = {
        "timestamp": str(datetime.datetime.now()),
        "agent": agent,
        "event": event_type,
        "details": details
    }
    LOGS.append(entry)
    # also update simple metrics
    key = agent + '_' + event_type
    METRICS.setdefault(key, 0)
    METRICS[key] += 1

# Helper to print latest logs (compact)
def show_logs(n=10):
    for e in LOGS[-n:]:
        print(e)

# -------------------------
# A2A MESSAGE PROTOCOL
# -------------------------
# Agents exchange JSON messages with fields: msg_id, from, to, session_id, timestamp, payload, metrics (optional)

def new_msg(frm, to, session_id, payload):
    return {
        "msg_id": str(uuid.uuid4()),
        "from": frm,
        "to": to,
        "session_id": session_id,
        "timestamp": str(datetime.datetime.now()),
        "payload": payload,
        "metrics": {}
    }

# -------------------------
# OCR UTIL
# -------------------------

def ocr_image_to_text_bytes(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    text = pytesseract.image_to_string(img)
    return text

# -------------------------
# PARSER / EXTRACTOR UTIL
# -------------------------

def parse_permissions_from_text(text, max_lookback=5):
    """
    Robust parser:
    - Mark permission-like lines (contain ':' or known permission keywords).
    - For each permission line, assign it to the nearest previous non-permission line (within max_lookback).
    - Build app records from those groupings.
    """
    import re
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    n = len(lines)
    if n == 0:
        return []

    # helper: is this a permission-like line?
    def is_permission_line(s):
        if ':' in s:
            return True
        s_low = s.lower()
        for k in ["contacts","location","camera","microphone","sms","internet","storage","notifications","call","phone","contacts"]:
            if k in s_low:
                return True
        return False

    # mark lines
    is_perm = [is_permission_line(ln) for ln in lines]

    # For each permission line, find nearest preceding non-permission line as app
    app_map = {}  # app_name -> list of permission dicts
    last_app_fallback = None

    for i, ln in enumerate(lines):
        if is_perm[i]:
            # parse permission name & state if possible
            m = re.search(r'(?P<perm>[A-Za-z /_]+)[\:\-\t ]+(?P<state>Granted|Denied|Allowed|On|Off|Yes|No|Blocked)', ln, re.I)
            if m:
                perm = m.group('perm').strip()
                state = m.group('state').strip()
            else:
                # fallback: take entire line as permission name
                perm = ln
                state = 'Unknown'
            # look back for app
            app_name = None
            for lookback in range(1, max_lookback+1):
                j = i - lookback
                if j < 0:
                    break
                if not is_perm[j]:
                    app_name = lines[j]
                    break
            if not app_name:
                # fallback to last seen app header
                app_name = last_app_fallback or "UnknownApp"
            # add permission to app_map
            app_map.setdefault(app_name, []).append({"permission": perm, "state": state})
        else:
            # non-permission line: treat as potential app header, update fallback
            last_app_fallback = ln

    # convert app_map to records list
    records = []
    for app, perms in app_map.items():
        records.append({"app": app, "permissions": perms})
    return records


# -------------------------
# NORMALIZE APP NAME
# -------------------------

def normalize_app_name(name):
    keys = list(APP_REPUTATION.keys())
    if not keys:
        return name.lower()
    best, score = process.extractOne(name.lower(), keys)
    return best if score > 75 else name.lower()

# -------------------------
# RISK SCORING & RECOMMENDATIONS
# -------------------------

def compute_rule_score(permission_list):
    score = 0.0
    granted = 0
    for p in permission_list:
        pname = p['permission'].lower()
        state = p.get('state','').lower()
        weight = PERM_WEIGHTS.get(pname, 0.2)
        if state in ('granted','on','allowed','yes'):
            score += weight
            granted += 1
        if any(k in pname for k in ['contacts','sms','location','call']):
            score += 0.05
    max_possible = max(1, granted * max(PERM_WEIGHTS.values()))
    return round((score / max_possible) * 100, 1)


def generate_recommendations(app_record):
    recs = []
    for p in app_record['permissions']:
        pname = p['permission'].lower()
        state = p.get('state','').lower()
        if pname in ('contacts','sms','call_log') and state in ('granted','on','allowed'):
            recs.append(f"Revoke {p['permission']} unless essential.")
        if pname=='location' and state in ('granted','on','allowed'):
            recs.append("Set Location to 'Only while using' or deny when not needed.")
        if pname in ('camera','microphone') and state in ('granted','on','allowed'):
            recs.append(f"Grant {p['permission']} only when using the feature.")
    norm = normalize_app_name(app_record['app'])
    rep = APP_REPUTATION.get(norm)
    if rep and 'adware' in rep.get('tags',[]):
        recs.append('Consider uninstalling — flagged as adware.')
    return list(dict.fromkeys(recs))

# -------------------------
# AGENT WRAPPERS (Multi-Agent System)
# -------------------------
# Each agent accepts an A2A message and returns an A2A message.

# Ingest Agent: accepts files / text and creates session

def ingest_agent(msg):
    # payload: {uploads: [{name, bytes}] OR {text: ...}}
    session_id = msg.get('session_id') or str(uuid.uuid4())
    payload = msg['payload']
    artifacts = {'raw_inputs': []}

    if 'uploads' in payload:
        for u in payload['uploads']:
            artifacts['raw_inputs'].append({'name': u['name'], 'bytes_len': len(u['bytes'])})
    elif 'text' in payload:
        artifacts['raw_inputs'].append({'text_len': len(payload['text'])})

    out = new_msg('ingest_agent','ocr_extractor_agent', session_id, {'status':'ingested','artifacts':artifacts})
    log_event('ingest_agent','ingested', {'session_id': session_id, 'num_inputs': len(artifacts['raw_inputs'])})
    return out

# OCR & Extractor Agent: does OCR (if images) and parses

def ocr_extractor_agent(msg):
    sess = msg['session_id']
    payload = msg['payload']
    # expect payload to supply 'uploads' or 'text'
    parsed_records = []
    inputs = payload.get('uploads') or [{'name':'inline_text','bytes':None,'text': payload.get('text')}]

    for inp in inputs:
        if inp.get('bytes'):
            text = ocr_image_to_text_bytes(inp['bytes'])
            log_event('ocr_extractor_agent','ocr_done', {'name': inp['name']})
        else:
            text = inp.get('text','')
        recs = parse_permissions_from_text(text)
        parsed_records.extend(recs)

    out = new_msg('ocr_extractor_agent','scoring_agent', sess, {'records': parsed_records, 'num_records': len(parsed_records)})
    out['metrics']['num_records'] = len(parsed_records)
    log_event('ocr_extractor_agent','parsed', {'session_id':sess, 'num_records': len(parsed_records)})
    return out

# Scoring Agent: computes rule scores and basic metadata

def scoring_agent(msg):
    sess = msg['session_id']
    records = msg['payload'].get('records', [])
    scored = []
    for r in records:
        score = compute_rule_score(r['permissions'])
        rep_name = normalize_app_name(r['app'])
        r_out = { 'app': r['app'], 'normalized': rep_name, 'permissions': r['permissions'], 'score': score }
        scored.append(r_out)
    out = new_msg('scoring_agent','reputation_agent', sess, {'scored': scored})
    out['metrics']['num_scored'] = len(scored)
    log_event('scoring_agent','scored', {'session_id':sess, 'num_scored': len(scored)})
    return out

# Reputation Agent: enriches with reputation info (local cache + optional web.run)

def reputation_agent(msg):
    sess = msg['session_id']
    scored = msg['payload'].get('scored', [])
    enriched = []
    for s in scored:
        rep = APP_REPUTATION.get(s['normalized'], {})
        s['reputation'] = rep
        enriched.append(s)
    out = new_msg('reputation_agent','recommendation_agent', sess, {'enriched': enriched})
    log_event('reputation_agent','enriched', {'session_id':sess, 'num': len(enriched)})
    return out

# Recommendation Agent: produce remediation per app

def recommendation_agent(msg):
    sess = msg['session_id']
    enriched = msg['payload'].get('enriched', [])
    cards = []
    for e in enriched:
        recs = generate_recommendations({'app': e['app'], 'permissions': e['permissions']})
        explanation = f"Rule score {e['score']}" + ("; reputation flags: " + ",".join(e['reputation'].get('tags',[])) if e.get('reputation') else "")
        cards.append({'app': e['app'], 'score': e['score'], 'explanation': explanation, 'recommendations': recs})
    out = new_msg('recommendation_agent','report_agent', sess, {'cards': cards})
    log_event('recommendation_agent','recommended', {'session_id':sess, 'num_cards': len(cards)})
    return out

# Report Agent: compiles final report and writes MemoryBank summary

def report_agent(msg):
    sess = msg['session_id']
    cards = msg['payload'].get('cards', [])
    overall = np.mean([c['score'] for c in cards]) if cards else 0.0
    top_apps = sorted(cards, key=lambda x:-x['score'])[:5]
    report = {
        'session_id': sess,
        'overall_score': float(overall),
        'num_apps': len(cards),
        'top_apps': [t['app'] for t in top_apps],
        'cards': cards
    }
    # save to memorybank
    user_hash = 'demo_user'
    week_start = str(datetime.date.today() - datetime.timedelta(days=datetime.date.today().weekday()))
    save_week_summary(user_hash, week_start, float(overall), [t['app'] for t in top_apps], {c['app']: c['recommendations'] for c in cards})
    out = new_msg('report_agent','done', sess, {'report': report})
    log_event('report_agent','report_generated', {'session_id':sess, 'overall': overall, 'num_apps': len(cards)})
    return out

# -------------------------
# MEMORYBANK FUNCTIONS
# -------------------------

def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute('''CREATE TABLE IF NOT EXISTS memory_bank
                 (user_hash TEXT, week_start DATE, total_risk REAL, top_apps TEXT, recommendations TEXT, PRIMARY KEY(user_hash, week_start))''')
    conn.commit()
    return conn

conn = init_db()

def save_week_summary(user_hash, week_start, total_risk, top_apps, recommendations, conn=conn):
    conn.execute('REPLACE INTO memory_bank VALUES (?,?,?,?,?)',
                 (user_hash, week_start, total_risk, json.dumps(top_apps), json.dumps(recommendations)))
    conn.commit()


def fetch_history(user_hash, conn=conn):
    cur = conn.execute('SELECT week_start, total_risk, top_apps, recommendations FROM memory_bank WHERE user_hash=? ORDER BY week_start', (user_hash,))
    rows = cur.fetchall()
    out = []
    for r in rows:
        out.append({'week_start': r[0], 'total_risk': r[1], 'top_apps': json.loads(r[2]), 'recommendations': json.loads(r[3])})
    return out

# -------------------------
# CONTEXT COMPACTION
# -------------------------
# Compact long contexts before sending to LLMs or other agents. Here we implement a simple compaction
# that keeps only top N apps and permission counts.

def compact_context(cards, max_apps=10):
    cards_sorted = sorted(cards, key=lambda x:-x['score'])
    top = cards_sorted[:max_apps]
    summary = {
        'total_apps': len(cards),
        'top_apps': [ {'app':c['app'], 'score': c['score'], 'top_reasons': c['explanation'][:120]} for c in top ]
    }
    return summary

# -------------------------
# EVALUATION (Tiny labeled set)
# -------------------------

def evaluate_tiny():
    labeled = [
        ({'app':'FlashPro','permissions':[{'permission':'Camera','state':'Granted'},{'permission':'SMS','state':'Granted'}]}, 'high'),
        ({'app':'WeatherNow','permissions':[{'permission':'Location','state':'Granted'},{'permission':'Internet','state':'Granted'}]}, 'medium'),
        ({'app':'NotesApp','permissions':[{'permission':'Storage','state':'Granted'}]}, 'low')
    ]
    def score_to_label(score):
        if score>=70: return 'high'
        if score>=40: return 'medium'
        return 'low'
    preds, trues = [], []
    for rec, true_label in labeled:
        s = compute_rule_score(rec['permissions'])
        preds.append(score_to_label(s))
        trues.append(true_label)
    acc = sum(p==t for p,t in zip(preds,trues)) / len(preds)
    return {'preds':preds,'trues':trues,'accuracy':acc}

# -------------------------
# STREAMLIT APP GENERATOR (writes app.py)
# -------------------------

def write_streamlit_app(path='app.py'):
    # NOTE: this string is NOT an f-string — it intentionally keeps {name} and {e}
    # so they are evaluated inside the generated app, not here.
    content = """import streamlit as st
from PIL import Image
import io, json
import pandas as pd
import matplotlib.pyplot as plt
import pytesseract

# Minimal helper implementations copied from notebook so the app is standalone.
# You can improve these later.

PERM_WEIGHTS = {
    "contacts": 0.9, "sms": 0.9, "call_log": 0.9, "location": 0.85,
    "camera": 0.7, "microphone": 0.7, "background_data": 0.6,
    "storage": 0.5, "internet": 0.4, "notifications": 0.3
}

def parse_permissions_from_text(text):
    import re
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    records = []
    current_app = None
    current_perms = []
    for ln in lines:
        if re.search(r'\\bApp\\b|\\bapp\\b', ln) or (len(ln.split())<=4 and ln[0].isupper() and ln==ln.title()):
            if current_app and current_perms:
                records.append({"app": current_app, "permissions": current_perms})
            current_app = ln
            current_perms = []
            continue
        m = re.search(r'(?P<perm>[A-Za-z /_]+)[\\:\\-\\t ]+(?P<state>Granted|Denied|Allowed|On|Off|Yes|No|Blocked)', ln, re.I)
        if m and current_app:
            perm = m.group('perm').strip()
            state = m.group('state').strip()
            current_perms.append({"permission": perm, "state": state})
            continue
        for k in PERM_WEIGHTS.keys():
            if k in ln.lower() and current_app:
                state = "Granted" if re.search(r'granted|on|allowed|yes', ln, re.I) else "Unknown"
                current_perms.append({"permission": k, "state": state})
                break
    if current_app and current_perms:
        records.append({"app": current_app, "permissions": current_perms})
    return records

def compute_rule_score(permission_list):
    score = 0.0
    granted = 0
    for p in permission_list:
        pname = p['permission'].lower()
        state = p.get('state','').lower()
        weight = PERM_WEIGHTS.get(pname, 0.2)
        if state in ('granted','on','allowed','yes'):
            score += weight
            granted += 1
        if any(k in pname for k in ['contacts','sms','location','call']):
            score += 0.05
    max_possible = max(1, granted * max(PERM_WEIGHTS.values()))
    return round((score / max_possible) * 100, 1)

def generate_recommendations(app_record):
    recs = []
    for p in app_record.get('permissions', []):
        pname = p['permission'].lower()
        state = p.get('state','').lower()
        if pname in ('contacts','sms','call_log') and state in ('granted','on','allowed'):
            recs.append(f"Revoke {p['permission']} unless essential.")
        if pname=='location' and state in ('granted','on','allowed'):
            recs.append("Set Location to 'Only while using' or deny when not needed.")
        if pname in ('camera','microphone') and state in ('granted','on','allowed'):
            recs.append(f"Grant {p['permission']} only when using the feature.")
    return list(dict.fromkeys(recs))

def score_and_explain(app_record):
    score = compute_rule_score(app_record.get('permissions', []))
    recs = generate_recommendations(app_record)
    explanation = f"Risk Score {score}. Higher = more risky."
    return {"app": app_record.get('app','Unknown'), "score": score, "explanation": explanation, "recommendations": recs}

# Streamlit UI
st.set_page_config(page_title='PrivacyGuard Demo', layout='wide')
st.title('PrivacyGuard — Permission Risk Analyzer')
st.markdown('Upload screenshots or JSON permission lists to analyze privacy risk.')

uploaded = st.file_uploader('Upload images or JSON files', accept_multiple_files=True)
if uploaded:
    cards = []
    for f in uploaded:
        name = f.name
        b = f.read()
        try:
            if name.lower().endswith(('.png','.jpg','.jpeg')):
                text = pytesseract.image_to_string(Image.open(io.BytesIO(b)).convert('RGB'))
                recs = parse_permissions_from_text(text)
                for r in recs:
                    c = score_and_explain(r)
                    cards.append(c)
            elif name.lower().endswith('.json'):
                data = json.loads(b.decode())
                c = score_and_explain(data)
                cards.append(c)
        except Exception as e:
            st.error('Error processing {}: {}'.format(name, e))
    if cards:
        df = pd.DataFrame([{'app':c['app'],'score':c['score']} for c in cards])
        st.dataframe(df)
        fig, ax = plt.subplots()
        ax.bar(df['app'], df['score'])
        st.pyplot(fig)
"""
    with open(path,'w') as f:
        f.write(content)
    print(f'Wrote Streamlit app to {path}')

# -------------------------
# DEMO RUN (simulate end-to-end agent flow)
# -------------------------
print("\n=== PrivacyGuard Demo (Agent Pipeline) ===\n")
# create a synthetic upload payload
sample_text = """
SuperFlashlight
Camera: On
SMS: Allowed
Location: Denied

Chatly
Location: On
Contacts: Granted
Microphone: Allowed

MyBank
Internet: Allowed
Contacts: Denied
"""

session_id = str(uuid.uuid4())
# Ingest
msg = new_msg('user','ingest_agent', session_id, {'text': sample_text})
resp1 = ingest_agent(msg)
# OCR & Extractor (we'll call extractor directly with text)
resp2 = ocr_extractor_agent({'msg_id':resp1['msg_id'],'from':'ingest_agent','to':'ocr_extractor_agent','session_id':session_id,'payload':{'text': sample_text}})
# Scoring
resp3 = scoring_agent(resp2)
# Reputation
resp4 = reputation_agent(resp3)
# Recommendation
resp5 = recommendation_agent(resp4)
# Report
resp6 = report_agent(resp5)

# Print final report summary
report = resp6['payload']['report']
print('Session:', report['session_id'])
print('Overall privacy risk score:', report['overall_score'])
print('Top apps:', report['top_apps'])
for c in report['cards']:
    print('-'*40)
    print('App:', c['app'])
    print('Score:', c['score'])
    print('Explanation:', c['explanation'])
    print('Recommendations:')
    for r in c['recommendations']:
        print('  -', r)

# Show logs and metrics
print("\n--- Observability Logs (last 10) ---")
show_logs(20)
print("\n--- Metrics ---")
print(json.dumps(METRICS, indent=2))

# Run tiny evaluation
eval_res = evaluate_tiny()
print("\n--- Tiny Evaluation ---")
print("Preds:", eval_res['preds'])
print("Trues:", eval_res['trues'])
print("Accuracy:", eval_res['accuracy'])

# Create Streamlit app file for deployment (optional)
write_streamlit_app()

print("\nPrivacyGuard agent pipeline complete. The MemoryBank contains:")
print(fetch_history('demo_user'))


