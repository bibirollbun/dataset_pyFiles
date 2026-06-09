# --- GEMINI SETUP ---
# Turn Internet ON in Kaggle Notebook Settings before running.
# Load API key from Kaggle Secrets at the beginning of the notebook.

!pip -q install -U google-generativeai

import os, json
import google.generativeai as genai

# Read from Kaggle Secrets (preferred). Fallback to environment.
API_KEY = None
try:
    from kaggle_secrets import UserSecretsClient
    API_KEY = UserSecretsClient().get_secret("GEMINI_API_KEY")
except Exception:
    API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("Add a Kaggle Secret named 'GEMINI_API_KEY' with your Google AI Studio key.")

genai.configure(api_key=API_KEY)

# Default model. You can set GEMINI_MODEL env to override.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

def _make_model(json_mode=False, temperature=0.2):
    cfg = {"temperature": temperature}
    if json_mode:
        cfg["response_mime_type"] = "application/json"
    return genai.GenerativeModel(GEMINI_MODEL, generation_config=cfg)

def gemini_text(prompt, temperature=0.2):
    m = _make_model(json_mode=False, temperature=temperature)
    resp = m.generate_content(prompt)
    return getattr(resp, "text", str(resp))

def gemini_json(prompt, temperature=0.1):
    m = _make_model(json_mode=True, temperature=temperature)
    resp = m.generate_content(prompt)
    try:
        return json.loads(resp.text)
    except Exception:
        # Fallback: try to parse best-effort JSON from text
        try:
            return json.loads(str(resp))
        except Exception:
            return {"error":"Could not parse JSON", "raw": str(resp)}

USE_GEMINI = True
print("Gemini configured. Using model:", GEMINI_MODEL)


# --- DATA & POLICY ---

import os, sys, json, glob, zipfile, shutil

KAGGLE_INPUT = "/kaggle/input"
WORK = "/kaggle/working"
BASE = os.path.join(WORK, "capstone_kit")

def _print_tree(root, max_chars=8000):
    lines = []
    for r, d, f in os.walk(root):
        level = r.replace(root, '').count(os.sep)
        lines.append("  " * level + os.path.basename(r) + "/")
        for name in f:
            lines.append("  " * (level + 1) + name)
    text = "\n".join(lines)
    print(text[:max_chars] + ("\n... (truncated)" if len(text) > max_chars else ""))

def _attach_from_input():
    if not os.path.isdir(KAGGLE_INPUT):
        return False
    for root, dirs, files in os.walk(KAGGLE_INPUT):
        if os.path.basename(root) == "capstone_kit":
            if os.path.exists(BASE): shutil.rmtree(BASE)
            shutil.copytree(root, BASE)
            return True
    zip_candidates = glob.glob(os.path.join(KAGGLE_INPUT, "**", "*.zip"), recursive=True)
    for zpath in zip_candidates:
        try:
            with zipfile.ZipFile(zpath, "r") as z:
                z.extractall(WORK)
            if os.path.isdir(BASE):
                return True
        except Exception as e:
            print("Zip read failed:", zpath, e)
            continue
    return False

def _bootstrap_locally():
    import pandas as pd
    try:
        import yaml
    except Exception:
        print("PyYAML not found. If this fails, enable Internet in Settings and: !pip -q install pyyaml")
        import yaml

    for sub in ["", "agent", "tools", "data", "scenarios", "artifacts"]:
        os.makedirs(os.path.join(BASE, sub), exist_ok=True)

    policy = {
        "currency": "USD",
        "hotel_max_by_city": {"NYC": 230, "PIT": 150, "SFO": 250},
        "meal_cap_per_day": 60,
        "receipt_required_min": 25,
        "duplicate_detection_window_days": 1,
        "flight": {"advance_purchase_min_days": 7},
        "price_drop": {"rebook_threshold": 30}
    }
    json.dump(policy, open(f"{BASE}/data/policy.json","w"), indent=2)

    pd.DataFrame([
        ["S1","TRIP01","flight","2025-10-07","2025-10-07","NYC","PIT-JFK","Delta Air Lines","",350,"Out 7 Oct"],
        ["S2","TRIP01","hotel","2025-10-07","2025-10-09","NYC","","Hilton Midtown",230,460,"2 nights @ $230 policy cap"],
        ["S3","TRIP01","ground","2025-10-08","2025-10-08","NYC","","Uber","",40,"In-city ride"],
        ["S4","TRIP01","meal","2025-10-08","2025-10-08","NYC","","Dinner","",60,"Per-diem cap $60"],
    ], columns=["segment_id","trip_id","type","start_date","end_date","city","route","merchant_expected","nightly_expected","amount_expected","notes"]
    ).to_csv(f"{BASE}/data/itinerary.csv", index=False)

    pd.DataFrame([
        ["R1","2025-10-07","Hilton Midtown",260.00,"hotel","NYC",True,"corp_card","Night 1 over ADR cap"],
        ["R2","2025-10-08","Hilton Midtown",260.00,"hotel","NYC",True,"corp_card","Night 2 over ADR cap"],
        ["R3","2025-10-08","Joe's Steakhouse",85.00,"meal","NYC",True,"personal_card","Over per-diem"],
        ["R4","2025-10-08","Uber",48.00,"ground","NYC",False,"corp_card","Missing receipt >= $25"],
        ["R5","2025-10-08","Taxi Corp",30.00,"ground","NYC",True,"personal_card","Duplicate A"],
        ["R6","2025-10-08","Taxi Corp",30.00,"ground","NYC",True,"personal_card","Duplicate B"],
        ["R7","2025-10-10","Starbucks",5.00,"meal","NYC",False,"personal_card","Outside itinerary dates"],
        ["R8","2025-10-08","Electronics World",120.00,"other","NYC",True,"personal_card","Unplanned category"],
    ], columns=["receipt_id","date","merchant","amount","category","city","has_receipt","payment_method","notes"]
    ).to_csv(f"{BASE}/data/receipts.csv", index=False)

    pd.DataFrame([
        ["B1","flight","PIT-JFK",350.0,50.0,1,"2025-09-20",340.0,True],
        ["B1","flight","PIT-JFK",350.0,50.0,1,"2025-09-23",280.0,True],
        ["B1","flight","PIT-JFK",350.0,50.0,1,"2025-09-25",250.0,True],
        ["B2","hotel","Hilton Midtown NYC",460.0,0.0,2,"2025-09-25",440.0,True],
        ["B2","hotel","Hilton Midtown NYC",460.0,0.0,2,"2025-09-28",420.0,True],
    ], columns=["booking_id","type","route_or_hotel","booked_price_total","change_fee","nights","observed_at","observed_price","refundable"]
    ).to_csv(f"{BASE}/data/price_series.csv", index=False)

    scenarios = {
        "policy_tests": [
            {"name":"NYC hotel over cap",
             "proposed":{"type":"hotel","city":"NYC","nightly_rate":260,"nights":2},
             "expected_decision":"deny","expect_reason_contains":["over city cap"]},
            {"name":"Pittsburgh hotel within cap",
             "proposed":{"type":"hotel","city":"PIT","nightly_rate":130,"nights":2},
             "expected_decision":"approve"},
            {"name":"Flight insufficient advance purchase",
             "proposed":{"type":"flight","route":"PIT-SFO","advance_purchase_days":3,"cabin":"economy"},
             "expected_decision":"needs_approval","expect_reason_contains":["advance purchase"]},
            {"name":"Meal over per-diem",
             "proposed":{"type":"meal","amount":85},
             "expected_decision":"deny","expect_reason_contains":["per-diem"]}
        ],
        "audit_tests": [
            {"name":"Baseline audit on sample receipts",
             "expected_flags":{"over_cap":3,"duplicate_pair":1,"missing_receipt":1,"off_itinerary":2,"date_mismatch":0},
             "expected_dollars_at_risk":85}
        ],
        "price_drop_tests": [
            {"name":"Flight PIT-JFK should rebook","booking_id":"B1","expected_rebook":True,"expect_min_net_savings":50},
            {"name":"Hotel NYC 2 nights should rebook","booking_id":"B2","expected_rebook":True,"expect_min_net_savings":40}
        ]
    }
    try:
        import yaml
        yaml.safe_dump(scenarios, open(f"{BASE}/scenarios/scenarios.yaml","w"), sort_keys=False)
    except Exception:
        json.dump(scenarios, open(f"{BASE}/scenarios/scenarios.json","w"), indent=2)

    open(f"{BASE}/tools/policy_copilot.py","w").write("""
import json
def load_policy(path): return json.load(open(path))
def check_proposal(policy, proposed):
    reasons, suggestions = [], []; decision = "approve"; t = proposed.get("type")
    if t == "hotel":
        city = proposed.get("city"); nightly = float(proposed.get("nightly_rate",0))
        cap = policy.get("hotel_max_by_city",{}).get(city)
        if cap is not None and nightly > cap:
            decision = "deny"; reasons.append(f"Hotel nightly rate ${nightly:.0f} exceeds {city} cap ${cap:.0f} (over city cap).")
            suggestions.append(f"Choose a property at or under ${cap:.0f} per night.")
    elif t == "meal":
        amt = float(proposed.get("amount",0)); per = policy.get("meal_cap_per_day",0)
        if amt > per:
            decision = "deny"; reasons.append(f"Meal amount ${amt:.0f} exceeds per-diem ${per:.0f}.")
            suggestions.append(f"Keep meals at or under ${per:.0f} per day.")
    elif t == "flight":
        ap_min = policy.get("flight",{}).get("advance_purchase_min_days",0)
        ap_days = int(proposed.get("advance_purchase_days",0))
        if ap_days < ap_min:
            decision = "needs_approval"; reasons.append(f"Advance purchase {ap_days}d < minimum {ap_min}d (advance purchase).")
            suggestions.append(f"Book at least {ap_min} days ahead or obtain manager approval.")
        if proposed.get("cabin","economy").lower() != "economy":
            decision = "needs_approval"; reasons.append("Cabin not economy."); suggestions.append("Select economy to auto-approve.")
    else:
        decision = "needs_approval"; reasons.append("Unknown request type; manual review required.")
    return decision, reasons, suggestions
""")
    open(f"{BASE}/tools/expense_auditor.py","w").write("""
import pandas as pd
def load_inputs(itinerary_path, receipts_path, card_path=None):
    it_df = pd.read_csv(itinerary_path); rc_df = pd.read_csv(receipts_path)
    it_df["start_date"] = pd.to_datetime(it_df["start_date"]).dt.date
    it_df["end_date"]   = pd.to_datetime(it_df["end_date"]).dt.date
    rc_df["date"]       = pd.to_datetime(rc_df["date"]).dt.date
    return it_df, rc_df, None
def audit(policy, itinerary_df, receipts_df):
    trip_start, trip_end = itinerary_df["start_date"].min(), itinerary_df["end_date"].max()
    hotel_caps = policy.get("hotel_max_by_city",{}); per_diem = policy.get("meal_cap_per_day",0); receipt_min = policy.get("receipt_required_min",0)
    flags=[]; dollars=0.0
    for _, r in receipts_df.iterrows():
        rec_flags=[]; amt=float(r["amount"]); cat=str(r["category"]).lower(); city=r.get("city",None); has_receipt=bool(r.get("has_receipt",True))
        if amt >= receipt_min and not has_receipt: rec_flags.append("missing_receipt")
        if (r["date"] < trip_start) or (r["date"] > trip_end): rec_flags.append("off_itinerary")
        if cat=="hotel":
            cap = hotel_caps.get(city); 
            if cap is not None and amt>cap: rec_flags.append("over_cap"); dollars += (amt-cap)
        elif cat=="meal":
            if amt>per_diem: rec_flags.append("over_cap"); dollars += (amt-per_diem)
        elif cat in ["ground","ride","taxi","uber","lyft","parking","toll"]: pass
        else: rec_flags.append("off_itinerary")
        flags.append({"receipt_id":r["receipt_id"],"date":str(r["date"]),"merchant":r["merchant"],"amount":amt,"category":cat,"flags":rec_flags})
    df = receipts_df[["date","merchant","amount","category"]].copy(); df["date"]=pd.to_datetime(df["date"]).dt.date
    grp = df[df["category"].str.lower().isin(["ground","ride","taxi","uber","lyft","parking","toll","meal"])].groupby(["merchant","amount","date"]).size()
    dup_pairs = int((grp>=2).sum())
    counts = {
        "over_cap": sum("over_cap" in f["flags"] for f in flags),
        "missing_receipt": sum("missing_receipt" in f["flags"] for f in flags),
        "off_itinerary": sum("off_itinerary" in f["flags"] for f in flags),
        "date_mismatch": 0,
        "duplicate_pair": dup_pairs
    }
    return flags, counts, round(dollars,2)
""")
    open(f"{BASE}/tools/price_drop_advisor.py","w").write("""
import pandas as pd
def decide_rebook(policy, price_series_csv, booking_id):
    df=pd.read_csv(price_series_csv); dfb=df[df['booking_id']==booking_id].copy()
    if dfb.empty: return {"booking_id":booking_id,"decision":False,"reason":"booking not found","net_savings":0.0}
    booked=float(dfb['booked_price_total'].iloc[0]); fee=float(dfb['change_fee'].iloc[0])
    min_price=float(dfb['observed_price'].min()); net= booked-(min_price+fee)
    threshold=float(policy.get('price_drop',{}).get('rebook_threshold',0))
    return {"booking_id":booking_id,"decision": bool(net>=threshold),"reason": f"min ${min_price:.0f}+fee ${fee:.0f} -> net ${net:.0f} vs thr ${threshold:.0f}","net_savings":round(net,2),"booked_total":booked,"min_observed":min_price}
""")
    open(f"{BASE}/agent/orchestrator.py","w").write("""
import json, yaml
from tools.policy_copilot import load_policy, check_proposal
from tools.expense_auditor import load_inputs, audit
from tools.price_drop_advisor import decide_rebook
def run_policy_tests(policy_path, scenarios_yaml):
    policy = load_policy(policy_path); tests = yaml.safe_load(open(scenarios_yaml)).get("policy_tests",[])
    out=[]; okn=0
    for t in tests:
        dec, reasons, sugg = check_proposal(policy, t["proposed"])
        ok = (dec==t["expected_decision"])
        if ok and t.get("expect_reason_contains"):
            exp=t["expect_reason_contains"]; ok = any(any(s in r for s in exp) for r in reasons)
        out.append({"name":t["name"],"decision":dec,"reasons":reasons,"ok":ok}); okn += int(ok)
    return out, okn, len(tests)
def run_audit_tests(policy_path, itinerary_csv, receipts_csv, scenarios_yaml, artifacts_dir):
    policy=load_policy(policy_path); it_df, rc_df, _ = load_inputs(itinerary_csv, receipts_csv)
    flags, counts, dollars = audit(policy, it_df, rc_df)
    json.dump(flags, open(f"{artifacts_dir}/audit_flags.json","w"), indent=2)
    tests = yaml.safe_load(open(scenarios_yaml)).get("audit_tests", [])
    ok=True
    if tests:
        exp=tests[0]["expected_flags"]; exp_d=float(tests[0]["expected_dollars_at_risk"])
        ok_counts = all(counts.get(k,0)==v for k,v in exp.items()); ok_dollars = abs(dollars-exp_d)<1e-6; ok = ok_counts and ok_dollars
    return counts, dollars, ok
def run_price_drop_tests(policy_path, price_series_csv, scenarios_yaml, artifacts_dir):
    policy=load_policy(policy_path); tests=yaml.safe_load(open(scenarios_yaml)).get("price_drop_tests",[])
    res=[]; okn=0
    for t in tests:
        r=decide_rebook(policy, price_series_csv, t["booking_id"])
        ok = (bool(r["decision"])==bool(t["expected_rebook"])) and (r["net_savings"]>=t.get("expect_min_net_savings",0))
        r["ok"]=ok; res.append(r); okn += int(ok)
    json.dump(res, open(f"{artifacts_dir}/price_drop_results.json","w"), indent=2)
    return res, okn, len(tests)
def build_scorecard(policy_path, itinerary_csv, receipts_csv, price_series_csv, scenarios_yaml, artifacts_dir):
    p_res, p_ok, p_tot = run_policy_tests(policy_path, scenarios_yaml)
    a_counts, a_dollars, a_ok = run_audit_tests(policy_path, itinerary_csv, receipts_csv, scenarios_yaml, artifacts_dir)
    d_res, d_ok, d_tot = run_price_drop_tests(policy_path, price_series_csv, scenarios_yaml, artifacts_dir)
    score = {"policy":{"passed":p_ok,"total":p_tot,"accuracy":round(p_ok/max(p_tot,1),2)},
             "audit":{"counts":a_counts,"dollars_at_risk":a_dollars,"ok":bool(a_ok)},
             "price_drop":{"passed":d_ok,"total":d_tot,"accuracy":round(d_ok/max(d_tot,1),2)}}
    json.dump(score, open(f"{artifacts_dir}/scorecard.json","w"), indent=2)
    return score, p_res, d_res
""")
    return True

attached = _attach_from_input()
if not attached:
    print("No attached kit detected; bootstrapping locally...")
    _bootstrap_locally()
else:
    print("Loaded kit from /kaggle/input.")

sys.path.insert(0, BASE)
POLICY = f"{BASE}/data/policy.json"
ITIN   = f"{BASE}/data/itinerary.csv"
RCPTS  = f"{BASE}/data/receipts.csv"
PRICES = f"{BASE}/data/price_series.csv"
SCEN   = f"{BASE}/scenarios/scenarios.yaml"
ART    = f"{BASE}/artifacts"

print("Kit location:", BASE)
_print_tree(BASE)



# --- LLM ROUTER AGENT ---
# The router takes a free-form message and returns a structured action for the orchestrator.

import json

ROUTER_SCHEMA = {
  "action": "policy | price | audit",
  "params": {
    "policy": {"type":"hotel|flight|meal|other", "...":"..."},
    "price": {"booking_id":"B1 or B2"},
    "audit": {}
  },
  "reason": "short rationale"
}

ROUTER_PROMPT_TMPL = """
You route user travel requests to one of three actions: policy, price, or audit.
Return ONLY JSON in this exact schema:
{{
  "action": "policy" | "price" | "audit",
  "params": <object>,
  "reason": "<one short sentence>"
}}

If action is "policy", params must include a "type" field and relevant keys:
- hotel: city, nightly_rate, nights
- flight: route, advance_purchase_days, cabin
- meal: amount
If action is "price", params must include "booking_id" like "B1" or "B2".
If action is "audit", params is an empty object.

User message:
{message}
"""

def llm_route(message: str):
    if not USE_GEMINI:
        # Naive offline default
        if "price" in message.lower() or "rebook" in message.lower():
            return {"action":"price","params":{"booking_id":"B1"},"reason":"offline default"}
        if "audit" in message.lower() or "receipt" in message.lower():
            return {"action":"audit","params":{},"reason":"offline default"}
        return {"action":"policy","params":{"type":"hotel","city":"NYC","nightly_rate":230,"nights":2},"reason":"offline default"}
    prompt = ROUTER_PROMPT_TMPL.format(message=message)
    out = gemini_json(prompt)
    return out


# --- AGENT ENTRYPOINT DEMO ---
# Requires Setup & Data cell ran (to define POLICY, ITIN, RCPTS, PRICES, SCEN, ART).

from tools.policy_copilot import load_policy, check_proposal
from tools.price_drop_advisor import decide_rebook
from tools.expense_auditor import load_inputs, audit

policy = load_policy(POLICY)

examples = [
    "I need a hotel in NYC at 260 per night for two nights next week",
    "Did the price drop on my flight booking B1",
    "Audit my trip receipts against the itinerary"
]

for msg in examples:
    print("\n=== User:", msg)
    route = llm_route(msg)
    print("Router:", route)

    if route.get("action") == "policy":
        proposed = route.get("params", {})
        decision, reasons, suggestions = check_proposal(policy, proposed)
        print("Decision:", decision)
        print("Reasons:", reasons)
        if USE_GEMINI:
            p = f"""Write a short traveler note that explains this decision and 1 suggestion.
Decision: {decision}
Reasons: {reasons}
Suggestions: {suggestions}
Proposed: {json.dumps(proposed)}"""
            print(gemini_text(p))
    elif route.get("action") == "price":
        bid = route.get("params", {}).get("booking_id", "B1")
        r = decide_rebook(policy, PRICES, bid)
        print("Price decision:", r)
        if USE_GEMINI:
            p = f"""Write a short memo to explain if we should rebook or not with the net savings math.
Data: {json.dumps(r)}"""
            print(gemini_text(p))
    elif route.get("action") == "audit":
        it_df, rc_df, _ = load_inputs(ITIN, RCPTS)
        flags, counts, dollars = audit(policy, it_df, rc_df)
        print("Audit counts:", counts, "| Dollars at risk:", dollars)
        if USE_GEMINI:
            p = f"""Summarize the audit as 3 bullet points for a CFO. Include dollars at risk.
Counts: {json.dumps(counts)} Dollars at risk: {dollars}"""
            print(gemini_text(p))
    else:
        print("Unknown action.")


# --- POLICY DETAILS ---
import json, pandas as pd
from tools.policy_copilot import load_policy

policy = load_policy(POLICY)

# Tabular view of hotel caps
caps = policy.get("hotel_max_by_city", {})
df_caps = pd.DataFrame([{"city":k, "hotel_cap_usd":v} for k,v in caps.items()]).sort_values("city")
display(df_caps)

print("Meal per diem USD:", policy.get("meal_cap_per_day"))
print("Receipt required min USD:", policy.get("receipt_required_min"))
print("Advance purchase min days:", policy.get("flight",{}).get("advance_purchase_min_days"))
print("Rebook threshold USD:", policy.get("price_drop",{}).get("rebook_threshold"))

# Optional Gemini summary for readability
if USE_GEMINI:
    prompt = f"""Summarize this travel policy for a small team in 4 to 6 bullet points.
Keep it plain and specific. Use USD.
{json.dumps(policy)}"""
    print("\nPolicy brief:")
    print(gemini_text(prompt))


# --- RUN UNIFIED SCORECARD ---
from agent.orchestrator import build_scorecard
import json

# Paths and ART are defined by the Setup & Data cell.
scorecard, policy_results, price_results = build_scorecard(POLICY, ITIN, RCPTS, PRICES, SCEN, ART)

print("Unified scorecard:\n", json.dumps(scorecard, indent=2))



# --- POLICY COPILOT DEMO ---
from tools.policy_copilot import load_policy, check_proposal

policy = load_policy(POLICY)
cases = [
    {"name":"NYC hotel @ $260","proposed":{"type":"hotel","city":"NYC","nightly_rate":260,"nights":2}},
    {"name":"PIT hotel @ $130","proposed":{"type":"hotel","city":"PIT","nightly_rate":130,"nights":2}},
    {"name":"Flight 3d advance","proposed":{"type":"flight","route":"PIT-SFO","advance_purchase_days":3,"cabin":"economy"}},
    {"name":"Meal $85","proposed":{"type":"meal","amount":85}},
]

for c in cases:
    dec, reasons, sugg = check_proposal(policy, c["proposed"])
    print(f"\nCase: {c['name']} â†’ Decision: {dec}")
    if reasons: print("  Reasons:", "; ".join(reasons))
    if sugg:    print("  Suggestions:", "; ".join(sugg))



# --- EXPENSE AUDITOR DEMO ---
import pandas as pd, matplotlib.pyplot as plt
from tools.expense_auditor import load_inputs, audit
from tools.policy_copilot import load_policy

policy = load_policy(POLICY)
it_df, rc_df, _ = load_inputs(ITIN, RCPTS)
flags, counts, dollars = audit(policy, it_df, rc_df)

# Peek at flagged items
flag_df = pd.DataFrame(flags)
display(flag_df.head())

# Simple counts bar chart (no custom colors, single chart as per Kaggle guidelines)
keys = ["over_cap","missing_receipt","off_itinerary","duplicate_pair"]
values = [counts.get(k,0) for k in keys]

plt.figure()
plt.bar(keys, values)
plt.title("Audit Flags Summary")
plt.xlabel("Flag")
plt.ylabel("Count")
plt.show()

print("Dollars at risk:", dollars)



# --- PRICE-DROP ADVISOR DEMO ---
from tools.price_drop_advisor import decide_rebook
from tools.policy_copilot import load_policy

policy = load_policy(POLICY)
for bid in ["B1","B2"]:
    r = decide_rebook(policy, PRICES, bid)
    print(f"{bid} â†’ decision={r['decision']} | reason={r['reason']} | net_savings=${r['net_savings']:.2f}")



# --- RESULTS CALL-OUT TABLE ---
import json, pandas as pd, pathlib

sc_path = pathlib.Path(ART) / "scorecard.json"
sc = json.load(open(sc_path))

rows = []
rows.append(("Policy accuracy", f"{sc['policy']['passed']}/{sc['policy']['total']} ({sc['policy']['accuracy']:.2f})"))
rows.append(("Price-drop accuracy", f"{sc['price_drop']['passed']}/{sc['price_drop']['total']} ({sc['price_drop']['accuracy']:.2f})"))
rows.append(("Audit: dollars_at_risk", f"${sc['audit']['dollars_at_risk']:.2f}"))
for k, v in sc["audit"]["counts"].items():
    rows.append((f"Audit: {k}", v))

pd.DataFrame(rows, columns=["Metric","Value"])


