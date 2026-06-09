# Kaggle notebook single cell: Personal Assistant Agent with SEQUENTIAL AGENTS (Interpreter -> Planner -> Action -> Responder)
# Paste the whole cell in a Kaggle notebook and run.
# Provides: time, book cab, appointment + .ics, start/diagnose car (simulated), generate 5-day plan.
# Uses sequential agents to orchestrate tools.

import sys, subprocess, importlib, os, json, re, uuid
from datetime import datetime, timedelta

# -------------------------
# Robust installer helper
# -------------------------
def base_pkg_name(spec: str) -> str:
    for op in ["==", ">=", "<=", ">", "<", "~="]:
        if op in spec:
            return spec.split(op)[0].strip()
    return spec.strip()

def ensure_installed(specs):
    for spec in specs:
        name = base_pkg_name(spec)
        try:
            importlib.import_module(name)
        except Exception:
            print(f"Installing {spec} ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", spec])

ensure_installed(["gradio>=3.0", "python-dateutil", "ics"])

from dateutil import parser as dtparser
from ics import Calendar, Event
import gradio as gr

# -------------------------
# Persistence (JSON DB)
# -------------------------
WORK_DIR = "/kaggle/working"
os.makedirs(WORK_DIR, exist_ok=True)
DB_PATH = os.path.join(WORK_DIR, "assistant_db.json")

def load_db():
    if os.path.exists(DB_PATH):
        try:
            with open(DB_PATH, "r") as f:
                return json.load(f)
        except:
            pass
    return {"bookings": [], "appointments": [], "plans": [], "next_booking_id": 1, "next_appointment_id": 1}

def save_db(db):
    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=2, default=str)

db = load_db()

# -------------------------
# Core tool functions (the "tools" that agents call)
# -------------------------
def add_booking(kind, title, dt_obj, details=""):
    bid = db.get("next_booking_id", 1)
    rec = {"id": bid, "kind": kind, "title": title, "datetime": dt_obj.isoformat(), "details": details, "created_at": datetime.utcnow().isoformat()}
    db["bookings"].append(rec); db["next_booking_id"] = bid + 1; save_db(db)
    return rec

def list_bookings():
    return db.get("bookings", [])

def cancel_booking(bid):
    before = len(db["bookings"])
    db["bookings"] = [b for b in db["bookings"] if b["id"] != bid]
    save_db(db)
    return len(db["bookings"]) < before

def add_appointment(title, dt_obj, details=""):
    aid = db.get("next_appointment_id", 1)
    rec = {"id": aid, "title": title, "datetime": dt_obj.isoformat(), "details": details, "created_at": datetime.utcnow().isoformat()}
    db["appointments"].append(rec); db["next_appointment_id"] = aid + 1; save_db(db)
    return rec

def list_appointments():
    return db.get("appointments", [])

def create_ics_for_appointment(app):
    cal = Calendar(); ev = Event()
    ev.name = app["title"]; ev.begin = dtparser.parse(app["datetime"]); ev.duration = timedelta(hours=1)
    ev.description = app.get("details",""); cal.events.add(ev)
    fname = os.path.join(WORK_DIR, f"appointment_{app['id']}_{uuid.uuid4().hex[:6]}.ics")
    with open(fname, "w") as f: f.writelines(cal)
    return fname

def simulated_car_start():
    return "Simulated: engine start command sent. (Demo only)"

def simulated_car_diagnose():
    report = {"engine":"OK","battery_voltage":"12.6V","tire_pressure":"Normal","errors":[]}
    return f"Simulated diagnostic complete. Engine={report['engine']}, Battery={report['battery_voltage']}, Tires={report['tire_pressure']}.", report

def generate_5day_plan(weekly_hours=10):
    topics = [
        "Day 1 - Foundations of AI Agents",
        "Day 2 - Tools & Integrations",
        "Day 3 - Planning & Orchestration",
        "Day 4 - Budgeting & Optimization",
        "Day 5 - Deployment & Observability"
    ]
    base_hours = round(weekly_hours / 5, 2)
    days = []; resources={}; flashcards={}; quizzes={}; info={}
    start = datetime.utcnow().date()
    for i,t in enumerate(topics, start=1):
        date = (start + timedelta(days=i-1)).isoformat()
        tasks = [
            {"type":"reading","title":f"Intro to {t}","duration_hours":round(base_hours*0.25,2)},
            {"type":"hands_on","title":f"Notebook for {t}","duration_hours":round(base_hours*0.5,2)},
            {"type":"practice","title":f"Exercises for {t}","duration_hours":round(base_hours*0.2,2)},
            {"type":"review","title":"Flashcards & review","duration_hours":round(base_hours*0.05,2)}
        ]
        days.append({"day_index":i,"date":date,"topic":t,"hours_allocated":base_hours,"tasks":tasks})
        resources[t] = [{"title":f"{t} — Intro video","url":"https://example.com/vid","summary":"Short intro"}]
        flashcards[t] = [{"q":f"What is key in {t} (card {j})","a":"short answer"} for j in range(1, max(5, int(base_hours*5))+1)]
        quizzes[t] = [{"q":f"MCQ about {t}","options":["A","B","C","D"],"answer":"A"} for _ in range(max(3,int(base_hours*2)))]
        info[t] = {"practical_prompt":f"3-step plan for {t}"}
    plan = {"created_at": datetime.utcnow().isoformat(), "weekly_hours": weekly_hours, "days": days, "resources":resources, "flashcards":flashcards, "quizzes":quizzes, "informative_queries":info}
    pid = f"plan_{len(db.get('plans',[]))+1}"
    db.get("plans", []).append({"id":pid, "plan":plan, "created_at":plan["created_at"]}); save_db(db)
    return plan

# -------------------------
# Datetime parsing (utilities)
# -------------------------
def parse_datetime_from_text(text, ref_dt=None):
    if ref_dt is None: ref_dt = datetime.now()
    t = (text or "").lower()
    if "tomorrow" in t:
        base = (ref_dt + timedelta(days=1)).date()
        m = re.search(r'(\d{1,2}(:\d{2})?\s*(am|pm)?)', t)
        if m:
            try: return dtparser.parse(f"{base} {m.group(1)}", fuzzy=True)
            except: pass
        return datetime.combine(base, datetime.strptime("09:00","%H:%M").time())
    if "today" in t:
        base = ref_dt.date()
        m = re.search(r'(\d{1,2}(:\d{2})?\s*(am|pm)?)', t)
        if m:
            try: return dtparser.parse(f"{base} {m.group(1)}", fuzzy=True)
            except: pass
        return datetime.combine(base, datetime.strptime("09:00","%H:%M").time())
    m_date = re.search(r'(\d{4}-\d{2}-\d{2})', t)
    m_time = re.search(r'(\d{1,2}(:\d{2})?\s*(am|pm)?)', t)
    try:
        if m_date:
            ds = m_date.group(1)
            if m_time: return dtparser.parse(f"{ds} {m_time.group(1)}", fuzzy=True)
            return dtparser.parse(ds)
        if m_time:
            dt = dtparser.parse(m_time.group(1), default=ref_dt)
            if dt < ref_dt: dt = dt + timedelta(days=1)
            return dt
    except:
        pass
    try:
        return dtparser.parse(t, fuzzy=True, default=ref_dt)
    except:
        return None

# -------------------------
# Sequential agent classes
# -------------------------
class InterpreterAgent:
    """Interprets user text into intent + slots (rule-based for now)."""
    @staticmethod
    def interpret(text):
        t = (text or "").lower()
        # reuse small detector
        slots = {}
        if any(k in t for k in ["what's the time","what is the time","time right now","current time","time now"]):
            return {"intent":"get_time","slots":{}}
        if any(k in t for k in ["book a cab","book cab","book taxi","book a taxi","call a cab","book ride","taxi"]):
            dt = parse_datetime_from_text(t)
            slots["datetime"] = dt.isoformat() if dt else None
            m = re.search(r'from\s+(.+?)\s+(to|for)\s+(.+)', t)
            if m: slots["pickup"]=m.group(1).strip(); slots["dropoff"]=m.group(3).strip()
            else:
                m2 = re.search(r'to\s+(.+)', t)
                if m2: slots["dropoff"]=m2.group(1).strip()
            return {"intent":"book_cab","slots":slots}
        if any(k in t for k in ["book appointment","book an appointment","dentist","dental","doctor appointment","schedule appointment"]):
            dt = parse_datetime_from_text(t)
            slots["datetime"] = dt.isoformat() if dt else None
            if "dent" in t or "tooth" in t: slots["type"]="dental"
            m_loc = re.search(r'at\s+(.+)', t)
            if m_loc: slots["location"]=m_loc.group(1).strip()
            return {"intent":"book_appointment","slots":slots}
        if any(k in t for k in ["list bookings","show bookings","my bookings","upcoming bookings"]):
            return {"intent":"list_bookings","slots":{}}
        if any(k in t for k in ["cancel booking","cancel my booking","cancel the booking"]):
            m = re.search(r'\b(\d{1,5})\b', t)
            return {"intent":"cancel_booking","slots":{"id":int(m.group(1))}} if m else {"intent":"cancel_booking","slots":{}}
        if any(k in t for k in ["start the car","start car","ignite car","turn on the car"]):
            return {"intent":"car_start","slots":{}}
        if any(k in t for k in ["diagnose the car","diagnose car","car diagnosis","car health","run diagnostics"]):
            return {"intent":"car_diagnose","slots":{}}
        if any(k in t for k in ["create 5-day study plan","create 5 day study plan","5-day study plan","generate 5-day ai plan","generate 5 day ai plan"]):
            return {"intent":"generate_5day_plan","slots":{}}
        if any(k in t for k in ["create calendar","create ics","export calendar","add to calendar","create .ics"]):
            return {"intent":"create_calendar_file","slots":{}}
        return {"intent":"unknown","slots":{}}

class PlannerAgent:
    """Creates an ordered plan of actions given intent+slots."""
    @staticmethod
    def plan(interp_result):
        intent = interp_result.get("intent")
        slots = interp_result.get("slots", {})
        actions = []
        # Map intents to small action plans (tool names + params)
        if intent == "get_time":
            actions.append({"tool":"tell_time","params":{}})
        elif intent == "book_cab":
            # if missing datetime -> ask follow-up
            if not slots.get("datetime"):
                actions.append({"tool":"ask_for_datetime","params":{"reason":"cab"}})
            else:
                actions.append({"tool":"book_cab","params":slots})
        elif intent == "book_appointment":
            if not slots.get("datetime"):
                actions.append({"tool":"ask_for_datetime","params":{"reason":"appointment"}})
            else:
                actions.append({"tool":"book_appointment","params":slots})
        elif intent == "list_bookings":
            actions.append({"tool":"list_bookings","params":{}})
        elif intent == "cancel_booking":
            if not slots.get("id"):
                actions.append({"tool":"ask_for_booking_id","params":{}})
            else:
                actions.append({"tool":"cancel_booking","params":slots})
        elif intent == "car_start":
            actions.append({"tool":"start_car","params":{}})
        elif intent == "car_diagnose":
            actions.append({"tool":"diagnose_car","params":{}})
        elif intent == "generate_5day_plan":
            actions.append({"tool":"generate_plan","params":{"weekly_hours":10}})
        elif intent == "create_calendar_file":
            actions.append({"tool":"create_calendar","params":{}})
        else:
            actions.append({"tool":"unknown","params":{}})
        return actions

class ActionAgent:
    """Executes the planned actions by calling tools and returns results list."""
    @staticmethod
    def execute(actions, incoming_user_text=None):
        results = []
        for act in actions:
            tool = act["tool"]; params = act.get("params", {})
            if tool == "tell_time":
                now = datetime.now()
                results.append({"tool":"tell_time","result":now.strftime('%Y-%m-%d %I:%M %p')})
            elif tool == "ask_for_datetime":
                reply = "I need the date and time — which day/time do you want?"
                results.append({"tool":"ask_for_datetime","result":reply})
            elif tool == "book_cab":
                dt_raw = params.get("datetime")
                dt_obj = dtparser.parse(dt_raw) if dt_raw else parse_datetime_from_text(incoming_user_text)
                pickup = params.get("pickup","your location"); dropoff = params.get("dropoff","destination")
                rec = add_booking("cab", f"Cab: {pickup} → {dropoff}", dt_obj, f"Pickup:{pickup};Dropoff:{dropoff}")
                results.append({"tool":"book_cab","result":rec})
            elif tool == "book_appointment":
                dt_raw = params.get("datetime")
                dt_obj = dtparser.parse(dt_raw) if dt_raw else parse_datetime_from_text(incoming_user_text)
                typ = params.get("type","appointment"); title = f"{typ.title()} appointment"
                rec = add_appointment(title, dt_obj, params.get("location",""))
                results.append({"tool":"book_appointment","result":rec})
            elif tool == "list_bookings":
                results.append({"tool":"list_bookings","result":list_bookings()})
            elif tool == "ask_for_booking_id":
                results.append({"tool":"ask_for_booking_id","result":"Please tell me the booking id to cancel (e.g., 'cancel booking 3')."})
            elif tool == "cancel_booking":
                ok = cancel_booking(params.get("id"))
                results.append({"tool":"cancel_booking","result":ok})
            elif tool == "start_car":
                results.append({"tool":"start_car","result":simulated_car_start()})
            elif tool == "diagnose_car":
                summary, report = simulated_car_diagnose()
                results.append({"tool":"diagnose_car","result":summary, "report":report})
            elif tool == "generate_plan":
                plan = generate_5day_plan(params.get("weekly_hours",10))
                results.append({"tool":"generate_plan","result":plan})
            elif tool == "create_calendar":
                apps = list_appointments()
                if not apps:
                    results.append({"tool":"create_calendar","result":None})
                else:
                    last = apps[-1]
                    path = create_ics_for_appointment(last)
                    results.append({"tool":"create_calendar","result":path})
            else:
                results.append({"tool":"unknown","result":"I don't know how to do that yet."})
        return results

class ResponderAgent:
    """Formats the result from action agent into a user-facing reply."""
    @staticmethod
    def respond(actions, results):
        # Build friendly text depending on results
        out = []
        for act, res in zip(actions, results):
            tool = act["tool"]
            r = res.get("result")
            if tool == "tell_time":
                out.append(f"The current time is {r}.")
            elif tool == "ask_for_datetime":
                out.append(r)
            elif tool == "book_cab":
                rec = r
                dt_obj = dtparser.parse(rec["datetime"])
                out.append(f"Booked (simulated) cab for {dt_obj.strftime('%Y-%m-%d %I:%M %p')}. Booking id: {rec['id']}.")
            elif tool == "book_appointment":
                rec = r
                dt_obj = dtparser.parse(rec["datetime"])
                out.append(f"Scheduled appointment for {dt_obj.strftime('%Y-%m-%d %I:%M %p')}. Appointment id: {rec['id']}.")
            elif tool == "list_bookings":
                rows = r
                if not rows: out.append("You have no bookings.")
                else:
                    lines = [f"#{b['id']} {b['kind'].title()} — {b['title']} at {dtparser.parse(b['datetime']).strftime('%Y-%m-%d %I:%M %p')}" for b in rows]
                    out.append("Your bookings:\n" + "\n".join(lines))
            elif tool == "ask_for_booking_id":
                out.append(r)
            elif tool == "cancel_booking":
                out.append("Cancelled booking." if r else "No booking found with that id.")
            elif tool == "start_car":
                out.append(r)
            elif tool == "diagnose_car":
                out.append(r)
            elif tool == "generate_plan":
                plan = r
                # short summary
                lines = [f"Created 5-day study plan (weekly_hours={plan['weekly_hours']})."]
                for d in plan["days"]:
                    lines.append(f"Day {d['day_index']}: {d['topic']} — {d['hours_allocated']} hrs")
                out.append("\n".join(lines))
            elif tool == "create_calendar":
                if r:
                    out.append(f"Created calendar file: {os.path.basename(r)} (path: {r}). Download from notebook files.")
                else:
                    out.append("No appointments to export to calendar.")
            else:
                out.append(str(r))
        # Combine replies into a single message
        return "\n\n".join(out)

# -------------------------
# Conversation & follow-up state
# -------------------------
conv_state = {"pending_intent": None, "slots": {}}

# -------------------------
# Main orchestrator function used by UI
# -------------------------
def orchestrate_user_message(user_text, history):
    # Step 1: Interpreter
    interp = InterpreterAgent.interpret(user_text)

    # If a follow-up was pending, try to fill slots
    if conv_state.get("pending_intent"):
        # merge incoming text into pending slots
        slots = conv_state["slots"]
        dt = parse_datetime_from_text(user_text)
        if dt and not slots.get("datetime"):
            slots["datetime"] = dt.isoformat()
        m = re.search(r'from\s+(.+?)\s+(to|for)\s+(.+)', user_text.lower())
        if m:
            slots["pickup"] = m.group(1).strip(); slots["dropoff"] = m.group(3).strip()
        else:
            m2 = re.search(r'to\s+(.+)', user_text.lower())
            if m2: slots["dropoff"] = m2.group(1).strip()
        # refill interp to match pending intent
        interp = {"intent": conv_state["pending_intent"], "slots": slots}
        # clear pending (we'll set later if still needed)
        conv_state["pending_intent"] = None
        conv_state["slots"] = {}

    # Step 2: Planner
    plan = PlannerAgent.plan(interp)

    # If planner asked to request more info (e.g., ask_for_datetime), set pending intent and return question
    needs_followup = any(a["tool"] in ("ask_for_datetime","ask_for_booking_id") for a in plan)
    if needs_followup:
        # store pending intent type so next user reply will be merged
        # decide intent to resume based on reason parameter
        for a in plan:
            if a["tool"] == "ask_for_datetime":
                # set pending to either book_cab or book_appointment depending on reason
                reason = a["params"].get("reason","")
                conv_state["pending_intent"] = "book_cab" if reason=="cab" else "book_appointment"
                conv_state["slots"] = interp.get("slots", {})
                # ask the user directly
                messages = [{"role":"user","content":user_text}, {"role":"assistant","content":"I need the date and time — which day and time do you want?"}]
                return messages, [{"user":user_text,"assistant":"I need the date and time — which day and time do you want?"}]
            if a["tool"] == "ask_for_booking_id":
                conv_state["pending_intent"] = "cancel_booking"; conv_state["slots"] = {}
                messages = [{"role":"user","content":user_text}, {"role":"assistant","content":"Please tell me the booking id to cancel (e.g., 'cancel booking 3')."}]
                return messages, [{"user":user_text,"assistant":"Please tell me the booking id to cancel (e.g., 'cancel booking 3')."}]

    # Step 3: Action execution
    results = ActionAgent.execute(plan, incoming_user_text=user_text)

    # Step 4: Responder formatting
    reply = ResponderAgent.respond(plan, results)

    # also return messages in gradio messages format
    messages = [{"role":"user","content":user_text}, {"role":"assistant","content":reply}]
    human_pair = [{"user":user_text,"assistant":reply}]
    return messages, human_pair

# -------------------------
# Gradio UI (messages format)
# -------------------------
with gr.Blocks() as demo:
    gr.Markdown("# Personal Assistant Agent — Sequential Agents (Interpreter → Planner → Action → Responder)\nTry: 'What's the time', 'Book a cab', 'Book a dental appointment', 'Generate 5-day plan', 'Start the car', 'Diagnose the car'.")
    chatbot = gr.Chatbot(type="messages")
    txt = gr.Textbox(placeholder="Type message (e.g. 'Book a cab for tomorrow 9am')", lines=1)
    state = gr.State([])

    def submit(user_text, history):
        messages, human_pairs = orchestrate_user_message(user_text, history or [])
        return messages, human_pairs

    txt.submit(submit, [txt, state], [chatbot, state])
    txt.submit(lambda: "", None, txt)

# If Kaggle blocks local serving, change share=True
demo.launch(share=False)


