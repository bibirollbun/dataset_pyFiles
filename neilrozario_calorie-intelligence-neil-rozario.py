# Cell 1 â€” Fully Quiet Install
import subprocess, sys

def run_quiet(package_args):
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", *package_args],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception:
        pass

# Required packages (quiet installation)
run_quiet(["install", "google-adk"])
run_quiet(["install", "google-generativeai", "--upgrade"])
run_quiet(["install", "rich", "matplotlib", "ipywidgets", "nest_asyncio", "pandas", "sqlite-utils"])

import nest_asyncio
nest_asyncio.apply()

print("âœ” Packages installed quietly. Restart kernel if needed, then run Cell 2 â†’ 15.")



# Cell 2 â€” Imports & basic config
import os, re, json, uuid, asyncio, time, random, tempfile, shutil, sqlite3, datetime
from collections import Counter, defaultdict
from threading import Lock
from rich.console import Console
from rich.panel import Panel
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import ipywidgets as widgets
import pandas as pd
import requests

# ADK imports
from kaggle_secrets import UserSecretsClient
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools import FunctionTool

# latest Google AI client import
from google import genai

console = Console()
MODEL = "gemini-2.5-flash"  # change if needed

# path to your uploaded notebook (for submission reference)
UPLOADED_NOTEBOOK_PATH = "/mnt/data/calorie-tracker.ipynb"
console.print(Panel(f"Uploaded notebook path: {UPLOADED_NOTEBOOK_PATH}", style="cyan"))



# Cell 3 â€” SAFE STARTUP: configure Gemini (latest) + ADK session + parser runner
# Run this cell first after kernel restart.

# 1) Load Gemini API key
api_key = None
try:
    api_key = UserSecretsClient().get_secret("GEMINI_API_KEY")
except Exception:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

if not api_key:
    console.print("[red]â�Œ ERROR: GEMINI_API_KEY not found! Add it to Kaggle Secrets or env variables.[/red]")
    raise ValueError("GEMINI_API_KEY missing")

# 2) Create genai client (latest API)
GENAI_CLIENT = genai.Client(api_key=api_key)
console.print("[green]âœ” genai.Client configured (latest API)[/green]")

# 3) ADK session service + parser agent + runner
session_service = InMemorySessionService()

parser_agent = LlmAgent(
    name="parser",
    model=MODEL,
    instruction='Extract only a JSON list of food items from the user message. Example: ["pizza","coke"]'
)

parser_runner = Runner(agent=parser_agent, app_name="calorie_tracker", session_service=session_service)
console.print("[green]âœ” Parser agent & Runner created[/green]")

# 4) Ensure session exists
try:
    asyncio.get_event_loop().run_until_complete(
        session_service.create_session(app_name="calorie_tracker", user_id="user1", session_id="session-main")
    )
    console.print("[green]âœ” ADK session-main created (user1)[/green]")
except Exception as e:
    console.print(f"[yellow]Warning creating session-main (may already exist): {e}[/yellow]")

# small wrapper to run parser via runner (async)
class Part:
    def __init__(self, text): self.text = text; self.function_call = None
class Content:
    def __init__(self, role, parts): self.role = role; self.parts = parts

def extract_output_from_events(events):
    for e in reversed(events):
        if getattr(e, "content", None) and getattr(e.content, "parts", None):
            parts = e.content.parts
            texts = [getattr(p, "text", None) for p in parts]
            txt = "\n".join([t for t in texts if t])
            if txt and txt.strip():
                return txt.strip()
    return ""

async def run_parser_async(message, user_id="user1", session_id="session-main"):
    content = Content(role="user", parts=[Part(message)])
    events = []
    try:
        async for ev in parser_runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
            events.append(ev)
    except Exception as e:
        console.print(f"[OBS ERROR] run_parser_async error: {e}")
    return extract_output_from_events(events)

def run_parser(message):
    return asyncio.get_event_loop().run_until_complete(run_parser_async(message))

console.print(Panel("Safe startup complete â€” parser ready.", style="green"))



# Cell 4 â€” State + cache + helpers
_STATE_FILE = os.path.expanduser("~/.calorie_tracker_state.json")
_CACHE_FILE = os.path.expanduser("~/.calorie_cache.json")
_MEMORY_DB = os.path.expanduser("~/.calorie_memory_bank.sqlite")
_state_lock = Lock()

def _atomic_write(path: str, data: str):
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        shutil.move(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass

def load_state():
    if not os.path.exists(_STATE_FILE):
        return {}
    try:
        with open(_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        console.print(f"[OBS ERROR] load_state failed: {e}")
        return {}

def save_state(state: dict):
    try:
        s = json.dumps(state, ensure_ascii=False, indent=2)
        _atomic_write(_STATE_FILE, s)
    except Exception as e:
        console.print(f"[OBS ERROR] save_state failed: {e}")

def today_iso():
    return datetime.date.today().isoformat()

# bootstrap
st = load_state()
st.setdefault("meals", [])
st.setdefault("daily_totals", {})
st.setdefault("profile", None)
st.setdefault("summaries", {})
save_state(st)

# cache
try:
    if os.path.exists(_CACHE_FILE):
        with open(_CACHE_FILE, 'r', encoding='utf-8') as f:
            calorie_cache = json.load(f)
    else:
        calorie_cache = {}
except Exception:
    calorie_cache = {}



# Cell 5 â€” Memory bank + observability
def init_memory_bank():
    conn = sqlite3.connect(_MEMORY_DB)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS meals (
                    id TEXT PRIMARY KEY,
                    date TEXT,
                    time TEXT,
                    foods TEXT,
                    calories INTEGER,
                    breakdown TEXT
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    ts REAL,
                    agent TEXT,
                    type TEXT,
                    payload TEXT
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS summaries (
                    id TEXT PRIMARY KEY,
                    created TEXT,
                    summary TEXT
                )''')
    conn.commit()
    conn.close()

def memory_insert_meal(entry: dict):
    conn = sqlite3.connect(_MEMORY_DB)
    c = conn.cursor()
    c.execute('INSERT OR REPLACE INTO meals (id, date, time, foods, calories, breakdown) VALUES (?,?,?,?,?,?)', (
        entry['id'], entry['date'], entry['time'], json.dumps(entry['foods']), entry['calories'], json.dumps(entry.get('breakdown', {}))
    ))
    conn.commit()
    conn.close()

def memory_log_event(agent, etype, payload):
    conn = sqlite3.connect(_MEMORY_DB)
    c = conn.cursor()
    eid = uuid.uuid4().hex
    c.execute('INSERT INTO events (id, ts, agent, type, payload) VALUES (?,?,?,?,?)', (
        eid, time.time(), agent, etype, json.dumps(payload)
    ))
    conn.commit()
    conn.close()

def memory_query_meals_by_date(date_iso):
    conn = sqlite3.connect(_MEMORY_DB)
    c = conn.cursor()
    c.execute('SELECT id, date, time, foods, calories FROM meals WHERE date=? ORDER BY time', (date_iso,))
    rows = c.fetchall()
    conn.close()
    out = []
    for r in rows:
        out.append({'id': r[0], 'date': r[1], 'time': r[2], 'foods': json.loads(r[3]), 'calories': r[4]})
    return out

init_memory_bank()
METRICS = defaultdict(int)

def log_event(agent, etype, payload):
    METRICS[f"{agent}.{etype}"] += 1
    memory_log_event(agent, etype, payload)



# Cell 6 â€” parsing & calorie extraction heuristics
def normalize_food_key(food: str) -> str:
    f = food.lower().strip()
    f = re.sub(r"\b(\d+(\.\d+)?\s?(g|gram|grams|ml|oz|l|litre|cup|cups|slice|serving|tbsp|tsp))\b", "", f)
    f = re.sub(r"[^\w\s]", "", f)
    f = re.sub(r"\s+", " ", f).strip()
    return f

def extract_calories_from_text(txt: str):
    if not txt:
        return None
    t = txt.replace(",", "")
    m = re.findall(r"(\d{1,5})\s*(?:kcal|calories|cal)\b", t, flags=re.IGNORECASE)
    if m:
        nums = [int(x) for x in m if 5 <= int(x) <= 10000]
        if nums:
            return Counter(nums).most_common(1)[0][0]
    m2 = re.findall(r"(\d{1,5})\s*[â€“\-]\s*(\d{1,5})\s*(?:kcal|calories|cal)?", t)
    if m2:
        nums = []
        for a,b in m2:
            a,b = int(a), int(b)
            if 5 <= a <= 10000: nums.append(a)
            if 5 <= b <= 10000: nums.append(b)
        if nums:
            return int(sum(nums)/len(nums))
    m3 = re.findall(r"\b(\d{2,5})\b", t)
    m3 = [int(x) for x in m3 if 5 <= int(x) <= 10000]
    if m3:
        return Counter(m3).most_common(1)[0][0]
    return None



# Cell 7 â€” Safe Gemini query function (synchronous with latest genai.Client)
def gemini_query(prompt: str, model: str = MODEL):
    """
    Synchronous, safe wrapper around genai.Client.models.generate_content
    Returns text (string) or None on failure.
    """
    try:
        resp = GENAI_CLIENT.models.generate_content(model=model, contents=prompt)
        # resp may have .text or .candidates
        text = getattr(resp, "text", None)
        if not text:
            candidates = getattr(resp, "candidates", None)
            if candidates and len(candidates) > 0:
                # candidate may have 'content' or 'text'
                text = getattr(candidates[0], "content", None) or getattr(candidates[0], "text", None) or str(candidates[0])
        return text
    except Exception as e:
        console.print(f"[OBS ERROR] gemini_query failed: {e}")
        METRICS['gemini_failures'] += 1
        return None



# Cell 8 â€” Calorie lookup logic (hybrid)
def google_calorie_search(food: str):
    key = normalize_food_key(food)
    if key in calorie_cache and calorie_cache[key] is not None:
        METRICS['cache_hits'] += 1
        return calorie_cache[key]
    prompt = f"How many kilocalories (kcal) are in ONE typical serving of {food}? Answer with a single integer only."
    text = gemini_query(prompt)
    if not text:
        # model fail: leave None to be handled by caller
        calorie_cache[key] = None
        persist_cache()
        return None
    cal = extract_calories_from_text(text)
    calorie_cache[key] = cal
    persist_cache()
    return cal

def google_calorie_search_batch(foods: list):
    # Ask for JSON mapping for robustness
    items_text = "\n".join([f"- {f}" for f in foods])
    prompt = (
        "For the list below, return a JSON object mapping the exact item string to a single integer of kcal per typical serving. "
        "Example: {\"apple\":95}\nList:\n" + items_text
    )
    text = gemini_query(prompt)
    results = {}
    if text:
        jmatch = re.search(r"\{[\s\S]*\}", text)
        if jmatch:
            try:
                parsed = json.loads(jmatch.group(0))
                for k,v in parsed.items():
                    try:
                        results[k] = int(v) if v is not None else None
                    except:
                        results[k] = extract_calories_from_text(str(v))
            except Exception:
                pass
    # fallback to per-item queries if any missing
    for f in foods:
        if f not in results:
            results[f] = google_calorie_search(f)
    # cache normalized keys
    for f,v in list(results.items()):
        k = normalize_food_key(f)
        calorie_cache[k] = v
    persist_cache()
    return results

def get_calories(food: str, prefer_batch=False, batch_list=None):
    """
    Unified accessor. If prefer_batch and batch_list provided (list of foods), will use batch.
    Otherwise single-item path.
    """
    if prefer_batch and batch_list and len(batch_list)>1:
        batch_res = google_calorie_search_batch(batch_list)
        return batch_res.get(food)
    return google_calorie_search(food)

# ensure persist_cache exists
def persist_cache():
    try:
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(calorie_cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        console.print(f"[OBS ERROR] persist_cache failed: {e}")



# Cell 9 â€” Orchestrator (auto-process on add)
def orchestrate_meal_entry(user_text: str, use_batch=True):
    log_event("orchestrator", "start", {"input": user_text})
    # 1) parser via ADK
    parser_out = run_parser(f"Extract foods as JSON list from: {user_text}")
    parser_out = (parser_out or "").replace("```json","").replace("```","").strip()
    try:
        foods = json.loads(parser_out)
        if not isinstance(foods, list): raise ValueError()
    except Exception:
        foods = [x.strip() for x in re.split(r",| and ", user_text) if x.strip()]
        log_event("orchestrator", "parser_fallback", {"input": user_text, "foods": foods})
    a2a_msg = {"from":"parser","to":"orchestrator","foods":foods, "ts": datetime.datetime.utcnow().isoformat()+"Z"}
    log_event("a2a", "parser->orchestrator", a2a_msg)

    # 2) estimator (hybrid)
    if use_batch and len(foods)>1:
        est = google_calorie_search_batch(foods)
    else:
        # parallel estimation (threads)
        tasks = [asyncio.to_thread(get_calories, f, False, None) for f in foods]
        results = asyncio.get_event_loop().run_until_complete(asyncio.gather(*tasks))
        est = {foods[i]: results[i] for i in range(len(foods))}

    # fill any None with secondary prompt fallback or default 150
    breakdown = {}
    for k,v in est.items():
        if v is None:
            prompt = f"Give a reasonable integer kcal estimate for one serving of {k} (single integer only)."
            t = gemini_query(prompt)
            guessed = extract_calories_from_text(t) if t else None
            breakdown[k] = int(guessed) if guessed is not None else 150
        else:
            breakdown[k] = int(v)

    total = sum(breakdown.values())

    # 3) save entry + update state + memory
    entry = {
        "id": uuid.uuid4().hex,
        "time": datetime.datetime.now().isoformat(),
        "foods": foods,
        "calories": int(total),
        "breakdown": breakdown,
        "date": today_iso()
    }
    st = load_state()
    st.setdefault("meals", []).append(entry)
    st.setdefault("daily_totals", {})
    st["daily_totals"][today_iso()] = st["daily_totals"].get(today_iso(), 0) + entry["calories"]
    save_state(st)
    memory_insert_meal(entry)
    log_event("orchestrator", "meal_saved", entry)

    # 4) advisor logic
    prof = st.get("profile")
    remaining = None
    exceeded = False
    if prof and prof.get("target_kcal"):
        remaining = prof["target_kcal"] - st["daily_totals"].get(today_iso(), 0)
        exceeded = remaining < 0

    result = {"entry": entry, "today_total": st["daily_totals"].get(today_iso(), 0), "remaining": remaining, "exceeded": exceeded, "profile": prof}
    log_event("orchestrator", "result", result)
    return result



# Cell 10 â€” Compact older meals into a short summary stored in summaries table
def compact_context_and_summarize(window_days=30):
    cutoff = (datetime.date.today() - datetime.timedelta(days=window_days)).isoformat()
    conn = sqlite3.connect(_MEMORY_DB)
    c = conn.cursor()
    c.execute('SELECT id, date, time, foods, calories FROM meals WHERE date <= ?', (cutoff,))
    rows = c.fetchall()
    if not rows:
        conn.close()
        return None
    texts = []
    for r in rows:
        texts.append(f"{r[1]} {r[2]}: {', '.join(json.loads(r[3]))} ({r[4]} kcal)")
    prompt = "Summarize the following meal history into one short paragraph focusing on patterns (frequent foods, average calories, snacks, time-of-day patterns):\n\n" + "\n".join(texts)
    summary_text = gemini_query(prompt) or "Summary not available"
    sid = uuid.uuid4().hex
    c.execute('INSERT OR REPLACE INTO summaries (id, created, summary) VALUES (?,?,?)', (sid, datetime.datetime.utcnow().isoformat()+"Z", summary_text))
    conn.commit()
    # OPTIONAL: delete old meal rows to reduce size (we keep them by default)
    conn.close()
    log_event("compactor", "summary_saved", {"summary_id": sid})
    # store in state for quick access
    st = load_state()
    st.setdefault("summaries", {})[sid] = {"created": datetime.datetime.utcnow().isoformat()+"Z", "summary": summary_text}
    save_state(st)
    return {"id": sid, "summary": summary_text}



# Cell 11 â€” small ground-truth and evaluation metrics
GROUND_TRUTH = {
    "apple": 95,
    "banana": 105,
    "large egg": 78,
    "slice of bread": 80,
    "rice (1 cup cooked)": 205,
    "pizza (slice)": 285,
    "broccoli (100g)": 34
}

def evaluate_estimator():
    foods = list(GROUND_TRUTH.keys())
    preds = []
    for f in foods:
        v = get_calories(f, prefer_batch=False)
        if v is None:
            v = google_calorie_search(f)  # fallback
        preds.append(v if v is not None else 150)
    gts = [GROUND_TRUTH[f] for f in foods]
    errors = [abs(preds[i]-gts[i]) for i in range(len(foods))]
    mae = sum(errors)/len(errors)
    mse = sum((preds[i]-gts[i])**2 for i in range(len(foods)))/len(foods)
    within_10pct = sum(1 for i in range(len(foods)) if abs(preds[i]-gts[i]) <= 0.1*gts[i]) / len(foods)
    report = {"foods": foods, "preds": preds, "gts": gts, "mae": mae, "mse": mse, "within_10pct": within_10pct}
    log_event("evaluation", "run", report)
    return report



# Cell 12 â€” GitHub JSON bridge helpers
# Reads public raw input JSON (frontend writes) and writes output.json to repo using PAT (if provided) otherwise writes local file.

def read_input_json_from_raw_url(url: str):
    """Read JSON from a public raw URL (e.g., https://raw.githubusercontent.com/user/repo/main/input.json)."""
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
        console.print(f"[OBS WARN] read_input_json_from_raw_url status {r.status_code}")
    except Exception as e:
        console.print(f"[OBS ERROR] read_input_json_from_raw_url: {e}")
    return None

def write_output_json_to_repo(owner: str, repo: str, path: str, data: dict, token: str = None):
    """
    If token provided, this will create/update the file in the repo via GitHub API.
    If token is None, writes a local file named output.json for manual push.
    """
    content = json.dumps(data, ensure_ascii=False, indent=2)
    if not token:
        # local fallback
        local_path = os.path.join(os.getcwd(), path)
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(content)
        console.print(f"[OBS] Wrote {path} locally. Push to GitHub manually if desired.")
        return {"status": "local_written", "path": local_path}
    # create/update via GitHub REST API (simple approach)
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    # get existing file sha (if exists)
    resp = requests.get(api_url, headers=headers)
    if resp.status_code == 200:
        sha = resp.json().get("sha")
    else:
        sha = None
    payload = {"message": "Update output.json from Kaggle notebook", "content": content.encode("utf-8").decode("utf-8")}
    # GitHub expects base64 content; do that:
    import base64
    payload = {"message": "Update output.json from Kaggle notebook", "content": base64.b64encode(content.encode("utf-8")).decode("utf-8")}
    if sha:
        payload["sha"] = sha
    r2 = requests.put(api_url, headers=headers, json=payload)
    if r2.status_code in (200, 201):
        console.print(f"[OBS] Wrote {path} to GitHub repo {owner}/{repo}.")
        return {"status": "written", "url": r2.json().get("content", {}).get("html_url")}
    else:
        console.print(f"[OBS ERROR] GitHub write failed: {r2.status_code} {r2.text}")
        return {"status": "error", "detail": r2.text}



# Cell 13 â€” Interactive UI (auto-process + graphs)

# -----------------------------
# Graph helper
# -----------------------------
def draw_progress_and_donut(target_kcal, current_kcal):
    remaining = target_kcal - current_kcal
    exceeded = remaining < 0

    fig, ax = plt.subplots(1, 2, figsize=(9, 3))

    # --- Progress Bar ---
    ax0 = ax[0]
    ax0.barh([0], [current_kcal], height=0.6)
    ax0.barh([0], [max(0, remaining)], left=[current_kcal], height=0.6, alpha=0.3)
    ax0.set_xlim(0, max(target_kcal, current_kcal) * 1.1)
    ax0.set_yticks([])
    ax0.set_title(
        f"Today: {current_kcal} / {target_kcal} kcal\n"
        f"{'Exceeded by ' + str(abs(remaining)) if exceeded else 'Remaining: ' + str(remaining)}"
    )

    # --- Donut Chart ---
    ax1 = ax[1]
    if exceeded:
        sizes = [target_kcal, abs(remaining)]
        colors = ["orange", "red"]
        ax1.pie(sizes, colors=colors, wedgeprops=dict(width=0.4))
        ax1.set_title("Exceeded!")
    else:
        sizes = [current_kcal, remaining]
        ax1.pie(sizes, wedgeprops=dict(width=0.4))
        ax1.set_title("Daily Progress")

    plt.tight_layout()
    plt.show()


# -----------------------------
# UI Widgets
# -----------------------------
age_input = widgets.BoundedIntText(value=30, min=10, max=100, description="Age")
sex_input = widgets.Dropdown(options=["male","female"], value="male", description="Sex")
height_input = widgets.BoundedFloatText(value=170.0, min=50, max=250, description="Height (cm)")
weight_input = widgets.BoundedFloatText(value=70.0, min=20, max=300, description="Weight (kg)")
activity_input = widgets.Dropdown(options=["sedentary","light","moderate","active","very active"], value="sedentary", description="Activity")
goal_input = widgets.Dropdown(options=["maintain","lose","gain"], value="maintain", description="Goal")

profile_save_btn = widgets.Button(description="Save Profile", button_style="success")
profile_delete_btn = widgets.Button(description="Delete Profile", button_style="danger")

profile_output = widgets.Output()
meals_output = widgets.Output()


# -----------------------------
# Profile save/delete functions
# -----------------------------
def compute_bmr_tdee(age:int, sex:str, height_cm:float, weight_kg:float, activity:str):
    s = 5 if sex.lower().startswith("m") else -161
    bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + s
    act_map = {"sedentary": 1.2, "light":1.375, "moderate":1.55, "active":1.725, "very active":1.9}
    tdee = bmr * act_map.get(activity, 1.2)
    return int(round(bmr)), int(round(tdee))

def macro_split(tdee:int, goal:str):
    if goal=="maintain": target = tdee
    elif goal=="lose": target = int(tdee - 300)
    else: target = int(tdee + 300)
    protein_kcal = int(0.25 * target)
    fat_kcal = int(0.25 * target)
    carb_kcal = target - protein_kcal - fat_kcal
    return {
        "target_kcal": target,
        "protein_g": protein_kcal//4,
        "carb_g": carb_kcal//4,
        "fat_g": fat_kcal//9
    }

def save_profile(b=None):
    st = load_state()
    age = int(age_input.value)
    sex = sex_input.value
    height_cm = float(height_input.value)
    weight_kg = float(weight_input.value)
    activity = activity_input.value
    goal = goal_input.value
    bmr, tdee = compute_bmr_tdee(age, sex, height_cm, weight_kg, activity)
    macros = macro_split(tdee, goal)
    prof = {
        "age": age, "sex": sex, "height_cm": height_cm, "weight_kg": weight_kg,
        "activity": activity, "goal": goal,
        "bmr": bmr, "tdee": tdee, "target_kcal": macros["target_kcal"],
        "protein_g": macros["protein_g"], "carb_g": macros["carb_g"], "fat_g": macros["fat_g"]
    }
    st["profile"] = prof
    save_state(st)
    with profile_output:
        clear_output()
        console.print(Panel("Profile Saved âœ”", style="green"))
        console.print(f"Target kcal: {prof['target_kcal']} â€¢ TDEE: {prof['tdee']}")
    log_event("profile", "saved", prof)

def delete_profile(b=None):
    st = load_state()
    st["profile"] = None
    save_state(st)
    with profile_output:
        clear_output()
        console.print(Panel("Profile deleted.", style="red"))
    log_event("profile", "deleted", {})

profile_save_btn.on_click(save_profile)
profile_delete_btn.on_click(delete_profile)


# -----------------------------
# Meal UI
# -----------------------------
meal_input = widgets.Text(value="", placeholder="Type meal like 'pizza, coke, fries'", description="Meal")
add_meal_btn = widgets.Button(description="Add Meal", button_style="success")
show_meals_btn = widgets.Button(description="Show Meals", button_style="info")
delete_meal_input = widgets.Text(value="", placeholder="Enter meal id to delete", description="Delete ID")
delete_meal_btn = widgets.Button(description="Delete Meal", button_style="danger")

def display_meals_table():
    st = load_state()
    meals = [m for m in st.get("meals", []) if m.get("date") == today_iso()]
    with meals_output:
        clear_output()
        if not meals:
            console.print(Panel("No meals logged today", title="Today's meals"))
            return
        lines = [
            f"{'id':8}  {'time':19}  {'foods':30}  {'kcal':5}",
            "-"*70
        ]
        for m in meals[-50:]:
            lines.append(f"{m['id'][:8]:8}  {m['time']:19}  {', '.join(m['foods'])[:30]:30}  {m['calories']:5}")
        console.print(Panel("\n".join(lines), title="Today's meals"))


# -----------------------------
# Add Meal (Graph enabled)
# -----------------------------
def on_add_meal(b=None):
    txt = meal_input.value.strip()
    if not txt:
        with meals_output:
            console.print("[red]Please type a meal[/red]")
        return

    with meals_output:
        clear_output()
        console.print(Panel(f"Processing meal: {txt} ...", style="cyan"))

    # Auto-process via orchestrator
    try:
        res = orchestrate_meal_entry(txt, use_batch=True)

        st = load_state()
        prof = st.get("profile")
        today_total = st["daily_totals"].get(today_iso(), 0)

        with meals_output:
            clear_output()

            # Meals table
            display_meals_table()

            # Show graphs if profile is set
            if prof and prof.get("target_kcal"):
                draw_progress_and_donut(prof["target_kcal"], today_total)

            # Success message
            console.print(Panel(
                f"Added meal: {res['entry']['id'][:8]} â€¢ {res['entry']['calories']} kcal",
                style="green"
            ))

    except Exception as e:
        with meals_output:
            clear_output()
            console.print(f"[red]Error processing meal: {e}[/red]")

    log_event("ui","add_meal", {"text": txt})


add_meal_btn.on_click(on_add_meal)
show_meals_btn.on_click(lambda b: display_meals_table())


# -----------------------------
# Delete meal
# -----------------------------
def on_delete_meal(b=None):
    mid = delete_meal_input.value.strip()
    if not mid:
        with meals_output:
            console.print("[red]Enter meal id to delete[/red]")
        return
    st = load_state()
    before = len(st.get("meals", []))
    st["meals"] = [m for m in st["meals"] if not m["id"].startswith(mid)]

    # Recompute totals
    daily = {}
    for m in st["meals"]:
        daily[m["date"]] = daily.get(m["date"], 0) + m["calories"]
    st["daily_totals"] = daily
    save_state(st)

    with meals_output:
        clear_output()
        console.print(f"[green]Deleted meals starting with {mid}. Before: {before}, Now: {len(st['meals'])}[/green]")
        display_meals_table()

delete_meal_btn.on_click(on_delete_meal)


# -----------------------------
# Layout
# -----------------------------
controls = widgets.VBox([
    widgets.HBox([age_input, sex_input, height_input, weight_input]),
    widgets.HBox([activity_input, goal_input, profile_save_btn, profile_delete_btn]),
    widgets.HBox([meal_input, add_meal_btn, show_meals_btn]),
    widgets.HBox([delete_meal_input, delete_meal_btn]),
    profile_output,
    meals_output
])
display(controls)

console.print(
    Panel(
        "Interactive tracker ready â€” add meals, visualize progress, delete entries.\nGraphs update automatically.",
        title="Interactive Tracker",
        style="bold cyan"
    )
)


