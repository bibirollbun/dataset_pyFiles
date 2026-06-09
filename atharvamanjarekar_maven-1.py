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


!pip install apscheduler ipywidgets gTTS





# ==========================================
# CELL 2 â€” Core Multi-Agent System for Maven
# ==========================================

import re
import json
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

# ------------------------------
# 1. Memory System (Long-term)
# ------------------------------
class MemoryBank:
    def __init__(self):
        self.data = []

    def remember(self, item):
        self.data.append({"memory": item, "time": str(datetime.now())})

    def recall(self):
        return self.data


memory = MemoryBank()

# ------------------------------
# 2. Safety Filter (Blocks adult)
# ------------------------------
class SafetyAgent:
    forbidden_words = ["sex", "nude", "porn", "xxx", "boobs", "dick", "pussy"]

    def check(self, text):
        text = text.lower()
        for w in self.forbidden_words:
            if w in text:
                return False
        return True


safety = SafetyAgent()

# ------------------------------
# 3. Reminder Agent
# ------------------------------
scheduler = BackgroundScheduler()
scheduler.start()

reminders = []

def set_reminder(text, time):
    reminders.append({"task": text, "time": time})
    print(f"â�° Reminder set for: {time} â†’ {text}")

# ------------------------------
# 4. Chat Agent (Maven Brain)
# ------------------------------
class MavenAgent:
    def respond(self, message):
        if not safety.check(message):
            return "âš ï¸� Sorry, I cannot discuss adult or unsafe content."

        if "remember" in message.lower():
            item = message.split("remember", 1)[1].strip()
            memory.remember(item)
            return f"ğŸ§  Got it! I'll remember: '{item}'"

        if "recall" in message.lower():
            return json.dumps(memory.recall(), indent=2)

        if "remind me" in message.lower():
            task = re.sub(r".*remind me to", "", message, flags=re.I).strip()
            set_reminder(task, "soon")
            return f"â�° Okay! I'll remind you to: {task}"

        return f"Maven: I hear you! You said â†’ {message}"


maven = MavenAgent()

print("Maven system initialized successfully!")





# ==========================================
# CELL 3 â€” Chat Interface
# ==========================================

def chat():
    print("Maven is ready! Type 'exit' to stop.\n")

    while True:
        user = input("You: ")

        if user.lower() == "exit":
            break

        reply = maven.respond(user)
        print(reply)

chat()





# =========================
# MAVEN UPGRADE â€” Richer Responses + Chat UI
# Replace previous simple MavenAgent + chat UI with this cell.
# Works offline on Kaggle (no transformers, no APIs).
# =========================

# Required (install cell should have run earlier):
# !pip install --no-deps "apscheduler" "ipywidgets" "gTTS"

import re, json, random, time
from datetime import datetime, timedelta
from IPython.display import display, clear_output
import ipywidgets as widgets

# --- Memory & reminders (lightweight) ---
class MemoryBank:
    def __init__(self, capacity=50):
        self.items = []  # list of (ts, text)
        self.capacity = capacity
    def remember(self, text):
        self.items.append((datetime.now().isoformat(), text))
        if len(self.items) > self.capacity:
            self.items.pop(0)
    def recent(self, n=5):
        return [t for ts,t in self.items[-n:]]
    def all(self):
        return list(self.items)
memory = MemoryBank()

# Simple reminder storage (in-memory)
reminders = []  # list of dicts {"time_iso":..., "text":...}

# --- Safety / moderation ---
BLOCKLIST = {"porn","xxx","sex","nude","incest","bestiality","hentai","rape","kill","suicide"}
def is_safe(text):
    t = text.lower()
    for w in BLOCKLIST:
        if w in t:
            return False, w
    return True, None

# --- Simple sentiment (rule-based) ---
POS = {"good","great","awesome","happy","love","nice","thanks","thank you","yay"}
NEG = {"bad","sad","angry","hate","frustrat","tired","upset","sucks"}
def sentiment(text):
    t = text.lower()
    p = sum(1 for w in POS if w in t)
    n = sum(1 for w in NEG if w in t)
    if p>n: return "positive"
    if n>p: return "negative"
    return "neutral"

# --- Humor & small utilities ---
JOKES = [
    "Why did the computer show up at work late? It had a hard drive. ğŸ˜‚",
    "I would tell a UDP joke, but you might not get it. ğŸ˜…",
    "I'm reading a book on anti-gravity â€” it's impossible to put down!"
]
def tell_joke():
    return random.choice(JOKES)

def time_in_words(dt_iso):
    try:
        dt = datetime.fromisoformat(dt_iso)
        return dt.strftime("%Y-%m-%d %H:%M")
    except:
        return str(dt_iso)

# --- Response templates (varied) ---
def choose_template(kind):
    templates = {
        "greeting": [
            "Hey! Maven here â€” what's up? ğŸ˜„",
            "Yo! Maven at your service â€” how can I help?",
            "Hello! Ready when you are. ğŸ‘‹"
        ],
        "thanks": [
            "Anytime! Glad to help. ğŸ˜Š",
            "You got it â€” happy to help!",
            "No problem â€” that's what I'm here for."
        ],
        "fallback": [
            "Got you â€” can you say a bit more so I can help?",
            "Hmm, I didn't catch that. Want to rephrase?",
            "I want to help â€” can you give me one more detail?"
        ],
        "confirm": [
            "Done âœ…",
            "All set ğŸ‘�",
            "Consider it handled."
        ],
        "info": [
            "Here's what I found:",
            "Try this:",
            "I recommend the following steps:"
        ]
    }
    return random.choice(templates.get(kind, templates["fallback"]))

# --- Naturalizer: craft a friendly reply using context ---
def craft_reply(user_text):
    # Safety check
    ok, reason = is_safe(user_text)
    if not ok:
        return f"ğŸš« Sorry â€” I can't help with that ({reason})."

    ut = user_text.strip()
    low = ut.lower()

    # Greetings
    if low in {"hi","hello","hey","yo","sup","hiya"}:
        return choose_template("greeting")

    # Small talk & mood
    if any(w in low for w in ["how are you","how's it going","how r you","how are u"]):
        return "I'm doing great â€” running smooth and ready to help! How about you?"

    # Jokes
    if "joke" in low or "tell me a joke" in low:
        return tell_joke()

    # Memory commands
    # "remember that I like coffee" or "remember my mom's birthday is 2025-12-01"
    if low.startswith("remember ") or low.startswith("remember that "):
        # extract main clause
        m = re.sub(r"^remember( that)?\s+", "", ut, flags=re.I).strip()
        if m:
            memory.remember(m)
            return f"ğŸ§  Saved to memory: \"{m}\""
        else:
            return "What would you like me to remember? Say: remember that I ..."

    # recall command
    if low.startswith("recall") or "what did i tell you" in low or "what do you remember" in low:
        recent = memory.recent(5)
        if not recent:
            return "I don't have anything in memory yet. Tell me something to remember!"
        bullets = "\n".join([f"- {r}" for r in recent])
        return f"I remember these recent things:\n{bullets}"

    # Reminder scheduling: "remind me to call mom at 2025-12-01 15:00"
    if low.startswith("remind me to ") and " at " in low:
        try:
            parts = re.split(r"\s+at\s+", ut, maxsplit=1, flags=re.I)
            task = re.sub(r"(?i)^remind me to\s+", "", parts[0]).strip()
            when = parts[1].strip()
            # try parse ISO-ish
            try:
                dt = datetime.fromisoformat(when)
            except:
                # fallback: try common format without seconds
                try:
                    dt = datetime.strptime(when, "%Y-%m-%d %H:%M")
                except:
                    # if parse fails, schedule in 1 minute
                    dt = datetime.now() + timedelta(minutes=1)
            reminders.append({"time_iso": dt.isoformat(), "text": task})
            return f"â�° Reminder set for {time_in_words(dt.isoformat())}: {task}"
        except Exception as e:
            return "I couldn't parse that time. Use: Remind me to <task> at YYYY-MM-DD HH:MM"

    # Ask to run small tool: "calc 5+3"
    if low.startswith("calc ") or low.startswith("calculate "):
        expr = re.sub(r"(?i)^(calc|calculate)\s+", "", ut)
        try:
            # safe-ish eval: allow digits and basic ops only
            if re.fullmatch(r"[0-9\.\s\+\-\*\/\(\)]+", expr):
                val = eval(expr)
                return f"ğŸ§® Result: {val}"
            else:
                return "I can calculate basic math like 2+2 or (5*3)."
        except Exception:
            return "Couldn't calculate that expression."

    # Ask to run python: "run: print('hi')" (we won't execute by default - ask confirm)
    if low.startswith("run code:") or low.startswith("run:"):
        return "I can run small Python snippets if you want. Reply 'run it' to confirm execution."

    # If user asks for help with a task (e.g., make meal plan)
    if any(w in low for w in ["how do i", "how to", "help me", "plan", "schedule", "steps to"]):
        # produce short step-by-step answer
        steps = [
            "1) Clarify your goal.",
            "2) Break it into 3 small steps.",
            "3) Try one step today and review tomorrow.",
            "Want me to make a specific plan now?"
        ]
        return choose_template("info") + "\n" + "\n".join(steps)

    # If user asks about memory usage
    if "memory" in low and ("how" in low or "what" in low or "show" in low):
        cnt = len(memory.all())
        return f"I currently remember {cnt} item(s). Use 'recall' to list recent ones."

    # default fallback: use context + friendly follow-up
    # include a short mention of last memory to feel contextual
    last_mem = (memory.recent(1)[0] if memory.recent(1) else None)
    follow = f" By the way, I remember: \"{last_mem}\"." if last_mem else ""
    return f"I hear you! {follow} How would you like me to help â€” a quick answer, steps, or save this to memory?"

# --- Chat UI (phone-friendly ipywidgets) ---
chat_log = []
chat_box = widgets.HTML(layout=widgets.Layout(max_height="60vh", overflow="auto"))
input_box = widgets.Text(placeholder="Say something to Mavenâ€¦")
send_btn = widgets.Button(description="Send", button_style="primary")

def render():
    html = "<div style='font-family: system-ui, Arial; max-width:700px;'>"
    for who,txt in chat_log[-80:]:
        if who=="You":
            html += f"<div style='text-align:right;margin:6px'><span style='background:#0084ff;color:white;padding:8px 12px;border-radius:12px;display:inline-block'>{txt}</span></div>"
        else:
            html += f"<div style='text-align:left;margin:6px'><span style='background:#f1f0f0;padding:8px 12px;border-radius:12px;display:inline-block'>{txt}</span></div>"
    html += "</div>"
    chat_box.value = html

def on_send(b):
    user_text = input_box.value.strip()
    if not user_text:
        return
    chat_log.append(("You", user_text))
    render()
    # compose reply
    reply = craft_reply(user_text)
    # handle special "run it" confirmation for code execution (requires explicit permission)
    if user_text.strip().lower() == "run it" and chat_log:
        # find last bot message that asked for confirmation
        for who, msg in reversed(chat_log):
            if who=="Maven" and "I can run small Python snippets" in msg:
                # find the snippet previously asked by user (naive approach)
                # find the last user message before the bot's ask that started with "run code:" or "run:"
                snippet = None
                for i in range(len(chat_log)-1, -1, -1):
                    if chat_log[i][0]=="You" and chat_log[i][1].lower().startswith(("run code:","run:")):
                        snippet = chat_log[i][1].split(":",1)[1].strip()
                        break
                if snippet:
                    # run safely via subprocess
                    import tempfile, subprocess, os
                    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
                        f.write(snippet)
                        fname = f.name
                    try:
                        proc = subprocess.run(["python", fname], capture_output=True, text=True, timeout=6)
                        output = proc.stdout.strip() + ("\nERR:"+proc.stderr.strip() if proc.stderr.strip() else "")
                        reply = "âœ… Execution result:\n" + output
                    except Exception as e:
                        reply = f"Execution failed: {e}"
                    finally:
                        try: os.unlink(fname)
                        except: pass
                else:
                    reply = "I couldn't find the code snippet to run."
                break

    # If reply sets a reminder, also add to reminders list (the craft_reply handles parsing and appended)
    chat_log.append(("Maven", reply))
    render()
    input_box.value = ""

send_btn.on_click(on_send)

display(chat_box)
display(widgets.HBox([input_box, send_btn]))
render()

print("Maven UI ready â€” chat above. Examples:\n- 'Hi'\n- 'Remember that my mom's birthday is 2025-12-01'\n- 'Remind me to call mom at 2025-12-01 15:00'\n- 'Tell me a joke'\n- 'Calc 5+7'\n")





Daddy: ## ğŸ§‘â€�ğŸ’» How to Use Maven

### ğŸ’¬ Talk Normally
Example:
 Maven replies in natural conversation.

### â�° Add a reminder
 ### ğŸ“‹ See reminders
 ### â�Œ Maven blocks adult content
â†’ â€œSorry, I cannot help with that.â€�

### ğŸ”� Continuous Chat
You can keep running the cell to talk more.










