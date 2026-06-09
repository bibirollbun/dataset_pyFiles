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
import time
import traceback
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import json
import pandas as pd
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets

# Attempt to import gemini/genai if available
try:
    import genai
except Exception:
    genai = None  # fallback to mock mode



# === Logging System ===

LOG_FILE = "project_logs.log"

log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
date_format = "%Y-%m-%d %H:%M:%S"

handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
handler.setFormatter(logging.Formatter(log_format, date_format))

logger = logging.getLogger("StudyVerseLogger")
logger.setLevel(logging.INFO)

# Avoid duplicate handlers
if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
    logger.addHandler(handler)

logger.info("Logging initialized.")

# Local execution log (simple list)
execution_log = []

def read_logs(severity=None, tail=5000):
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()[-tail:]
    except FileNotFoundError:
        return pd.DataFrame()

    data = []
    for line in lines:
        parts = line.strip().split(" | ")
        if len(parts) >= 4:
            timestamp, level, name, message = parts[0], parts[1], parts[2], " | ".join(parts[3:])
            if severity is None or severity == level:
                data.append({"timestamp": timestamp, "level": level, "logger": name, "message": message})

    return pd.DataFrame(data)



# === Timing Decorator ===
def timed(fn):
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            res = fn(*args, **kwargs)
            logger.info(f"{fn.__name__} finished in {round(time.time()-start,4)}s.")
            return res
        except Exception as e:
            logger.error(f"{fn.__name__} crashed: {e}\n{traceback.format_exc()}")
            return None
    return wrapper

# === Safe Execute Wrapping Utility ===
def safe_execute(agent_name, func, *args, **kwargs):
    logger.info(f"{agent_name} started.")
    start = time.time()
    try:
        result = func(*args, **kwargs)
        logger.info(f"{agent_name} completed in {round(time.time()-start,4)}s.")
        execution_log.append({
            "agent": agent_name,
            "time": round(time.time()-start,4),
            "status": "success"
        })
        return result
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"{agent_name} FAILED: {e}\n{error_trace}")
        execution_log.append({
            "agent": agent_name,
            "time": round(time.time()-start,4),
            "status": "error",
            "error": str(e)
        })
        return None



class NotebookMemory:
    def __init__(self, file="memory.json", max_items=200):
        self.file = file
        self.max_items = max_items
        self._mem = []
        self.load()

    def remember(self, key, value, tags=None):
        item = {
            "timestamp": datetime.now().isoformat(),
            "key": key,
            "value": value,
            "tags": tags or []
        }
        self._mem.append(item)
        self._mem = self._mem[-self.max_items:]
        self.save()

    def recall(self, query=None, tag=None, limit=10):
        items = list(reversed(self._mem))
        if tag:
            items = [i for i in items if tag in i["tags"]]
        if query:
            items = [i for i in items if query.lower() in json.dumps(i).lower()]
        return items[:limit]

    def summarize(self):
        tags = {}
        for m in self._mem:
            for t in m["tags"]:
                tags[t] = tags.get(t, 0) + 1
        return {"total": len(self._mem), "tags": tags}

    def save(self):
        try:
            with open(self.file, "w") as f:
                json.dump(self._mem, f, indent=2)
        except:
            pass

    def load(self):
        if os.path.exists(self.file):
            try:
                self._mem = json.load(open(self.file))
            except:
                self._mem = []

memory = NotebookMemory()
logger.info("Memory module initialized.")



agent_metrics = {
    "summarizer_runs": 0,
    "task_extractor_runs": 0,
    "study_plan_runs": 0,
    "fandom_style_runs": 0
}

def llm_generate(prompt):
    if genai is None:
        logger.warning("LLM not available â€” using mock output.")
        return "[MOCK OUTPUT]\n" + prompt[:300]
    try:
        model = genai.GenerativeModel("models/gemini-2.0-flash")
        res = model.generate_content(prompt)
        return res.text
    except Exception as e:
        logger.error(f"LLM Error: {e}")
        return f"[LLM ERROR] {e}"



@timed
def summarizer_agent(text):
    agent_metrics["summarizer_runs"] += 1
    def task():
        prompt = f"""
You are the Summarizer Agent.

Summarize the following content into:
- A concise summary
- Bullet points
- 3 possible exam Qs

Text:
{text}
"""
        result = llm_generate(prompt)
        memory.remember("summary", result, tags=["summary"])
        return result

    return safe_execute("summarizer_agent", task)



@timed
def task_extractor_agent(text):
    agent_metrics["task_extractor_runs"] += 1
    def task():
        prompt = f"""
Extract actionable study tasks from the text.
Add priority labels: High, Medium, Low.

Text:
{text}
"""
        result = llm_generate(prompt)
        memory.remember("tasks", result, tags=["tasks"])
        return result
    return safe_execute("task_extractor_agent", task)



@timed
def study_planner_agent(tasks):
    agent_metrics["study_plan_runs"] += 1
    def task():
        prompt = f"""
Create a structured study plan based on:

{tasks}

Include:
- Timeline
- Priority ordering
- Daily schedule sample
"""
        result = llm_generate(prompt)
        memory.remember("plan", result, tags=["plan"])
        return result
    return safe_execute("study_planner_agent", task)



@timed
def fandom_stylist_agent(plan, mode="default"):
    agent_metrics["fandom_style_runs"] += 1
    def task():
        style = style_prompts.get(mode, style_prompts["default"])
        prompt = f"""
Transform the study plan using this fandom style:

FANDOM MODE: {mode}
STYLE GUIDE:
{style}

Plan:
{plan}
"""
        result = llm_generate(prompt)
        memory.remember("styled", result, tags=["styled", mode])
        return result
    return safe_execute("fandom_stylist_agent", task)



# === style_prompts: FULL Fandom Dictionary ===

style_prompts = {
    "default": "Rewrite the study plan in a clear, friendly, motivational tone.",

    # Major fandoms
    "barbie": "Rewrite the study plan in sparkly, bubbly, pink Dreamhouse energy with empowering, positive glam.",
    "bridgerton": "Rewrite in elegant Regency-era charm with manners, romance, and refined style.",
    "disney_princess": "Rewrite with fairytale magic, optimism, sparkles, whimsical charm.",

    # Bollywood sass & vibes
    "poo": "Rewrite in extremely sassy, queen B energy like Poo from Kabhi Khushi Kabhie Gham.",
    "geet": "Rewrite with bubbly, dramatic, over-sharing energy like Geet from Jab We Met.",
    "rahul": "Rewrite with charming, emotional Bollywood hero vibes like Rahul from K3G.",
    "dsp": "Rewrite with intense, dramatic, no-nonsense cop energy like Singham / Dabangg style.",
    "bunny": "Rewrite with dreamy, travel-loving, motivational vibes like Bunny from YJHD.",
    "don": "Rewrite with stylish, dangerous confidence and boss-level attitude like Don.",
    "munna": "Rewrite with lovable goon energy and emotional wisdom like Munna Bhai M.B.B.S.",
    "rancho": "Rewrite with intelligent, rebellious, life-advice energy like Rancho from 3 Idiots.",

    # Hollywood / Web series
    "modern_family": "Rewrite with warm, chaotic family humor, love, and witty sarcasm like Modern Family.",
    "brooklyn_99": "Rewrite with fast-paced comedy, team spirit, and detective-style motivation like Brooklyn Nine-Nine.",
    "emily_in_paris": "Rewrite with fashionable, dreamy, upbeat Parisian influencer energy.",
    "gossip_girl": "Rewrite with dramatic Upper East Side glam, secrets, luxury, and attitude.",

    # Extra fun tones
    "harry_potter": "Rewrite with magical school vibes, friendship, and destiny energy.",
    "friends": "Rewrite with cozy friendship, humor, and coffee-shop comfort vibes.",
    "office": "Rewrite in mockumentary-style awkward humor like The Office.",
    "stranger_things": "Rewrite with 80s adventure, courage, and mystery energy.",
    "money_heist": "Rewrite with strategic, intense, mastermind energy like a heist mission.",
    "squid_game": "Rewrite with high-stakes survival and urgency tone (no violence).",
    "kdrama": "Rewrite in emotional, soft-romantic Korean drama style.",
    "anime_main_character": "Rewrite like an overpowered anime protagonist training for the final arc.",

    # Mood-based styles (bonus)
    "soft_aesthetic": "Rewrite in calm, cozy, soft-girl aesthetic with gentle motivation.",
    "dark_academia": "Rewrite with intellectual, mysterious, candle-lit library vibes.",
    "that_girl": "Rewrite in clean-girl, productive, motivational influencer tone.",
    "villain_arc": "Rewrite in villain-origin-story mode with hunger for success and dominance."
}



# === ui_themes: FULL Fandom UI Themes ===

ui_themes = {
    "default": {
        "primary": "#222",
        "background": "#fafafa",
        "accent": "#4a90e2",
        "font": "Inter, sans-serif",
        "effect": "none"
    },

    # Hollywood / Web shows
    "modern_family": {
        "primary": "#2b4f60",
        "background": "#f2f7f5",
        "accent": "#f4b41a",
        "font": "'Quicksand', sans-serif",
        "effect": "warm_glow"
    },

    "brooklyn_99": {
        "primary": "#002244",
        "background": "#f1f5f9",
        "accent": "#ffc72c",
        "font": "'Montserrat', sans-serif",
        "effect": "badge_flash"
    },

    "emily_in_paris": {
        "primary": "#a4133c",
        "background": "#fff1f4",
        "accent": "#fcbf49",
        "font": "'Playfair Display', serif",
        "effect": "soft_blur"
    },

    "gossip_girl": {
        "primary": "#1c1c1c",
        "background": "#f6f1ec",
        "accent": "#bfa76f",
        "font": "'Didot', serif",
        "effect": "gold_shimmer"
    },

    "stranger_things": {
        "primary": "#d10f0f",
        "background": "#0c0c0c",
        "accent": "#ffb3b3",
        "font": "'Special Elite', monospace",
        "effect": "vhs"
    },

    "friends": {
        "primary": "#6f4e37",
        "background": "#fdf3e7",
        "accent": "#e85d04",
        "font": "'Nunito', sans-serif",
        "effect": "soft_pop"
    },

    "office": {
        "primary": "#3a3a3a",
        "background": "#ffffff",
        "accent": "#0077b6",
        "font": "'Roboto', sans-serif",
        "effect": "paper_slide"
    },

    "money_heist": {
        "primary": "#b11226",
        "background": "#111111",
        "accent": "#e63946",
        "font": "'Oswald', sans-serif",
        "effect": "glitch"
    },

    "squid_game": {
        "primary": "#ff2f92",
        "background": "#0d1b2a",
        "accent": "#21e6c1",
        "font": "'Rubik', sans-serif",
        "effect": "neon_pulse"
    },

    "harry_potter": {
        "primary": "#5d2e8c",
        "background": "#f4eee4",
        "accent": "#cfae70",
        "font": "'Cinzel', serif",
        "effect": "magic_dust"
    },

    "kdrama": {
        "primary": "#ff88a5",
        "background": "#fff7fa",
        "accent": "#ffd6e0",
        "font": "'Noto Serif KR', serif",
        "effect": "petal_fade"
    },

    # Barbie & Aesthetic
    "barbie": {
        "primary": "#ff4dc4",
        "background": "#ffe0f5",
        "accent": "#ff9adf",
        "font": "'Poppins', sans-serif",
        "effect": "sparkle"
    },

    "soft_aesthetic": {
        "primary": "#9c89b8",
        "background": "#f9f4ff",
        "accent": "#f0a6ca",
        "font": "'DM Sans', sans-serif",
        "effect": "blur"
    },

    "dark_academia": {
        "primary": "#2b1d0f",
        "background": "#e6dccf",
        "accent": "#7a5c3d",
        "font": "'Libre Baskerville', serif",
        "effect": "grain"
    },

    "that_girl": {
        "primary": "#4e5d42",
        "background": "#f0f5ed",
        "accent": "#cdb4db",
        "font": "'Inter', sans-serif",
        "effect": "glow"
    },

    "villain_arc": {
        "primary": "#7a0000",
        "background": "#0f0f0f",
        "accent": "#c1121f",
        "font": "'Bebas Neue', sans-serif",
        "effect": "ember"
    },

    "anime_main_character": {
        "primary": "#ff9ecb",
        "background": "#e7f0ff",
        "accent": "#ffe1ef",
        "font": "'Comic Neue', cursive",
        "effect": "shimmer"
    },

    # Bollywood themes
    "poo": {
        "primary": "#ff69b4",
        "background": "#fff0f7",
        "accent": "#ffd700",
        "font": "'Pacifico', cursive",
        "effect": "glitter"
    },

    "geet": {
        "primary": "#ff6f91",
        "background": "#fff5eb",
        "accent": "#ffc75f",
        "font": "'Baloo 2', cursive",
        "effect": "bounce"
    },

    "rahul": {
        "primary": "#8d5c2f",
        "background": "#fff1dc",
        "accent": "#c68642",
        "font": "'Playfair Display', serif",
        "effect": "romance_glow"
    },

    "dsp": {
        "primary": "#1f3c88",
        "background": "#eaeef7",
        "accent": "#ffcc00",
        "font": "'Teko', sans-serif",
        "effect": "impact"
    },

    "bunny": {
        "primary": "#0081a7",
        "background": "#f1faee",
        "accent": "#fcbf49",
        "font": "'Raleway', sans-serif",
        "effect": "drift"
    },

    "don": {
        "primary": "#111827",
        "background": "#030712",
        "accent": "#e11d48",
        "font": "'Cormorant Garamond', serif",
        "effect": "smoke"
    },

    "munna": {
        "primary": "#4a7c59",
        "background": "#e8f3ee",
        "accent": "#f4a261",
        "font": "'Nunito', sans-serif",
        "effect": "warm_heartbeat"
    },

    "rancho": {
        "primary": "#005f73",
        "background": "#edf6f9",
        "accent": "#94d2bd",
        "font": "'Source Sans 3', sans-serif",
        "effect": "lightburst"
    },

    # Extra vibes
    "bridgerton": {
        "primary": "#6a4080",
        "background": "#f7effa",
        "accent": "#e2cfea",
        "font": "'Cormorant Infant', serif",
        "effect": "silk"
    },

    "disney_princess": {
        "primary": "#88c1d8",
        "background": "#fdf9ff",
        "accent": "#ffd6ef",
        "font": "'Dancing Script', cursive",
        "effect": "fairy_dust"
    }
}



THEME_CSS = """
<style>
:root {
    --primary: #111;
    --background: #fafafa;
    --radius: 12px;
}
.theme-card {
    padding: 14px;
    border-radius: var(--radius);
    background: var(--background);
    color: var(--primary);
    margin: 8px 0;
    font-size: 0.95rem;
}
</style>
"""
display(HTML(THEME_CSS))

def render_card(title, body, theme_key="default"):
    theme = ui_themes.get(theme_key, ui_themes["default"])
    html = f"""
    <div class='theme-card' style="background:{theme['background']}; color:{theme['primary']}; font-family:{theme['font']}">
        <h4>{title}</h4>
        <div>{body}</div>
    </div>
    """
    display(HTML(html))



import random

def random_fandom():
    return random.choice(list(style_prompts.keys()))

def random_fandom_run(plan):
    mode = random_fandom()
    styled = fandom_stylist_agent(plan, mode)
    render_card(f"Random Fandom: {mode}", styled[:800].replace("\n","<br/>"), theme_key=mode)
    return styled

print("Random fandom mode ready! Use: random_fandom_run(plan)")



fandom_list = sorted(ui_themes.keys())
index = 0

next_btn = widgets.Button(description="Next â†’")
prev_btn = widgets.Button(description="â†� Previous")
carousel_output = widgets.Output()

def show_fandom(i):
    mode = fandom_list[i]
    theme = ui_themes.get(mode)
    with carousel_output:
        clear_output()
        render_card(f"Fandom Preview: {mode}", f"Primary: {theme['primary']}<br>Background: {theme['background']}", theme_key=mode)

show_fandom(index)

def next_click(_):
    global index
    index = (index + 1) % len(fandom_list)
    show_fandom(index)

def prev_click(_):
    global index
    index = (index - 1) % len(fandom_list)
    show_fandom(index)

next_btn.on_click(next_click)
prev_btn.on_click(prev_click)

display(widgets.HBox([prev_btn, next_btn]))
display(carousel_output)



severity = widgets.Dropdown(options=["ALL","INFO","WARNING","ERROR","CRITICAL"], description="Severity")
show_btn = widgets.Button(description="Show Logs", button_style="info")
metrics_btn = widgets.Button(description="Show Metrics", button_style="success")
out = widgets.Output()

def show_logs(_):
    with out:
        clear_output()
        sev = None if severity.value=="ALL" else severity.value
        df = read_logs(sev, tail=200)
        display(df.tail(20))

def show_metrics(_):
    with out:
        clear_output()
        display(pd.DataFrame([
            ["Summaries", agent_metrics["summarizer_runs"]],
            ["Tasks", agent_metrics["task_extractor_runs"]],
            ["Plans", agent_metrics["study_plan_runs"]],
            ["Styled", agent_metrics["fandom_style_runs"]],
            ["Memory items", memory.summarize()["total"]]
        ], columns=["Metric","Value"]))

show_btn.on_click(show_logs)
metrics_btn.on_click(show_metrics)
display(widgets.HBox([severity, show_btn, metrics_btn]))
display(out)



# === Fandom Styler for User Input ===

print("ğŸ�­ Choose a fandom aesthetic to transform your study plan!\n")

# Dropdown for selecting fandoms
fandom_dropdown = widgets.Dropdown(
    options=sorted(style_prompts.keys()),
    value="barbie",
    description="Fandom:"
)

style_button = widgets.Button(description="Apply Fandom Style", button_style="success")

styled_output_area = widgets.Output()

def apply_fandom_style(_):
    with styled_output_area:
        styled_output_area.clear_output()
        
        # We need the userâ€™s latest plan from the previous cell
        try:
            latest_plan = memory.recall(tag="plan", limit=1)[0]["value"]
        except:
            print("âš ï¸� Generate a study plan first using the Submit button above.")
            return
        
        fandom = fandom_dropdown.value
        styled = fandom_stylist_agent(latest_plan, fandom)
        
        print(f"âœ¨ Styled in {fandom} universe:\n")
        print(styled)
        
        # Show a beautiful themed card preview
        render_card(f"{fandom.title()} Aesthetic Output", styled[:800].replace("\n","<br/>"), theme_key=fandom)

style_button.on_click(apply_fandom_style)

display(fandom_dropdown, style_button, styled_output_area)



# === Try It Yourself: User Input Text Box ===

print("ğŸ“¥ Enter your study text below and click 'Submit' to generate results.\n")

user_input = widgets.Textarea(
    value="",
    placeholder="Paste your notes, textbook paragraphs, or concepts here...",
    description="Study Text:",
    layout=widgets.Layout(width="100%", height="150px")
)

submit_btn = widgets.Button(description="Submit", button_style="primary")
user_output = widgets.Output()

def on_submit(_):
    with user_output:
        user_output.clear_output()
        text = user_input.value

        if not text.strip():
            print("âš ï¸� Please enter some text first.")
            return
        
        print("Processing your input...\n")
        
        summary = summarizer_agent(text)
        tasks = task_extractor_agent(summary)
        plan = study_planner_agent(tasks)

        print("âœ” Summary generated")
        print("âœ” Tasks generated")
        print("âœ” Study plan created")
        
        print("\nBelow is your study plan:\n")
        print(plan)

submit_btn.on_click(on_submit)

display(user_input, submit_btn, user_output)



# === Sample Fandom Showcase Gallery ===

sample_plan_text = """
This study plan focuses on improving biology fundamentals.
Key tasks:
- Review photosynthesis and respiration
- Practice diagrams of chloroplasts, mitochondria
- Solve previous year questions
- Revise formulas and definitions
"""

fandom_samples = [
    "barbie",
    "poo",
    "disney_princess",
    "dark_academia",
    "cottagecore"
]

print("Generating fandom previews...\n")

for fandom in fandom_samples:
    styled = fandom_stylist_agent(sample_plan_text, fandom)
    render_card(f"{fandom.title()} Style", styled[:600].replace("\n","<br/>"), theme_key=fandom)
    print(f"âœ“ Rendered: {fandom}")



sample = "Photosynthesis is the process through which plants convert light energy..."

print("\n--- SUMMARY ---\n")
summary = summarizer_agent(sample)
print(summary)

print("\n--- TASKS ---\n")
tasks = task_extractor_agent(summary)
print(tasks)

print("\n--- PLAN ---\n")
plan = study_planner_agent(tasks)
print(plan)

print("\n--- FANFOM STYLE (Barbie) ---\n")
styled = fandom_stylist_agent(plan, "barbie")
print(styled)

render_card("Styled (Barbie)", styled[:800], "barbie")





