# === Setup & Imports ===

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



# === style_prompts: FULL 45+ Fandom Dictionary ===

style_prompts = {
    "default": "Rewrite the study plan in a clear, friendly, motivational tone.",

    # Major fandoms
    "stranger_things": "Rewrite the plan with eerie tension, neon red glow, static distortion, retro mystery, and Upside Down energy.",
    "barbie": "Rewrite the study plan in sparkly, bubbly, pink Dreamhouse energy with empowering, positive glam.",
    "anime": "Rewrite this like an anime mentor inspiring the protagonist. Dramatic, sparkly, high-energy narrative.",
    "anime_shonen": "Rewrite as a shonen training arc with power levels, grit, and hype.",
    "kpop": "Rewrite with idol trainee discipline, neon stage lights, fan chants, and confident punchlines.",
    
    "harry_potter": "Rewrite in wizarding school style, full of magic metaphors, bravery, houses, and enchanted vibes.",
    "taylor_swift": "Rewrite inspired by Taylor Swiftâ€™s lyricism â€” poetic, emotional, easter-egg filled, era-coded.",
    "star_wars": "Rewrite with epic galactic energy â€” the Force, light vs dark balance, heroic determination.",
    "game_of_thrones": "Rewrite in medieval royal prose â€” strategy, houses, honor, fire & ice aesthetics.",
    "bridgerton": "Rewrite in elegant Regency-era charm with manners, romance, and refined style.",
    "twilight": "Rewrite with moody romance, soft darkness, dramatic devotion, and forest stillness.",
    "ghibli": "Rewrite in Studio Ghibli style: soft, warm, whimsical nature magic, gentle emotional depth.",
    
    "minecraft": "Rewrite as a gamerâ€™s crafting guide â€” biomes, blocks, XP, upgrades, tools, enchantments.",
    "pokemon": "Rewrite like a PokÃ©mon trainer guide â€” XP, leveling up, adventures, gym battles.",
    "retro_arcade": "Rewrite with neon lights, 8-bit energy, pixel intensity, boss battles.",
    "cyberpunk": "Rewrite with neon dystopia, glitch phrases, chrome, circuitry, and rain-soaked mood.",
    
    "doctor_who": "Rewrite with cosmic time-travel excitement, paradoxes, and quirky sci-fi humor.",
    "star_trek": "Rewrite with futuristic optimism, exploration, crew teamwork, and stardate structure.",
    "rick_and_morty": "Rewrite with chaotic science humor, dimension-hopping energy, weird metaphors.",
    
    "moana": "Rewrite with oceanic courage, waves, journeys, the heart of the sea imagery.",
    "frozen": "Rewrite with icy elegance, snowflakes, warmth, inner strength.",
    "avatar_tla": "Rewrite using the four nationsâ€™ themes â€” earth, fire, water, air, harmony.",
    "percy_jackson": "Rewrite with demigod quest humor, monsters, camp-half blood energy.",
    "hunger_games": "Rewrite with rebellion focus, survival tone, determination, fire symbolism.",
    "lotr": "Rewrite with ancient high-fantasy tone, quests, courage, elvish elegance.",
    "sherlock": "Rewrite with deduction tone â€” observational logic, sharp clues, intellect.",
    
    "gilmore_girls": "Rewrite with cozy coffee-shop humor, friendly tone, fast-paced wit.",
    "disney_princess": "Rewrite with fairytale magic, optimism, sparkles, whimsical charm.",
    "cottagecore": "Rewrite with soft nature imagery â€” flowers, tea, meadows, calm vibes.",
    "fairycore": "Rewrite with magic dust, wings, sparkles, pastel woodland charm.",
    "mermaidcore": "Rewrite with ocean shimmer, pearls, coral colors, gentle tides.",
    "witchcore": "Rewrite in spellbook fantasy â€” potions, moonlight, candles, mystical vibes.",
    "royalcore": "Rewrite in regal majestic elegance, royal duties, gold accents.",
    "gothic_romance": "Rewrite with dark passion, candlelit rooms, poetic melancholy.",
    
    "y2k": "Rewrite in shiny retro pop tone â€” sass, glitter, bold statements.",
    "vaporwave": "Rewrite with dreamy neon synth mood â€” soft, surreal, ambient.",
    
    "minecraft_survival": "Rewrite with survival crafting toneâ€”resource gathering, mobs, endurance.",
    
    # BTS individual themes
    "bts": "Rewrite in soft poetic warmth with comforting idol-style encouragement.",
    "bts_jimin": "Rewrite in pastel comfort energy â€” soft peach, angelic warmth.",
    "bts_v": "Rewrite in retro aesthetic poetry with moody sophistication.",
    "bts_jk": "Rewrite in high-energy gamer/cyberpunk athletic tone.",
    "bts_rm": "Rewrite in philosophical calm, earthy tones, wise reflections.",
    
    # Additional vibes
    "tumblr": "Rewrite in nostalgic aesthetic â€” soft quotes, poetic moodboards.",
    "dark_academia": "Rewrite with intellectual melancholy, candlelight, vintage books.",
    "softcore": "Rewrite with soft pastels, gentle wording, kawaii energy.",
}



# === ui_themes: Visual Themes for Each Fandom ===

ui_themes = {
    "default": {
        "primary": "#222",
        "background": "#fafafa",
        "accent": "#4a90e2",
        "font": "Inter, sans-serif",
        "effect": "none"
    },

    "stranger_things": {
        "primary": "#d10f0f",
        "background": "#0c0c0c",
        "accent": "#ffb3b3",
        "font": "'Special Elite', monospace",
        "effect": "vhs"
    },

    "barbie": {
        "primary": "#ff4dc4",
        "background": "#ffe0f5",
        "accent": "#ff9adf",
        "font": "'Poppins', sans-serif",
        "effect": "sparkle"
    },

    "anime": {
        "primary": "#ff9ecb",
        "background": "#e7f0ff",
        "accent": "#ffe1ef",
        "font": "'Comic Neue', cursive",
        "effect": "shimmer"
    },

    "kpop": {
        "primary": "#ff93e8",
        "background": "#f7eaff",
        "accent": "#b56cff",
        "font": "'Poppins', sans-serif",
        "effect": "glow"
    },

    "harry_potter": {
        "primary": "#7f0909",
        "background": "#f7f4e8",
        "accent": "#d2a100",
        "font": "'IM Fell English', serif",
        "effect": "spark"
    },

    "taylor_swift": {
        "primary": "#c49ccf",
        "background": "#fdeaff",
        "accent": "#deb4e7",
        "font": "'Homemade Apple', cursive",
        "effect": "glitter"
    },

    "star_wars": {
        "primary": "#00b7ff",
        "background": "#000",
        "accent": "#fff",
        "font": "'Oxanium', sans-serif",
        "effect": "starfield"
    },

    "game_of_thrones": {
        "primary": "#1f1f1f",
        "background": "#ececec",
        "accent": "#b9b9b9",
        "font": "'Cinzel', serif",
        "effect": "frost"
    },

    "bridgerton": {
        "primary": "#6a86c1",
        "background": "#f4f0ff",
        "accent": "#c9b3ff",
        "font": "'Playfair Display', serif",
        "effect": "floral"
    },

    "twilight": {
        "primary": "#8b0000",
        "background": "#f7f7f7",
        "accent": "#e3e3e3",
        "font": "'Lora', serif",
        "effect": "fog"
    },

    "ghibli": {
        "primary": "#7c9c7c",
        "background": "#fff7e1",
        "accent": "#d6e7c8",
        "font": "'Quicksand', sans-serif",
        "effect": "soft"
    },

    "minecraft": {
        "primary": "#3c8527",
        "background": "#cdebb5",
        "accent": "#8dcf5e",
        "font": "'Press Start 2P', monospace",
        "effect": "pixel"
    },

    "pokemon": {
        "primary": "#ffcb05",
        "background": "#f5f5f5",
        "accent": "#3c5aa6",
        "font": "'Nunito', sans-serif",
        "effect": "bounce"
    },

    "retro_arcade": {
        "primary": "#00ffff",
        "background": "#0b0b1a",
        "accent": "#ff00ff",
        "font": "'Press Start 2P', monospace",
        "effect": "crt"
    },

    "cyberpunk": {
        "primary": "#ff00c8",
        "background": "#080808",
        "accent": "#00e5ff",
        "font": "'Orbitron', sans-serif",
        "effect": "neon"
    },

    "rick_and_morty": {
        "primary": "#00ff7f",
        "background": "#222222",
        "accent": "#9d4edd",
        "font": "'Roboto Mono', monospace",
        "effect": "portal"
    },

    "moana": {
        "primary": "#00b7c2",
        "background": "#eefcff",
        "accent": "#ffd1b3",
        "font": "'Comfortaa', cursive",
        "effect": "wave"
    },

    "frozen": {
        "primary": "#5dade2",
        "background": "#f0f8ff",
        "accent": "#aed6f1",
        "font": "'Nunito', sans-serif",
        "effect": "snow"
    },

    "avatar_tla": {
        "primary": "#3b8ea5",
        "background": "#f4f4f4",
        "accent": "#d8c39f",
        "font": "'Merriweather', serif",
        "effect": "element"
    },

    "percy_jackson": {
        "primary": "#0e4d92",
        "background": "#eaf4ff",
        "accent": "#d1d9ff",
        "font": "'Cardo', serif",
        "effect": "water"
    },

    "hunger_games": {
        "primary": "#d35400",
        "background": "#1b1b1b",
        "accent": "#f39c12",
        "font": "'Cinzel', serif",
        "effect": "embers"
    },

    "lotr": {
        "primary": "#355e3b",
        "background": "#f6f2e6",
        "accent": "#c9b27a",
        "font": "'Cardo', serif",
        "effect": "ancient"
    },

    "sherlock": {
        "primary": "#1c1c1c",
        "background": "#f5f5f5",
        "accent": "#afafaf",
        "font": "'Merriweather', serif",
        "effect": "paper"
    },

    "cottagecore": {
        "primary": "#7a9a6d",
        "background": "#fff9f0",
        "accent": "#e5d8c5",
        "font": "'Playfair Display', serif",
        "effect": "leaves"
    },

    "fairycore": {
        "primary": "#ffdefa",
        "background": "#fff8fb",
        "accent": "#e0f7fa",
        "font": "'Merriweather', serif",
        "effect": "sparkle"
    },

    "mermaidcore": {
        "primary": "#64b6ac",
        "background": "#e8ffff",
        "accent": "#adebeb",
        "font": "'Nunito', sans-serif",
        "effect": "shimmer"
    },

    "witchcore": {
        "primary": "#4c2a4c",
        "background": "#f3e9f7",
        "accent": "#d6b4d8",
        "font": "'Cormorant Garamond', serif",
        "effect": "moon"
    },

    "royalcore": {
        "primary": "#9a7b4f",
        "background": "#fff9ef",
        "accent": "#d4b483",
        "font": "'Playfair Display', serif",
        "effect": "glow"
    },

    "gothic_romance": {
        "primary": "#5a0c0c",
        "background": "#f2f2f2",
        "accent": "#a87f7f",
        "font": "'Lora', serif",
        "effect": "mist"
    },

    # BTS Themings
    "bts": {
        "primary": "#c084fc",
        "background": "#f7e8ff",
        "accent": "#e9d5ff",
        "font": "'Nunito', sans-serif",
        "effect": "softglow"
    },
    "bts_jimin": {
        "primary": "#f7a8c5",
        "background": "#fff0f6",
        "accent": "#ffd9e8",
        "font": "'Poppins', sans-serif",
        "effect": "fairyspark"
    },
    "bts_v": {
        "primary": "#9a8fff",
        "background": "#f2ecff",
        "accent": "#cabdff",
        "font": "'Noto Serif Display', serif",
        "effect": "vsoft"
    },

    "tumblr": {
        "primary": "#35465c",
        "background": "#fafafa",
        "accent": "#7b9dbd",
        "font": "'Quicksand', sans-serif",
        "effect": "grain"
    },

    "dark_academia": {
        "primary": "#3e2f1b",
        "background": "#f0e6d2",
        "accent": "#7a5a3c",
        "font": "'Cormorant Garamond', serif",
        "effect": "grain"
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



dark_toggle = widgets.ToggleButton(description="Dark Mode", icon="moon")

def toggle_dark(change):
    if dark_toggle.value:
        display(HTML("<style> body { filter: invert(1) hue-rotate(180deg);} </style>"))
    else:
        display(HTML("<style> body { filter: none;} </style>"))

dark_toggle.observe(toggle_dark, names='value')
display(dark_toggle)



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
    "stranger_things",
    "anime",
    "harry_potter",
    "ghibli",
    "star_wars",
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



import json

def save_output(name, content):
    filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump({"name": name, "content": content}, f, indent=2)
    print(f"Saved â†’ {filename}")

save_button = widgets.Button(description="Save Styled Output", button_style="success")

def save_click(_):
    save_output("styled_output", styled)

save_button.on_click(save_click)
display(save_button)



# === Sanity Test: Quick Agent Health Check ===

print("Running sanity check...\n")

sample = "The mitochondria is the powerhouse of the cell."

try:
    s = summarizer_agent(sample)
    print("âœ“ Summarizer OK")

    t = task_extractor_agent(s)
    print("âœ“ Task Extractor OK")

    p = study_planner_agent(t)
    print("âœ“ Study Planner OK")

    st = fandom_stylist_agent(p, "anime")
    print("âœ“ Fandom Stylist OK")

    print("\nAll agents successfully executed.")
except Exception as e:
    print("â�Œ Sanity Test Failed:", e)





