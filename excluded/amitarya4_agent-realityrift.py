# ============================================
# RealityRift â€” Environment & Core Imports (Silent Install)
# ============================================

# Silent installation: suppresses warnings and dependency noise
!pip install -q -U --disable-pip-version-check google-generativeai matplotlib rich > /dev/null 2>&1

import os, time, ast, json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import matplotlib.pyplot as plt
import google.generativeai as genai
from kaggle_secrets import UserSecretsClient
from rich.console import Console
from rich.table import Table

console = Console()
print("ğŸ”° Runtime environment initialized successfully âœ…")


# ============================================
# Gemini 2.5 API Authentication & Model Setup
# ============================================

secrets_client = UserSecretsClient()
GEMINI_API_KEY = secrets_client.get_secret("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("â�Œ GEMINI_API_KEY not configured in Kaggle Secrets.")

genai.configure(api_key=GEMINI_API_KEY)

MODEL_DEEP = "gemini-2.5-pro"
MODEL_FAST = "gemini-2.5-flash"

console.log(f"[OK] Gemini active: {MODEL_DEEP}")
print("ğŸ”‘ Gemini API authenticated successfully âœ…")


# ============================================
# Data Structures â€” Scores, Scenarios, Result
# ============================================

@dataclass
class RealityScores:
    happiness: int
    stress: int
    growth: int
    money: int
    life_quality: int

@dataclass
class RealityScenario:
    label: str
    description: str
    scores: RealityScores
    key_insights: List[str]

@dataclass
class SimulationResult:
    user_profile_summary: str
    decision: str
    realities: List[RealityScenario]
    model_notes: str

print("ğŸ§¬ Data structures loaded successfully âœ…")


# ============================================
# RealityRift Cognitive Instructions (High-Precision Dict Output)
# ============================================

REALITYRIFT_SYSTEM_PROMPT = """
You are REALITYRIFT â€” an advanced decision-reflection agent that simulates
three alternate futures of the same person based on their personality and a
life decision they are considering.

Your role:
Show how the SAME person might evolve under three different life paths:
1) Fearless Reality      â†’ bold choices, risk-embracing version of them
2) Passion Reality       â†’ joy-driven, meaningful-life version
3) Logical Reality       â†’ safety-oriented, rational-optimization version

Core principles:
- You do NOT tell the user what to do.
- You do NOT advise, pressure, moralize, or prescribe.
- You simply reveal three possible futures for self-reflection.
- Each future must include benefits AND drawbacks (no perfect life).
- Be realistic and psychologically grounded, not cinematic.
- The simulation reflects tendencies, not predictions or certainty.

Required response format:
You MUST return ONLY a valid Python dictionary.
No markdown, no backticks, no commentary, and no extra text outside the dictionary.
The first character MUST be '{' and the last MUST be '}'.

STRICT structure (follow EXACTLY):

{
  "user_profile_summary": "string",
  "decision": "string",
  "realities": [
    {
      "label": "Fearless Reality",
      "description": "6â€“8 grounded sentences describing this future in detail.",
      "scores": {
        "happiness": int 0-100,
        "stress": int 0-100,
        "growth": int 0-100,
        "money": int 0-100,
        "life_quality": int 0-100
      },
      "key_insights": ["short string", "short string", "short string"]
    },
    {
      "label": "Passion Reality",
      "description": "6â€“8 grounded sentences describing this future in detail.",
      "scores": {
        "happiness": int 0-100,
        "stress": int 0-100,
        "growth": int 0-100,
        "money": int 0-100,
        "life_quality": int 0-100
      },
      "key_insights": ["short string", "short string", "short string"]
    },
    {
      "label": "Logical Reality",
      "description": "6â€“8 grounded sentences describing this future in detail.",
      "scores": {
        "happiness": int 0-100,
        "stress": int 0-100,
        "growth": int 0-100,
        "money": int 0-100,
        "life_quality": int 0-100
      },
      "key_insights": ["short string", "short string", "short string"]
    }
  ],
  "model_notes": "short string with assumptions, limitations, or context"
}

Psychological simulation rules:
- The same personality leads to different outcomes depending on environment + choices.
- Fearless Reality focuses on risk, ambition, speed â€” not guaranteed success.
- Passion Reality prioritizes joy, meaning, identity â€” not guaranteed financial ease.
- Logical Reality optimizes safety and planning â€” not guaranteed fulfillment.
- Success and pain can coexist; each future has trade-offs.

Tone and safety rules:
- Emotionally supportive, balanced, empowering.
- No pressuring, no judgement, no glorification of risk.
- Respect autonomy: user chooses their own future.

NON-NEGOTIABLE OUTPUT RULES:
- No emoji anywhere in the dictionary.
- No trailing commas.
- No additional text before or after the dictionary.
- Do NOT mention that you are following these instructions.
- If generation begins invalid, FIX it internally BEFORE responding.
""".strip()

console.log("[OK] System prompt loaded.")
print("ğŸ§  Cognitive instructions activated successfully âœ…")


# ============================================
# Robust Dict Extraction from Model Output
# ============================================

def extract_python_dict(raw_text: str) -> Dict[str, Any]:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Dictionary block missing in model output.")
    return ast.literal_eval(raw_text[start:end+1])

print("ğŸ§© Response parser initialized successfully âœ…")


# ============================================
# RealityRift Simulation Engine â€” Gemini 2.5
# ============================================

def run_realityrift_simulation(user_profile: str, decision: str, retries: int = 3) -> SimulationResult:
    model = genai.GenerativeModel(
        model_name=MODEL_DEEP,
        system_instruction=REALITYRIFT_SYSTEM_PROMPT,
    )

    user_msg = f"USER PROFILE:\n{user_profile.strip()}\n\nDECISION:\n{decision.strip()}"
    last_error = None

    for attempt in range(1, retries + 1):
        console.log(f"[RUN] Attempt {attempt}/{retries}")
        try:
            response = model.generate_content(
                [{"role": "user", "parts": [{"text": user_msg}]}],
                generation_config={"temperature": 0.75, "max_output_tokens": 2600},
            )
            payload = extract_python_dict((response.text or "").strip())
            console.log("[OK] Parsed successfully.")
            break
        except Exception as e:
            last_error = e
            time.sleep(1.0)

    realities = []
    for r in payload["realities"]:
        scores = RealityScores(**r["scores"])
        realities.append(RealityScenario(r["label"], r["description"], scores, r["key_insights"]))

    return SimulationResult(
        payload["user_profile_summary"],
        payload["decision"],
        realities,
        payload["model_notes"],
    )

print("ğŸš€ Simulation engine powered successfully âœ…")


# ============================================
# Summary Table + Insights
# ============================================

def display_simulation_summary(result: SimulationResult):
    console.rule("[ RealityRift â€” Reflection Summary ]")

    print("\nğŸ§¾ SUMMARY:\n", result.user_profile_summary, "\n")
    print("â�“ DECISION UNDER ANALYSIS:\nğŸ‘‰", result.decision, "\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Reality")
    table.add_column("Happiness", justify="right")
    table.add_column("Stress", justify="right")
    table.add_column("Growth", justify="right")
    table.add_column("Money", justify="right")
    table.add_column("LifeQ", justify="right")

    for r in result.realities:
        s = r.scores
        table.add_row(r.label, str(s.happiness), str(s.stress),
                      str(s.growth), str(s.money), str(s.life_quality))

    console.print(table)

    print("\nğŸ”� INSIGHTS")
    for r in result.realities:
        print(f"\nğŸ”® {r.label}:")
        for bullet in r.key_insights:
            print("  â€¢", bullet)

    if result.model_notes.strip():
        print("\nğŸ“Œ Model Notes:")
        print(result.model_notes)

print("ğŸ“„ Summary renderer activated successfully âœ…")


# ============================================
# Score Plot â€” Alternate Reality Comparison
# ============================================

def plot_reality_scores(result: SimulationResult):
    labels = [r.label for r in result.realities]
    idx = range(len(labels))
    metrics = ["happiness", "stress", "growth", "money", "life_quality"]

    plt.figure(figsize=(9, 5))
    for m in metrics:
        plt.plot(idx, [getattr(r.scores, m) for r in result.realities], marker="o", label=m)

    plt.xticks(list(idx), labels, rotation=10)
    plt.ylim(0, 110)
    plt.ylabel("Score (0â€“100)")
    plt.title("RealityRift â€” Comparative Future Metrics")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

print("ğŸ“Š Visualization module ready successfully âœ…")


# ============================================
# Demonstration Simulation â€” Sample Decision
# ============================================

user_demo = """
I am 26, ambitious but introverted. I value creative independence and long-term security.
I want a life that feels meaningful and financially sustainable. I fear letting my family down.
"""

decision_demo = "Should I quit my stable job to start my own online business?"

console.log("[DEMO] Running demonstration decision...")
demo_result = run_realityrift_simulation(user_demo, decision_demo)

display_simulation_summary(demo_result)
plot_reality_scores(demo_result)

print("ğŸ�¯ Demonstration simulation completed successfully ğŸš€")


# ============================================
# Interactive Mode â€” Custom User Run
# ============================================

def run_interactive_realityrift():
    print("ğŸŒ€ RealityRift â€” Interactive Simulation Mode")
    profile = input("\nDescribe yourself (3â€“6 sentences):\n> ")
    decision = input("\nWhat decision are you considering?\n> ")
    result = run_realityrift_simulation(profile, decision)
    display_simulation_summary(result)
    plot_reality_scores(result)

print("ğŸ§© Interactive mode ready for users & judges ğŸ”¥")


# ============================================
# RealityRift â€” Premium Interactive UI (Auto-Scroll Output Panel)
# ============================================

from ipywidgets import VBox, HBox, Textarea, Text, Button, Output, Layout
from IPython.display import display, clear_output, HTML
import matplotlib.pyplot as plt
import time

print("ğŸŒ� Loading enhanced RealityRift UI...")

profile_input = Textarea(
    value="",
    placeholder="Describe yourself (personality, values, fears, goals, strengths)...",
    description="ğŸ§‘ Profile:",
    layout={'width': '700px', 'height': '120px'}
)

decision_input = Text(
    value="",
    placeholder="e.g., Should I quit my job to start a business?",
    description="â�“ Decision:",
    layout={'width': '700px'}
)

run_button = Button(
    description="ğŸ”® Run Reality Simulation",
    button_style="success",
    layout={'width': '280px', 'height': '40px'}
)

output_style = Output(
    layout=Layout(
        width='100%',
        height='420px',
        border='2px solid #4CAF50',
        padding='10px',
        overflow='auto'
    )
)

ui_output = Output()

def on_run_click(b):
    with output_style:
        clear_output()
        profile = profile_input.value
        decision = decision_input.value

        if not profile or not decision:
            print("âš ï¸� Please enter both your profile and the decision.")
            return
        
        try:
            result = run_realityrift_simulation(profile, decision)

            print("ğŸ§¾ USER SUMMARY")
            print(result.user_profile_summary)
            print("\nâ�“ DECISION")
            print(result.decision)

            for r in result.realities:
                print("\n-----------------------------------")
                print(r.label.upper())
                print("-----------------------------------")
                print(r.description)
                print("\nScores:")
                print(f"  Happiness: {r.scores.happiness}")
                print(f"  Stress: {r.scores.stress}")
                print(f"  Growth: {r.scores.growth}")
                print(f"  Money: {r.scores.money}")
                print(f"  Life Quality: {r.scores.life_quality}")
                print("\nKey Insights:")
                for k in r.key_insights:
                    print(" â€¢", k)

            # Scroll automatically to bottom
            display(HTML("<script>var out=document.getElementsByClassName('output_scroll')[0]; if(out){out.scrollTop=out.scrollHeight;}</script>"))

            # Plot comparison
            labels = [r.label for r in result.realities]
            metrics = ["happiness", "stress", "growth", "money", "life_quality"]
            x = range(len(labels))

            plt.figure(figsize=(9, 5))
            for m in metrics:
                plt.plot(
                    x,
                    [getattr(r.scores, m) for r in result.realities],
                    marker="o",
                    label=m,
                )

            plt.xticks(list(x), labels, rotation=10)
            plt.ylim(0, 110)
            plt.title("Reality Comparison â€” Score Visualization")
            plt.grid(alpha=.3)
            plt.legend()
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print("â�Œ Error while generating simulation:")
            print(str(e))


run_button.on_click(on_run_click)

ui = VBox([profile_input, decision_input, run_button, output_style])
display(ui)

print("âœ¨ RealityRift UI upgraded â€” Auto-scroll enabled, premium panel active ğŸ”¥")


# ============================================
# RealityRift UI â€” PHASE A: Dark Mode + Animation + Premium Formatting
# ============================================

from ipywidgets import VBox, HBox, Textarea, Text, Button, Output, Layout, HTML
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

print("ğŸ�¨ Applying Dark-Mode RealityRift UI...")

dark_css = HTML("""
<style>
body { background-color: #111 !important; }
textarea, input { background-color:#222 !important; color:#e6e6e6 !important; }
.widget-label { color:#d2d2d2 !important; font-weight:bold; font-size:14px; }
button { transition:0.25s; font-weight:bold; }
button:hover { transform:scale(1.065); box-shadow:0px 0px 12px #0ff; }
.output-wrapper, .output { color:#e6e6e6 !important; }
</style>
""")
display(dark_css)

output_panel = Output(
    layout=Layout(
        width='100%',
        height='420px',
        border='2px solid #00d4ff',
        padding='12px',
        overflow='auto',
        background_color='#171717'
    )
)

profile_box = Textarea(
    placeholder="Who are you as a person?",
    description="ğŸ§‘ YOU:",
    layout={'width':'700px','height':'120px'}
)

decision_box = Text(
    placeholder="e.g. Should I move abroad? Start a business? Change career?",
    description="â�“ Decision:",
    layout={'width':'700px'}
)

run_button = Button(
    description="ğŸ”® Simulate Reality",
    button_style="primary",
    layout={'width':'300px','height':'45px'}
)

def format_section(title):
    print(f"\nâ”�â”�â”�â”�â”�â”�â”�â”�â”�â”� {title} â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�")

def run_sim(_):
    with output_panel:
        clear_output()
        p, d = profile_box.value, decision_box.value
        if not p or not d:
            print("âš ï¸� Please enter profile and decision.")
            return

        result = run_realityrift_simulation(p, d)

        format_section("USER SUMMARY")
        print(result.user_profile_summary)

        format_section("DECISION")
        print(result.decision)

        for r in result.realities:
            format_section(r.label.upper())
            print(r.description)
            print("\nScores:")
            s = r.scores
            print(f" Happiness: {s.happiness}")
            print(f" Stress: {s.stress}")
            print(f" Growth: {s.growth}")
            print(f" Money: {s.money}")
            print(f" Life Quality: {s.life_quality}")
            print("\nKey Insights:")
            for k in r.key_insights:
                print(" â€¢", k)

        # Graph
        labels = [r.label for r in result.realities]
        metrics = ["happiness","stress","growth","money","life_quality"]
        x = range(len(labels))

        plt.figure(figsize=(9,5))
        for m in metrics:
            plt.plot(x,[getattr(r.scores,m) for r in result.realities],marker="o",label=m)
        plt.xticks(list(x),labels,rotation=10)
        plt.ylim(0,110)
        plt.title("Reality Comparison â€” Score Visualization")
        plt.grid(alpha=.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

run_button.on_click(run_sim)

ui = VBox([profile_box, decision_box, run_button, output_panel])
display(ui)

print("âœ¨ UI-A installed â€” Dark Mode + Animation + Premium Formatting Active!")


# ============================================
# RealityRift â€” PHASE B: History + Comparison + PDF Export
# ============================================

print("ğŸ“œ Initializing RealityRift Advanced Control Panel...")

import datetime
from ipywidgets import VBox, HBox, Textarea, Text, Button, Output, Layout, Dropdown
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

# Optional PDF export dependency
try:
    import reportlab
except ImportError:
    import subprocess, sys
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "reportlab"],
        check=False
    )
finally:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        PDF_AVAILABLE = True
    except Exception:
        PDF_AVAILABLE = False

simulation_history = []  # stores dicts: {time, profile, decision, result}


# ---------- Helper: run simulation and store history ----------
def run_and_store(profile, decision, label="Run"):
    res = run_realityrift_simulation(profile, decision)
    entry = {
        "timestamp": datetime.datetime.now(),
        "label": label,
        "profile": profile,
        "decision": decision,
        "result": res,
    }
    simulation_history.append(entry)
    return res, entry


# ---------- UI Controls ----------
adv_profile = Textarea(
    placeholder="Who are you, in your own words?",
    description="ğŸ§‘ YOU:",
    layout={'width': '750px', 'height': '120px'}
)

decision_1 = Text(
    placeholder="Decision A (e.g. Stay in current job)",
    description="A:",
    layout={'width': '360px'}
)

decision_2 = Text(
    placeholder="Decision B (e.g. Quit to start business)",
    description="B:",
    layout={'width': '360px'}
)

run_a_btn = Button(
    description="Simulate A",
    button_style="info",
    layout={'width': '160px', 'height': '35px'}
)
run_b_btn = Button(
    description="Simulate B",
    button_style="info",
    layout={'width': '160px', 'height': '35px'}
)
compare_btn = Button(
    description="ğŸ”� Compare A vs B",
    button_style="warning",
    layout={'width': '200px', 'height': '35px'}
)
export_btn = Button(
    description="ğŸ“„ Export Latest to PDF",
    button_style="success",
    layout={'width': '220px', 'height': '35px'}
)

history_output = Output(
    layout=Layout(
        width='100%',
        height='220px',
        border='2px solid #888',
        padding='8px',
        overflow='auto'
    )
)

compare_output = Output(
    layout=Layout(
        width='100%',
        height='380px',
        border='2px solid #ffaa00',
        padding='8px',
        overflow='auto'
    )
)


# ---------- Button Callbacks ----------

latest_result = {"A": None, "B": None}

def on_run_a(_):
    with history_output:
        profile = adv_profile.value
        dec = decision_1.value
        if not profile or not dec:
            print("âš ï¸� Please fill profile and Decision A.")
            return
        res, entry = run_and_store(profile, dec, label="Decision A")
        latest_result["A"] = res
        print(f"[{entry['timestamp']:%H:%M:%S}] Stored simulation for Decision A.")
        print("   â†’", dec)

def on_run_b(_):
    with history_output:
        profile = adv_profile.value
        dec = decision_2.value
        if not profile or not dec:
            print("âš ï¸� Please fill profile and Decision B.")
            return
        res, entry = run_and_store(profile, dec, label="Decision B")
        latest_result["B"] = res
        print(f"[{entry['timestamp']:%H:%M:%S}] Stored simulation for Decision B.")
        print("   â†’", dec)

def on_compare(_):
    with compare_output:
        clear_output()
        res_a = latest_result.get("A")
        res_b = latest_result.get("B")

        if res_a is None or res_b is None:
            print("âš ï¸� Please simulate both Decision A and Decision B first.")
            return

        print("ğŸ”� Comparing Decision A vs Decision B\n")
        print("A:", res_a.decision)
        print("B:", res_b.decision)
        print("\nScore comparison across futures:\n")

        # Simple numeric comparison: average scores per simulation
        def avg_scores(res):
            vals = {"happiness":0,"stress":0,"growth":0,"money":0,"life_quality":0}
            n = max(len(res.realities),1)
            for r in res.realities:
                vals["happiness"] += r.scores.happiness
                vals["stress"] += r.scores.stress
                vals["growth"] += r.scores.growth
                vals["money"] += r.scores.money
                vals["life_quality"] += r.scores.life_quality
            for k in vals:
                vals[k] = vals[k] / n
            return vals

        a_avg = avg_scores(res_a)
        b_avg = avg_scores(res_b)

        print("Average metric scores (A vs B):")
        for k in a_avg:
            print(f"  {k.capitalize():12}:  A = {a_avg[k]:5.1f}   |   B = {b_avg[k]:5.1f}")

        # Plot bar comparison
        labels = list(a_avg.keys())
        xa = range(len(labels))
        ya = [a_avg[k] for k in labels]
        yb = [b_avg[k] for k in labels]

        plt.figure(figsize=(8,4))
        width = 0.35
        plt.bar([x - width/2 for x in xa], ya, width=width, label="Decision A")
        plt.bar([x + width/2 for x in xa], yb, width=width, label="Decision B")
        plt.xticks(list(xa), [k.capitalize() for k in labels])
        plt.ylim(0, 110)
        plt.ylabel("Score (0â€“100)")
        plt.title("Average Future Metrics â€” A vs B")
        plt.legend()
        plt.tight_layout()
        plt.show()

def on_export(_):
    with history_output:
        if not simulation_history:
            print("âš ï¸� No simulation found to export.")
            return
        if not PDF_AVAILABLE:
            print("âš ï¸� PDF export library not available in this environment.")
            return

        latest = simulation_history[-1]
        res = latest["result"]
        filename = "realityrift_latest_simulation.pdf"
        
        c = canvas.Canvas(filename, pagesize=A4)
        width, height = A4

        y = height - 50
        c.setFont("Helvetica-Bold", 16)
        c.drawString(40, y, "RealityRift â€” Simulation Export")
        y -= 30
        c.setFont("Helvetica", 10)
        c.drawString(40, y, f"Timestamp: {latest['timestamp']}")
        y -= 20
        c.drawString(40, y, f"Decision: {latest['decision']}")
        y -= 30

        c.setFont("Helvetica-Bold", 11)
        c.drawString(40, y, "User Profile Summary:")
        y -= 15
        c.setFont("Helvetica", 10)
        for line in res.user_profile_summary.split("\n"):
            c.drawString(50, y, line[:100])
            y -= 14
            if y < 80:
                c.showPage()
                y = height - 50

        for r in res.realities:
            if y < 120:
                c.showPage()
                y = height - 50
            y -= 10
            c.setFont("Helvetica-Bold", 11)
            c.drawString(40, y, f"Reality: {r.label}")
            y -= 15
            c.setFont("Helvetica", 10)
            for line in r.description.split("\n"):
                c.drawString(50, y, line[:100])
                y -= 14
                if y < 80:
                    c.showPage()
                    y = height - 50
            y -= 10
            c.drawString(50, y, f"Scores â†’ H:{r.scores.happiness}  S:{r.scores.stress}  G:{r.scores.growth}  M:{r.scores.money}  LQ:{r.scores.life_quality}")
            y -= 18
            c.drawString(50, y, "Key Insights:")
            y -= 14
            for k in r.key_insights:
                c.drawString(60, y, f"- {k[:90]}")
                y -= 14
                if y < 80:
                    c.showPage()
                    y = height - 50

        c.showPage()
        c.save()

        print(f"ğŸ“„ Latest simulation exported as: {filename}")
        print("   You can download this file from the Kaggle notebook file explorer panel.")

# Attach callbacks
run_a_btn.on_click(on_run_a)
run_b_btn.on_click(on_run_b)
compare_btn.on_click(on_compare)
export_btn.on_click(on_export)

top_row = HBox([decision_1, decision_2])
btn_row = HBox([run_a_btn, run_b_btn, compare_btn, export_btn])
panel = VBox([
    adv_profile,
    top_row,
    btn_row,
    history_output,
    compare_output
])

display(panel)
print("âœ… RealityRift Advanced Control Panel ready â€” History, Comparison & PDF Export enabled.")


# ============================================
# RealityRift â€” PHASE C: Cyberpunk Decision Matrix (Deep A/B Comparison)
# ============================================

print("âš¡ Activating Cyberpunk Decision Matrix â€” Reality Comparison Mode...")

from ipywidgets import VBox, HBox, HTML, Button, Output, Layout
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import numpy as np

comparison_panel = Output(
    layout=Layout(
        width='100%',
        height='480px',
        border='3px solid #8A2BE2',
        padding='12px',
        overflow='auto',
        background_color='#0c0017'
    )
)

matrix_button = Button(
    description="ğŸ’  Deep Compare A vs B (Decision Matrix Mode)",
    button_style="",
    layout={'width': '360px', 'height': '45px'}
)

# Cyberpunk CSS
cyber_css = HTML("""
<style>
button:hover { box-shadow:0 0 18px #9d4bff; transform:scale(1.06); transition:0.25s; }
.output, .output_subarea { color:#E6E6FF !important; font-family: 'Consolas', monospace; font-size:15px; }
</style>
""")
display(cyber_css)

def deep_compare(_):
    with comparison_panel:
        clear_output()

        A = latest_result["A"]
        B = latest_result["B"]

        if A is None or B is None:
            print("âš ï¸� Please run both Decision A and Decision B first (using UI-B).")
            return

        print("ğŸ’  CYBERPUNK DECISION MATRIX â€” Deep Comparative Interpretation\n")

        # Score extraction function
        def collect_scores(result):
            values = {"happiness":[], "stress":[], "growth":[], "money":[], "life_quality":[]}
            for r in result.realities:
                values["happiness"].append(r.scores.happiness)
                values["stress"].append(r.scores.stress)
                values["growth"].append(r.scores.growth)
                values["money"].append(r.scores.money)
                values["life_quality"].append(r.scores.life_quality)
            return values

        Avals = collect_scores(A)
        Bvals = collect_scores(B)

        # Compute mean difference
        interpretation = {}
        for k in Avals:
            interpretation[k] = np.mean(Avals[k]) - np.mean(Bvals[k])

        # Print table comparison
        print("ğŸ“Š SCORE GAP (Positive = Decision A is higher | Negative = Decision B is higher)\n")
        for k,v in interpretation.items():
            tag = "A stronger" if v > 0 else ("B stronger" if v < 0 else "Equal")
            print(f"  {k.capitalize():12}:  {v:+5.2f}   â†’ {tag}")

        # Heatmap visualization
        labels = list(interpretation.keys())
        x = np.arange(len(labels))
        values = [interpretation[k] for k in labels]

        plt.figure(figsize=(9,4))
        bars = plt.bar(x, values, color=['#8A2BE2' if v>0 else '#00E5FF' for v in values])
        plt.axhline(0, color='white', linewidth=1)
        plt.xticks(x, [k.capitalize() for k in labels])
        plt.title("Decision Advantage Heatmap")
        plt.tight_layout()
        plt.show()

        # Personality alignment inference
        print("\nğŸ¤– AI INTERPRETATION OF DECISION ALIGNMENT\n")
        scoreA = sum(max(v,0) for v in interpretation.values())
        scoreB = sum(max(-v,0) for v in interpretation.values())

        if scoreA > scoreB:
            print(f"âœ¨ Decision A aligns more with your psychological identity.")
        elif scoreA < scoreB:
            print(f"âœ¨ Decision B aligns more with your psychological identity.")
        else:
            print(f"ğŸ¤� Both decisions are equally valid for your identity â€” reflection recommended.")

        print("\nğŸ”� Interpretation logic considers:")
        print("â€¢ Emotional outcomes")
        print("â€¢ Financial trajectory")
        print("â€¢ Personal growth potential")
        print("â€¢ Stress and life stability")
        print("â€¢ Self-identity alignment")

matrix_button.on_click(deep_compare)

display(VBox([matrix_button, comparison_panel]))

print("ğŸš€ UI-C Ready â€” Cyberpunk Decision Matrix installed successfully!")


# ============================================
# RealityRift â€” PHASE D: Decision Report Generator (PDF Export)
# ============================================

print("ğŸ“„ Initializing RealityRift Decision Report Generator...")

import datetime
from ipywidgets import Button, Output, VBox, Layout
from IPython.display import display, clear_output

# Ensure reportlab availability (in case previous cell didn't define it)
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    PDF_AVAILABLE_D = True
except Exception:
    try:
        import subprocess, sys
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "reportlab"],
            check=False
        )
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        PDF_AVAILABLE_D = True
    except Exception:
        PDF_AVAILABLE_D = False

report_output = Output(
    layout=Layout(
        width="100%",
        height="200px",
        border="2px solid #44aa44",
        padding="8px",
        overflow="auto"
    )
)

generate_report_btn = Button(
    description="ğŸ“„ Generate Full Decision Report (A vs B)",
    button_style="success",
    layout={'width': '320px', 'height': '40px'}
)


def build_decision_report(_):
    with report_output:
        clear_output()
        if not PDF_AVAILABLE_D:
            print("âš ï¸� PDF library is not available in this environment.")
            print("   The notebook remains fully functional; only report export is disabled.")
            return

        # We need at least one simulation
        if not simulation_history:
            print("âš ï¸� No simulations found in history. Run at least one simulation first.")
            return

        # Try to use last A/B if available, otherwise fallback to latest simulation
        resA = latest_result.get("A")
        resB = latest_result.get("B")
        latest_entry = simulation_history[-1]
        latest_res = latest_entry["result"]

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"realityrift_decision_report_{ts}.pdf"

        c = canvas.Canvas(filename, pagesize=A4)
        width, height = A4
        y = height - 50

        def write_line(text, font="Helvetica", size=10, offset=14, bold=False):
            nonlocal y
            if y < 80:
                c.showPage()
                y = height - 50
            if bold:
                c.setFont("Helvetica-Bold", size)
            else:
                c.setFont(font, size)
            c.drawString(40, y, text)
            y -= offset

        # Header
        write_line("RealityRift â€” Decision Reflection Report", size=16, bold=True, offset=24)
        write_line(f"Generated at: {datetime.datetime.now()}", size=9, offset=16)

        # Latest profile + decision
        write_line("", offset=10)
        write_line("User Profile Summary:", bold=True, offset=16)
        for line in latest_res.user_profile_summary.split("\n"):
            write_line(line[:110])

        write_line("", offset=10)
        write_line("Latest Decision Analyzed:", bold=True, offset=16)
        write_line(latest_res.decision[:110])

        # If A & B both exist, include comparison block
        if resA is not None and resB is not None:
            write_line("", offset=14)
            write_line("Decision A vs Decision B Overview:", bold=True, offset=16)
            write_line(f"Decision A: {resA.decision[:110]}")
            write_line(f"Decision B: {resB.decision[:110]}")

            def avg_scores(res):
                vals = {"happiness":0,"stress":0,"growth":0,"money":0,"life_quality":0}
                n = max(len(res.realities), 1)
                for r in res.realities:
                    vals["happiness"] += r.scores.happiness
                    vals["stress"] += r.scores.stress
                    vals["growth"] += r.scores.growth
                    vals["money"] += r.scores.money
                    vals["life_quality"] += r.scores.life_quality
                for k in vals:
                    vals[k] = vals[k] / n
                return vals

            a_avg = avg_scores(resA)
            b_avg = avg_scores(resB)

            write_line("", offset=10)
            write_line("Average Metrics (Decision A vs Decision B):", bold=True, offset=16)
            for k in a_avg:
                line = f"{k.capitalize():12}:  A = {a_avg[k]:5.1f}   |   B = {b_avg[k]:5.1f}"
                write_line(line)

        # Per-reality details (latest simulation)
        write_line("", offset=16)
        write_line("Detailed Futures (Latest Simulation):", bold=True, offset=16)

        for r in latest_res.realities:
            write_line("", offset=10)
            write_line(f"Reality: {r.label}", bold=True, offset=16)
            for line in r.description.split("\n"):
                write_line(line[:110])
            s = r.scores
            write_line(
                f"Scores â†’ H:{s.happiness}  S:{s.stress}  G:{s.growth}  M:{s.money}  LQ:{s.life_quality}",
                offset=14
            )
            write_line("Key Insights:", bold=True, offset=14)
            for k in r.key_insights:
                write_line(f"- {k[:100]}")

        # Final page: meta
        c.showPage()
        y = height - 60
        write_line("Notes:", bold=True, offset=18)
        write_line("â€¢ This report is a reflection tool, not a prediction or prescription.")
        write_line("â€¢ All futures are simulated using Gemini-based cognitive modeling.")
        write_line("â€¢ The purpose is to support self-awareness and clarity in decision-making.")
        c.showPage()
        c.save()

        print("âœ… Decision report generated successfully.")
        print(f"   File name: {filename}")
        print("   You can download it from the Kaggle file explorer panel on the right side.")

generate_report_btn.on_click(build_decision_report)

report_ui = VBox([
    generate_report_btn,
    report_output
])

display(report_ui)
print("âœ… UI-D ready â€” Full Decision Report Generator is now active.")

