# Life Balance Agent: Health + Productivity + Mood Analyzer
# New Version — Random data changes every run

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import textwrap
import os

# ------------------------------
# 1. Synthetic data generator
# ------------------------------

def generate_synthetic_wellness_data(days: int = 60) -> pd.DataFrame:
    """Create a 60-day log with unique random data every run."""
    
    # ===========================
    # RANDOMNESS MODES
    # ===========================
    
    # Mode 1: Different every run (simple)
    # rng = np.random.default_rng()

    # Mode 2: Time-based
    # rng = np.random.default_rng(int(datetime.now().timestamp()))

    # Mode 3: Cryptographically strong randomness (BEST, default)
    rng = np.random.default_rng(int.from_bytes(os.urandom(8), "little"))
    
    # ===========================

    start_date = datetime.today() - timedelta(days=days-1)
    dates = [start_date + timedelta(days=i) for i in range(days)]

    # Health signals
    sleep_hours = rng.normal(7, 1, days).clip(4, 10)
    steps = rng.normal(8000, 2500, days).clip(1000, 16000)
    workout_minutes = rng.normal(25, 15, days).clip(0, 90)

    # Productivity signals
    deep_work = rng.normal(3, 1.2, days).clip(0, 7)
    screen_time = rng.normal(5, 1.5, days).clip(2, 10)
    break_count = rng.normal(8, 3, days).clip(1, 20)

    # Mood signals
    mood = rng.normal(7, 1.5, days).clip(1, 10)
    stress = rng.normal(4.5, 1.8, days).clip(1, 10)
    caffeine = rng.normal(180, 70, days).clip(0, 400)

    df = pd.DataFrame({
        "date": dates,
        "sleep_hours": sleep_hours,
        "steps": steps.astype(int),
        "workout_minutes": workout_minutes,
        "deep_work_hours": deep_work,
        "screen_time_hours": screen_time,
        "break_count": break_count.astype(int),
        "mood_score": mood,
        "stress_level": stress,
        "caffeine_mg": caffeine.astype(int)
    })
    
    return df


# ------------------------------
# 2. Analysis Agents
# ------------------------------

def health_agent(df):
    df = df.copy()
    sleep_score = np.clip((df["sleep_hours"] - 6) / 2, 0, 1)
    steps_score = np.clip(df["steps"] / 10000, 0, 1)
    workout_score = np.clip(df["workout_minutes"] / 30, 0, 1)
    
    df["health_score"] = (
        0.45*sleep_score + 0.35*steps_score + 0.20*workout_score
    ) * 100
    return df

def productivity_agent(df):
    df = df.copy()
    deep_score = np.clip(df["deep_work_hours"] / 4, 0, 1)
    screen_score = 1 - np.clip((df["screen_time_hours"] - 3) / 5, 0, 1)
    breaks_score = 1 - np.clip(abs(df["break_count"] - 8) / 10, 0, 1)
    
    df["productivity_score"] = (
        0.5*deep_score + 0.3*screen_score + 0.2*breaks_score
    ) * 100
    return df

def mood_agent(df):
    df = df.copy()
    mood_norm = df["mood_score"] / 10
    stress_norm = 1 - (df["stress_level"] / 10)
    caffeine_penalty = np.clip(df["caffeine_mg"] / 400, 0, 1)
    
    df["mood_wellbeing_score"] = (
        0.6*mood_norm + 0.3*stress_norm - 0.1*caffeine_penalty
    ) * 100
    return df

def classification_agent(df):
    df = df.copy()
    
    df["overall_score"] = (
        0.4*df["health_score"] +
        0.35*df["productivity_score"] +
        0.25*df["mood_wellbeing_score"]
    )
    
    df["day_label"] = np.select(
        [
            df["overall_score"] >= 75,
            df["overall_score"] >= 55,
        ],
        ["Great", "OK"],
        default="Warning"
    )
    
    return df

def planning_agent(df, lookback_days=14):
    recent = df.sort_values("date").tail(lookback_days)

    avg_s = recent["sleep_hours"].mean()
    avg_steps = recent["steps"].mean()
    avg_w = recent["workout_minutes"].mean()
    avg_screen = recent["screen_time_hours"].mean()
    avg_deep = recent["deep_work_hours"].mean()
    avg_mood = recent["mood_score"].mean()
    avg_stress = recent["stress_level"].mean()

    actions = []

    # Sleep
    if avg_s < 7:
        actions.append(("Daily", "Sleep", f"Sleep target: {round(avg_s+0.8,1)} hours"))
    else:
        actions.append(("Daily", "Sleep", "Maintain 7–8 hours sleep."))

    # Workout
    steps_target = max(8000, int(avg_steps+1200))
    actions.append(("3x/week", "Exercise", f"Hit {steps_target} steps daily."))

    # Screen time
    if avg_screen > 6:
        actions.append(("Daily", "Screen time", "Cut 1 hour from evening screen time."))

    # Deep work
    if avg_deep < 3:
        actions.append(("Weekdays", "Deep Work", "Add one 90-min deep work block."))

    # Mood / Stress
    if avg_mood < 7 or avg_stress > 6:
        actions.append(("Daily", "Mood", "Do 10 minutes of journaling or meditation."))

    # Build plan
    days = [f"Day {i}" for i in range(1, 8)]
    rows = []

    for d in days:
        for freq, area, act in actions:
            rows.append({"day": d, "frequency": freq, "focus_area": area, "action": act})

    return pd.DataFrame(rows)


def report_agent(df, plan_df, filename="wellness_report.md"):
    last_14 = df.sort_values("date").tail(14)

    label_counts = last_14["day_label"].value_counts()
    avg_health = last_14["health_score"].mean()
    avg_prod = last_14["productivity_score"].mean()
    avg_mood = last_14["mood_wellbeing_score"].mean()

    worst = last_14.sort_values("overall_score").head(3)

    buf = []
    buf.append("# Life Balance Agent Report\n")
    buf.append(f"Generated: {datetime.now()}\n")

    buf.append("\n## Past 14-Day Summary\n")
    buf.append(f"- Avg Health Score: {avg_health:.1f}\n")
    buf.append(f"- Avg Productivity Score: {avg_prod:.1f}\n")
    buf.append(f"- Avg Mood Score: {avg_mood:.1f}\n")

    buf.append("\n### Day Labels\n")
    for lbl in ["Great","OK","Warning"]:
        buf.append(f"- {lbl}: {label_counts.get(lbl,0)} days\n")

    buf.append("\n### Toughest Days\n")
    for _, r in worst.iterrows():
        buf.append(f"- {r['date'].date()} — {r['overall_score']:.1f} ({r['day_label']})\n")

    buf.append("\n## 7-Day Improvement Plan\n")
    for day in plan_df["day"].unique():
        buf.append(f"\n### {day}\n")
        rows = plan_df[plan_df["day"] == day]
        for _, r in rows.iterrows():
            buf.append(f"- **{r['focus_area']}** ({r['frequency']}): {r['action']}\n")

    with open(filename, "w") as f:
        f.write("\n".join(buf))


# ------------------------------
# 3. Run the pipeline
# ------------------------------

df = generate_synthetic_wellness_data(60)
df = health_agent(df)
df = productivity_agent(df)
df = mood_agent(df)
df = classification_agent(df)

display(df.tail())

plan = planning_agent(df)
df.to_csv("life_balance_data.csv", index=False)
plan.to_csv("weekly_plan.csv", index=False)
report_agent(df, plan)

print("Generated:")
print("- life_balance_data.csv")
print("- weekly_plan.csv")
print("- wellness_report.md")

# ------------------------------
# Chart
# ------------------------------

plt.figure(figsize=(10,5))
recent = df.tail(21)
plt.plot(recent["date"], recent["health_score"])
plt.plot(recent["date"], recent["productivity_score"])
plt.plot(recent["date"], recent["mood_wellbeing_score"])
plt.legend(["Health","Productivity","Mood"])
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("metrics_overview.png")
plt.show()

print("Saved metrics_overview.png")



from IPython.display import FileLink, display
import os

print("Generated files:\n")

for f in os.listdir("/kaggle/working"):
    full_path = f"/kaggle/working/{f}"
    if os.path.isfile(full_path):   # skip folders
        display(FileLink(full_path))



import shutil

# Create ZIP with all files
shutil.make_archive("life_balance_outputs", "zip", "/kaggle/working")

from IPython.display import FileLink
FileLink("/kaggle/working/life_balance_outputs.zip")


