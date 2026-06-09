


# Install dependencies (only if missing). On Kaggle this is usually not required.
import sys, subprocess, pkgutil

required = ["openai","numpy","pandas","python-dateutil","pytz","scikit-learn","matplotlib"]
for p in required:
    if not pkgutil.find_loader(p):
        print(f"Installing {p}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", p])
print("Dependencies check done")



import os, json, random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# OpenAI client if key is set
if "OPENAI_API_KEY" in os.environ:
    try:
        import openai
        openai.api_key = os.environ["OPENAI_API_KEY"]
    except Exception:
        openai = None
else:
    openai = None

NOTEBOOK_TITLE = "Personal AI Health & Lifestyle Concierge"



sample_data = [
    {"date":"2025-11-01","sleep_hours":7.0,"steps":3000,"water_l":1.2,"mood":0,"meals":"oats; rice & dal; roti & sabzi","notes":"felt okay"},
    {"date":"2025-11-02","sleep_hours":6.0,"steps":1500,"water_l":0.8,"mood":-1,"meals":"skipped breakfast; burger; noodles","notes":"tired"},
    {"date":"2025-11-03","sleep_hours":8.0,"steps":5000,"water_l":1.5,"mood":1,"meals":"fruit & salad","notes":"energetic"},
    {"date":"2025-11-04","sleep_hours":5.5,"steps":1000,"water_l":0.6,"mood":-2,"meals":"fast food","notes":"stressed"}
]

df = pd.DataFrame(sample_data)
df["date"] = pd.to_datetime(df["date"]).dt.date
df



def compute_metrics(df):
    df_sorted = df.sort_values("date").copy()
    df_sorted["sleep_7d_avg"] = df_sorted["sleep_hours"].rolling(7,min_periods=1).mean()
    df_sorted["steps_7d_avg"] = df_sorted["steps"].rolling(7,min_periods=1).mean()
    df_sorted["water_7d_avg"] = df_sorted["water_l"].rolling(7,min_periods=1).mean()
    df_sorted["mood_7d_avg"] = df_sorted["mood"].rolling(7,min_periods=1).mean()
    return df_sorted

def detect_risks(latest_row):
    risks = []
    if latest_row["sleep_hours"] < 6:
        risks.append("Low sleep: <6 hours last night")
    if latest_row["water_l"] < 1.0:
        risks.append("Low hydration: <1 L today")
    if latest_row["steps"] < 2000:
        risks.append("Low activity: <2000 steps today")
    if latest_row["mood"] <= -2:
        risks.append("Low mood / possible high stress")
    return risks



DEFAULT_PROMPT_TEMPLATE = """
You are a helpful health and lifestyle assistant. A user provides daily metrics with short notes.
Produce a short, friendly personalised plan for the next day.
Include:
- One-sentence summary of their current state
- 3 specific, easy actions for tomorrow (sleep, hydration, movement)
- One dietary suggestion
- One motivational line
Use the following data:
{context}
"""

def generate_plan_with_openai(context_text, model="gpt-4o-mini"):
    if not openai:
        raise RuntimeError("OpenAI not configured")
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role":"system","content":"You are a concise health & lifestyle assistant."},
            {"role":"user","content":context_text}
        ],
        max_tokens=300,
        temperature=0.7,
    )
    return resp['choices'][0]['message']['content'].strip()



def generate_plan_rule_based(latest_row, metrics_row=None):
    lines = []
    lines.append(f"Summary: Yesterday you slept {latest_row['sleep_hours']}h, walked {latest_row['steps']} steps, drank {latest_row['water_l']}L, mood {latest_row['mood']}.")
    if latest_row['sleep_hours'] < 7:
        lines.append("Action: Aim for 7.5-8 hours; try a calming pre-sleep routine (no screens 30 min before).")
    else:
        lines.append("Action: Keep your sleep schedule consistent; maintain it for recovery.")
    if latest_row['water_l'] < 1.5:
        lines.append("Action: Carry a 1L bottle and set 3 small hourly water goals.")
    else:
        lines.append("Action: Good hydration — keep it up.")
    if latest_row['steps'] < 3000:
        lines.append("Action: Do two 10-minute brisk walks tomorrow.")
    else:
        lines.append("Action: Add a 15-minute bodyweight routine.")
    if 'fast food' in latest_row.get('meals','').lower() or 'burger' in latest_row.get('meals','').lower():
        lines.append("Diet: Add vegetables/fruit and reduce deep-fried items.")
    else:
        lines.append("Diet: Keep protein in at least two meals and include a fruit snack.")
    lines.append("Motivation: Small consistent changes beat big one-time efforts — you can do this!")
    return "\n".join(lines)



def make_context_from_df(df):
    metrics = compute_metrics(df)
    latest = metrics.iloc[-1]
    lines = []
    lines.append(f"Recent averages (7d): sleep {latest['sleep_7d_avg']:.1f}h, steps {int(latest['steps_7d_avg'])} steps, water {latest['water_7d_avg']:.1f}L, mood {latest['mood_7d_avg']:.2f}.")
    lines.append("Last entries:\n")
    for _, row in df.tail(5).iterrows():
        lines.append(f"{row['date']}: sleep {row['sleep_hours']}h, steps {row['steps']}, water {row['water_l']}L, mood {row['mood']}, notes: {row.get('notes','')}")
    return "\n".join(lines), metrics

def produce_plan_and_report(df):
    context_text, metrics = make_context_from_df(df)
    latest = df.sort_values('date').iloc[-1]
    risks = detect_risks(latest)
    if openai:
        prompt = DEFAULT_PROMPT_TEMPLATE.format(context=context_text)
        try:
            llm_output = generate_plan_with_openai(prompt)
            plan_text = llm_output
        except Exception as e:
            plan_text = "[LLM error] " + str(e)
    else:
        plan_text = generate_plan_rule_based(latest, metrics.iloc[-1])
    report = {
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'summary_context': context_text,
        'detected_risks': risks,
        'plan': plan_text,
    }
    return report



report = produce_plan_and_report(df)
print("=== Generated Report ===")
print("Generated at:", report['generated_at'])
print("\n-- Summary Context --\n")
print(report['summary_context'])
print("\n-- Detected Risks --")
print(report['detected_risks'])
print("\n-- Plan --\n")
print(report['plan'])

# Save sample report
out_path = "/kaggle/working/health_concierge_report_sample.json"
with open(out_path, "w") as f:
    json.dump(report, f, indent=2)
print("\nSaved sample report to:", out_path)



def predict_stress(latest_row):
    score = 0
    if latest_row.get('mood',0) <= -2: score += 40
    elif latest_row.get('mood',0) == -1: score += 20
    if latest_row.get('sleep_hours',0) < 6: score += 20
    elif latest_row.get('sleep_hours',0) < 7: score += 10
    if latest_row.get('water_l',0) < 1: score += 15
    if latest_row.get('steps',0) < 2000: score += 15
    return min(score,100)

def compute_habit_score(metrics):
    latest = metrics.iloc[-1]
    sleep_score = min(100,(latest['sleep_7d_avg']/8)*100)
    hydration_score = min(100,(latest['water_7d_avg']/2)*100)
    steps_score = min(100,(latest['steps_7d_avg']/6000)*100)
    mood_score = ((latest['mood_7d_avg'] + 2) / 4) * 100
    final = (sleep_score*0.35 + hydration_score*0.25 + steps_score*0.25 + mood_score*0.15)
    return round(final,2)

def analyze_trends(df):
    trends = {}
    if len(df) < 4:
        return {"message":"Not enough data for trend analysis"}
    trends['sleep_trend'] = 'increasing' if df['sleep_hours'].diff().mean() > 0 else 'decreasing'
    trends['steps_trend'] = 'increasing' if df['steps'].diff().mean() > 0 else 'decreasing'
    trends['water_trend'] = 'increasing' if df['water_l'].diff().mean() > 0 else 'decreasing'
    trends['mood_trend'] = 'improving' if df['mood'].diff().mean() > 0 else 'declining'
    return trends

def produce_advanced_report(df):
    base = produce_plan_and_report(df)
    ctx, metrics = make_context_from_df(df)
    latest = df.sort_values('date').iloc[-1]
    stress = predict_stress(latest)
    score = compute_habit_score(metrics)
    trends = analyze_trends(df)
    base['advanced_agent'] = {
        'stress_prediction_score': stress,
        'habit_score_0_100': score,
        'health_trends': trends,
    }
    return base

# Demo advanced
adv = produce_advanced_report(df)
print("\n=== ADVANCED AGENT REPORT ===\n")
print(json.dumps(adv, indent=2))



def plot_trend(df, column, title, ylabel):
    plt.figure(figsize=(8,4))
    plt.plot(df['date'], df[column], marker='o')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Use sample df (or synthetic_df later)
plot_trend(df, 'sleep_hours', 'Sleep Trend', 'Hours')
plot_trend(df, 'steps', 'Steps Trend', 'Steps')
plot_trend(df, 'water_l', 'Hydration Trend', 'Liters')
plot_trend(df, 'mood', 'Mood Trend', 'Mood Level')



def conversation_agent():
    print("AI Health Assistant: Hello! How was your day?")
    try:
        sleep = float(input("How many hours did you sleep? "))
        water = float(input("How many liters of water did you drink? "))
        steps = int(input("How many steps did you walk? "))
        mood = int(input("Mood (-2 to 2): "))
    except Exception as e:
        print("Input error:", e)
        return
    row = {'sleep_hours': sleep, 'water_l': water, 'steps': steps, 'mood': mood}
    stress = predict_stress(row)
    print(f"Your stress score: {stress}/100")

# To run: conversation_agent()



def generate_60_day_data(start_date=datetime(2025,9,1)):
    rows = []
    for i in range(60):
        day = start_date + timedelta(days=i)
        rows.append({
            'date': day.date(),
            'sleep_hours': round(random.uniform(5, 8.5),1),
            'steps': random.randint(1000,7000),
            'water_l': round(random.uniform(0.7,2.0),1),
            'mood': random.randint(-2,2),
            'meals': 'varied meals',
            'notes': 'auto-generated'
        })
    return pd.DataFrame(rows)

synthetic_df = generate_60_day_data()
synthetic_df.head()



def plot_habit_score(df):
    ctx, metrics = make_context_from_df(df)
    scores = []
    for i in range(len(metrics)):
        part = metrics.iloc[:i+1]
        scores.append(compute_habit_score(part))
    plt.figure(figsize=(8,4))
    plt.plot(df['date'], scores, marker='o')
    plt.title('Habit Score Trend')
    plt.xlabel('Date')
    plt.ylabel('Habit Score (0-100)')
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_habit_score(synthetic_df)



def advanced_risk_engine(df):
    latest = df.sort_values('date').iloc[-1]
    risks = []
    if latest['sleep_hours'] < 6:
        risks.append('Sleep Debt Risk')
    if latest['water_l'] < 1:
        risks.append('Dehydration Risk')
    if latest['steps'] < 2000:
        risks.append('Activity Deficit Risk')
    if latest['mood'] <= -1:
        risks.append('Mood Decline Warning')
    return risks

print("Advanced risks for sample data:", advanced_risk_engine(df))





