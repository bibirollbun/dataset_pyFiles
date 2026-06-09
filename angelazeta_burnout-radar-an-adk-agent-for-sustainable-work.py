import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (8, 4)

print("Libraries ready âœ…")



import google.generativeai as genai
from kaggle_secrets import UserSecretsClient

try:
    user_secrets = UserSecretsClient()
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
    GEMINI_AVAILABLE = True
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    print("âœ… Gemini configured")
except Exception as e:
    GEMINI_AVAILABLE = False
    model = None
    print("âš ï¸� Gemini not available:", e)



def generate_synthetic_work_log(
    start_date: str = "2025-09-01",
    n_weeks: int = 10,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate a synthetic daily work log over n_weeks.
    Columns:
      date, project, hours_worked, meetings_count, context_switches,
      evening_work, weekend_work, mood, notes
    """
    rng = np.random.default_rng(seed)
    start = datetime.fromisoformat(start_date)
    days = n_weeks * 7

    projects = ["Platform", "UX Research", "Data Pipelines", "Support", "Internal"]
    rows = []

    for i in range(days):
        day = start + timedelta(days=i)
        weekday = day.weekday()  # 0=Mon ... 6=Sun
        is_weekend = weekday >= 5

        # Baseline hours
        if is_weekend:
            base_hours = rng.choice([0, 0, 0, 2, 4])  # mostly 0, sometimes some weekend work
        else:
            base_hours = rng.normal(8, 1.5)
            base_hours = max(0, min(base_hours, 12))

        # Add some "crunch" periods for realism (week 4-5)
        week_idx = i // 7
        if 3 <= week_idx <= 4 and not is_weekend:
            base_hours += rng.choice([1, 2])  # crunch weeks

        hours_worked = round(float(max(0, base_hours)), 1)

        # meetings & context switches
        if hours_worked == 0:
            meetings_count = 0
            context_switches = 0
        else:
            meetings_count = int(rng.integers(0, 6))
            context_switches = int(rng.integers(1, 8))

        evening_work = int(hours_worked > 9 and not is_weekend)
        weekend_work = int(is_weekend and hours_worked > 0)

        # mood: influenced a bit by hours & weekend work
        mood_base = 4.0
        if hours_worked > 9:
            mood_base -= 0.7
        if weekend_work:
            mood_base -= 0.5
        if hours_worked == 0 and not is_weekend:
            mood_base -= 0.3  # sick / unproductive day?
        mood_base += rng.normal(0, 0.5)
        mood = int(np.clip(round(mood_base), 1, 5))

        # notes (very simple)
        note = ""
        if weekend_work:
            note = "Worked on weekend"
        elif hours_worked > 10:
            note = "Long intense day"
        elif hours_worked == 0:
            note = "Day off"

        project = rng.choice(projects)

        rows.append(
            {
                "date": day.date().isoformat(),
                "project": project,
                "hours_worked": hours_worked,
                "meetings_count": meetings_count,
                "context_switches": context_switches,
                "evening_work": evening_work,
                "weekend_work": weekend_work,
                "mood": mood,
                "notes": note,
            }
        )

    df = pd.DataFrame(rows)
    df.head()
    return df


work_log = generate_synthetic_work_log()
work_log.head()



print("Shape:", work_log.shape)
display(work_log.head())

print("\nBasic stats (hours, mood):")
display(work_log[["hours_worked", "mood"]].describe())

work_log["date_dt"] = pd.to_datetime(work_log["date"])

# Hours over time
work_log.sort_values("date_dt", inplace=True)

fig, ax = plt.subplots()
ax.plot(work_log["date_dt"], work_log["hours_worked"], marker="o")
ax.set_title("Daily hours worked over time")
ax.set_xlabel("Date")
ax.set_ylabel("Hours worked")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Mood over time
fig, ax = plt.subplots()
ax.plot(work_log["date_dt"], work_log["mood"], marker="o")
ax.set_title("Daily mood over time (1â€“5)")
ax.set_xlabel("Date")
ax.set_ylabel("Mood")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



def add_week_info(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["date_dt"] = pd.to_datetime(d["date"])
    iso = d["date_dt"].dt.isocalendar()
    d["year_week"] = iso["year"].astype(str) + "-" + iso["week"].astype(str).str.zfill(2)
    return d


def compute_weekly_stats(df: pd.DataFrame) -> pd.DataFrame:
    d = add_week_info(df)
    agg = (
        d.groupby("year_week")
        .agg(
            total_hours=("hours_worked", "sum"),
            avg_hours_per_day=("hours_worked", "mean"),
            meetings_total=("meetings_count", "sum"),
            weekend_days_worked=("weekend_work", "sum"),
            evening_days_worked=("evening_work", "sum"),
            avg_mood=("mood", "mean"),
            days_logged=("date", "count"),
        )
        .reset_index()
    )
    agg["avg_hours_per_day"] = agg["avg_hours_per_day"].round(2)
    agg["avg_mood"] = agg["avg_mood"].round(2)
    return agg


def detect_patterns(weekly_stats: pd.DataFrame) -> dict:
    patterns = {}
    patterns["n_weeks"] = len(weekly_stats)

    patterns["overload_weeks"] = int((weekly_stats["total_hours"] > 45.0).sum())
    patterns["very_high_weeks"] = int((weekly_stats["total_hours"] > 50.0).sum())
    patterns["weeks_with_weekend_work"] = int((weekly_stats["weekend_days_worked"] > 0).sum())
    patterns["weeks_with_evening_work"] = int((weekly_stats["evening_days_worked"] > 0).sum())

    patterns["avg_mood_overall"] = float(weekly_stats["avg_mood"].mean().round(2))
    patterns["low_mood_weeks"] = int((weekly_stats["avg_mood"] <= 2.5).sum())

    first_mood = weekly_stats["avg_mood"].iloc[0]
    last_mood = weekly_stats["avg_mood"].iloc[-1]
    patterns["mood_trend"] = float((last_mood - first_mood).round(2))

    return patterns


weekly_stats = compute_weekly_stats(work_log)
patterns = detect_patterns(weekly_stats)

display(weekly_stats)
patterns



def estimate_burnout_risk(patterns: dict) -> dict:
    """
    Estimate burnout risk level ("low", "medium", "high") and explain drivers,
    using the transparent rule-based model defined in the notebook.
    """

    overload_weeks = patterns.get("overload_weeks", 0)
    very_high_weeks = patterns.get("very_high_weeks", 0)
    weeks_with_weekend_work = patterns.get("weeks_with_weekend_work", 0)
    weeks_with_evening_work = patterns.get("weeks_with_evening_work", 0)
    avg_mood_overall = patterns.get("avg_mood_overall", 3.5)
    low_mood_weeks = patterns.get("low_mood_weeks", 0)
    mood_trend = patterns.get("mood_trend", 0.0)
    n_weeks = patterns.get("n_weeks", None)

    drivers = []

    # Build driver messages
    if overload_weeks > 0:
        if overload_weeks == 1:
            drivers.append("One week with total hours above 45h.")
        else:
            drivers.append(f"{overload_weeks} weeks with total hours above 45h.")
    if very_high_weeks > 0:
        drivers.append("At least one week with total hours above 50h.")
    if weeks_with_weekend_work > 0:
        drivers.append(f"Weekend work in {weeks_with_weekend_work} week(s).")
    if weeks_with_evening_work > 0:
        drivers.append(f"Evening work (after 19:00) in {weeks_with_evening_work} week(s).")
    if low_mood_weeks > 0:
        drivers.append(f"{low_mood_weeks} week(s) with low average mood (â‰¤ 2.5).")
    if avg_mood_overall < 3.5:
        drivers.append(f"Overall average mood is below 3.5 (current: {avg_mood_overall}).")
    if mood_trend < 0:
        drivers.append(f"Mood trend is decreasing over time (Î” = {round(mood_trend, 2)}).")

    # High risk conditions
    strong_overload = (overload_weeks >= 2) or (very_high_weeks >= 1)
    invasive_work = (weeks_with_weekend_work >= 2) or (weeks_with_evening_work >= 3)
    mood_impact = (low_mood_weeks >= 2) or (avg_mood_overall <= 3.0) or (mood_trend < -0.5)

    # Medium risk conditions
    has_overload_signal = (overload_weeks >= 1) or (very_high_weeks >= 1)
    has_invasive_signal = (weeks_with_weekend_work >= 1) or (weeks_with_evening_work >= 1)
    has_mood_signal = (low_mood_weeks >= 1) or (avg_mood_overall < 3.5) or (mood_trend < 0)

    risk_level = "low"

    # High risk: overload + invasiveness + mood impact
    if strong_overload and invasive_work and mood_impact:
        risk_level = "high"
    # Medium risk: at least one signal, but not enough for high
    elif has_overload_signal or has_invasive_signal or has_mood_signal:
        risk_level = "medium"

    # Positive driver if no issues
    if not drivers and risk_level == "low":
        drivers.append("No strong negative patterns detected. Workload and mood look broadly sustainable.")

    return {
        "risk_level": risk_level,
        "drivers": drivers,
        "patterns": patterns,
    }


risk_info = estimate_burnout_risk(patterns)
risk_info



def generate_weekly_report(weekly_stats: pd.DataFrame, risk_info: dict) -> str:
    risk_level = risk_info["risk_level"]
    patterns = risk_info["patterns"]
    drivers = risk_info["drivers"]

    n_weeks = patterns["n_weeks"]
    avg_hours = weekly_stats["total_hours"].mean().round(1)
    avg_mood = patterns["avg_mood_overall"]

    lines = []

    # Header
    lines.append("ğŸ”¥ Burnout Radar â€“ Weekly Work Patterns Review")
    lines.append("")
    lines.append(f"- Time span analyzed: {n_weeks} week(s)")
    lines.append(f"- Average total hours/week: {avg_hours}")
    lines.append(f"- Overall average mood (1â€“5): {avg_mood}")
    lines.append("")
    lines.append(f"Estimated burnout risk: **{risk_level.upper()}**")
    lines.append("")
    lines.append("What I see in your work patterns:")

    if drivers:
        for d in drivers:
            lines.append(f"- {d}")
    else:
        lines.append("- No strong negative patterns detected.")

    # Suggestions
    lines.append("")
    lines.append("Experiments for next week (non-medical, small and realistic):")

    if risk_level == "low":
        lines.append("- Protect the habits that are working: keep clear boundaries for evenings and weekends.")
        lines.append("- Schedule a short weekly check-in with yourself to notice early if the pattern changes.")
    elif risk_level == "medium":
        lines.append("- Identify 1â€“2 sources of overload (meetings, extra projects) and reduce them by ~10â€“20%.")
        lines.append("- Protect at least one evening as a real recovery space (no work, no email).")
        lines.append("- If possible, talk with a manager or colleague about your current load using these data points.")
    else:  # high
        lines.append("- Your current way of working looks hard to sustain. See if you can urgently reduce or say no to some tasks.")
        lines.append("- Plan real recovery time (a full day off or several evenings without work) and protect it as much as possible.")
        lines.append("- If you feel overwhelmed, consider talking to a professional (doctor, therapist, coach). "
                     "This agent is not a medical tool, but your health matters more than any deadline.")

    return "\n".join(lines)


report = generate_weekly_report(weekly_stats, risk_info)
print(report)



def generate_weekly_report_gemini(weekly_stats: pd.DataFrame, risk_info: dict) -> str:
    """
    Use Gemini to generate an empathetic weekly report from structured data.
    Falls back to the rule-based version if Gemini is not available.
    """
    if not GEMINI_AVAILABLE or model is None:
        return generate_weekly_report(weekly_stats, risk_info)

    patterns = risk_info["patterns"]
    risk_level = risk_info["risk_level"]

    summary = {
        "n_weeks": patterns["n_weeks"],
        "risk_level": risk_level,
        "overload_weeks": patterns["overload_weeks"],
        "very_high_weeks": patterns["very_high_weeks"],
        "weeks_with_weekend_work": patterns["weeks_with_weekend_work"],
        "weeks_with_evening_work": patterns["weeks_with_evening_work"],
        "avg_mood_overall": patterns["avg_mood_overall"],
        "low_mood_weeks": patterns["low_mood_weeks"],
        "mood_trend": patterns["mood_trend"],
    }

    prompt = f"""
You are a burnout-aware work coach.

You receive:
- A summary of weekly work patterns (hours, weekend/evening work, mood).
- A burnout risk level estimated by a transparent rule-based model (low/medium/high).

Your task:
- Write a short weekly reflection for the user.
- Explain in simple language what you see in their work patterns.
- Mention why the risk level ({risk_level}) makes sense.
- Propose 3â€“5 small, realistic experiments for next week to make work more sustainable.
- Be empathetic and non-judgemental.
- Do NOT give medical or clinical advice.

Here is the structured summary (JSON):

{summary}

Write the report in English, in markdown format.
    """

    response = model.generate_content(prompt)
    return response.text.strip()


def plot_weekly_hours(weekly_stats: pd.DataFrame):
    fig, ax = plt.subplots()
    ax.plot(weekly_stats["year_week"], weekly_stats["total_hours"], marker="o")
    ax.axhline(40, linestyle="--")
    ax.set_title("Total hours per week")
    ax.set_xlabel("Year-week")
    ax.set_ylabel("Hours")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_weekly_mood(weekly_stats: pd.DataFrame):
    fig, ax = plt.subplots()
    ax.plot(weekly_stats["year_week"], weekly_stats["avg_mood"], marker="o")
    ax.set_title("Average mood per week (1â€“5)")
    ax.set_xlabel("Year-week")
    ax.set_ylabel("Mood")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def run_burnout_radar_from_df(df: pd.DataFrame) -> dict:
    """
    End-to-end pipeline:
      1) compute weekly stats
      2) detect patterns
      3) estimate risk
      4) generate report
    """
    weekly_stats = compute_weekly_stats(df)
    patterns = detect_patterns(weekly_stats)
    risk_info = estimate_burnout_risk(patterns)
    report = generate_weekly_report(weekly_stats, risk_info)

    return {
        "weekly_stats": weekly_stats,
        "patterns": patterns,
        "risk_info": risk_info,
        "report": report,
    }


results = run_burnout_radar_from_df(work_log)

display(results["weekly_stats"])
plot_weekly_hours(results["weekly_stats"])
plot_weekly_mood(results["weekly_stats"])
print()
print(results["report"])



class DataAgent:
    """Agent that turns raw daily work logs into weekly stats and patterns."""
    def run(self, work_log: pd.DataFrame) -> dict:
        weekly_stats = compute_weekly_stats(work_log)
        patterns = detect_patterns(weekly_stats)
        return {
            "weekly_stats": weekly_stats,
            "patterns": patterns,
        }


class RiskAgent:
    """Agent that estimates burnout risk from patterns."""
    def run(self, patterns: dict) -> dict:
        risk_info = estimate_burnout_risk(patterns)
        return risk_info


class CoachAgent:
    """Agent that generates an empathetic weekly report."""
    def run(self, weekly_stats: pd.DataFrame, risk_info: dict) -> str:
        report = generate_weekly_report(weekly_stats, risk_info)
        return report


class BurnoutRadarWorkflow:
    """
    Simple orchestrator that wires the three agents together:
    DataAgent -> RiskAgent -> CoachAgent.
    """
    def __init__(self):
        self.data_agent = DataAgent()
        self.risk_agent = RiskAgent()
        self.coach_agent = CoachAgent()

    def run(self, work_log: pd.DataFrame) -> dict:
        # 1) DataAgent: raw log -> weekly stats + patterns
        data_result = self.data_agent.run(work_log)
        weekly_stats = data_result["weekly_stats"]
        patterns = data_result["patterns"]

        # 2) RiskAgent: patterns -> risk_info
        risk_info = self.risk_agent.run(patterns)

        # 3) CoachAgent: weekly stats + risk_info -> report
        report = self.coach_agent.run(weekly_stats, risk_info)

        return {
            "weekly_stats": weekly_stats,
            "patterns": patterns,
            "risk_info": risk_info,
            "report": report,
        }


# ---- Demo of the workflow ----

workflow = BurnoutRadarWorkflow()
results = workflow.run(work_log)

display(results["weekly_stats"])
plot_weekly_hours(results["weekly_stats"])
plot_weekly_mood(results["weekly_stats"])
print()
print(results["report"])



from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()
deployment_workflow = BurnoutRadarWorkflow()


class WorkLogItem(BaseModel):
    date: str
    project: str
    hours_worked: float
    meetings_count: int
    context_switches: int
    evening_work: int
    weekend_work: int
    mood: int
    notes: str = ""


@app.post("/burnout-radar")
def analyze_worklog(items: List[WorkLogItem]):
    """
    Example deployment endpoint:
    receives a list of daily entries and returns
    the burnout radar analysis (risk + report).
    """
    df = pd.DataFrame([item.dict() for item in items])
    results = deployment_workflow.run(df)
    return {
        "risk_level": results["risk_info"]["risk_level"],
        "drivers": results["risk_info"]["drivers"],
        "report": results["report"],
    }


